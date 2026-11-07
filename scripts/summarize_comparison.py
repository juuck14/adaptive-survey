"""Aggregate saved two-policy runs without exposing respondent-level records."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from adaptive_survey.metrics import ScoredCheckpoint, paired_uuid_bootstrap, summarize_scores


POLICIES = ("fixed", "weighted_js")
METRICS = ("nll", "accuracy", "brier")


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values) -> dict:
    values = [float(x) for x in values]
    if not values or not np.isfinite(values).all():
        raise ValueError("Need nonempty finite replicate values")
    return {"mean": mean(values), "seed_sd": stdev(values) if len(values) > 1 else None,
            "per_seed": values}


def score_identity(row: ScoredCheckpoint) -> tuple:
    return (row.scenario, row.policy, row.reader, row.uuid, row.context, row.budget)


def average_training_seeds(runs: list[list[ScoredCheckpoint]]) -> list[ScoredCheckpoint]:
    """Each UUID/context/budget cell appears once, averaged over training seeds."""
    if not runs:
        raise ValueError("No training runs")
    maps = []
    for rows in runs:
        mapping = {score_identity(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate score cell")
        maps.append(mapping)
    if any(mapping.keys() != maps[0].keys() for mapping in maps):
        raise ValueError("Training seeds have different respondent/context/budget grids")
    result = []
    for key, first in sorted(maps[0].items()):
        cells = [mapping[key] for mapping in maps]
        if len({row.n_tasks for row in cells}) != 1:
            raise ValueError("Protected task counts differ between seeds")
        result.append(replace(first, **{metric: mean(getattr(row, metric) for row in cells)
                                       for metric in METRICS}))
    return result


def check_paths(run: Path, summary: dict, expected_ids: set[str]) -> dict:
    """Check paired initial responses and complete, unique two-track rollouts."""
    paths, steps = {}, 0
    for line in (run / "evaluation/traces.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["track"], row["uuid"], row["context"], row["policy"])
        if (row["uuid"] not in expected_ids or row["context"] not in summary["contexts"]
                or row["policy"] not in POLICIES or row["scenario"] != summary["scenario"]):
            raise ValueError("A trace lies outside its declared evaluation grid")
        seq = paths.setdefault(key, [])
        if row["step"] != len(seq) + 1 or row["task_id"] in {x[0] for x in seq}:
            raise ValueError("Repeated task or invalid step sequence")
        seq.append((row["task_id"], row["choice"], row["initial"]))
        steps += 1
    diagnostics = {}
    initial_count = summary["protocol"]["initial_per_block"] * 2
    expected_keys = set()
    for track, settings in summary["tracks"].items():
        if not settings["budgets"]:
            continue
        different_order, different_next = [], []
        subset_difference = {budget: [] for budget in settings["budgets"]}
        for uid in sorted(expected_ids):
            for context in summary["contexts"]:
                expected_keys.update((track, uid, context, policy) for policy in POLICIES)
                left, right = [paths[(track, uid, context, policy)] for policy in POLICIES]
                if any(len(seq) != max(settings["budgets"]) for seq in (left, right)):
                    raise ValueError("Incomplete rollout")
                if left[:initial_count] != right[:initial_count]:
                    raise ValueError("Initial questions or actual responses differ between policies")
                if any([x[2] for x in seq] != [i < initial_count for i in range(len(seq))]
                       for seq in (left, right)):
                    raise ValueError("Initial-question count differs from the declared protocol")
                different_order.append([x[0] for x in left] != [x[0] for x in right])
                if len(left) > initial_count:
                    different_next.append(left[initial_count][0] != right[initial_count][0])
                for budget in settings["budgets"]:
                    subset_difference[budget].append({x[0] for x in left[:budget]} != {x[0] for x in right[:budget]})
                if track == "preference" and max(settings["budgets"]) == 18:
                    if {x[0] for x in left} != {x[0] for x in right}:
                        raise ValueError("Full-budget methods collected different task sets")
        diagnostics[track] = {"episodes": len(different_order),
                              "different_order_fraction": mean(different_order),
                              "different_first_adaptive_question_fraction": mean(different_next) if different_next else None,
                              "different_subset_fraction": {str(b): mean(v) for b, v in subset_difference.items()}}
    if paths.keys() != expected_keys:
        raise ValueError("The rollout grid is incomplete or contains extra episodes")
    return {"revealed_steps": steps, "initial_questions": initial_count, "paths": diagnostics}


def summarize(root: Path, *, n_bootstrap: int = 5000, seed: int = 2026) -> dict:
    manifest, status, split = [read(root / name) for name in ("manifest.json", "status.json", "split.json")]
    if status["errors"] or len(status["completed"]) != len(manifest["cases"]):
        raise ValueError("The planned comparison has unfinished or failed cases")
    if digest(root / "split.json") != manifest["split_sha256"]:
        raise ValueError("Prepared split changed after launch")
    expected_ids = set(split["uuids"]["test"])
    output = {"schema_version": 1, "code_revision": manifest["code_revision"],
              "source_sha256": manifest["source_sha256"], "counts": manifest["counts"],
              "fingerprints": manifest["fingerprints"], "protocol": split["protocol"],
              "training_seeds": sorted(manifest["config"]["seeds"]), "scenarios": {},
              "bootstrap": {"unit": "UUID", "replicates": n_bootstrap, "seed": seed,
                            "training_seeds": "Average each UUID cell over training seeds before resampling; report seed SD separately",
                            "scope": "Conditional on the saved response realization and trained models"}}
    for scenario in manifest["config"]["scenarios"]:
        cases = sorted((case for case in manifest["cases"] if case["scenario"] == scenario), key=lambda case: case["seed"])
        if [case["seed"] for case in cases] != sorted(manifest["config"]["seeds"]):
            raise ValueError("Incomplete scenario/training-seed grid")
        summaries, populations, score_runs, case_reports = [], [], [], []
        for case in cases:
            run = root / case["path"]
            summary = read(run / "evaluation/summary.json")
            population = read(run / "population.json")
            training = read(run / "checkpoint.pt.training.json")["metadata"]
            if (summary["schema_version"] != 2 or summary["scenario"] != scenario
                    or summary["policies"] != list(POLICIES)
                    or summary["fingerprints"] != manifest["fingerprints"]
                    or summary["protocol"] != split["protocol"]
                    or summary["n_uuids"] != len(expected_ids)
                    or training["training_config"]["seed"] != case["seed"]
                    or training["scenario"] != scenario):
                raise ValueError("Case configuration differs from the frozen experiment")
            if digest(run / "evaluation/fixed-plans.json") != summary["fixed_plans_sha256"]:
                raise ValueError("Fixed plans changed after evaluation")
            if summaries and any(summary[key] != summaries[0][key] for key in
                                 ("tracks", "contexts", "split_hash", "fixed_plans_sha256", "fixed_design")):
                raise ValueError("A controlled evaluation condition differs between training seeds")
            scores = [ScoredCheckpoint(**row) for row in read(run / "evaluation/scores.json")]
            expected_grid = {(scenario, policy, "native", uid, context, budget)
                             for policy in POLICIES for uid in expected_ids for context in summary["contexts"]
                             for budget in summary["tracks"]["prediction"]["budgets"]}
            if len(scores) != len(expected_grid) or {score_identity(row) for row in scores} != expected_grid:
                raise ValueError("Prediction scores do not form a complete paired grid")
            if any(row.n_tasks != summary["tracks"]["prediction"]["holdout_count"] for row in scores):
                raise ValueError("Incorrect protected task count")
            integrity = check_paths(run, summary, expected_ids)
            expected_population = {(policy, budget) for policy in POLICIES
                                   for budget in summary["tracks"]["preference"]["budgets"]}
            if (population["track"] != "preference" or
                    len(population["results"]) != len(expected_population) or
                    {(r["policy"], r["budget"]) for r in population["results"]} != expected_population):
                raise ValueError("Incomplete population-estimation grid")
            if any(abs(r["agreement"]["rmse"]) > 1e-12 for r in population["results"] if r["budget"] == 18):
                raise ValueError("Full-budget population fit differs from its reference")
            summaries.append(summary)
            populations.append(population)
            score_runs.append(scores)
            case_reports.append({"seed": case["seed"], "checkpoint_sha256": summary["checkpoint_sha256"],
                                 "fixed_plans_sha256": summary["fixed_plans_sha256"],
                                 "best_epochs": training["best_epochs"],
                                 "best_optimizer_steps": training["best_optimizer_steps"],
                                 "optimizer_steps_run": training["optimizer_steps_run"],
                                 "validation_bank_hash": training["validation_bank_hash"],
                                 "integrity": integrity, "timing": read(run / "timing.json"),
                                 "rollout_timing": summary["timing"],
                                 "fixed_planning_timing": summary["fixed_planning_timing"],
                                 "evaluation_summary_sha256": digest(run / "evaluation/summary.json"),
                                 "population_sha256": digest(run / "population.json")})
        averaged = average_training_seeds(score_runs)
        per_seed = [summarize_scores(rows) for rows in score_runs]
        prediction = []
        for budget in summaries[0]["tracks"]["prediction"]["budgets"]:
            policy_stats = {}
            for policy in POLICIES:
                values = [next(row for row in rows if row["policy"] == policy and row["budget"] == budget
                               and row["context"] == "macro") for rows in per_seed]
                policy_stats[policy] = {metric: statistics(row[metric] for row in values) for metric in METRICS}
            difference = paired_uuid_bootstrap(averaged, "weighted_js", "fixed", reader="native", scenario=scenario,
                                               budget=budget, n_bootstrap=n_bootstrap, seed=seed)
            prediction.append({"budget": budget, **policy_stats, "paired_nll_difference": difference})
        overall = paired_uuid_bootstrap(averaged, "weighted_js", "fixed", reader="native", scenario=scenario,
                                       n_bootstrap=n_bootstrap, seed=seed)
        averages = {policy: statistics(next(row["mean_nll"] for row in summary["budget_average"]
                                           if row["policy"] == policy and row["reader"] == "native")
                                      for summary in summaries) for policy in POLICIES}
        preference = []
        for budget in summaries[0]["tracks"]["preference"]["budgets"]:
            policy_stats = {}
            for policy in POLICIES:
                cells = [next(row for row in value["results"] if row["policy"] == policy and row["budget"] == budget)
                         for value in populations]
                policy_stats[policy] = {metric: statistics(row["agreement"][metric] for row in cells)
                                        for metric in ("rmse", "l2_difference", "relative_l2", "sign_agreement")}
            preference.append({"budget": budget, **policy_stats})
        output["scenarios"][scenario] = {"tracks": summaries[0]["tracks"], "prediction": prediction,
                                          "budget_average_nll": averages, "paired_nll_difference": overall,
                                          "preference": preference, "cases": case_reports,
                                          "fixed_design": summaries[0]["fixed_design"],
                                          "population_interpretation": "Common pooled-MNL agreement with the same full-survey cohort; not latent MMNL recovery"}
    return output


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    result = summarize(args.root, n_bootstrap=args.bootstrap, seed=args.seed)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "scenarios": list(result["scenarios"]), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
