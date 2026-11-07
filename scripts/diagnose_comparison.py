"""Check history use and answer-dependent selection on validation respondents.

This is a descriptive sensitivity check, not another test-set performance claim.
All models stay frozen. Changing initial answers keeps questions, attributes,
profiles, candidate masks and tie-breaking RNG identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from adaptive_survey.cli import _load, _predictor, _protocol, _split
from adaptive_survey.data import ResponseWorld
from adaptive_survey.metrics import choice_metrics
from adaptive_survey.policies import WeightedJSPolicy
from adaptive_survey.protocol import make_episode
from adaptive_survey.runner import episode_seed
from adaptive_survey.schema import CONTEXTS


def diagnose(root: Path, *, limit: int | None = None) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    split = _split(root / "split.json")
    args = argparse.Namespace(data=Path(manifest["config"]["data"]))
    bundle = _load(args, split)
    config = _protocol(split)
    ids = split["uuids"]["validation"]
    if limit is not None:
        if limit < 1:
            raise ValueError("Validation diagnostic limit must be positive")
        ids = ids[:limit]
    result = {"schema_version": 1, "cohort": "validation", "respondents": len(ids),
              "contexts": list(CONTEXTS), "initial_questions": config.initial_per_block * 2,
              "intervention": "Cyclic shift +1 modulo 4 of every initial choice; same questions, profile, pool and RNG",
              "interpretation": "Descriptive validation sensitivity after checkpoint selection; no optimization or test-score feedback",
              "cases": [], "scenarios": {}}
    for case in manifest["cases"]:
        scenario, seed = case["scenario"], case["seed"]
        checkpoint = root / case["path"] / "checkpoint.pt"
        predictor = _predictor("non_graph", str(checkpoint), split, scenario)
        world = ResponseWorld(bundle, scenario)
        changed, losses_empty, losses_initial, score_maxima = [], [], [], []
        for uid in ids:
            for context in CONTEXTS:
                episode = make_episode(uid, context, bundle.tasks, config, track="prediction")
                empty = predictor.reset(bundle.profiles[uid], context)
                actual, altered = empty, empty
                for task in episode.initial:
                    answer = world.reveal(uid, task.task_id, episode.acquisition_ids)
                    actual = predictor.observe(actual, task, answer)
                    altered = predictor.observe(altered, task, (answer + 1) % 4)
                seen = {item.task.task_id for item in actual.history}
                remaining = tuple(task for task in episode.candidates if task.task_id not in seen)
                rng_seed = episode_seed(manifest["config"]["rollout_seed"], uid, context)
                policy = WeightedJSPolicy()
                left = policy.select(predictor, actual, remaining, np.random.default_rng(rng_seed))
                right = policy.select(predictor, altered, remaining, np.random.default_rng(rng_seed))
                changed.append(left.task_id != right.task_id)
                score_maxima.append(max(left.scores.values()))
                # Hidden labels are accessed only after selection and prediction.
                p_empty = predictor.predict(empty, episode.holdout)
                p_initial = predictor.predict(actual, episode.holdout)
                answers = [world.answer(uid, task.task_id) for task in episode.holdout]
                losses_empty.append(choice_metrics(p_empty, answers)["nll"])
                losses_initial.append(choice_metrics(p_initial, answers)["nll"])
        row = {"scenario": scenario, "seed": seed, "episodes": len(changed),
               "changed_first_adaptive_question": sum(changed),
               "changed_fraction": mean(changed), "mean_max_weighted_js": mean(score_maxima),
               "nll_empty": mean(losses_empty), "nll_initial": mean(losses_initial),
               "nll_initial_minus_empty": mean(b - a for a, b in zip(losses_empty, losses_initial))}
        result["cases"].append(row)
        print(json.dumps(row), flush=True)
    for scenario in manifest["config"]["scenarios"]:
        rows = [row for row in result["cases"] if row["scenario"] == scenario]
        result["scenarios"][scenario] = {
            "changed_fraction_mean_over_training_seeds": mean(row["changed_fraction"] for row in rows),
            "nll_empty_mean_over_training_seeds": mean(row["nll_empty"] for row in rows),
            "nll_initial_mean_over_training_seeds": mean(row["nll_initial"] for row in rows),
            "nll_initial_minus_empty_mean_over_training_seeds": mean(row["nll_initial_minus_empty"] for row in rows)}
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new diagnostic output file")
    result = diagnose(args.root, limit=args.limit)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
