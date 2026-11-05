#!/usr/bin/env python3
r"""Aggregate the prespecified rerun across frozen learner training seeds.

Each input directory must contain summary.json, scores.json, and
predictions.jsonl from the common evaluator. Losses and accuracies are averaged
over the trained learners; their probability predictions are never pooled.
The primary contrast is weighted_js minus random, averaged over budgets
3/5/7/9 and the three contexts. Its percentile bootstrap samples UUIDs after
averaging each UUID's paired losses across learner seeds. That interval is
conditional on these frozen learners and this response realization. Sample
standard deviations across training seeds are reported separately.

Example:
    python scripts/summarize_recheck.py \
        --evaluations outputs/recheck/seed2026 outputs/recheck/seed2027 outputs/recheck/seed2028 \
        --seed-labels 2026 2027 2028 --output outputs/recheck/aggregate.json

The default grid is three training seeds, 300 UUIDs, three contexts, four
budgets, and four policies. An explicit --expected-uuids override supports
smoke fixtures; it does not change the reported aggregation formulas.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CONTEXTS = ("transfer", "layover", "stopover")
BUDGETS = (3, 5, 7, 9)
POLICIES = ("fixed", "random", "entropy", "weighted_js")
METRICS = ("nll", "accuracy")
Key = tuple[str, str, int, str]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(values: Any, length: int, label: str) -> tuple[str, ...]:
    if (not isinstance(values, list) or len(values) != length
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != length):
        raise ValueError(f"{label} must contain {length} distinct nonempty strings")
    return tuple(values)


@dataclass(frozen=True)
class Evaluation:
    directory: Path
    label: str
    summary: dict[str, Any]
    scores: dict[Key, tuple[float, float]]
    holdouts: dict[tuple[str, str], tuple[str, ...]]
    queries: dict[Key, tuple[str, ...]]
    uuids: tuple[str, ...]
    source_hashes: dict[str, str]


def _load_evaluation(directory: Path, label: str, reader: str, expected_uuids: int) -> Evaluation:
    summary_path = directory / "summary.json"
    scores_path = directory / "scores.json"
    predictions_path = directory / "predictions.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary.get("policies", ())) != set(POLICIES):
        raise ValueError(f"{directory}: expected the four prespecified policies")
    if set(summary.get("contexts", ())) != set(CONTEXTS):
        raise ValueError(f"{directory}: expected all three contexts")
    protocol = summary.get("protocol", {})
    if (tuple(protocol.get("budgets", ())) != BUDGETS
            or protocol.get("holdout_per_block") != 2
            or protocol.get("initial_per_block") != 1):
        raise ValueError(f"{directory}: protocol differs from the prespecified 4-holdout/2-initial grid")
    scenario = summary.get("scenario")
    if scenario not in ("mnl", "mmnl"):
        raise ValueError(f"{directory}: missing or invalid response scenario")
    for field in ("split_hash", "checkpoint_sha256", "fingerprints", "rollout_seed", "fixed_rankings", "predictor"):
        if field not in summary or summary[field] is None:
            raise ValueError(f"{directory}: missing provenance field {field}")
    fingerprints = summary["fingerprints"]
    if (not isinstance(fingerprints, dict)
            or not {"profiles.csv", "tasks.json", "responses.csv"} <= fingerprints.keys()):
        raise ValueError(f"{directory}: incomplete data fingerprints")
    if set(summary["fixed_rankings"]) != set(CONTEXTS):
        raise ValueError(f"{directory}: fixed ranking must cover each context")
    for context in CONTEXTS:
        _strings(summary["fixed_rankings"][context], 18, f"{directory}: fixed {context} ranking")

    raw_scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if not isinstance(raw_scores, list) or not raw_scores:
        raise ValueError(f"{directory}: scores.json must be a nonempty list")
    scores: dict[Key, tuple[float, float]] = {}
    for row in raw_scores:
        if row.get("scenario") != scenario:
            raise ValueError(f"{directory}: mixed response scenarios in scores")
        if row.get("reader") != reader:
            continue
        key = (row["uuid"], row["context"], row["budget"], row["policy"])
        if key in scores:
            raise ValueError(f"{directory}: duplicate UUID/context/budget/policy score")
        if (not isinstance(key[0], str) or not key[0] or key[1] not in CONTEXTS
                or type(key[2]) is not int or key[2] not in BUDGETS or key[3] not in POLICIES):
            raise ValueError(f"{directory}: score outside the prespecified evaluation grid")
        values = (float(row["nll"]), float(row["accuracy"]))
        if (not np.isfinite(values).all() or values[0] < 0 or not 0 <= values[1] <= 1
                or row.get("n_tasks") != 4):
            raise ValueError(f"{directory}: invalid scores or wrong holdout count")
        scores[key] = values
    uuids = tuple(sorted({key[0] for key in scores}))
    if len(uuids) != expected_uuids or summary.get("n_uuids") != expected_uuids:
        raise ValueError(f"{directory}: expected {expected_uuids} UUIDs, found {len(uuids)}")
    expected = {(uuid, context, budget, policy) for uuid in uuids for context in CONTEXTS
                for budget in BUDGETS for policy in POLICIES}
    if set(scores) != expected:
        raise ValueError(f"{directory}: incomplete UUID/context/budget/policy score grid")

    holdouts: dict[tuple[str, str], tuple[str, ...]] = {}
    queries: dict[Key, tuple[str, ...]] = {}
    for line_number, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{directory}: blank prediction record at line {line_number}")
        row = json.loads(line)
        if row.get("scenario") != scenario:
            raise ValueError(f"{directory}: mixed response scenarios in predictions")
        key = (row["uuid"], row["context"], row["budget"], row["policy"])
        if key not in expected or key in queries:
            raise ValueError(f"{directory}: duplicate or unexpected prediction episode")
        held = _strings(row["holdout_task_ids"], 4, f"{directory}: holdout")
        queried = _strings(row["queried_task_ids"], key[2], f"{directory}: query history")
        if set(held) & set(queried):
            raise ValueError(f"{directory}: a holdout task was queried")
        pair = key[:2]
        if pair in holdouts and holdouts[pair] != held:
            raise ValueError(f"{directory}: holdouts differ across policies or budgets")
        holdouts[pair] = held
        queries[key] = queried
        probability_field = "probabilities" if reader == "native" else "common_probabilities"
        p = np.asarray(row.get(probability_field), dtype=float)
        if (p.shape != (4, 4) or not np.isfinite(p).all() or (p < 0).any()
                or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError(f"{directory}: invalid {reader} holdout probabilities")
    if set(queries) != expected:
        raise ValueError(f"{directory}: predictions and scores have different episode grids")
    for uuid in uuids:
        for context in CONTEXTS:
            initial = queries[uuid, context, BUDGETS[0], POLICIES[0]][:2]
            for policy in POLICIES:
                final = queries[uuid, context, BUDGETS[-1], policy]
                for budget in BUDGETS:
                    queried = queries[uuid, context, budget, policy]
                    if queried[:2] != initial or queried != final[:budget]:
                        raise ValueError(f"{directory}: initial tasks or budget histories are inconsistent")
    source_hashes = {name: _sha256_file(directory / name)
                     for name in ("summary.json", "scores.json", "predictions.jsonl")}
    return Evaluation(directory, label, summary, scores, holdouts, queries, uuids, source_hashes)


def _seed_stat(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std_across_training_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
            "seed_values": values.tolist()}


def _paired_contrast(
    tensor: np.ndarray, policy_a: str, policy_b: str, metric: str,
    budget: int | None, bootstrap_indices: np.ndarray, confidence: float,
) -> dict[str, Any]:
    # tensor axes: training seed, UUID, context, budget, policy, metric.
    a, b, m = POLICIES.index(policy_a), POLICIES.index(policy_b), METRICS.index(metric)
    paired = tensor[..., a, m] - tensor[..., b, m]
    if budget is not None:
        paired = paired[:, :, :, BUDGETS.index(budget):BUDGETS.index(budget) + 1]
    per_seed = paired.mean(axis=(1, 2, 3))
    per_uuid = paired.mean(axis=(0, 2, 3))
    bootstrap_means = per_uuid[bootstrap_indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "policy_a": policy_a, "policy_b": policy_b, "metric": metric,
        "budget": budget, "budgets_averaged": list(BUDGETS) if budget is None else [budget],
        "difference_a_minus_b": float(per_uuid.mean()),
        "ci_lower": float(np.quantile(bootstrap_means, alpha)),
        "ci_upper": float(np.quantile(bootstrap_means, 1 - alpha)),
        "confidence": confidence,
        "favors_a_when": "negative" if metric == "nll" else "positive",
        "per_seed_differences": per_seed.tolist(),
        "std_of_differences_across_training_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
        "n_uuid_clusters": len(per_uuid), "n_training_seeds": tensor.shape[0],
        "n_bootstrap": len(bootstrap_indices),
        "interval_scope": "UUID sampling conditional on the average loss of these frozen learners and one response realization",
        "analysis_role": "primary" if (policy_a, policy_b, metric, budget) ==
                         ("weighted_js", "random", "nll", None) else "secondary_descriptive",
    }


def summarize_evaluations(
    evaluations: Sequence[str | Path], *, seed_labels: Sequence[str] | None = None,
    expected_uuids: int = 300, expected_seeds: int = 3, reader: str = "native",
    n_bootstrap: int = 5000, bootstrap_seed: int = 2026, confidence: float = .95,
) -> dict[str, Any]:
    if (expected_uuids < 1 or expected_seeds < 1 or n_bootstrap < 1
            or not 0 < confidence < 1 or reader not in ("native", "common")):
        raise ValueError("Invalid aggregation configuration")
    directories = [Path(path).resolve(strict=True) for path in evaluations]
    if len(directories) != expected_seeds or len(set(directories)) != len(directories):
        raise ValueError(f"Need {expected_seeds} distinct evaluation directories")
    if seed_labels is not None:
        labels = [str(label) for label in seed_labels]
    else:
        labels = [p.name for p in directories]
        if len(set(labels)) != len(labels):
            labels = [p.parent.name for p in directories]
        if len(set(labels)) != len(labels):
            labels = [str(p) for p in directories]
    if len(labels) != len(directories) or len(set(labels)) != len(labels):
        raise ValueError("Training seed labels must be unique and match the evaluation directories")
    runs = [_load_evaluation(path, str(label), reader, expected_uuids)
            for path, label in zip(directories, labels)]
    reference = runs[0]
    if len({run.summary["checkpoint_sha256"] for run in runs}) != len(runs):
        raise ValueError("Each training-seed run must use a distinct learner checkpoint")
    shared_fields = ("scenario", "split_hash", "fingerprints", "protocol", "rollout_seed", "fixed_rankings", "predictor")
    for run in runs[1:]:
        if run.uuids != reference.uuids or run.holdouts != reference.holdouts:
            raise ValueError("Training-seed runs have different UUIDs, contexts, or holdout tasks")
        for field in shared_fields:
            if _canonical(run.summary[field]) != _canonical(reference.summary[field]):
                raise ValueError(f"Training-seed runs differ in {field}; do not mix response worlds or protocols")
        for key in reference.queries:
            if key[3] in ("fixed", "random") and run.queries[key] != reference.queries[key]:
                raise ValueError("Fixed/random histories must remain identical across learner training seeds")
    tensor = np.asarray([[[[[run.scores[uuid, context, budget, policy]
                             for policy in POLICIES] for budget in BUDGETS]
                           for context in CONTEXTS] for uuid in reference.uuids] for run in runs])
    expected_shape = (expected_seeds, expected_uuids, len(CONTEXTS), len(BUDGETS), len(POLICIES), len(METRICS))
    if tensor.shape != expected_shape or not np.isfinite(tensor).all():
        raise ValueError("Internal aggregation grid is invalid")
    # Balanced cells make these equal to context-macro then budget-macro means.
    seed_budget_average = tensor.mean(axis=(1, 2, 3))
    seed_curves = tensor.mean(axis=(1, 2))
    seed_context_curves = tensor.mean(axis=1)
    per_seed = []
    for si, run in enumerate(runs):
        per_seed.append({
            "seed_label": run.label, "evaluation_directory": str(run.directory),
            "checkpoint_sha256": run.summary["checkpoint_sha256"],
            "source_sha256": run.source_hashes,
            "budget_average": [{"policy": policy, **{metric: float(seed_budget_average[si, pi, mi])
                                 for mi, metric in enumerate(METRICS)}} for pi, policy in enumerate(POLICIES)],
            "budget_curves": [{"policy": policy, "budget": budget,
                               **{metric: float(seed_curves[si, bi, pi, mi]) for mi, metric in enumerate(METRICS)}}
                              for pi, policy in enumerate(POLICIES) for bi, budget in enumerate(BUDGETS)],
        })
    across_seed = {
        "budget_average": [{"policy": policy, **{metric: _seed_stat(seed_budget_average[:, pi, mi])
                             for mi, metric in enumerate(METRICS)}} for pi, policy in enumerate(POLICIES)],
        "budget_curves": [{"policy": policy, "budget": budget,
                           **{metric: _seed_stat(seed_curves[:, bi, pi, mi]) for mi, metric in enumerate(METRICS)}}
                          for pi, policy in enumerate(POLICIES) for bi, budget in enumerate(BUDGETS)],
        "context_budget_curves": [{"policy": policy, "context": context, "budget": budget,
                                   **{metric: _seed_stat(seed_context_curves[:, ci, bi, pi, mi])
                                      for mi, metric in enumerate(METRICS)}}
                                  for pi, policy in enumerate(POLICIES) for ci, context in enumerate(CONTEXTS)
                                  for bi, budget in enumerate(BUDGETS)],
    }
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(0, expected_uuids, size=(n_bootstrap, expected_uuids), dtype=np.int32)
    contrasts = [("weighted_js", "random"), ("weighted_js", "fixed"), ("fixed", "random"), ("entropy", "random")]
    comparisons = [_paired_contrast(tensor, a, b, metric, budget, bootstrap_indices, confidence)
                   for a, b in contrasts for metric in METRICS for budget in (None, *BUDGETS)]
    return {
        "schema_version": 1,
        "scenario": reference.summary["scenario"], "reader": reader,
        "n_training_seeds": len(runs), "seed_labels": labels,
        "n_uuids": expected_uuids, "contexts": list(CONTEXTS), "budgets": list(BUDGETS),
        "policies": list(POLICIES), "n_holdout_tasks_per_episode": 4,
        "n_holdout_answers_per_policy_budget": expected_uuids * len(CONTEXTS) * 4,
        "n_scores_per_seed": expected_uuids * len(CONTEXTS) * len(BUDGETS) * len(POLICIES),
        "bootstrap_seed": bootstrap_seed, "n_bootstrap": n_bootstrap,
        "estimand": "Equal mean of per-learner losses/accuracies across frozen training seeds, UUIDs, three contexts, and prespecified budgets; predictions are not pooled across training seeds",
        "uncertainty": "Percentile paired UUID bootstrap after averaging each UUID across frozen learner seeds; training-seed sample SD (ddof=1) is reported separately and is not the bootstrap interval",
        "primary_hypothesis": "weighted_js minus random NLL averaged over budgets 3/5/7/9 and three contexts",
        "secondary_intervals": "Other contrasts, accuracy, and individual-budget intervals are descriptive and not adjusted for multiple comparisons",
        "provenance": {"split_hash": reference.summary["split_hash"],
                       "fingerprints": reference.summary["fingerprints"],
                       "protocol": reference.summary["protocol"],
                       "rollout_seed": reference.summary["rollout_seed"],
                       "fixed_rankings_sha256": _digest(reference.summary["fixed_rankings"]),
                       "verified": ["complete score/prediction grid", "identical UUIDs/holdouts/world/protocol across training seeds",
                                    "same initial tasks across policies", "nested budget histories",
                                    "identical fixed ranking and fixed/random histories across learner seeds",
                                    "distinct frozen learner checkpoints"]},
        "per_seed": per_seed, "across_seed": across_seed,
        "paired_comparisons": comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluations", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-labels", nargs="+")
    parser.add_argument("--expected-uuids", type=int, default=300)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--reader", choices=("native", "common"), default="native")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--confidence", type=float, default=.95)
    args = parser.parse_args(argv)
    try:
        report = summarize_evaluations(args.evaluations, seed_labels=args.seed_labels,
            expected_uuids=args.expected_uuids, expected_seeds=args.expected_seeds,
            reader=args.reader, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    if any(output == (Path(path).resolve() / name) for path in args.evaluations
           for name in ("summary.json", "scores.json", "predictions.jsonl", "traces.jsonl")):
        parser.error("Output must not overwrite evaluator input files")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    primary = next(r for r in report["paired_comparisons"] if r["analysis_role"] == "primary")
    print(json.dumps({"output": str(output), "n_uuids": report["n_uuids"],
                      "n_training_seeds": report["n_training_seeds"], "primary": primary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
