#!/usr/bin/env python3
"""Diagnose history use in frozen recheck learners, using saved B9 rollouts.

H9 predictions come from weighted_js evaluation/predictions.jsonl. H0 uses
the same frozen checkpoint and exactly those four holdout tasks after reset.
Labels enter evaluator scores only. Output contains aggregates, never UUIDs,
individual responses, profile attributes, or probability rows.

The paired interval first averages each UUID over the frozen training seeds
and three contexts, then bootstraps UUID clusters. Training-seed sample SD is
reported separately. It is not additional respondent sample size.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adaptive_survey.data import load_bundle
from adaptive_survey.metrics import choice_metrics
from adaptive_survey.neural import NeuralPredictor
from adaptive_survey.policies import WeightedJSPolicy
from adaptive_survey.protocol import ProtocolConfig, make_episode
from adaptive_survey.runner import episode_seed
from adaptive_survey.schema import CONTEXTS, checked_probabilities


POLICIES = ("weighted_js", "random")
METRIC_NAMES = ("h0_nll", "h9_nll", "h0_accuracy", "h9_accuracy", "query_tv", "argmax_change")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Input JSONL contains a blank record")
            yield json.loads(line)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite aggregate vector")
    return {"n_values": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max())}


def _seed_stat(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "per_training_seed": x.tolist(),
            "std_across_training_seeds": float(x.std(ddof=1)) if len(x) > 1 else None}


def _strings(value: Any, n: int, label: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or len(value) != n or len(set(value)) != n
            or any(not isinstance(item, str) or not item for item in value)):
        raise ValueError(f"Invalid {label} task identifiers")
    return tuple(value)


def _split(path: Path, expected_uuids: int | None) -> tuple[dict, ProtocolConfig, tuple[str, ...]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    groups = split.get("uuids", {})
    if split.get("schema_version") != 1 or set(groups) != {"train", "validation", "test"}:
        raise ValueError("Invalid split manifest")
    all_ids = [uuid for group in groups.values() for uuid in group]
    if (not all_ids or len(set(all_ids)) != len(all_ids)
            or any(not isinstance(uuid, str) or not uuid for uuid in all_ids)):
        raise ValueError("Split UUID cohorts must be unique and disjoint")
    ids = tuple(sorted(groups["test"]))
    if not ids or (expected_uuids is not None and len(ids) != expected_uuids):
        raise ValueError("The test cohort does not have the expected respondent count")
    fields = dict(split["protocol"])
    fields["budgets"] = tuple(fields["budgets"])
    protocol = ProtocolConfig(**fields)
    if protocol.holdout_per_block != 2 or protocol.initial_per_block != 1 or 9 not in protocol.budgets:
        raise ValueError("Diagnostics require four holdout tasks, two initial tasks and budget nine")
    return split, protocol, ids


def _load_run(run: Path, split: dict, protocol: ProtocolConfig, ids: tuple[str, ...], bundle):
    evaluation = run / "evaluation"
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    checkpoint = run / "checkpoint.pt"
    checkpoint_hash = _file_hash(checkpoint)
    split_hash = hashlib.sha256(_canonical(split["uuids"]).encode()).hexdigest()
    scenario = summary.get("scenario")
    if scenario not in ("mnl", "mmnl"):
        raise ValueError("Evaluation has no valid response scenario")
    if (summary.get("checkpoint_sha256") != checkpoint_hash
            or summary.get("split_hash") != split_hash
            or summary.get("fingerprints") != split["fingerprints"]
            or _canonical(summary.get("protocol")) != _canonical(asdict(protocol))
            or summary.get("n_uuids") != len(ids)
            or set(summary.get("contexts", ())) != set(CONTEXTS)
            or not set(POLICIES) <= set(summary.get("policies", ()))):
        raise ValueError("Evaluation provenance, protocol or cohort differs from the requested inputs")
    predictor = NeuralPredictor.load(checkpoint)
    metadata = predictor.metadata
    if (metadata.get("scenario") != scenario or metadata.get("split_hash") != split_hash
            or metadata.get("fingerprints") != split["fingerprints"]
            or any(metadata.get("protocol", {}).get(key) != getattr(protocol, key)
                   for key in ("seed", "holdout_per_block", "initial_per_block"))):
        raise ValueError("Frozen checkpoint metadata differs from the evaluation inputs")
    seed = metadata.get("training_config", {}).get("seed")
    if type(seed) is not int or type(summary.get("rollout_seed")) is not int:
        raise ValueError("Training and rollout seeds must be recorded")
    expected = {(uuid, context) for uuid in ids for context in CONTEXTS}
    predictions = {policy: {} for policy in POLICIES}
    for row in _read_jsonl(evaluation / "predictions.jsonl"):
        if row.get("scenario") != scenario:
            raise ValueError("Prediction records mix response scenarios")
        if row.get("policy") not in POLICIES or row.get("budget") != 9:
            continue
        policy, key = row["policy"], (row["uuid"], row["context"])
        if key not in expected or key in predictions[policy]:
            raise ValueError("Duplicate or unexpected budget-nine prediction episode")
        held = _strings(row.get("holdout_task_ids"), 4, "holdout")
        queried = _strings(row.get("queried_task_ids"), 9, "queried")
        episode = make_episode(key[0], key[1], bundle.tasks, protocol)
        if (set(held) != {task.task_id for task in episode.holdout}
                or not set(queried) <= episode.acquisition_ids
                or queried[:2] != tuple(task.task_id for task in episode.initial)):
            raise ValueError("Saved predictions do not follow the requested holdout/initial protocol")
        probabilities = checked_probabilities(np.asarray(row.get("probabilities")), 4)
        predictions[policy][key] = (held, queried, probabilities)
    if any(set(values) != expected for values in predictions.values()):
        raise ValueError("Incomplete test UUID/context budget-nine prediction grid")
    traces = {policy: defaultdict(dict) for policy in POLICIES}
    for row in _read_jsonl(evaluation / "traces.jsonl"):
        if row.get("scenario") != scenario:
            raise ValueError("Trace records mix response scenarios")
        if row.get("policy") not in POLICIES:
            continue
        step = row.get("step")
        if type(step) is not int or not 1 <= step <= 9:
            raise ValueError("Trace is outside the prespecified nine-step rollout")
        policy, key = row["policy"], (row["uuid"], row["context"])
        if key not in expected or step in traces[policy][key]:
            raise ValueError("Duplicate or unexpected trace episode/step")
        traces[policy][key][step] = row
    score_values, scores_by_step, order_diff, set_diff, set_distance = [], defaultdict(list), [], [], []
    h2_states = {}
    for key in sorted(expected):
        js_held, js_history, _ = predictions["weighted_js"][key]
        random_held, random_history, _ = predictions["random"][key]
        if set(js_held) != set(random_held):
            raise ValueError("Weighted JS and random use different holdout tasks")
        order_diff.append(js_history != random_history)
        set_diff.append(set(js_history) != set(random_history))
        set_distance.append(len(set(js_history) ^ set(random_history)))
        for policy in POLICIES:
            records = traces[policy].get(key, {})
            if set(records) != set(range(1, 10)):
                raise ValueError("Incomplete test UUID/context trace grid")
            state = predictor.reset(bundle.profiles[key[0]], key[1])
            episode = make_episode(key[0], key[1], bundle.tasks, protocol)
            for step in range(1, 10):
                row = records[step]
                remaining = episode.acquisition_ids - {item.task.task_id for item in state.history}
                candidate_ids = row.get("candidate_task_ids")
                if (not isinstance(candidate_ids, list) or len(set(candidate_ids)) != len(candidate_ids)
                        or set(candidate_ids) != remaining or row.get("task_id") not in remaining
                        or row.get("initial") != (step <= 2)):
                    raise ValueError("Trace candidates or selected task do not match the public history")
                choice = row.get("choice")
                if type(choice) is not int or not 0 <= choice < 4:
                    raise ValueError("Trace contains an invalid revealed choice")
                if policy == "weighted_js" and step > 2:
                    scores = row.get("scores")
                    if not isinstance(scores, dict) or set(scores) != remaining:
                        raise ValueError("JS trace must score every remaining candidate")
                    values = np.asarray(list(scores.values()), dtype=float)
                    if (not np.isfinite(values).all() or (values < -1e-10).any()
                            or (values > np.log(4) + 1e-10).any()):
                        raise ValueError("Trace JS scores are outside their mathematical bounds")
                    score_values.extend(values.tolist())
                    scores_by_step[step].extend(values.tolist())
                state = predictor.observe(state, bundle.tasks[row["task_id"]], choice)
                if policy == "weighted_js" and step == 2:
                    h2_states[key] = state
            if tuple(item.task.task_id for item in state.history) != predictions[policy][key][1]:
                raise ValueError("Trace replay and saved budget-nine query history differ")
    source_hashes = {name: _file_hash(evaluation / name)
                     for name in ("summary.json", "predictions.jsonl", "traces.jsonl")}
    diagnostics = {"js_scores": {**_stats(score_values),
                                "by_step": [{"step": step, **_stats(scores_by_step[step])}
                                            for step in range(3, 10)]},
                   "js_vs_random_history": {
                       "budget": 9, "order_diff_fraction": float(np.mean(order_diff)),
                       "set_diff_fraction": float(np.mean(set_diff)),
                       "mean_symmetric_set_difference_count": float(np.mean(set_distance))}}
    return predictor, summary, predictions, traces, h2_states, diagnostics, source_hashes


def _training_summary(predictor: NeuralPredictor) -> dict[str, Any]:
    metadata = predictor.metadata
    n_models = len(predictor.models)
    result = {"ensemble_size": n_models, "config": metadata.get("training_config", {})}
    for key in ("best_epochs", "best_optimizer_steps", "optimizer_steps_run"):
        values = metadata.get(key)
        if (not isinstance(values, list) or len(values) != n_models
                or any(type(value) is not int or value < 0 for value in values)):
            raise ValueError("Checkpoint lacks per-member epoch/optimizer-step accounting")
        result[key] = values
        result[key + "_summary"] = _stats(values)
    if any(best > total for best, total in zip(result["best_optimizer_steps"], result["optimizer_steps_run"])):
        raise ValueError("Best optimizer step exceeds completed training steps")
    for key in ("optimizer_steps_per_epoch", "max_optimizer_steps_per_member",
                "validation_respondents", "validation_bank_respondents", "validation_bank_episodes",
                "validation_bank_respondent_context_pairs", "validation_bank_context_counts",
                "validation_bank_context_respondents", "validation_bank_hash", "validation_bank_sampling"):
        result[key] = metadata.get(key)
    return result


def _answer_sensitivity(predictor, summary, traces, states, bundle, protocol,
                        keys, n_episodes: int, sample_seed: int) -> dict[str, Any]:
    selected = sorted(keys, key=lambda key: hashlib.sha256(
        _canonical([sample_seed, key[0], key[1]]).encode()).hexdigest())[:n_episodes]
    policy = WeightedJSPolicy()
    changed, score_changes, score_tv = [], [], []
    for key in selected:
        state = states[key]
        episode = make_episode(key[0], key[1], bundle.tasks, protocol)
        seen = {item.task.task_id for item in state.history}
        candidates = tuple(task for task in episode.candidates if task.task_id not in seen)
        rng_seed = episode_seed(summary["rollout_seed"], key[0], key[1])
        actual = policy.select(predictor, state, candidates, np.random.default_rng(rng_seed))
        if actual.task_id != traces["weighted_js"][key][3]["task_id"]:
            raise ValueError("Frozen checkpoint could not reproduce the saved step-three selection")
        counterfactual = predictor.reset(state.profile, state.context)
        for item in state.history:
            counterfactual = predictor.observe(counterfactual, item.task, (item.choice + 1) % 4)
        flipped = policy.select(predictor, counterfactual, candidates, np.random.default_rng(rng_seed))
        changed.append(actual.task_id != flipped.task_id)
        a = np.asarray([actual.scores[task.task_id] for task in candidates])
        b = np.asarray([flipped.scores[task.task_id] for task in candidates])
        score_changes.append(float(np.max(np.abs(a - b))))
        # This is score-vector change, not a probability distribution TV.
        score_tv.append(float(np.mean(np.abs(a - b))))
    return {"n_episodes": len(selected), "n_available_episodes": len(keys),
            "sampling": "smallest SHA256 of sample-seed, UUID, context; same sample across training seeds",
            "sample_seed": sample_seed, "intervention": "both revealed initial choices changed to (choice+1)%4",
            "held_fixed": ["profile", "context", "two initial tasks", "remaining candidates", "policy RNG seed", "frozen learner"],
            "actual_selection_reproduced_fraction": 1.0,
            "next_task_changed_fraction": float(np.mean(changed)),
            "maximum_absolute_score_change_per_episode": _stats(score_changes),
            "mean_absolute_score_change_per_episode": _stats(score_tv),
            "interpretation": "Computational response intervention: policy reacts to changed observed answers; this does not establish true information gain or policy benefit"}


def _paired_interval(differences: np.ndarray, indices: np.ndarray, confidence: float) -> dict[str, Any]:
    # Axes: frozen training seed, UUID, context. Do not count seeds/contexts
    # as independent respondents or mix probabilities across trained learners.
    per_seed = differences.mean(axis=(1, 2))
    per_uuid = differences.mean(axis=(0, 2))
    sampled = per_uuid[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {"difference_H9_minus_H0": float(per_uuid.mean()),
            "ci_lower": float(np.quantile(sampled, alpha)), "ci_upper": float(np.quantile(sampled, 1 - alpha)),
            "confidence": confidence, "n_bootstrap": len(indices), "n_uuid_clusters": len(per_uuid),
            "per_training_seed": per_seed.tolist(),
            "std_across_training_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
            "interval_scope": "UUID bootstrap after seed/context averaging; conditional on these frozen learners and one response realization"}


def diagnose_recheck(data: str | Path, split_path: str | Path, runs: Sequence[str | Path], *,
                     expected_uuids: int | None = 300, expected_seeds: int | None = 3,
                     n_bootstrap: int = 5000, bootstrap_seed: int = 2026, confidence: float = .95,
                     check_answer_sensitivity: bool = False, sensitivity_episodes: int = 180,
                     sensitivity_seed: int = 2026) -> dict[str, Any]:
    if (n_bootstrap < 1 or not 0 < confidence < 1 or sensitivity_episodes < 1
            or (expected_uuids is not None and expected_uuids < 1)
            or (expected_seeds is not None and expected_seeds < 1)):
        raise ValueError("Invalid diagnostic configuration")
    data, split_path = Path(data).resolve(), Path(split_path).resolve()
    directories = [Path(run).resolve(strict=True) for run in runs]
    if not directories or len(set(directories)) != len(directories) or (
            expected_seeds is not None and len(directories) != expected_seeds):
        raise ValueError("Expected distinct completed training-seed run directories")
    split, protocol, ids = _split(split_path, expected_uuids)
    paths = [data / name for name in ("profiles.csv", "tasks.json", "responses.csv")]
    fingerprints = {path.name: _file_hash(path) for path in paths}
    if fingerprints != split["fingerprints"]:
        raise ValueError("Data files changed after the split was frozen")
    bundle = load_bundle(*paths, uuids=ids)
    keys = [(uuid, context) for uuid in ids for context in CONTEXTS]
    tensors, reports, seeds, hashes, holdout_reference, rollout_seed, scenario = [], [], [], [], None, None, None
    for run in directories:
        predictor, summary, predictions, traces, h2_states, diagnostics, source_hashes = _load_run(
            run, split, protocol, ids, bundle)
        current_scenario = summary["scenario"]
        current_seed = predictor.metadata["training_config"]["seed"]
        checkpoint_hash = summary["checkpoint_sha256"]
        if current_seed in seeds or checkpoint_hash in hashes:
            raise ValueError("Each run must contain a distinct training seed and frozen checkpoint")
        current_holdouts = {key: tuple(sorted(predictions["weighted_js"][key][0])) for key in keys}
        if holdout_reference is not None and (current_holdouts != holdout_reference
                or current_scenario != scenario or summary["rollout_seed"] != rollout_seed):
            raise ValueError("Runs differ in holdouts, response scenario or rollout seed")
        holdout_reference, scenario, rollout_seed = current_holdouts, current_scenario, summary["rollout_seed"]
        # The learner receives only an empty public state and requested task
        # attributes. Hidden labels are looked up below, after prediction.
        requests = [(predictor.reset(bundle.profiles[uuid], context),
                     tuple(bundle.tasks[task_id] for task_id in predictions["weighted_js"][uuid, context][0]))
                    for uuid, context in keys]
        p0_values = predictor.predict_many(requests)
        values = []
        for key, p0 in zip(keys, p0_values):
            held, _, p9 = predictions["weighted_js"][key]
            # Evaluator-only hidden-label access; never supplied to a learner,
            # acquisition function, or hypothetical response state.
            labels = [bundle.responses[scenario][key[0], task_id] for task_id in held]
            h0, h9 = choice_metrics(p0, labels), choice_metrics(p9, labels)
            values.append((h0["nll"], h9["nll"], h0["accuracy"], h9["accuracy"],
                           float((.5 * np.abs(p0 - p9).sum(axis=1)).mean()),
                           float((p0.argmax(1) != p9.argmax(1)).mean())))
        tensor = np.asarray(values).reshape(len(ids), len(CONTEXTS), len(METRIC_NAMES))
        macro = tensor.mean(axis=(0, 1))
        report = {"training_seed": current_seed, "checkpoint_sha256": checkpoint_hash,
                  "source_sha256": source_hashes, "training": _training_summary(predictor),
                  "H0": {"nll": float(macro[0]), "accuracy": float(macro[2])},
                  "H9": {"nll": float(macro[1]), "accuracy": float(macro[3])},
                  "H9_minus_H0": {"nll": float(macro[1] - macro[0]), "accuracy": float(macro[3] - macro[2])},
                  "mean_query_total_variation": float(macro[4]), "argmax_change_fraction": float(macro[5]),
                  "by_context": [{"context": context, **dict(zip(METRIC_NAMES, tensor[:, ci].mean(0).tolist()))}
                                 for ci, context in enumerate(CONTEXTS)], **diagnostics}
        if check_answer_sensitivity:
            report["answer_sensitivity"] = _answer_sensitivity(
                predictor, summary, traces, h2_states, bundle, protocol, keys,
                sensitivity_episodes, sensitivity_seed)
        reports.append(report)
        tensors.append(tensor)
        seeds.append(current_seed)
        hashes.append(checkpoint_hash)
    tensor = np.asarray(tensors)
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(ids), size=(n_bootstrap, len(ids)), dtype=np.int32)
    macro = tensor.mean(axis=(1, 2))
    aggregate = {"H0": {"nll": _seed_stat(macro[:, 0]), "accuracy": _seed_stat(macro[:, 2])},
                 "H9": {"nll": _seed_stat(macro[:, 1]), "accuracy": _seed_stat(macro[:, 3])},
                 "H9_minus_H0": {
                     "nll": _paired_interval(tensor[..., 1] - tensor[..., 0], indices, confidence),
                     "accuracy": _paired_interval(tensor[..., 3] - tensor[..., 2], indices, confidence)},
                 "mean_query_total_variation": _seed_stat(macro[:, 4]),
                 "argmax_change_fraction": _seed_stat(macro[:, 5]),
                 "js_vs_random_order_diff_fraction": _seed_stat([
                     report["js_vs_random_history"]["order_diff_fraction"] for report in reports]),
                 "js_vs_random_set_diff_fraction": _seed_stat([
                     report["js_vs_random_history"]["set_diff_fraction"] for report in reports])}
    if check_answer_sensitivity:
        aggregate["answer_intervention_next_task_changed_fraction"] = _seed_stat([
            report["answer_sensitivity"]["next_task_changed_fraction"] for report in reports])
    return {"schema_version": 1, "scenario": scenario, "training_seeds": seeds,
            "n_training_seeds": len(reports), "n_uuids": len(ids), "contexts": list(CONTEXTS),
            "episodes_per_seed": len(keys), "holdout_queries_per_seed": 4 * len(keys),
            "budget": 9, "policy_history": "weighted_js", "n_holdout_tasks_per_episode": 4,
            "n_bootstrap": n_bootstrap, "bootstrap_seed": bootstrap_seed,
            "provenance": {"data_fingerprints": fingerprints, "split_sha256": _file_hash(split_path),
                           "protocol": asdict(protocol), "rollout_seed": rollout_seed,
                           "verified": ["complete test UUID/context grid", "saved H9 traces replay to saved query histories",
                                        "same four held-out tasks for H0/H9 and across training seeds",
                                        "checkpoint/evaluation/data/split provenance", "distinct frozen learner seeds"]},
            "aggregate": aggregate, "per_seed": reports,
            "interpretation": "History-use diagnostic, not a policy comparison: H9 is the saved adaptive history and H0 is the same model without observations",
            "scope": "Scores averaged after prediction; probabilities are not pooled across training seeds. UUID bootstrap keeps contexts together. Training-seed SD is separate. No training or checkpoint selection is performed."}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-uuids", type=int, default=300)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--check-answer-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-episodes", type=int, default=180)
    parser.add_argument("--sensitivity-seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Diagnostic output exists; choose a new output file")
    try:
        report = diagnose_recheck(args.data, args.split, args.runs,
            expected_uuids=args.expected_uuids, expected_seeds=args.expected_seeds,
            n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed, confidence=args.confidence,
            check_answer_sensitivity=args.check_answer_sensitivity,
            sensitivity_episodes=args.sensitivity_episodes, sensitivity_seed=args.sensitivity_seed)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "n_uuids": report["n_uuids"],
                      "n_training_seeds": report["n_training_seeds"],
                      "H9_minus_H0_nll": report["aggregate"]["H9_minus_H0"]["nll"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
