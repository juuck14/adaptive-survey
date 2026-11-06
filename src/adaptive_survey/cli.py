"""Command-line preparation, training and evaluation for the common protocol."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from . import __version__
from .data import load_bundle, load_profiles, load_tasks, ResponseWorld, validate_response_csv
from .protocol import DEFAULT_BUDGETS, TRACKS, ProtocolConfig, make_episode, split_uuids
from .schema import CONTEXTS


def _hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: str | Path, value: Any) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _paths(directory: str | Path) -> tuple[Path, Path, Path]:
    root = Path(directory)
    return root / "profiles.csv", root / "tasks.json", root / "responses.csv"


def _fingerprints(directory: str | Path) -> dict[str, str]:
    return {path.name: _hash(path) for path in _paths(directory)}


def _runtime() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _cohort(responses: Path) -> list[str]:
    # Only UUIDs are retained during this scan, never a full response panel.
    with responses.open(encoding="utf-8-sig", newline="") as stream:
        return sorted({row["uuid"].strip() for row in csv.DictReader(stream)})


def _split(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("uuids"), dict):
        raise ValueError("Unsupported split manifest")
    groups = value["uuids"]
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Split must contain train, validation and test")
    combined = [uid for key in groups for uid in groups[key]]
    if not combined or len(combined) != len(set(combined)) or any(not isinstance(u, str) or not u for u in combined):
        raise ValueError("Split UUIDs must be unique and disjoint")
    return value


def _split_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value["uuids"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _protocol(value: dict) -> ProtocolConfig:
    fields = dict(value["protocol"])
    fields["budgets"] = tuple(fields["budgets"])
    return ProtocolConfig(**fields)


def _load(args: argparse.Namespace, split: dict | None = None):
    ids = None if split is None else [u for group in split["uuids"].values() for u in group]
    paths = _paths(args.data)
    bundle = load_bundle(*paths, uuids=ids)
    if split is not None and split["fingerprints"] != _fingerprints(args.data):
        raise ValueError("Input files changed after the split was created; create a new manifest")
    return bundle


def _records(bundle, uuids, scenario):
    from .estimation import ChoiceRecord
    return [ChoiceRecord(bundle.profiles[u], task, bundle.responses[scenario][(u, task.task_id)])
            for u in uuids for task in bundle.tasks.values()]


def command_demo(args):
    from .demo import generate_demo
    print(json.dumps(generate_demo(args.output, n_respondents=args.respondents, seed=args.seed), default=str))


def command_validate(args):
    p, t, r = _paths(args.data)
    profiles, tasks = load_profiles(p), load_tasks(t)
    report = validate_response_csv(r, tasks, profiles)
    result = {"version": __version__, "validation": report.to_dict(), "fingerprints": _fingerprints(args.data)}
    if args.output:
        _json(args.output, result)
    print(json.dumps(result, allow_nan=False))


def command_split(args):
    p, t, r = _paths(args.data)
    profiles, tasks = load_profiles(p), load_tasks(t)
    report = validate_response_csv(r, tasks, profiles)
    groups = split_uuids(_cohort(r), args.train, args.validation, args.test, seed=args.seed)
    protocol = ProtocolConfig(seed=args.seed, budgets=tuple(args.budgets),
                              initial_per_block=args.initial_per_block,
                              holdout_per_block=args.holdout_per_block)
    if Path(args.output).exists():
        raise FileExistsError("Split manifest exists; choose a new output")
    value = {"schema_version": 1, "version": __version__, "uuids": groups,
             "protocol": asdict(protocol), "fingerprints": _fingerprints(args.data),
             "validation": report.to_dict()}
    _json(args.output, value)
    print(json.dumps({"output": args.output, "counts": {k: len(v) for k, v in groups.items()}}))


def command_train(args):
    from .training import TrainingConfig, train_ensemble
    split = _split(args.split)
    bundle = _load(args, split)
    if Path(args.output).exists() or Path(str(args.output) + ".training.json").exists():
        raise FileExistsError("Checkpoint exists; choose a new output")
    config = TrainingConfig(ensemble_size=args.ensemble, epochs=args.epochs,
                            episodes_per_epoch=args.episodes, validation_episodes=args.validation_episodes,
                            batch_size=args.batch_size, learning_rate=args.learning_rate,
                            patience=args.patience, max_history=args.max_history,
                            seed=args.seed, num_threads=args.threads)
    started = perf_counter()
    result = train_ensemble(bundle.profiles, bundle.tasks, bundle.responses[args.scenario],
                            split["uuids"]["train"], split["uuids"]["validation"],
                            config=config, protocol_config=_protocol(split))
    result.predictor.metadata.update({"training_seconds": perf_counter() - started,
                                      "runtime": _runtime(),
                                      "scenario": args.scenario, "split_hash": _split_hash(split),
                                      "fingerprints": split["fingerprints"], "protocol": split["protocol"]})
    result.predictor.save(args.output)
    _json(str(args.output) + ".training.json", {"history": result.history, "metadata": result.predictor.metadata})
    print(json.dumps({"checkpoint": args.output, "ensemble_size": args.ensemble,
                      "epochs_recorded": len(result.history)}))


def command_fit_baseline(args):
    from .baselines import PanelMAPPredictor
    split = _split(args.split)
    bundle = _load(args, split)
    if Path(args.output).exists() or Path(str(args.output) + ".manifest.json").exists():
        raise FileExistsError("Baseline checkpoint exists; choose a new output")
    predictor = PanelMAPPredictor.fit(_records(bundle, split["uuids"]["train"], args.scenario),
                                      prior_scale=args.prior_scale)
    predictor.save(args.output)
    _json(str(args.output) + ".manifest.json", {"scenario": args.scenario,
           "checkpoint_sha256": _hash(args.output),
           "split_hash": _split_hash(split), "fingerprints": split["fingerprints"],
           "protocol": split["protocol"],
           "fit": predictor.fit_summary})
    print(json.dumps({"checkpoint": args.output, "fit": predictor.fit_summary}, allow_nan=False))


def _predictor(kind: str, checkpoint: str | None, split: dict, scenario: str):
    if kind == "uniform":
        from .baselines import UniformPredictor
        return UniformPredictor()
    if not checkpoint:
        raise ValueError("A trained checkpoint is required")
    if kind == "non_graph":
        from .neural import NeuralPredictor
        predictor = NeuralPredictor.load(checkpoint)
        manifest = predictor.metadata
    else:
        from .baselines import PanelMAPPredictor
        predictor = PanelMAPPredictor.load(checkpoint)
        manifest = json.loads(Path(checkpoint + ".manifest.json").read_text())
        if manifest.get("checkpoint_sha256") != _hash(checkpoint):
            raise ValueError("Baseline checkpoint and its manifest do not match")
    if manifest.get("split_hash") != _split_hash(split) or manifest.get("scenario") != scenario:
        raise ValueError("Checkpoint and evaluation split/scenario do not match")
    if manifest.get("fingerprints") != split["fingerprints"]:
        raise ValueError("Checkpoint was trained against different source files")
    protocol_keys = ("seed", "holdout_per_block", "initial_per_block")
    if any(manifest.get("protocol", {}).get(k) != split["protocol"].get(k) for k in protocol_keys):
        raise ValueError("Checkpoint history protocol differs from the evaluation manifest")
    return predictor


def command_evaluate(args):
    from .baselines import FixedEfficientDesign
    from .estimation import fit_pooled_mnl
    from .metrics import paired_uuid_bootstrap, score_checkpoint, summarize_scores
    from .policies import make_policy
    from .runner import run_episode
    split = _split(args.split)
    bundle = _load(args, split)
    config = _protocol(split)
    if args.budgets:
        config = replace(config, budgets=tuple(args.budgets))
    predictor = _predictor(args.predictor, args.checkpoint, split, args.scenario)
    common = (_predictor("non_graph", args.common_checkpoint, split, args.scenario)
              if args.common_checkpoint else None)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Evaluation output exists; choose a new directory")
    ids = split["uuids"]["test"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Limit must be positive")
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("Evaluation split is empty")
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("Policies must be unique")
    if len(set(args.contexts)) != len(args.contexts) or len(set(args.tracks)) != len(args.tracks):
        raise ValueError("Contexts and tracks must be unique")
    tracks = {}
    for track in args.tracks:
        budgets = config.budgets_for(track)
        tracks[track] = {"budgets": list(budgets),
                         "unavailable_budgets": sorted(set(config.budgets) - set(budgets)),
                         "candidate_count": 18 - 2 * config.holdout_per_block if track == "prediction" else 18,
                         "holdout_count": 2 * config.holdout_per_block if track == "prediction" else 0,
                         "status": "completed" if budgets else "no_feasible_budgets"}
        if track == "prediction":
            tracks[track]["unavailable_reason"] = "Held-out tasks must remain unasked; no NLL is computed at unavailable budgets"
    prepared = {track: [make_episode(uid, context, bundle.tasks, config, track=track)
                        for uid in ids for context in args.contexts]
                for track, settings in tracks.items() if settings["budgets"]}
    if not prepared:
        raise ValueError("No feasible budgets in the requested evaluation tracks")
    design, fixed_fit = None, None
    planning_timing = None
    if "fixed" in args.policies:
        # This prior uses training answers only, not a generator model file.
        started = perf_counter()
        fixed_fit = fit_pooled_mnl(_records(bundle, split["uuids"]["train"], args.scenario),
                                  drop_unidentified=True)
        if not fixed_fit.converged:
            raise ValueError("Training-only fixed-design prior did not converge")
        planning_timing = {"training_prior_seconds": perf_counter() - started}
        started = perf_counter()
        design = FixedEfficientDesign(bundle.tasks, [bundle.profiles[u] for u in split["uuids"]["train"]],
                                      fixed_fit.coefficients, ridge=args.fixed_ridge,
                                      max_profiles=args.fixed_max_profiles)
        planning_timing["fisher_information_seconds"] = perf_counter() - started
    # Preplan before any test response is revealed. Different public masks may
    # require different fixed plans; none depend on a personal feature or answer.
    plans, episode_plans = {}, {}
    if design is not None:
        started = perf_counter()
        for track, episodes in prepared.items():
            for episode in episodes:
                plan = design.plan(episode.context, episode.candidates, episode.initial)
                payload = asdict(plan)
                plan_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                plans[plan_id] = payload
                episode_plans[(track, episode.uuid, episode.context)] = plan_id
        planning_timing["fixed_plans_seconds"] = perf_counter() - started
        planning_timing["unique_plans"] = len(plans)
    world = ResponseWorld(bundle, args.scenario)
    out.mkdir(parents=True, exist_ok=False)
    scored, time_totals = [], {}
    with ((out / "traces.jsonl").open("x", encoding="utf-8") as traces,
          (out / "predictions.jsonl").open("x", encoding="utf-8") as preds,
          (out / "collections.jsonl").open("x", encoding="utf-8") as collections):
        for policy_name in args.policies:
            seconds, steps = 0.0, 0
            prediction_seconds, common_prediction_seconds = 0.0, 0.0
            for track, episodes in prepared.items():
                for episode in episodes:
                    ranking = None
                    if design is not None and policy_name == "fixed":
                        plan_id = episode_plans[(track, episode.uuid, episode.context)]
                        ranking = plans[plan_id]["task_ids"]
                    policy = make_policy(policy_name, rankings=ranking)
                    result = run_episode(bundle.profiles[episode.uuid], episode.context, world, episode, predictor,
                                         policy, common, seed=args.seed)
                    for row in result.trace:
                        traces.write(json.dumps(asdict(row), allow_nan=False) + "\n")
                        seconds += row.selection_seconds + row.update_seconds
                        steps += 1
                    for cp in result.checkpoints:
                        prediction_seconds += cp.prediction_seconds
                        common_prediction_seconds += cp.common_prediction_seconds
                        stream = preds if track == "prediction" else collections
                        stream.write(json.dumps(asdict(cp), allow_nan=False) + "\n")
                        scored.extend(score_checkpoint(cp, world))
            time_totals[policy_name] = {"selection_and_update_seconds": seconds, "revealed_steps": steps,
                                       "prediction_seconds": prediction_seconds,
                                       "common_prediction_seconds": common_prediction_seconds}
    summary = summarize_scores(scored)
    comparisons = []
    main_budgets = set(tracks.get("prediction", {}).get("budgets", []))
    reference = "fixed" if "fixed" in args.policies else "random" if "random" in args.policies else None
    if reference and main_budgets:
        selected = [r for r in scored if r.budget in main_budgets]
        for reader in ("native", "common") if common else ("native",):
            for name in args.policies:
                if name != reference:
                    comparisons.append(paired_uuid_bootstrap(selected, name, reference, reader=reader,
                                       scenario=args.scenario, n_bootstrap=args.bootstrap, seed=args.seed))
    averages = []
    for policy in args.policies:
        for reader in ("native", "common") if common else ("native",):
            values = [r["nll"] for r in summary if r["context"] == "macro" and r["policy"] == policy
                      and r["reader"] == reader and r["budget"] in main_budgets]
            if values:
                averages.append({"policy": policy, "reader": reader, "budgets": sorted(main_budgets),
                                 "mean_nll": sum(values) / len(values)})
    _json(out / "scores.json", [asdict(s) for s in scored])
    if design is not None:
        _json(out / "fixed-plans.json", {"plans": plans,
              "episode_plans": [{"track": track, "uuid": uid, "context": context, "plan_id": plan_id}
                                for (track, uid, context), plan_id in episode_plans.items()]})
    result = {"schema_version": 2, "version": __version__, "scenario": args.scenario,
              "predictor": args.predictor, "protocol": asdict(config), "n_uuids": len(ids),
              "rollout_seed": args.seed, "bootstrap_seed": args.seed,
              "n_bootstrap": args.bootstrap, "policies": args.policies, "limit": args.limit,
              "runtime": _runtime(), "tracks": tracks,
              "learner_comparison": "the same frozen choice learner and checkpoint for every selection policy",
              "contexts": args.contexts, "split_hash": _split_hash(split), "fingerprints": split["fingerprints"],
              "checkpoint_sha256": _hash(args.checkpoint) if args.checkpoint else None,
              "common_checkpoint_sha256": _hash(args.common_checkpoint) if common else None,
              "common_reader_distinct_checkpoint": (None if common is None else
                  not args.checkpoint or _hash(args.common_checkpoint) != _hash(args.checkpoint)),
              "metrics": summary, "budget_average": averages, "paired_comparisons": comparisons,
              "fixed_prior": fixed_fit.as_dict() if fixed_fit else None,
              "fixed_plans_sha256": _hash(out / "fixed-plans.json") if design is not None else None,
              "timing": time_totals, "fixed_planning_timing": planning_timing,
              "fixed_design": design.metadata() if design is not None else None,
              "scope": "new respondents, known task bank; one fixed response realization"}
    _json(out / "summary.json", result)
    print(json.dumps({"output": str(out), "n_uuids": len(ids), "tracks": tracks,
                      "budget_average": averages}, allow_nan=False))


def command_generate(args):
    from .generator import generate_responses
    profiles, tasks = load_profiles(args.profiles), load_tasks(args.tasks)
    ids = sorted(profiles, key=lambda u: hashlib.sha256(f"{args.seed}:{u}".encode()).hexdigest())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Limit must be positive")
        ids = ids[:args.limit]
    print(json.dumps(generate_responses(args.handoff, profiles, tasks, args.output, uuids=ids, seed=args.seed)))


def command_estimate(args):
    """Offline population analysis; it never updates the rollout learner."""
    from .estimation import ChoiceRecord, fit_selected_vs_full
    split = _split(args.split)
    bundle = _load(args, split)
    root = Path(args.evaluation)
    summary = json.loads((root / "summary.json").read_text())
    if summary["split_hash"] != _split_hash(split) or summary["fingerprints"] != split["fingerprints"]:
        raise ValueError("Evaluation and population-analysis data/splits differ")
    config = _protocol(summary)
    if any(summary["protocol"].get(key) != split["protocol"].get(key)
           for key in ("seed", "holdout_per_block", "initial_per_block")):
        raise ValueError("Evaluation and population-analysis history protocols differ")
    track = args.track or ("preference" if summary.get("tracks", {}).get("preference", {}).get("budgets") else "prediction")
    track_budgets = summary.get("tracks", {}).get(track, {}).get("budgets", [])
    if not summary.get("tracks") and track == "prediction":
        track_budgets = summary["protocol"]["budgets"]
    if not track_budgets:
        raise ValueError("Requested population-analysis track has no collected trajectories")
    if Path(args.output).exists():
        raise FileExistsError("Analysis output exists; choose a new path")
    grouped = {}
    with (root / "traces.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("track", "prediction") != track:
                continue
            key = (row["policy"], row["uuid"], row["context"])
            grouped.setdefault(key, []).append(row)
    results = []
    scenario = summary["scenario"]
    for policy in args.policies or summary["policies"]:
        groups = {key: sorted(rows, key=lambda row: row["step"]) for key, rows in grouped.items() if key[0] == policy}
        if not groups:
            raise ValueError("Requested policy has no survey traces")
        expected_ids = split["uuids"]["test"][:summary["n_uuids"]]
        expected = {(policy, uid, c) for uid in expected_ids for c in summary["contexts"]}
        if set(groups) != expected:
            raise ValueError("Population traces must contain the complete evaluated UUID/context grid")
        full = []
        for (_, uid, context), rows in sorted(groups.items()):
            if uid not in split["uuids"]["test"] or context not in summary["contexts"]:
                raise ValueError("Trace UUID/context is not part of this evaluation")
            if [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("Trace has duplicated or missing steps")
            if len(rows) != max(track_budgets) or len({r["task_id"] for r in rows}) != len(rows):
                raise ValueError("Trace must contain a complete trajectory of unique questions")
            episode = make_episode(uid, context, bundle.tasks, config, track=track)
            if any(r["scenario"] != scenario or r["task_id"] not in episode.acquisition_ids
                   or bundle.responses[scenario][(uid, r["task_id"])] != r["choice"] for r in rows):
                raise ValueError("Trace and hidden response environment disagree")
            full.extend(ChoiceRecord(bundle.profiles[uid], task, bundle.responses[scenario][(uid, task.task_id)])
                        for task in bundle.tasks.values() if task.context == context)
        for budget in args.budgets or track_budgets:
            if budget <= 0 or any(len(rows) < budget for rows in groups.values()):
                raise ValueError("Population budget needs collected responses for every episode")
            selected = [ChoiceRecord(bundle.profiles[uid], bundle.tasks[row["task_id"]], row["choice"])
                        for (_, uid, _), rows in groups.items() for row in rows[:budget]]
            comparison = fit_selected_vs_full(selected, full, drop_unidentified=True,
                                               cluster_se=args.cluster_se, l2=args.l2)
            results.append({"policy": policy, "budget": budget, **comparison})
    value = {"schema_version": 2, "scenario": scenario, "track": track, "results": results,
             "evaluation_summary_sha256": _hash(root / "summary.json"), "runtime": _runtime(),
             "interpretation": "uniform-task full-survey MNL reference agreement; not MMNL parameter recovery"}
    _json(args.output, value)
    print(json.dumps({"output": args.output, "comparisons": len(results)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Create artificial, shareable development data")
    demo.add_argument("--output", required=True); demo.add_argument("--respondents", type=int, default=90)
    demo.add_argument("--seed", type=int, default=2026); demo.set_defaults(func=command_demo)
    validate = sub.add_parser("validate", help="Stream-check a complete response panel")
    validate.add_argument("--data", required=True); validate.add_argument("--output")
    validate.set_defaults(func=command_validate)
    split = sub.add_parser("split", help="Validate inputs and freeze an exact UUID split")
    split.add_argument("--data", required=True); split.add_argument("--output", required=True)
    split.add_argument("--train", type=int, required=True); split.add_argument("--validation", type=int, required=True)
    split.add_argument("--test", type=int, required=True); split.add_argument("--seed", type=int, default=2026)
    split.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    split.add_argument("--initial-per-block", type=int, default=2)
    split.add_argument("--holdout-per-block", type=int, default=2)
    split.set_defaults(func=command_split)
    train = sub.add_parser("train", help="Fit the history-conditioned Non-graph ensemble")
    for name in ("data", "split", "output"): train.add_argument("--"+name, required=True)
    train.add_argument("--scenario", choices=["mnl","mmnl"], default="mmnl")
    train.add_argument("--ensemble", type=int, default=5); train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--episodes", type=int, default=1024); train.add_argument("--validation-episodes", type=int, default=720)
    train.add_argument("--batch-size", type=int, default=64); train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=30); train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--max-history", type=int, default=17)
    train.add_argument("--threads", type=int, default=1); train.set_defaults(func=command_train)
    baseline = sub.add_parser("fit-baseline", help="Fit a training-only panel MAP readout")
    for name in ("data", "split", "output"): baseline.add_argument("--"+name, required=True)
    baseline.add_argument("--scenario", choices=["mnl","mmnl"], default="mmnl")
    baseline.add_argument("--prior-scale", type=float, default=.5); baseline.set_defaults(func=command_fit_baseline)
    evaluate = sub.add_parser("evaluate", help="Paired, leakage-controlled policy evaluation")
    for name in ("data", "split", "output"): evaluate.add_argument("--"+name, required=True)
    evaluate.add_argument("--predictor", choices=["non_graph","panel_map","uniform"], default="non_graph")
    evaluate.add_argument("--checkpoint"); evaluate.add_argument("--common-checkpoint")
    evaluate.add_argument("--scenario", choices=["mnl","mmnl"], default="mmnl")
    evaluate.add_argument("--policies", nargs="+", choices=["fixed","random","entropy","weighted_js"], default=["fixed","weighted_js"])
    evaluate.add_argument("--tracks", nargs="+", choices=list(TRACKS), default=list(TRACKS))
    evaluate.add_argument("--fixed-ridge", type=float, default=1.0)
    evaluate.add_argument("--fixed-max-profiles", type=int, default=256)
    evaluate.add_argument("--contexts", nargs="+", choices=list(CONTEXTS), default=list(CONTEXTS))
    evaluate.add_argument("--budgets", nargs="+", type=int); evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--bootstrap", type=int, default=2000); evaluate.add_argument("--seed", type=int, default=2026)
    evaluate.set_defaults(func=command_evaluate)
    generate = sub.add_parser("generate", help="Offline generation using an external trusted handoff")
    for name in ("handoff","profiles","tasks","output"): generate.add_argument("--"+name, required=True)
    generate.add_argument("--limit", type=int); generate.add_argument("--seed", type=int, default=2026)
    generate.set_defaults(func=command_generate)
    estimate = sub.add_parser("estimate", help="Offline pooled-MNL full-survey reference analysis")
    for name in ("data", "split", "evaluation", "output"): estimate.add_argument("--"+name, required=True)
    estimate.add_argument("--policies", nargs="+"); estimate.add_argument("--budgets", nargs="+", type=int)
    estimate.add_argument("--track", choices=list(TRACKS))
    estimate.add_argument("--l2", type=float, default=1e-4); estimate.add_argument("--cluster-se", action="store_true")
    estimate.set_defaults(func=command_estimate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
