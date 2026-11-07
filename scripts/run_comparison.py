"""Retrain Fixed/Non-graph comparisons with a new protocol and a saved cohort.

Only the protocol changes in the copied split; UUID partitions and source
responses remain fixed. Raw inputs, checkpoints, traces and manifests stay in
ignored runtime directories. Publish aggregate results separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from adaptive_survey.cli import _fingerprints, _load, _split
from adaptive_survey.protocol import DEFAULT_BUDGETS, ProtocolConfig


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source-split", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--scenarios", choices=["mmnl", "mnl"], nargs="+", default=["mmnl", "mnl"])
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--validation-episodes", type=int, default=720)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-history", type=int, default=17)
    p.add_argument("--rollout-seed", type=int, default=2026)
    p.add_argument("--protocol-seed", type=int, default=2026)
    p.add_argument("--initial-per-block", type=int, default=2)
    p.add_argument("--holdout-per-block", type=int, default=2)
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--fixed-ridge", type=float, default=1.0)
    p.add_argument("--fixed-max-profiles", type=int, default=256)
    return p


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def commands(args: argparse.Namespace, scenario: str, seed: int,
             run: Path, split: Path) -> dict[str, list[str]]:
    common = ["--data", str(args.data.resolve()), "--split", str(split), "--scenario", scenario]
    checkpoint = run / "checkpoint.pt"
    return {
        "train": [sys.executable, "-m", "adaptive_survey", "train", *common,
                  "--output", str(checkpoint), "--ensemble", str(args.ensemble),
                  "--epochs", str(args.epochs), "--episodes", str(args.episodes),
                  "--validation-episodes", str(args.validation_episodes),
                  "--batch-size", str(args.batch_size), "--patience", str(args.patience),
                  "--learning-rate", str(args.learning_rate), "--max-history", str(args.max_history),
                  "--seed", str(seed), "--threads", "1"],
        "evaluate": [sys.executable, "-m", "adaptive_survey", "evaluate", *common,
                     "--checkpoint", str(checkpoint), "--output", str(run / "evaluation"),
                     "--policies", "fixed", "weighted_js", "--tracks", "prediction", "preference",
                     "--bootstrap", str(args.bootstrap), "--seed", str(args.rollout_seed),
                     "--fixed-ridge", str(args.fixed_ridge),
                     "--fixed-max-profiles", str(args.fixed_max_profiles)],
        "estimate": [sys.executable, "-m", "adaptive_survey", "estimate",
                     "--data", str(args.data.resolve()), "--split", str(split),
                     "--evaluation", str(run / "evaluation"), "--track", "preference",
                     "--output", str(run / "population.json"), "--cluster-se"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    for key in ("jobs", "ensemble", "epochs", "episodes", "validation_episodes",
                "batch_size", "patience", "bootstrap", "fixed_max_profiles"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.learning_rate <= 0 or args.fixed_ridge <= 0 or not 0 <= args.max_history <= 17:
        raise ValueError("Invalid learning rate, ridge or training history limit")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("Seeds and scenarios must be unique")
    protocol = ProtocolConfig(args.holdout_per_block, args.initial_per_block,
                              tuple(args.budgets), args.protocol_seed)
    source = _split(args.source_split)
    # A complete panel is checked before creating any experiment output.
    bundle = _load(args, source)
    if any(not ids for ids in source["uuids"].values()):
        raise ValueError("Every respondent partition must be nonempty")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    split = root / "split.json"
    prepared = dict(source, protocol=asdict(protocol))
    write_json(split, prepared, exclusive=True)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    code_paths = [*sorted((repo / "src/adaptive_survey").glob("*.py")), Path(__file__).resolve()]
    manifest = {
        "schema_version": 1,
        "code_revision": revision,
        "source_sha256": {str(path.relative_to(repo)): file_hash(path) for path in code_paths},
        "config": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": {name: len(ids) for name, ids in source["uuids"].items()},
        "response_rows": bundle.validation.rows,
        "source_split_sha256": file_hash(args.source_split),
        "source_protocol": source["protocol"],
        "split_sha256": file_hash(split),
        "fingerprints": _fingerprints(args.data),
        "cohort_rule": "Preserve all source split UUIDs and all response values; change protocol only",
        "seed_interpretation": "Training procedure seeds vary; response realization, cohort, protocol and rollout seed stay fixed",
        "cases": [],
    }
    del bundle
    cases = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            run = root / scenario / f"seed-{seed}"
            run.mkdir(parents=True)
            phase_commands = commands(args, scenario, seed, run, split)
            cases.append((scenario, seed, run, phase_commands))
            manifest["cases"].append({"scenario": scenario, "seed": seed,
                                      "path": str(run.relative_to(root)), "commands": phase_commands})
    write_json(root / "manifest.json", manifest, exclusive=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")

    def worker(case: tuple) -> dict:
        scenario, seed, run, phase_commands = case
        timings = {}
        for phase, command in phase_commands.items():
            event = {"scenario": scenario, "seed": seed, "phase": phase, "status": "started"}
            write_json(run / "phase.json", event)
            print(json.dumps(event), flush=True)
            started = perf_counter()
            with (run / f"{phase}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            timings[phase + "_seconds"] = perf_counter() - started
            event.update(status="completed", seconds=timings[phase + "_seconds"])
            write_json(run / "phase.json", event)
            print(json.dumps(event), flush=True)
        result = {"scenario": scenario, "seed": seed, **timings}
        write_json(run / "timing.json", result, exclusive=True)
        return result

    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(cases))) as pool:
        pending = {pool.submit(worker, case): (case[0], case[1]) for case in cases}
        for future in as_completed(pending):
            scenario, seed = pending[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failure = {"scenario": scenario, "seed": seed, "status": "failed", "error": str(error)}
                errors.append(failure)
                write_json(root / scenario / f"seed-{seed}" / "failure.json", failure)
                print(json.dumps(failure), flush=True)
    write_json(root / "status.json", {"completed": completed, "errors": errors}, exclusive=True)
    if errors:
        raise RuntimeError(f"{len(errors)} cases failed; inspect retained phase logs")


if __name__ == "__main__":
    main()
