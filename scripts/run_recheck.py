"""Run prespecified training seeds and real policy rollouts in separate processes.

Inputs and outputs contain private respondent data and must remain ignored.
The rollout seed stays fixed while the training procedure seed varies. A run
root is exclusive: failed/partial runs are retained for inspection, never reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--validation-episodes", type=int, default=720)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--scenario", choices=["mnl", "mmnl"], default="mmnl")
    p.add_argument("--rollout-seed", type=int, default=2026)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p


def commands(args: argparse.Namespace, seed: int, run: Path) -> tuple[list[str], list[str]]:
    common = ["--data", str(args.data.resolve()), "--split", str(args.split.resolve()),
              "--scenario", args.scenario]
    checkpoint = run / "checkpoint.pt"
    train = [sys.executable, "-m", "adaptive_survey", "train", *common,
             "--output", str(checkpoint), "--ensemble", str(args.ensemble),
             "--epochs", str(args.epochs), "--episodes", str(args.episodes),
             "--validation-episodes", str(args.validation_episodes),
             "--batch-size", str(args.batch_size), "--patience", str(args.patience),
             "--learning-rate", str(args.learning_rate), "--seed", str(seed), "--threads", "1"]
    evaluate = [sys.executable, "-m", "adaptive_survey", "evaluate", *common,
                "--checkpoint", str(checkpoint), "--output", str(run / "evaluation"),
                "--policies", "fixed", "random", "entropy", "weighted_js",
                "--bootstrap", str(args.bootstrap), "--seed", str(args.rollout_seed)]
    return train, evaluate


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    positive = ("jobs", "ensemble", "epochs", "episodes", "validation_episodes",
                "batch_size", "patience", "bootstrap")
    if any(getattr(args, name) < 1 for name in positive) or args.learning_rate <= 0:
        raise ValueError("Run sizes and learning rate must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if not args.data.is_dir() or not args.split.is_file():
        raise FileNotFoundError("Prepared data directory and split are required")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    manifest["split_sha256"] = hashlib.sha256(args.split.read_bytes()).hexdigest()
    manifest["training_seed_interpretation"] = (
        "Training initialization, respondent bootstrap, episode draws, and validation bank vary; "
        "the protocol and rollout seed are fixed. Each model is selected using validation only.")
    manifest["commands"] = {}
    for seed in args.seeds:
        run = root / f"seed-{seed}"
        run.mkdir()
        manifest["commands"][str(seed)] = commands(args, seed, run)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")

    def worker(seed: int) -> dict:
        run = root / f"seed-{seed}"
        timings = {}
        for phase, command in zip(("train", "evaluate"), commands(args, seed, run)):
            print(json.dumps({"seed": seed, "phase": phase, "status": "started"}), flush=True)
            start = perf_counter()
            with (run / f"{phase}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            timings[phase + "_seconds"] = perf_counter() - start
            print(json.dumps({"seed": seed, "phase": phase, "status": "completed",
                              "seconds": timings[phase + "_seconds"]}), flush=True)
        (run / "timing.json").write_text(json.dumps(timings, indent=2) + "\n")
        return {"seed": seed, **timings}

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(args.seeds))) as pool:
        pending = {pool.submit(worker, seed): seed for seed in args.seeds}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"seed": pending[future], "error": str(exc)})
                print(json.dumps(errors[-1]), flush=True)
    status = {"completed": sorted(results, key=lambda row: row["seed"]), "errors": errors}
    (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"{len(errors)} runs failed; inspect the retained phase logs")


if __name__ == "__main__":
    main()
