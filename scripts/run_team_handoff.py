#!/usr/bin/env python3
"""Prepare, train, run and collect the November 12 team handoff protocol."""
import argparse
import json

from adaptive_survey.team_handoff import prepare_team, train_team, run_team, collect_run_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--inputs", required=True)
    prepare.add_argument("--handoff", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--train", type=int, default=1200)
    prepare.add_argument("--validation", type=int, default=300)
    prepare.add_argument("--seed", type=int, default=20261118)
    train = sub.add_parser("train")
    train.add_argument("--prepared", required=True)
    train.add_argument("--generator", choices=("mnl", "panel_mmnl"), required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--ensemble", type=int, default=5)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--episodes", type=int, default=1024)
    train.add_argument("--validation-episodes", type=int, default=900)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--seed", type=int, default=20261118)
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True)
    run.add_argument("--generator", choices=("mnl", "panel_mmnl"), required=True)
    run.add_argument("--method", choices=("fixed", "non_graph"), required=True)
    run.add_argument("--replicate", type=int, required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--checkpoint")
    run.add_argument("--seed", type=int, default=20261118)
    run.add_argument("--limit-per-context", type=int)
    run.add_argument("--accelerated", action="store_true")
    collect = sub.add_parser("collect")
    collect.add_argument("--output", required=True)
    collect.add_argument("--handoff", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_team(args.inputs, args.handoff, args.output, train_size=args.train,
                              validation_size=args.validation, seed=args.seed)
    elif args.command == "train":
        from adaptive_survey.training import TrainingConfig
        config = TrainingConfig(ensemble_size=args.ensemble, epochs=args.epochs,
                                episodes_per_epoch=args.episodes, validation_episodes=args.validation_episodes,
                                patience=args.patience, seed=args.seed, max_history=8)
        result = train_team(args.prepared, args.generator, args.checkpoint, config=config)
    elif args.command == "run":
        result = run_team(args.prepared, args.generator, args.method, args.replicate, args.output,
                          checkpoint=args.checkpoint, seed=args.seed,
                          limit_per_context=args.limit_per_context, accelerated=args.accelerated)
    else:
        result = collect_run_manifest(args.output, args.handoff)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
