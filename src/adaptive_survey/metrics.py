"""Proper choice scores and paired respondent-level uncertainty."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .runner import Checkpoint
from .schema import checked_probabilities


def choice_metrics(probabilities: np.ndarray, choices: Sequence[int]) -> dict[str, float]:
    y = np.asarray(choices)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= 4)):
        raise ValueError("Choices must be a nonempty vector of integers in [0, 3]")
    p = checked_probabilities(probabilities, len(y))
    target = np.eye(4, dtype=float)[y]
    return {
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
        "brier": float(np.square(p - target).sum(axis=1).mean()),
    }


def log_loss(probabilities: np.ndarray, choices: Sequence[int]) -> float:
    return choice_metrics(probabilities, choices)["nll"]


def accuracy(probabilities: np.ndarray, choices: Sequence[int]) -> float:
    return choice_metrics(probabilities, choices)["accuracy"]


def brier_score(probabilities: np.ndarray, choices: Sequence[int]) -> float:
    return choice_metrics(probabilities, choices)["brier"]


@dataclass(frozen=True, slots=True)
class ScoredCheckpoint:
    uuid: str
    context: str
    scenario: str
    policy: str
    budget: int
    reader: str
    nll: float
    accuracy: float
    brier: float
    n_tasks: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_checkpoint(checkpoint: Checkpoint, world: Any) -> tuple[ScoredCheckpoint, ...]:
    """Evaluator-only boundary: hidden labels are read here, after rollout."""
    scenario = str(getattr(world, "scenario", checkpoint.scenario))
    if scenario != checkpoint.scenario:
        raise ValueError("Prediction and response-world scenarios differ")
    choices = [world.answer(checkpoint.uuid, task_id) for task_id in checkpoint.holdout_task_ids]
    readers = [("native", checkpoint.probabilities)]
    if checkpoint.common_probabilities is not None:
        readers.append(("common", checkpoint.common_probabilities))
    return tuple(ScoredCheckpoint(checkpoint.uuid, checkpoint.context, checkpoint.scenario,
                                  checkpoint.policy, checkpoint.budget, reader,
                                  **choice_metrics(np.asarray(p), choices), n_tasks=len(choices))
                 for reader, p in readers)


def summarize_scores(records: Iterable[ScoredCheckpoint]) -> list[dict[str, Any]]:
    """Equal respondents within each context; equal contexts in the macro row."""
    groups: dict[tuple[Any, ...], list[ScoredCheckpoint]] = defaultdict(list)
    macro_uuids: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        identity = (record.scenario, record.policy, record.reader, record.budget, record.context, record.uuid)
        if identity in seen:
            raise ValueError("Duplicate scored episode; summarize training/world replicates separately")
        seen.add(identity)
        groups[(record.scenario, record.policy, record.reader, record.budget, record.context)].append(record)
        macro_uuids[(record.scenario, record.policy, record.reader, record.budget)].add(record.uuid)
    result: list[dict[str, Any]] = []
    macro_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for key, values in sorted(groups.items()):
        scenario, policy, reader, budget, context = key
        row = dict(scenario=scenario, policy=policy, reader=reader, budget=budget,
                   context=context, n_respondents=len(values), n_episodes=len(values))
        row.update({metric: float(np.mean([getattr(r, metric) for r in values]))
                    for metric in ("nll", "accuracy", "brier")})
        result.append(row)
        macro_groups[key[:4]].append(row)
    for key, values in sorted(macro_groups.items()):
        scenario, policy, reader, budget = key
        result.append(dict(scenario=scenario, policy=policy, reader=reader, budget=budget,
                           context="macro", n_respondents=len(macro_uuids[key]),
                           n_episodes=sum(r["n_episodes"] for r in values),
                           **{m: float(np.mean([r[m] for r in values])) for m in ("nll", "accuracy", "brier")}))
    return result


def context_macro(records: Iterable[ScoredCheckpoint], metric: str = "nll") -> dict[tuple[str, str, str, int], float]:
    if metric not in ("nll", "accuracy", "brier"):
        raise ValueError("Unknown metric")
    return {(r["scenario"], r["policy"], r["reader"], r["budget"]): r[metric]
            for r in summarize_scores(records) if r["context"] == "macro"}


def budget_average(values: Mapping[int, float], budgets: Sequence[int] = (3, 5, 7, 9)) -> float:
    if not budgets or any(b not in values or not np.isfinite(values[b]) for b in budgets):
        raise ValueError("All prespecified budgets need finite scores")
    return float(np.mean([values[b] for b in budgets]))


def paired_uuid_bootstrap(
    records: Iterable[ScoredCheckpoint], policy_a: str, policy_b: str,
    metric: str = "nll", reader: str = "common", budget: int | None = None,
    scenario: str | None = None, n_bootstrap: int = 2000,
    seed: int = 2026, confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap paired UUID clusters, retaining contexts/budgets together.

    Difference is A minus B. It is conditional on the input world and trained
    checkpoints; training seeds are not disguised as extra respondents.
    """
    if (policy_a == policy_b or metric not in ("nll", "accuracy", "brier")
            or n_bootstrap < 1 or not 0 < confidence < 1):
        raise ValueError("Invalid bootstrap configuration")
    selected = [r for r in records if r.reader == reader and r.policy in (policy_a, policy_b)
                and (budget is None or r.budget == budget) and (scenario is None or r.scenario == scenario)]
    worlds = {r.scenario for r in selected}
    if len(worlds) != 1:
        raise ValueError("Pair one response world/scenario at a time")
    grouped: dict[str, dict[tuple[str, str, int], float]] = {policy_a: {}, policy_b: {}}
    for r in selected:
        key = (r.uuid, r.context, r.budget)
        if key in grouped[r.policy]:
            raise ValueError("Duplicate paired score")
        value = float(getattr(r, metric))
        if not np.isfinite(value):
            raise ValueError("Nonfinite paired score")
        grouped[r.policy][key] = value
    if not grouped[policy_a] or grouped[policy_a].keys() != grouped[policy_b].keys():
        raise ValueError("Policies must contain identical UUID/context/budget episodes")
    by_uuid: dict[str, list[float]] = defaultdict(list)
    grids: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for key, value in grouped[policy_a].items():
        by_uuid[key[0]].append(value - grouped[policy_b][key])
        grids[key[0]].add((key[1], key[2]))
    if len({frozenset(grid) for grid in grids.values()}) != 1:
        raise ValueError("Every UUID must contain the same context/budget grid")
    # Each UUID contributes one mean, not 54 independent rows.
    differences = np.asarray([np.mean(by_uuid[u]) for u in sorted(by_uuid)], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        means[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    alpha = (1 - confidence) / 2
    return dict(policy_a=policy_a, policy_b=policy_b, reader=reader, metric=metric,
                scenario=next(iter(worlds)), budget=budget, n_uuids=len(differences),
                difference=float(differences.mean()),
                ci_lower=float(np.quantile(means, alpha)), ci_upper=float(np.quantile(means, 1 - alpha)),
                confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
