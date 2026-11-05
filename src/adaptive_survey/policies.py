"""Policies share the same public state and remaining-candidate interface."""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .acquisition import entropy, validate_candidates, weighted_js_scores
from .baselines import greedy_fisher_rankings
from .schema import Predictor, Selection, SurveyState, Task, checked_probabilities


def _choose(scores: Mapping[str, float], rng: np.random.Generator, tie_atol: float) -> Selection:
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError("tie_atol must be finite and nonnegative")
    ids = sorted(scores)
    values = np.asarray([scores[task_id] for task_id in ids], dtype=np.float64)
    if not len(ids) or not np.all(np.isfinite(values)):
        raise ValueError("Candidate scores must be nonempty and finite")
    best = np.flatnonzero(values >= values.max() - tie_atol)
    chosen = ids[int(best[int(rng.integers(len(best)))])]
    return Selection(chosen, MappingProxyType(dict(scores)))


class RandomPolicy:
    name = "random"

    def select(self, predictor: Predictor, state: SurveyState,
               candidates: Sequence[Task], rng: np.random.Generator) -> Selection:
        tasks = validate_candidates(state, candidates)
        return _choose({task.task_id: 0.0 for task in tasks}, rng, 0.0)


class FixedPolicy:
    """A preplanned ranking, fixed before evaluation histories are observed."""
    name = "fixed"

    def __init__(self, rankings: Mapping[str, Sequence[str]] | Sequence[str]):
        if isinstance(rankings, Mapping):
            self.rankings = {context: self._ranking(order) for context, order in rankings.items()}
            self.default_ranking: tuple[str, ...] | None = None
        else:
            self.rankings = {}
            self.default_ranking = self._ranking(rankings)

    @staticmethod
    def _ranking(order: Sequence[str]) -> tuple[str, ...]:
        if isinstance(order, str):
            raise ValueError("A fixed ranking must be a sequence of task IDs")
        result = tuple(order)
        if not result or len(set(result)) != len(result) or any(not isinstance(x, str) for x in result):
            raise ValueError("A fixed ranking must contain unique task IDs")
        return result

    def select(self, predictor: Predictor, state: SurveyState,
               candidates: Sequence[Task], rng: np.random.Generator) -> Selection:
        tasks = validate_candidates(state, candidates)
        ranking = self.rankings.get(state.context, self.default_ranking)
        if ranking is None:
            raise ValueError(f"No fixed ranking for context {state.context}")
        positions = {task_id: index for index, task_id in enumerate(ranking)}
        if any(task.task_id not in positions for task in tasks):
            raise ValueError("Fixed ranking does not cover every remaining candidate")
        scores = {task.task_id: -float(positions[task.task_id]) for task in tasks}
        return _choose(scores, rng, 0.0)


class EntropyPolicy:
    name = "entropy"

    def __init__(self, tie_atol: float = 1e-12):
        self.tie_atol = tie_atol

    def select(self, predictor: Predictor, state: SurveyState,
               candidates: Sequence[Task], rng: np.random.Generator) -> Selection:
        tasks = validate_candidates(state, candidates)
        p = checked_probabilities(predictor.predict(state, tasks), len(tasks))
        scores = {task.task_id: float(score) for task, score in zip(tasks, entropy(p), strict=True)}
        return _choose(scores, rng, self.tie_atol)


class WeightedJSPolicy:
    name = "weighted_js"

    def __init__(self, diagnostics: bool = False, tie_atol: float = 1e-12):
        self.diagnostics = diagnostics
        self.tie_atol = tie_atol
        self.last_consistency_tv: Mapping[str, float] | None = None

    def select(self, predictor: Predictor, state: SurveyState,
               candidates: Sequence[Task], rng: np.random.Generator) -> Selection:
        result = weighted_js_scores(predictor, state, candidates, diagnostics=self.diagnostics)
        self.last_consistency_tv = result.consistency_tv
        return _choose(result.scores, rng, self.tie_atol)


def make_policy(name: str, *, rankings: Mapping[str, Sequence[str]] | Sequence[str] | None = None):
    """Use explicit names: weighted JS is neither BALD nor Bayesian EIG."""
    if name == "random":
        return RandomPolicy()
    if name == "entropy":
        return EntropyPolicy()
    if name == "weighted_js":
        return WeightedJSPolicy()
    if name == "fixed":
        if rankings is None:
            raise ValueError("The fixed policy requires a training-derived ranking")
        return FixedPolicy(rankings)
    raise ValueError(f"Unknown policy: {name}")
