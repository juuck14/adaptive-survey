"""Candidate-only fantasy scores; these are predictive proxies, not posterior EIG."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .schema import Predictor, SurveyState, Task, checked_probabilities


def _probabilities(values: np.ndarray, n_tasks: int) -> np.ndarray:
    """Validate first, then remove harmless floating point sum error."""
    p = checked_probabilities(values, n_tasks)
    if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Choice probabilities must sum to one")
    return p / p.sum(axis=-1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    """Natural-log categorical entropy along the last (four-choice) axis."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] != 4:
        raise ValueError("Expected a final axis of four choice probabilities")
    flat = _probabilities(p.reshape(-1, 4), p.size // 4)
    logs = np.zeros_like(flat)
    np.log(flat, out=logs, where=flat > 0)
    return -(flat * logs).sum(axis=-1).reshape(p.shape[:-1])


def validate_candidates(state: SurveyState, candidates: Sequence[Task]) -> tuple[Task, ...]:
    """Canonical ordering makes seeded ties independent of input container order."""
    ordered = tuple(sorted(candidates, key=lambda task: task.task_id))
    if not ordered:
        raise ValueError("At least one remaining candidate is required")
    ids = [task.task_id for task in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate task IDs must be unique")
    observed = {observation.task.task_id for observation in state.history}
    if observed.intersection(ids):
        raise ValueError("An observed task cannot be a remaining candidate")
    if any(task.context != state.context for task in ordered):
        raise ValueError("Candidates must belong to the current context")
    return ordered


@dataclass(frozen=True, slots=True)
class WeightedJSResult:
    scores: Mapping[str, float]
    consistency_tv: Mapping[str, float] | None = None


def weighted_js_scores(
    predictor: Predictor,
    state: SurveyState,
    candidates: Sequence[Task],
    *,
    diagnostics: bool = False,
    negative_tolerance: float = 1e-10,
) -> WeightedJSResult:
    """Score q by average weighted JS over the other remaining candidates.

    The mixture uses the four possible answers to q, weighted by the current
    predictive probabilities. The supplied candidate pool is the entire target
    pool for acquisition: the evaluator must never pass held-out tasks here.
    A discriminative history model need not satisfy the law of total
    probability across its fantasy predictions, so optional TV diagnostics
    measure that mismatch without subtracting it from the nonnegative score.
    """
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0:
        raise ValueError("negative_tolerance must be finite and nonnegative")
    tasks = validate_candidates(state, candidates)
    if len(tasks) == 1:
        zero = MappingProxyType({tasks[0].task_id: 0.0})
        return WeightedJSResult(zero, zero if diagnostics else None)

    current = _probabilities(predictor.predict(state, tasks), len(tasks))
    requests: list[tuple[SurveyState, tuple[Task, ...]]] = []
    for q_index, query in enumerate(tasks):
        others = tasks[:q_index] + tasks[q_index + 1:]
        for answer in range(4):
            fantasy = predictor.fantasize(state, query, answer)
            expected = state.append(query, answer)
            if fantasy != expected:
                raise ValueError("A fantasy must only append its stated public observation")
            requests.append((fantasy, others))

    predict_many = getattr(predictor, "predict_many", None)
    if callable(predict_many):
        predictions = list(predict_many(requests))
    else:
        predictions = [predictor.predict(fantasy, others) for fantasy, others in requests]
    if len(predictions) != len(requests):
        raise ValueError("Batched predictor returned the wrong number of predictions")
    branches = np.stack([_probabilities(p, len(tasks) - 1) for p in predictions])

    scores: dict[str, float] = {}
    consistency: dict[str, float] = {}
    for q_index, query in enumerate(tasks):
        weights = current[q_index]
        conditionals = branches[4 * q_index:4 * q_index + 4]
        mixture = np.einsum("y,yrk->rk", weights, conditionals)
        gains = entropy(mixture) - np.einsum("y,yr->r", weights, entropy(conditionals))
        # JS is bounded by both H(answer to q) and log(4).
        upper_bound = float(entropy(weights))
        if (not np.all(np.isfinite(gains)) or np.any(gains < -negative_tolerance)
                or np.any(gains > upper_bound + negative_tolerance)):
            raise ValueError("Weighted JS fell outside its mathematical bounds")
        score = float(np.clip(gains, 0.0, upper_bound).mean())
        scores[query.task_id] = score
        if diagnostics:
            before = np.concatenate((current[:q_index], current[q_index + 1:]), axis=0)
            consistency[query.task_id] = float((0.5 * np.abs(mixture - before).sum(axis=-1)).mean())
    return WeightedJSResult(
        MappingProxyType(scores), MappingProxyType(consistency) if diagnostics else None
    )
