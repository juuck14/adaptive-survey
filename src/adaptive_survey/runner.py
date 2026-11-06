"""Sequential survey execution with public traces and evaluator-only labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import Observation, Policy, Predictor, Profile, Task, checked_probabilities


@dataclass(frozen=True, slots=True)
class StepTrace:
    uuid: str
    context: str
    scenario: str
    policy: str
    step: int
    task_id: str
    choice: int
    initial: bool
    candidate_task_ids: tuple[str, ...]
    scores: Mapping[str, float]
    selection_seconds: float
    update_seconds: float
    track: str = "prediction"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    uuid: str
    context: str
    scenario: str
    policy: str
    budget: int
    holdout_task_ids: tuple[str, ...]
    queried_task_ids: tuple[str, ...]
    probabilities: tuple[tuple[float, ...], ...]
    common_probabilities: tuple[tuple[float, ...], ...] | None = None
    prediction_seconds: float = 0.0
    common_prediction_seconds: float = 0.0
    track: str = "prediction"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    checkpoints: tuple[Checkpoint, ...]
    trace: tuple[StepTrace, ...]
    observations: tuple[Observation, ...]


def episode_seed(seed: int, uuid: str, context: str) -> int:
    """Stable across Python processes; independent of traversal order."""
    data = json.dumps([int(seed), uuid, context], ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _probability_rows(values: np.ndarray, n: int) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in checked_probabilities(values, n))


def run_episode(
    profile: Profile,
    context: str,
    world: Any,
    episode: Any,
    predictor: Predictor,
    policy: Policy,
    common_predictor: Predictor | None = None,
    seed: int = 2026,
    budgets: Sequence[int] | None = None,
) -> EpisodeResult:
    """Run one independent UUID/context episode.

    ``episode.candidates`` includes initial tasks. The world is passed only to
    this executor; neither predictors nor policies receive it. Hidden labels
    are never read by this function, including at prediction checkpoints.
    """
    if getattr(episode, "uuid", profile.uuid) != profile.uuid or episode.context != context:
        raise ValueError("Episode does not match profile/context")
    candidates: tuple[Task, ...] = tuple(episode.candidates)
    initial: tuple[Task, ...] = tuple(episode.initial)
    holdout: tuple[Task, ...] = tuple(episode.holdout)
    candidate_ids = tuple(t.task_id for t in candidates)
    initial_ids = tuple(t.task_id for t in initial)
    holdout_ids = tuple(t.task_id for t in holdout)
    if len(set(candidate_ids)) != len(candidates) or len(set(holdout_ids)) != len(holdout):
        raise ValueError("Duplicate candidate or holdout tasks")
    track = getattr(episode, "track", "prediction")
    if track not in ("prediction", "preference"):
        raise ValueError("Unknown evaluation track")
    if set(candidate_ids) & set(holdout_ids) or (track == "prediction" and not holdout):
        raise ValueError("Prediction holdout must be nonempty and disjoint from candidates")
    if track == "preference" and holdout:
        raise ValueError("Preference collection must not reserve prediction tasks")
    if len(set(initial_ids)) != len(initial) or not set(initial_ids) <= set(candidate_ids):
        raise ValueError("Initial tasks must be unique candidate tasks")
    candidate_map = {t.task_id: t for t in candidates}
    if any(task != candidate_map[task.task_id] for task in initial):
        raise ValueError("Initial task attributes differ from the candidate catalog")
    if any(t.context != context for t in candidates + holdout):
        raise ValueError("Episode contains tasks from another context")
    requested = tuple(sorted(set(episode.budgets if budgets is None else budgets)))
    if not requested or any(not isinstance(b, (int, np.integer)) or b < 0 or b > len(candidates) for b in requested):
        raise ValueError("Budgets must be integers between zero and candidate count")
    if any(0 < b < len(initial) for b in requested):
        raise ValueError("A positive budget must include all initial questions")
    rng = np.random.default_rng(episode_seed(seed, profile.uuid, context))
    state = predictor.reset(profile, context)
    common_state = common_predictor.reset(profile, context) if common_predictor is not None else None
    if state.profile != profile or state.context != context or state.history:
        raise ValueError("Predictor reset must create an empty matching personal state")
    if common_state is not None and (common_state.profile != profile or common_state.context != context or common_state.history):
        raise ValueError("Common predictor reset must create an empty matching personal state")
    scenario = str(getattr(world, "scenario", "unspecified"))
    policy_name = str(policy.name)
    checkpoints: list[Checkpoint] = []
    trace: list[StepTrace] = []

    def save_checkpoint() -> None:
        current_budget = len(state.history)
        if current_budget not in requested:
            return
        before = perf_counter()
        native = _probability_rows(predictor.predict(state, holdout), len(holdout)) if holdout else ()
        prediction_seconds = perf_counter() - before
        common = None
        common_prediction_seconds = 0.0
        if common_predictor is not None and holdout:
            before = perf_counter()
            common = _probability_rows(common_predictor.predict(common_state, holdout), len(holdout))
            common_prediction_seconds = perf_counter() - before
        checkpoints.append(Checkpoint(profile.uuid, context, scenario, policy_name,
                                      current_budget, holdout_ids,
                                      tuple(o.task.task_id for o in state.history), native, common,
                                      prediction_seconds, common_prediction_seconds, track))

    save_checkpoint()
    for step in range(1, max(requested) + 1):
        observed_ids = {o.task.task_id for o in state.history}
        remaining = tuple(t for t in candidates if t.task_id not in observed_ids)
        is_initial = step <= len(initial)
        before = perf_counter()
        if is_initial:
            task = initial[step - 1]
            scores: dict[str, float] = {}
        else:
            selection = policy.select(predictor, state, remaining, rng)
            by_id = {t.task_id: t for t in remaining}
            if selection.task_id not in by_id:
                raise ValueError("Policy selected a hidden, repeated, or unknown task")
            task = by_id[selection.task_id]
            scores = {str(k): float(v) for k, v in selection.scores.items()}
            if not set(scores) <= set(by_id) or not all(np.isfinite(v) for v in scores.values()):
                raise ValueError("Policy returned invalid acquisition scores")
        selection_seconds = perf_counter() - before
        # Scope is the public candidate set, never the holdout set.
        choice = world.reveal(profile.uuid, task.task_id, candidate_ids)
        before = perf_counter()
        expected_history = state.history + (Observation(task, int(choice)),)
        state = predictor.observe(state, task, choice)
        if common_predictor is not None:
            common_state = common_predictor.observe(common_state, task, choice)
        update_seconds = perf_counter() - before
        if state.profile != profile or state.context != context or state.history != expected_history:
            raise ValueError("Predictor did not preserve the public observation history")
        if common_state is not None and common_state.history != state.history:
            raise ValueError("Common and native predictors received different histories")
        trace.append(StepTrace(profile.uuid, context, scenario, policy_name, step,
                               task.task_id, int(choice), is_initial,
                               tuple(t.task_id for t in remaining), scores,
                               selection_seconds, update_seconds, track))
        save_checkpoint()
    return EpisodeResult(tuple(checkpoints), tuple(trace), state.history)


def run_survey(
    profiles: Mapping[str, Profile],
    episodes: Iterable[Any],
    world: Any,
    predictor: Predictor,
    policy: Policy,
    common_predictor: Predictor | None = None,
    seed: int = 2026,
    budgets: Sequence[int] | None = None,
) -> tuple[EpisodeResult, ...]:
    """Reset all personal state for each episode, including across contexts."""
    return tuple(run_episode(profiles[e.uuid], e.context, world, e, predictor, policy,
                             common_predictor, seed, budgets) for e in episodes)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Write explicit public records; reject nonfinite JSON values."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            value = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
