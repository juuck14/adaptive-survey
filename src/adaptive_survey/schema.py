"""Public, immutable inputs shared by every learner and policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

ATTRIBUTES = ("DNS_B", "DNS_A", "MSA_P", "MSA_F", "WIFI_B", "WIFI_H", "PERF_A", "PERF_F", "ENG_B", "ENG_F", "ENV_2", "ENV_3", "CTSA_A")
PERSON_FIELDS = ("GEN_Z", "GEND_F", "GEND_O", "GEND_PNO", "EDU_1", "EDU_2", "EDU_3", "EDU_5", "EDU_6")
ALTERNATIVES = ("A", "B", "C", "NONE")
CONTEXTS = ("transfer", "layover", "stopover")
CONTEXT_BLOCKS = {"transfer": (1, 2), "layover": (3, 4), "stopover": (5, 6)}
CONTEXT_TIMES = {"transfer": "2H", "layover": "12H", "stopover": "24H"}


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    context: str
    block: int
    cset: int
    time_code: str
    attributes: tuple[tuple[float, ...], ...]

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.attributes, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Profile:
    uuid: str
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    task: Task
    choice: int


@dataclass(frozen=True, slots=True)
class SurveyState:
    profile: Profile
    context: str
    history: tuple[Observation, ...] = ()

    def append(self, task: Task, choice: int) -> SurveyState:
        if task.context != self.context:
            raise ValueError("Observation belongs to another context")
        if not isinstance(choice, (int, np.integer)) or not 0 <= choice < 4:
            raise ValueError("Choice must be an integer in [0, 3]")
        if any(o.task.task_id == task.task_id for o in self.history):
            raise ValueError("A task cannot be observed twice")
        return SurveyState(self.profile, self.context, self.history + (Observation(task, choice),))


class Predictor(Protocol):
    def reset(self, profile: Profile, context: str) -> SurveyState: ...
    def predict(self, state: SurveyState, tasks: Sequence[Task]) -> np.ndarray: ...
    def observe(self, state: SurveyState, task: Task, choice: int) -> SurveyState: ...
    def fantasize(self, state: SurveyState, task: Task, choice: int) -> SurveyState: ...


class BasePredictor:
    """Only public history changes online; model weights remain frozen."""
    def reset(self, profile: Profile, context: str) -> SurveyState:
        if context not in CONTEXTS:
            raise ValueError("Unknown context")
        return SurveyState(profile, context)

    def observe(self, state: SurveyState, task: Task, choice: int) -> SurveyState:
        return state.append(task, choice)

    def fantasize(self, state: SurveyState, task: Task, choice: int) -> SurveyState:
        return state.append(task, choice)


@dataclass(frozen=True, slots=True)
class Selection:
    task_id: str
    scores: Mapping[str, float]


class Policy(Protocol):
    name: str
    def select(self, predictor: Predictor, state: SurveyState,
               candidates: Sequence[Task], rng: np.random.Generator) -> Selection: ...


def checked_probabilities(values: np.ndarray, n_tasks: int) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (n_tasks, 4) or not np.all(np.isfinite(p)):
        raise ValueError("Expected finite probabilities with shape [tasks, 4]")
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid choice probabilities")
    return p
