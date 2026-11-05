"""Transparent baselines trained only on the permitted population data."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .schema import BasePredictor, CONTEXTS, Profile, SurveyState, Task, checked_probabilities


class UniformPredictor(BasePredictor):
    """Useful for pipeline sanity checks, not a learned benchmark."""
    def predict(self, state: SurveyState, tasks: Sequence[Task]) -> np.ndarray:
        if any(task.context != state.context for task in tasks):
            raise ValueError("Prediction tasks must belong to the current context")
        return np.full((len(tasks), 4), 0.25, dtype=np.float64)

    def predict_many(self, requests):
        return [self.predict(state, tasks) for state, tasks in requests]


class PanelMAPPredictor(BasePredictor):
    """Population MNL plus a Gaussian-regularized personal coefficient deviation.

    Population coefficients are frozen. Revealed personal observations update
    only a MAP deviation, using the same documented feature map as estimation.
    This is a point-estimate baseline, not posterior integration or MMNL truth.
    """
    def __init__(self, coefficients: Sequence[float], prior_scale: float = 0.5,
                 max_iter: int = 200):
        from .estimation import DEFAULT_FEATURE_NAMES

        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (len(DEFAULT_FEATURE_NAMES),) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Population coefficients must match the estimation feature map")
        if not np.isfinite(prior_scale) or prior_scale <= 0:
            raise ValueError("prior_scale must be finite and positive")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        self.coefficients = coefficients.copy()
        self.coefficients.setflags(write=False)
        self.prior_scale = float(prior_scale)
        self.max_iter = max_iter
        self.fit_summary: dict = {}
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()

    @classmethod
    def fit(cls, records, *, prior_scale: float = 0.5, l2: float = 1e-4,
            max_iter: int = 500, drop_unidentified: bool = True):
        from .estimation import fit_pooled_mnl

        fitted = fit_pooled_mnl(records, l2=l2, drop_unidentified=drop_unidentified, max_iter=max_iter)
        result = cls(fitted.coefficients, prior_scale=prior_scale)
        result.fit_summary = {
            "rank": int(fitted.rank), "converged": bool(fitted.converged),
            "feature_names": list(fitted.feature_names),
            "active_features": list(fitted.active_features),
            "dropped_features": list(fitted.dropped_features),
            "log_likelihood": float(fitted.log_likelihood),
            "n_observations": int(fitted.n_observations),
            "n_respondents": int(fitted.n_respondents),
            "l2": float(l2), "population_max_iter": int(max_iter),
            "drop_unidentified": bool(drop_unidentified),
        }
        return result

    def _personal_coefficients(self, state: SurveyState) -> np.ndarray:
        from .estimation import design_matrix

        if not state.history:
            return self.coefficients
        ordered_history = tuple(sorted(state.history, key=lambda obs: obs.task.task_id))
        key = (state.profile.features, state.context, ordered_history)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if any(obs.task.context != state.context for obs in ordered_history):
            raise ValueError("History must belong to the current context")
        x = np.stack([design_matrix(obs.task, state.profile) for obs in ordered_history])
        labels = np.asarray([obs.choice for obs in ordered_history], dtype=int)
        if any(not isinstance(obs.choice, (int, np.integer)) or not 0 <= obs.choice < 4
               for obs in ordered_history):
            raise ValueError("History choices must be integer alternatives in [0, 3]")
        one_hot = np.eye(4)[labels]
        precision = 1.0 / self.prior_scale ** 2

        def objective(delta):
            logits = np.einsum("naf,f->na", x, self.coefficients + delta)
            loss = (logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]).sum()
            loss += 0.5 * precision * np.dot(delta, delta)
            gradient = np.einsum("naf,na->f", x, softmax(logits, axis=1) - one_hot) + precision * delta
            return float(loss), gradient

        result = minimize(objective, np.zeros_like(self.coefficients), jac=True,
                          method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7})
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Personal MAP optimization failed: {result.message}")
        coefficients = self.coefficients + result.x
        coefficients.setflags(write=False)
        self._cache[key] = coefficients
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return coefficients

    def predict(self, state: SurveyState, tasks: Sequence[Task]) -> np.ndarray:
        from .estimation import design_matrix

        if any(task.context != state.context for task in tasks):
            raise ValueError("Prediction tasks must belong to the current context")
        if not tasks:
            return np.empty((0, 4), dtype=np.float64)
        coefficients = self._personal_coefficients(state)
        x = np.stack([design_matrix(task, state.profile) for task in tasks])
        probabilities = softmax(np.einsum("naf,f->na", x, coefficients), axis=1)
        return checked_probabilities(probabilities, len(tasks))

    def predict_many(self, requests):
        return [self.predict(state, tasks) for state, tasks in requests]

    def save(self, path: str | Path) -> None:
        from .estimation import DEFAULT_FEATURE_NAMES

        payload = {
            "kind": "panel_map", "version": 1, "feature_names": list(DEFAULT_FEATURE_NAMES),
            "coefficients": self.coefficients.tolist(), "prior_scale": self.prior_scale,
            "max_iter": self.max_iter, "fit_summary": self.fit_summary,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path):
        from .estimation import DEFAULT_FEATURE_NAMES

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload.get("kind") != "panel_map" or payload.get("version") != 1
                or payload.get("feature_names") != list(DEFAULT_FEATURE_NAMES)):
            raise ValueError("Incompatible panel MAP checkpoint or feature map")
        result = cls(payload["coefficients"], payload["prior_scale"], payload["max_iter"])
        result.fit_summary = payload.get("fit_summary", {})
        return result


def greedy_fisher_rankings(tasks: Mapping[str, Task] | Sequence[Task],
                           profiles: Mapping[str, Profile] | Sequence[Profile],
                           coefficients: Sequence[float], ridge: float = 1.0,
                           max_profiles: int = 256) -> dict[str, list[str]]:
    """Preplan greedy log-det rankings from train-profile average MNL Fisher.

    This uses an estimated population coefficient vector. It does not claim
    the original task bank's design efficiency or access generator parameters.
    """
    from .estimation import DEFAULT_FEATURE_NAMES, design_matrix

    beta = np.asarray(coefficients, dtype=np.float64)
    if beta.shape != (len(DEFAULT_FEATURE_NAMES),) or not np.all(np.isfinite(beta)):
        raise ValueError("Coefficients must match the estimation feature map")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Fisher ridge must be finite and positive")
    if not isinstance(max_profiles, int) or max_profiles <= 0:
        raise ValueError("max_profiles must be a positive integer")
    profile_values = profiles.values() if isinstance(profiles, Mapping) else profiles
    task_values = tasks.values() if isinstance(tasks, Mapping) else tasks
    profiles = tuple(sorted(profile_values, key=lambda profile: profile.uuid))
    tasks = tuple(sorted(task_values, key=lambda task: task.task_id))
    if not profiles or not tasks:
        raise ValueError("Training profiles and tasks must be nonempty")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique")
    if any(task.context not in CONTEXTS for task in tasks):
        raise ValueError("Unknown task context")
    if len(profiles) > max_profiles:
        indices = np.linspace(0, len(profiles) - 1, max_profiles, dtype=int)
        profiles = tuple(profiles[int(index)] for index in indices)

    fisher: dict[str, np.ndarray] = {}
    for task in tasks:
        information = np.zeros((len(beta), len(beta)), dtype=np.float64)
        for profile in profiles:
            x = np.asarray(design_matrix(task, profile), dtype=np.float64)
            probabilities = softmax(x @ beta)
            centered = x - probabilities @ x
            information += (centered.T * probabilities) @ centered
        fisher[task.task_id] = information / len(profiles)

    rankings: dict[str, list[str]] = {}
    for context in CONTEXTS:
        remaining = [task.task_id for task in tasks if task.context == context]
        if not remaining:
            continue
        precision = ridge * np.eye(len(beta))
        ranking: list[str] = []
        while remaining:
            gains = []
            for task_id in remaining:
                sign, logdet = np.linalg.slogdet(precision + fisher[task_id])
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError("Regularized Fisher matrix must be positive definite")
                gains.append(float(logdet))
            maximum = max(gains)
            index = next(index for index, gain in enumerate(gains) if gain >= maximum - 1e-12)
            task_id = remaining.pop(index)
            ranking.append(task_id)
            precision = precision + fisher[task_id]
        rankings[context] = ranking
    return rankings
