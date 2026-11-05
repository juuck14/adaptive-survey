"""A frozen, history-conditioned DeepSets choice predictor.

Only public profile fields, choice-set attributes, context and observed answers
enter the network. Identifiers are used for bookkeeping, never as features.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .schema import (
    ALTERNATIVES, ATTRIBUTES, CONTEXTS, PERSON_FIELDS, BasePredictor,
    SurveyState, Task, checked_probabilities,
)


CHECKPOINT_VERSION = 1
ARCHITECTURE = {
    "profile": 9, "context": 3, "alternatives": 4, "attributes": 13,
    "event": [59, 64, 64], "state": [78, 64, 32],
    "query_set": [52, 64, 32], "utility": [84, 64, 1],
}


@dataclass(frozen=True)
class ChoiceBatch:
    profile: Tensor                 # [B, 9]
    context: Tensor                 # [B, 3]
    history_events: Tensor          # [B, H, 59]
    history_mask: Tensor            # [B, H]
    query_attributes: Tensor       # [B, Q, 4, 13]
    query_mask: Tensor              # [B, Q]
    lengths: tuple[int, ...]


def _task_matrix(task: Task, context: str) -> np.ndarray:
    if task.context != context:
        raise ValueError("A query/history task belongs to another context")
    matrix = task.matrix
    if matrix.shape != (4, 13) or not np.isfinite(matrix).all():
        raise ValueError("Every task needs finite attributes with shape [4, 13]")
    if np.any(matrix[3] != 0):
        raise ValueError("NONE must have thirteen zero attributes")
    return matrix


def collate_requests(
    requests: Sequence[tuple[SurveyState, Sequence[Task]]],
) -> ChoiceBatch:
    """Pad histories/queries; neither UUID nor task ID becomes a tensor input."""
    if not requests:
        raise ValueError("At least one prediction request is required")
    batch_size = len(requests)
    max_h = max(len(state.history) for state, _ in requests)
    max_q = max(len(tasks) for _, tasks in requests)
    profile = np.zeros((batch_size, 9), dtype=np.float32)
    context = np.zeros((batch_size, 3), dtype=np.float32)
    events = np.zeros((batch_size, max_h, 59), dtype=np.float32)
    history_mask = np.zeros((batch_size, max_h), dtype=np.float32)
    attributes = np.zeros((batch_size, max_q, 4, 13), dtype=np.float32)
    query_mask = np.zeros((batch_size, max_q), dtype=np.float32)
    lengths: list[int] = []
    labels = np.eye(4, dtype=np.float32)
    for row, (state, tasks) in enumerate(requests):
        features = np.asarray(state.profile.features, dtype=np.float32)
        if features.shape != (9,) or not np.isfinite(features).all():
            raise ValueError("Profile must contain nine finite public features")
        if state.context not in CONTEXTS:
            raise ValueError("Unknown context")
        profile[row] = features
        context[row, CONTEXTS.index(state.context)] = 1
        seen: set[str] = set()
        for column, observation in enumerate(state.history):
            if observation.task.task_id in seen:
                raise ValueError("A history task cannot occur twice")
            seen.add(observation.task.task_id)
            choice = observation.choice
            if not isinstance(choice, (int, np.integer)) or not 0 <= choice < 4:
                raise ValueError("Observed choice must be an integer in [0, 3]")
            matrix = _task_matrix(observation.task, state.context)
            events[row, column] = np.concatenate((matrix.reshape(-1), context[row], labels[choice]))
            history_mask[row, column] = 1
        lengths.append(len(tasks))
        for column, task in enumerate(tasks):
            attributes[row, column] = _task_matrix(task, state.context)
            query_mask[row, column] = 1
    return ChoiceBatch(
        *(torch.from_numpy(array) for array in
          (profile, context, events, history_mask, attributes, query_mask)),
        tuple(lengths),
    )


def _mlp(n_input: int, n_hidden: int, n_output: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))


class ChoiceNetwork(nn.Module):
    """Shared alternative utility, conditioned on a permutation-invariant history."""

    def __init__(self) -> None:
        super().__init__()
        self.event_encoder = _mlp(59, 64, 64)
        self.state_encoder = _mlp(78, 64, 32)
        self.query_encoder = _mlp(52, 64, 32)
        self.utility = _mlp(84, 64, 1)

    def state_features(self, batch: ChoiceBatch) -> Tensor:
        """Expose the explicit count/empty indicators for inspection and tests."""
        count = batch.history_mask.sum(dim=1, keepdim=True)
        if batch.history_events.shape[1]:
            encoded = self.event_encoder(batch.history_events)
            pooled = (encoded * batch.history_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / count.clamp_min(1)
        else:
            pooled = batch.profile.new_zeros((batch.profile.shape[0], 64))
        return torch.cat((pooled, batch.profile, batch.context,
                          torch.log1p(count), (count == 0).to(count.dtype)), dim=-1)

    def forward(self, batch: ChoiceBatch) -> Tensor:
        n_batch, n_queries = batch.query_attributes.shape[:2]
        if not n_queries:
            return batch.profile.new_empty((n_batch, 0, 4))
        state = self.state_encoder(self.state_features(batch))
        query = self.query_encoder(batch.query_attributes.flatten(start_dim=2))
        alternative_labels = torch.eye(4, dtype=state.dtype, device=state.device)
        alternative_labels = alternative_labels.view(1, 1, 4, 4).expand(n_batch, n_queries, -1, -1)
        utilities = torch.cat((
            state[:, None, None, :].expand(-1, n_queries, 4, -1),
            query[:, :, None, :].expand(-1, -1, 4, -1),
            batch.query_attributes,
            alternative_labels,
            batch.context[:, None, None, :].expand(-1, n_queries, 4, -1),
        ), dim=-1)
        return self.utility(utilities).squeeze(-1)


def _json_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    # Primitive-only metadata also keeps torch.load(weights_only=True) usable.
    try:
        return json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint metadata must contain finite JSON values") from error


class NeuralPredictor(BasePredictor):
    """Average softmax predictions of frozen networks; observations update state only.

    ``predict_many`` batches virtual histories for acquisition. Ensemble disagreement
    alone is not an estimate of personal information gain.
    """

    def __init__(self, models: Sequence[ChoiceNetwork], *,
                 metadata: Mapping[str, Any] | None = None,
                 num_threads: int = 1) -> None:
        if not models:
            raise ValueError("An ensemble must contain at least one model")
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)
        self.models = tuple(model.to("cpu").eval().requires_grad_(False) for model in models)
        self.metadata = _json_metadata(metadata)

    def predict(self, state: SurveyState, tasks: Sequence[Task]) -> np.ndarray:
        return self.predict_many([(state, tasks)])[0]

    def predict_many(
        self,
        states: Sequence[SurveyState] | Sequence[tuple[SurveyState, Sequence[Task]]],
        tasks_per_state: Sequence[Sequence[Task]] | None = None,
    ) -> list[np.ndarray]:
        """Accept either ``[(state, tasks), ...]`` or two parallel sequences."""
        if tasks_per_state is None:
            requests = list(states)
        else:
            if len(states) != len(tasks_per_state):
                raise ValueError("states and tasks_per_state must have equal length")
            requests = list(zip(states, tasks_per_state))
        if not requests:
            return []
        batch = collate_requests(requests)  # type: ignore[arg-type]
        with torch.inference_mode():
            probabilities = torch.stack([model(batch).softmax(dim=-1) for model in self.models]).mean(dim=0)
        output: list[np.ndarray] = []
        for row, length in enumerate(batch.lengths):
            p = probabilities[row, :length].numpy().astype(np.float64)
            if length:
                p /= p.sum(axis=1, keepdims=True)
            output.append(checked_probabilities(p, length))
        return output

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "feature_names": list(PERSON_FIELDS),
            "attribute_names": list(ATTRIBUTES),
            "alternative_order": list(ALTERNATIVES),
            "context_order": list(CONTEXTS),
            "metadata": _json_metadata(self.metadata),
            "state_dicts": [model.state_dict() for model in self.models],
        }
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path, *, num_threads: int = 1) -> NeuralPredictor:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "feature_names": list(PERSON_FIELDS),
            "attribute_names": list(ATTRIBUTES),
            "alternative_order": list(ALTERNATIVES),
            "context_order": list(CONTEXTS),
        }
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint version, dimensions or feature schema is incompatible")
        state_dicts = payload.get("state_dicts")
        if not isinstance(state_dicts, list) or not state_dicts:
            raise ValueError("Checkpoint has no ensemble weights")
        models = []
        for state_dict in state_dicts:
            model = ChoiceNetwork()
            model.load_state_dict(state_dict, strict=True)
            models.append(model)
        return cls(models, metadata=payload.get("metadata", {}), num_threads=num_threads)
