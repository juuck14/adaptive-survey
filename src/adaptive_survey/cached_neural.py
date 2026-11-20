"""Frozen-network inference with cached task/event encodings.

The same float32 layers are evaluated with the first utility layer factored
into its state and task contributions. Rounding can differ at float32 scale;
probabilities are checked against the original evaluator before production.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .neural import NeuralPredictor, _task_matrix
from .schema import CONTEXTS, checked_probabilities


class CachedNeuralPredictor(NeuralPredictor):
    def __init__(self, predictor, tasks):
        super().__init__(predictor.models, metadata=predictor.metadata)
        self._catalog_objects = tuple(tasks.values())
        self._tasks = {self._key(t): t for t in tasks.values()}
        self._indices = {key: index for index, key in enumerate(self._tasks)}
        values = list(self._tasks.values())
        self._object_indices = {id(t): self._indices[self._key(t)] for t in tasks.values()}
        contexts = np.eye(3, dtype=np.float32)[[CONTEXTS.index(t.context) for t in values]]
        matrices = np.stack([_task_matrix(t, t.context) for t in values])
        self._events, self._query_offsets = [], []
        with torch.inference_mode():
            for model in self.models:
                events = np.concatenate((
                    np.repeat(matrices.reshape(-1, 52), 4, axis=0),
                    np.repeat(contexts, 4, axis=0), np.tile(np.eye(4, dtype=np.float32), (len(values), 1)),
                ), axis=1)
                self._events.append(model.event_encoder(torch.from_numpy(events)).reshape(-1, 4, 64))
                query = model.query_encoder(torch.from_numpy(matrices.reshape(-1, 52)))
                tail = torch.cat((query[:, None, :].expand(-1, 4, -1), torch.from_numpy(matrices),
                                  torch.eye(4)[None].expand(len(values), -1, -1),
                                  torch.from_numpy(contexts)[:, None, :].expand(-1, 4, -1)), dim=-1)
                first = model.utility[0]
                self._query_offsets.append(F.linear(tail, first.weight[:, 32:], first.bias))

    @staticmethod
    def _key(task):
        # Identifiers are intentionally excluded: equal public content is equal input.
        return task.context, task.attributes

    def _index(self, task):
        index = self._object_indices.get(id(task))
        return self._indices[self._key(task)] if index is None else index

    def predict_many(self, states, tasks_per_state=None):
        if tasks_per_state is not None and len(states) != len(tasks_per_state):
            raise ValueError("states and tasks_per_state must have equal length")
        requests = list(states) if tasks_per_state is None else list(zip(states, tasks_per_state))
        if not requests:
            return []
        features, owners, queries, lengths = [], [], [], []
        max_history = max(len(s.history) for s, _ in requests)
        history_indices = np.zeros((len(requests), max_history), dtype=np.int64)
        history_mask = np.zeros((len(requests), max_history, 1), dtype=np.float32)
        for owner, (state, tasks) in enumerate(requests):
            if state.context not in CONTEXTS or len(state.profile.features) != 9:
                raise ValueError("Invalid public state")
            if not np.isfinite(state.profile.features).all():
                raise ValueError("Invalid public profile")
            if len({o.task.task_id for o in state.history}) != len(state.history):
                raise ValueError("A history task cannot occur twice")
            for task in (*[o.task for o in state.history], *tasks):
                if task.context != state.context:
                    raise ValueError("Unknown or cross-context public task")
                self._index(task)
            for position, observation in enumerate(state.history):
                if not isinstance(observation.choice, (int, np.integer)) or not 0 <= observation.choice < 4:
                    raise ValueError("Invalid observed choice")
                history_indices[owner, position] = 4 * self._index(observation.task) + observation.choice
                history_mask[owner, position, 0] = 1
            features.append(state.profile.features)
            lengths.append(len(tasks))
            owners.extend([owner] * len(tasks))
            queries.extend(self._index(t) for t in tasks)
        if not owners:
            return [np.empty((0, 4)) for _ in requests]
        profiles = torch.tensor(features, dtype=torch.float32)
        contexts = torch.eye(3)[[CONTEXTS.index(s.context) for s, _ in requests]]
        count = torch.tensor([[len(s.history)] for s, _ in requests], dtype=torch.float32)
        owner_index, query_index = torch.tensor(owners), torch.tensor(queries)
        history_index = torch.from_numpy(history_indices)
        mask = torch.from_numpy(history_mask)
        probabilities = []
        with torch.inference_mode():
            for member, model in enumerate(self.models):
                pooled = ((self._events[member].reshape(-1, 64)[history_index] * mask).sum(dim=1)
                          / count.clamp_min(1))
                encoded = model.state_encoder(torch.cat((pooled, profiles, contexts,
                                                          torch.log1p(count), (count == 0).float()), dim=-1))
                state_contribution = F.linear(encoded, model.utility[0].weight[:, :32])
                hidden = self._query_offsets[member][query_index] + state_contribution[owner_index, None, :]
                logits = model.utility[2](hidden.relu()).squeeze(-1)
                probabilities.append(logits.softmax(dim=-1))
            averaged = torch.stack(probabilities).mean(dim=0).numpy().astype(np.float64)
        result, offset = [], 0
        for length in lengths:
            part = averaged[offset:offset + length]
            if length:
                part /= part.sum(axis=1, keepdims=True)
            result.append(checked_probabilities(part, length))
            offset += length
        return result
