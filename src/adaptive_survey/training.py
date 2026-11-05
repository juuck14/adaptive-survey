"""Offline respondent-bootstrap training for the frozen DeepSets predictor.

Each supervised example is a respondent/context episode: public profile + a
labelled history, with every task outside that history as a prediction target.
Only training respondents' labels are used for optimizer updates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .neural import ChoiceNetwork, NeuralPredictor, collate_requests
from .schema import CONTEXTS, Observation, Profile, SurveyState, Task


@dataclass(frozen=True)
class TrainingConfig:
    ensemble_size: int = 5
    epochs: int = 120
    episodes_per_epoch: int = 1024
    batch_size: int = 64
    validation_episodes: int = 720
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    min_delta: float = 1e-5
    max_history: int = 14
    seed: int = 2026
    num_threads: int = 1

    def __post_init__(self) -> None:
        for field in ("ensemble_size", "epochs", "episodes_per_epoch", "batch_size",
                      "validation_episodes", "patience", "num_threads"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if not 0 <= self.max_history <= 14:
            raise ValueError("max_history must be between zero and fourteen")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("Invalid optimizer/early-stopping parameters")


@dataclass(frozen=True)
class TrainingResult:
    predictor: NeuralPredictor
    history: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _EpisodeSpec:
    profile: Profile
    context: str
    all_tasks: tuple[Task, ...]
    candidate_tasks: tuple[Task, ...]
    initial_tasks: tuple[Task, ...]
    choices: Mapping[str, int]


@dataclass(frozen=True)
class _Example:
    state: SurveyState
    targets: tuple[Task, ...]
    labels: tuple[int, ...]


def _ids(values: Sequence[Task | str], tasks: Mapping[str, Task]) -> tuple[Task, ...]:
    return tuple(tasks[value] if isinstance(value, str) else value for value in values)


def _build_specs(
    profiles: Mapping[str, Profile], tasks: Mapping[str, Task],
    responses: Mapping[tuple[str, str], int], uuids: Sequence[str],
    protocol_config: Any,
    candidate_task_ids: Mapping[str, Sequence[str]] | None,
    initial_task_ids: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[_EpisodeSpec, ...]]:
    by_context = {context: tuple(sorted(
        (task for task in tasks.values() if task.context == context), key=lambda task: task.task_id,
    )) for context in CONTEXTS}
    if protocol_config is not None:
        # Imported lazily: neural training can also run standalone on small toys.
        from .protocol import make_episode
    result: dict[str, tuple[_EpisodeSpec, ...]] = {}
    for uuid in uuids:
        if uuid not in profiles:
            raise ValueError(f"Missing public profile for respondent {uuid}")
        specs = []
        for context, context_tasks in by_context.items():
            if not context_tasks:
                continue
            if len(context_tasks) < 2:
                raise ValueError("Training requires at least two tasks per context")
            choices = {}
            for task in context_tasks:
                key = (uuid, task.task_id)
                if key not in responses:
                    raise ValueError(f"Missing training/validation response for {key}")
                value = responses[key]
                if not isinstance(value, (int, np.integer)) or not 0 <= value < 4:
                    raise ValueError(f"Invalid choice for {key}")
                choices[task.task_id] = int(value)
            if protocol_config is not None:
                episode = make_episode(profiles[uuid], context, tasks, protocol_config)
                candidates = _ids(episode.candidates, tasks)
                initials = _ids(episode.initial, tasks)
            else:
                candidates = _ids(candidate_task_ids[context], tasks) if candidate_task_ids else context_tasks
                initials = _ids(initial_task_ids.get(context, ()), tasks) if initial_task_ids else ()
            # Accommodate protocols returning either the full pool or unasked pool.
            candidate_lookup = {task.task_id: task for task in (*initials, *candidates)}
            candidates = tuple(candidate_lookup.values())
            context_ids = {task.task_id for task in context_tasks}
            if not candidates or not set(candidate_lookup).issubset(context_ids):
                raise ValueError("History candidates must be nonempty tasks in their context")
            if len({task.task_id for task in initials}) != len(initials):
                raise ValueError("Initial history tasks cannot repeat")
            specs.append(_EpisodeSpec(profiles[uuid], context, context_tasks, candidates, initials, choices))
        if not specs:
            raise ValueError("No contexts are available for training")
        result[uuid] = tuple(specs)
    return result


def _sample_example(spec: _EpisodeSpec, rng: np.random.Generator, max_history: int,
                    *, history_size: int | None = None) -> _Example:
    # A target must remain even for a tiny toy with fewer than eighteen tasks.
    limit = min(max_history, len(spec.candidate_tasks), len(spec.all_tasks) - 1)
    size = int(rng.integers(limit + 1)) if history_size is None else min(history_size, limit)
    initial = list(spec.initial_tasks[:size])
    remaining = [task for task in spec.candidate_tasks
                 if task.task_id not in {item.task_id for item in initial}]
    n_extra = size - len(initial)
    if n_extra:
        initial.extend(remaining[int(index)] for index in rng.choice(len(remaining), n_extra, replace=False))
    # Shuffling does not affect the network; it also prevents a position convention
    # from becoming an accidental implementation dependency.
    if initial:
        rng.shuffle(initial)
    seen = {task.task_id for task in initial}
    state = SurveyState(spec.profile, spec.context, tuple(
        Observation(task, spec.choices[task.task_id]) for task in initial
    ))
    targets = tuple(task for task in spec.all_tasks if task.task_id not in seen)
    return _Example(state, targets, tuple(spec.choices[task.task_id] for task in targets))


def _make_validation_bank(specs: Sequence[_EpisodeSpec], n_episodes: int,
                          max_history: int, seed: int) -> list[_Example]:
    """Use shuffled complete cycles, avoiding a fixed respondent-list prefix.

    Every respondent/context pair appears once per complete cycle. A smaller
    bank is a seeded sample without replacement from the entire validation
    cohort. Canonical input ordering makes the result stable under list reorder.
    """
    if not specs or n_episodes < 1:
        raise ValueError("The validation bank needs specs and a positive size")
    ordered = sorted(specs, key=lambda spec: (spec.profile.uuid, spec.context))
    rng = np.random.default_rng(seed)
    bank: list[_Example] = []
    while len(bank) < n_episodes:
        order = rng.permutation(len(ordered))
        for index in order[:n_episodes - len(bank)]:
            bank.append(_sample_example(ordered[int(index)], rng, max_history))
    return bank


def _validation_bank_hash(bank: Sequence[_Example]) -> str:
    """Hash sampled episode identities, histories, targets and their labels."""
    payload = [{
        "uuid": example.state.profile.uuid,
        "context": example.state.context,
        "history": [(item.task.task_id, item.choice) for item in example.state.history],
        "targets": [(task.task_id, choice) for task, choice in zip(example.targets, example.labels)],
    } for example in bank]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _loss(model: ChoiceNetwork, examples: Sequence[_Example]) -> torch.Tensor:
    batch = collate_requests([(example.state, example.targets) for example in examples])
    labels = torch.zeros(batch.query_mask.shape, dtype=torch.long)
    for row, example in enumerate(examples):
        labels[row, :len(example.labels)] = torch.tensor(example.labels, dtype=torch.long)
    logits = model(batch)
    per_target = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
    per_target = per_target.reshape(labels.shape) * batch.query_mask
    # Uniform episode weights: longer target lists do not overweight short histories.
    return (per_target.sum(dim=1) / batch.query_mask.sum(dim=1)).mean()


def _evaluate(model: ChoiceNetwork, examples: Sequence[_Example], batch_size: int) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            part = examples[start:start + batch_size]
            total += float(_loss(model, part)) * len(part)
    return total / len(examples)


def _uuid_hash(uuids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(uuids)).encode()).hexdigest()


def train_ensemble(
    profiles: Mapping[str, Profile], tasks: Mapping[str, Task],
    responses: Mapping[tuple[str, str], int],
    train_uuids: Sequence[str], val_uuids: Sequence[str], *,
    config: TrainingConfig | None = None,
    protocol_config: Any = None,
    candidate_task_ids: Mapping[str, Sequence[str]] | None = None,
    initial_task_ids: Mapping[str, Sequence[str]] | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train offline; UUID bootstraps keep all three contexts in one cluster.

    Pass ``protocol_config`` to use each respondent's evaluation-compatible
    holdout/initial split. Training respondents' holdout-labelled tasks remain
    supervised targets, but never history inputs. Validation labels are used only
    for a fixed-bank NLL and early stopping. Test UUIDs must be absent from both
    supplied splits. Policy-rollout history augmentation is deliberately deferred;
    this version uses uniform history cardinalities 0..14.
    """
    config = config or TrainingConfig()
    train_uuids = tuple(train_uuids)
    val_uuids = tuple(val_uuids)
    if not train_uuids or not val_uuids:
        raise ValueError("Nonempty respondent-level training and validation splits are required")
    if len(set(train_uuids)) != len(train_uuids) or len(set(val_uuids)) != len(val_uuids):
        raise ValueError("Respondent split lists cannot contain duplicates")
    if set(train_uuids) & set(val_uuids):
        raise ValueError("Training and validation respondents must be disjoint")
    if protocol_config is not None and (candidate_task_ids is not None or initial_task_ids is not None):
        raise ValueError("Use protocol_config or explicit task pools, not both")
    torch.set_num_threads(config.num_threads)
    specs = _build_specs(profiles, tasks, responses, (*train_uuids, *val_uuids),
                         protocol_config, candidate_task_ids, initial_task_ids)
    validation_specs = [spec for uuid in val_uuids for spec in specs[uuid]]
    # The bank is materialized once and shared by every epoch and ensemble member.
    validation_bank = _make_validation_bank(validation_specs, config.validation_episodes,
                                            config.max_history, config.seed + 900_001)
    histories: list[dict[str, Any]] = []
    models: list[ChoiceNetwork] = []
    bootstraps: list[dict[str, int]] = []
    best_epochs: list[int] = []
    best_optimizer_steps: list[int] = []
    optimizer_steps_run: list[int] = []
    for member in range(config.ensemble_size):
        member_seed = config.seed + 10_007 * (member + 1)
        rng = np.random.default_rng(member_seed)
        torch.manual_seed(member_seed)
        bootstrap = [train_uuids[int(index)] for index in rng.integers(len(train_uuids), size=len(train_uuids))]
        bootstraps.append(dict(Counter(bootstrap)))
        model = ChoiceNetwork()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        best_loss = _evaluate(model, validation_bank, config.batch_size)
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_epoch, best_step, stale, optimizer_steps = 0, 0, 0, 0
        histories.append({"member": member, "epoch": 0, "train_loss": None,
                          "optimizer_steps": 0, "val_loss": best_loss, "best": True})
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0.0
            n_completed = 0
            while n_completed < config.episodes_per_epoch:
                count = min(config.batch_size, config.episodes_per_epoch - n_completed)
                examples = []
                for _ in range(count):
                    uuid = bootstrap[int(rng.integers(len(bootstrap)))]
                    respondent_specs = specs[uuid]
                    spec = respondent_specs[int(rng.integers(len(respondent_specs)))]
                    examples.append(_sample_example(spec, rng, config.max_history))
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, examples)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite neural training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer_steps += 1
                total_loss += float(loss.detach()) * count
                n_completed += count
            val_loss = _evaluate(model, validation_bank, config.batch_size)
            improved = val_loss < best_loss - config.min_delta
            if improved:
                best_loss, best_epoch, best_step, stale = val_loss, epoch, optimizer_steps, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            histories.append({"member": member, "epoch": epoch,
                              "optimizer_steps": optimizer_steps,
                              "train_loss": total_loss / n_completed,
                              "val_loss": val_loss, "best": improved})
            if stale >= config.patience:
                break
        model.load_state_dict(best_state)
        models.append(model)
        best_epochs.append(best_epoch)
        best_optimizer_steps.append(best_step)
        optimizer_steps_run.append(optimizer_steps)
    protocol_metadata = asdict(protocol_config) if is_dataclass(protocol_config) else None
    bank_pairs = {(example.state.profile.uuid, example.state.context) for example in validation_bank}
    bank_context_counts = Counter(example.state.context for example in validation_bank)
    metadata = {
        "model_type": "deepsets_conditional_choice",
        "training_config": asdict(config),
        "protocol_config": protocol_metadata,
        "train_uuid_hash": _uuid_hash(train_uuids),
        "validation_uuid_hash": _uuid_hash(val_uuids),
        "train_respondents": len(train_uuids),
        "validation_respondents": len(val_uuids),
        "bootstrap_unit": "respondent_all_contexts",
        "bootstrap_counts": bootstraps,
        "best_epochs": best_epochs,
        "best_optimizer_steps": best_optimizer_steps,
        "optimizer_steps_run": optimizer_steps_run,
        "optimizer_steps_per_epoch": (config.episodes_per_epoch + config.batch_size - 1) // config.batch_size,
        "max_optimizer_steps_per_member": config.epochs * (
            (config.episodes_per_epoch + config.batch_size - 1) // config.batch_size
        ),
        "history_sampling": "uniform_cardinality_with_protocol_initial_prefix",
        "validation_bank_seed": config.seed + 900_001,
        "validation_bank_episodes": len(validation_bank),
        "validation_bank_respondents": len({uuid for uuid, _ in bank_pairs}),
        "validation_bank_respondent_context_pairs": len(bank_pairs),
        "validation_bank_context_counts": {context: bank_context_counts[context] for context in CONTEXTS},
        "validation_bank_context_respondents": {
            context: len({uuid for uuid, current in bank_pairs if current == context})
            for context in CONTEXTS
        },
        "validation_bank_hash": _validation_bank_hash(validation_bank),
        "validation_bank_sampling": "seeded_shuffled_spec_cycles",
        "rollout_augmentation": False,
        "target_weighting": "mean_per_episode",
    }
    predictor = NeuralPredictor(models, metadata=metadata, num_threads=config.num_threads)
    if checkpoint_path is not None:
        predictor.save(checkpoint_path)
    return TrainingResult(predictor, histories, metadata)
