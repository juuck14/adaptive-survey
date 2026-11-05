"""Behavioral checks for history conditioning, frozen inference and training."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_survey.neural import ChoiceNetwork, NeuralPredictor, collate_requests
from adaptive_survey.protocol import ProtocolConfig, make_episode
from adaptive_survey.schema import Observation, Profile, SurveyState, Task
from adaptive_survey.training import (
    TrainingConfig, _build_specs, _make_validation_bank, _sample_example,
    _validation_bank_hash, train_ensemble,
)


def make_tasks(n: int = 6, *, identical: bool = False) -> dict[str, Task]:
    rng = np.random.default_rng(42)
    result = {}
    for index in range(n):
        matrix = np.zeros((4, 13), dtype=float)
        if not identical:
            matrix[:3] = rng.integers(0, 2, size=(3, 13))
        task = Task(f"task-{index}", "transfer", 1 + index // 9,
                    1 + index % 9, "2H", tuple(map(tuple, matrix)))
        result[task.task_id] = task
    return result


def make_predictor() -> NeuralPredictor:
    torch.manual_seed(7)
    return NeuralPredictor([ChoiceNetwork(), ChoiceNetwork()])


def test_training_defaults_match_cli_and_record_actual_optimizer_steps() -> None:
    from adaptive_survey.cli import build_parser

    defaults = TrainingConfig()
    args = build_parser().parse_args(['train', '--data', 'unused-data', '--split', 'unused-split',
                                     '--output', 'unused-output'])
    assert defaults.epochs == args.epochs == 120
    assert defaults.patience == args.patience == 30
    assert defaults.validation_episodes == args.validation_episodes == 720
    assert defaults.episodes_per_epoch == args.episodes == 1024
    assert defaults.batch_size == args.batch_size == 64

    tasks = make_tasks(3, identical=True)
    train, validation = ('train-0', 'train-1'), ('validation-0',)
    profiles = {uuid: Profile(uuid, (0.0,) * 9) for uuid in (*train, *validation)}
    responses = {(uuid, task_id): 0 for uuid in profiles for task_id in tasks}
    # A partial final batch must still count as an optimizer update; each member
    # has its own cumulative counter and selected-checkpoint update count.
    config = TrainingConfig(ensemble_size=2, epochs=3, episodes_per_epoch=5, batch_size=3,
                            validation_episodes=3, patience=1, max_history=2, seed=17)
    result = train_ensemble(profiles, tasks, responses, train, validation, config=config)
    assert result.metadata['optimizer_steps_per_epoch'] == 2
    assert result.metadata['max_optimizer_steps_per_member'] == 6
    for member in range(2):
        rows = [row for row in result.history if row['member'] == member]
        assert rows[0]['optimizer_steps'] == 0
        assert all(row['optimizer_steps'] == 2 * row['epoch'] for row in rows)
        assert result.metadata['optimizer_steps_run'][member] == rows[-1]['optimizer_steps']
        best_row = [row for row in rows if row['best']][-1]
        assert result.metadata['best_optimizer_steps'][member] == best_row['optimizer_steps']
        assert result.metadata['best_epochs'][member] == best_row['epoch']


def test_history_order_and_identifiers_do_not_change_predictions() -> None:
    tasks = list(make_tasks().values())
    predictor = make_predictor()
    profile = Profile("person", (0.0,) * 9)
    observations = (Observation(tasks[0], 1), Observation(tasks[1], 3), Observation(tasks[2], 0))
    state = SurveyState(profile, "transfer", observations)
    reordered = replace(state, history=tuple(reversed(observations)))
    np.testing.assert_allclose(predictor.predict(state, tasks[3:]),
                               predictor.predict(reordered, tasks[3:]), atol=1e-7)
    renamed = replace(state, profile=replace(profile, uuid="unseen-identifier"), history=tuple(
        Observation(replace(item.task, task_id=f"renamed-{index}"), item.choice)
        for index, item in enumerate(observations)))
    queries = [replace(task, task_id=f"query-{index}") for index, task in enumerate(tasks[3:])]
    np.testing.assert_allclose(predictor.predict(state, tasks[3:]),
                               predictor.predict(renamed, queries), atol=1e-7)


def test_empty_history_has_explicit_count_and_empty_features() -> None:
    tasks = list(make_tasks(3, identical=True).values())
    model = ChoiceNetwork()
    state = SurveyState(Profile("p", (0.0,) * 9), "transfer")
    states = [state, state.append(tasks[0], 0), state.append(tasks[0], 0).append(tasks[1], 0)]
    features = model.state_features(collate_requests([(item, tasks[2:]) for item in states]))
    assert features.shape == (3, 78)
    np.testing.assert_allclose(features[:, -2].detach().numpy(), np.log1p([0, 1, 2]), rtol=1e-6)
    np.testing.assert_array_equal(features[:, -1].detach().numpy(), [1, 0, 0])
    # Repeated identical events have the same mean; count remains distinguishable.
    torch.testing.assert_close(features[1, :64], features[2, :64])
    assert torch.count_nonzero(features[0, :64]) == 0


def test_fantasies_are_immutable_and_do_not_change_global_weights() -> None:
    tasks = list(make_tasks().values())
    predictor = make_predictor()
    state = predictor.reset(Profile("p", (0.0,) * 9), "transfer")
    before = [{name: tensor.clone() for name, tensor in model.state_dict().items()}
              for model in predictor.models]
    original_predictions = predictor.predict(state, tasks)
    fantasy = predictor.fantasize(state, tasks[0], 2)
    observed = predictor.observe(state, tasks[0], 1)
    assert state.history == ()
    assert fantasy.history[0].choice == 2
    assert observed.history[0].choice == 1
    batched = predictor.predict_many([(state, ()), (fantasy, tasks[1:]), (observed, tasks[2:])])
    assert [p.shape for p in batched] == [(0, 4), (5, 4), (4, 4)]
    parallel = predictor.predict_many([fantasy, observed], [tasks[1:], tasks[2:]])
    for actual, expected in zip(parallel, batched[1:]):
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(predictor.predict(state, tasks), original_predictions)
    for model, previous in zip(predictor.models, before):
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, previous[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="twice"):
        predictor.observe(fantasy, tasks[0], 1)


def test_safe_checkpoint_roundtrip_and_schema_validation(tmp_path) -> None:
    tasks = list(make_tasks().values())
    predictor = make_predictor()
    predictor.metadata = {"seed": 3, "rollout_augmentation": False}
    state = predictor.reset(Profile("p", (0.0,) * 9), "transfer").append(tasks[0], 3)
    path = tmp_path / "ensemble.pt"
    predictor.save(path)
    restored = NeuralPredictor.load(path)
    assert restored.metadata == predictor.metadata
    assert len(restored.models) == 2
    np.testing.assert_array_equal(predictor.predict(state, tasks[1:]), restored.predict(state, tasks[1:]))
    payload = torch.load(path, weights_only=True)
    payload["architecture"]["profile"] = 10
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incompatible"):
        NeuralPredictor.load(path)


def test_training_histories_obey_uuid_protocol_holdout() -> None:
    tasks = make_tasks(18)
    profiles = {uuid: Profile(uuid, (0.0,) * 9) for uuid in ("train", "validation")}
    responses = {(uuid, task_id): index % 4 for uuid in profiles
                 for index, task_id in enumerate(tasks)}
    protocol = ProtocolConfig()
    specs = _build_specs(profiles, tasks, responses, tuple(profiles), protocol, None, None)
    rng = np.random.default_rng(13)
    for uuid, context_specs in specs.items():
        episode = make_episode(uuid, "transfer", tasks, protocol)
        held_ids = {task.task_id for task in episode.holdout}
        for size in range(15):
            example = _sample_example(context_specs[0], rng, 14, history_size=size)
            history_ids = {item.task.task_id for item in example.state.history}
            target_ids = {task.task_id for task in example.targets}
            assert len(history_ids) == size
            assert history_ids <= episode.acquisition_ids
            assert not history_ids & target_ids
            assert history_ids | target_ids == set(tasks)
            assert held_ids <= target_ids
            if size >= 2:
                assert {task.task_id for task in episode.initial} <= history_ids


def test_validation_bank_samples_whole_cohort_reproducibly() -> None:
    base_tasks = make_tasks(3)
    tasks = {}
    for context in ("transfer", "layover", "stopover"):
        for task in base_tasks.values():
            changed = replace(task, task_id=f"{context}-{task.task_id}", context=context)
            tasks[changed.task_id] = changed
    uuids = tuple(f"validation-{index:03d}" for index in range(120))
    profiles = {uuid: Profile(uuid, (0.0,) * 9) for uuid in uuids}
    responses = {(uuid, task_id): 0 for uuid in uuids for task_id in tasks}
    by_uuid = _build_specs(profiles, tasks, responses, uuids, None, None, None)
    specs = [spec for uuid in uuids for spec in by_uuid[uuid]]
    bank = _make_validation_bank(specs, 256, 2, 42)
    pairs = [(example.state.profile.uuid, example.state.context) for example in bank]
    assert len(set(pairs)) == 256  # Sampling without replacement within a cycle.
    # A prefix implementation would never reach these final thirty UUIDs.
    assert any(uuid in uuids[90:] for uuid, _ in pairs)
    assert _validation_bank_hash(bank) == _validation_bank_hash(
        _make_validation_bank(list(reversed(specs)), 256, 2, 42))
    assert _validation_bank_hash(bank) != _validation_bank_hash(
        _make_validation_bank(specs, 256, 2, 43))
    complete = _make_validation_bank(specs, len(specs) + 17, 2, 42)
    complete_pairs = [(example.state.profile.uuid, example.state.context) for example in complete]
    expected = {(uuid, context) for uuid in uuids for context in ("transfer", "layover", "stopover")}
    assert set(complete_pairs[:len(specs)]) == expected
    assert len(set(complete_pairs[len(specs):])) == 17


def test_offline_training_learns_personal_history_on_unseen_uuids() -> None:
    # Public profiles and query attributes are identical for both preference
    # groups. Only previous answers reveal each person's persistent latent class.
    tasks = make_tasks(6, identical=True)
    train = tuple(f"train-{index}" for index in range(16))
    validation = tuple(f"val-{index}" for index in range(8))
    profiles = {uuid: Profile(uuid, (0.0,) * 9) for uuid in (*train, *validation)}
    preferences = {uuid: index % 2 for cohort in (train, validation)
                   for index, uuid in enumerate(cohort)}
    responses = {(uuid, task_id): preferences[uuid] for uuid in profiles for task_id in tasks}
    config = TrainingConfig(ensemble_size=1, epochs=18, episodes_per_epoch=384,
                            validation_episodes=96, batch_size=32,
                            learning_rate=0.003, patience=5, max_history=5, seed=23)
    result = train_ensemble(profiles, tasks, responses, train, validation, config=config)
    initial_loss = result.history[0]["val_loss"]
    best_loss = min(row["val_loss"] for row in result.history)
    assert best_loss < initial_loss * 0.45
    query_tasks = list(tasks.values())
    empty_nll, observed_nll = [], []
    for uuid in validation:
        choice = preferences[uuid]
        empty = result.predictor.reset(profiles[uuid], "transfer")
        observed = result.predictor.observe(empty, query_tasks[0], choice)
        empty_nll.append(-np.log(result.predictor.predict(empty, query_tasks[1:])[:, choice]).mean())
        observed_nll.append(-np.log(result.predictor.predict(observed, query_tasks[1:])[:, choice]).mean())
    assert np.mean(observed_nll) < 0.2
    assert np.mean(empty_nll) > 0.5
    assert np.mean(observed_nll) < np.mean(empty_nll) * 0.3
    assert result.metadata["bootstrap_unit"] == "respondent_all_contexts"
    assert result.metadata["validation_respondents"] == len(validation)
    assert result.metadata["validation_bank_respondents"] == len(validation)
    assert result.metadata["validation_bank_context_counts"] == {"transfer": 96, "layover": 0, "stopover": 0}
    assert len(result.metadata["validation_bank_hash"]) == 64
    assert sum(result.metadata["bootstrap_counts"][0].values()) == len(train)
    assert not set(result.metadata["bootstrap_counts"][0]) & set(validation)
    with pytest.raises(ValueError, match="disjoint"):
        train_ensemble(profiles, tasks, responses, train, (train[0],), config=config)
