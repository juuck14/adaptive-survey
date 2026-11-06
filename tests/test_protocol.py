from collections import Counter
from dataclasses import replace

import pytest

from adaptive_survey.data import load_tasks
from adaptive_survey.demo import generate_demo
from adaptive_survey.protocol import ProtocolConfig, make_episode, split_uuids
from adaptive_survey.schema import CONTEXT_BLOCKS, CONTEXTS, Profile


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    paths = generate_demo(tmp_path_factory.mktemp("protocol"), n_respondents=1)
    return load_tasks(paths["tasks"])


@pytest.mark.parametrize("context", CONTEXTS)
def test_context_episode_has_balanced_disjoint_pools_and_initial_subset(tasks, context):
    episode = make_episode("person", context, tasks)
    assert len(episode.holdout) == 4
    assert len(episode.candidates) == 14
    assert len(episode.initial) == 4
    assert Counter(task.block for task in episode.holdout) == dict.fromkeys(CONTEXT_BLOCKS[context], 2)
    assert Counter(task.block for task in episode.initial) == dict.fromkeys(CONTEXT_BLOCKS[context], 2)
    held = {task.task_id for task in episode.holdout}
    initial = {task.task_id for task in episode.initial}
    assert held.isdisjoint(episode.acquisition_ids)
    assert initial <= episode.acquisition_ids
    assert len(held | episode.acquisition_ids) == 18
    assert episode.budgets == (6, 9, 12)


def test_protocol_is_input_order_independent_and_ignores_profile_features(tasks):
    baseline = make_episode("person", "transfer", tasks)
    reordered = make_episode("person", "transfer", list(reversed(list(tasks.values()))))
    profile = Profile("person", (1.,) * 9)
    changed_features = make_episode(profile, "transfer", tasks)
    assert baseline == reordered == changed_features
    alternatives = {tuple(task.task_id for task in make_episode(str(i), "transfer", tasks).holdout)
                    for i in range(10)}
    assert len(alternatives) > 1


def test_predeclared_fixed_ranking_selects_earliest_eligible_in_each_block(tasks):
    rank = [task.task_id for task in reversed(list(tasks.values()))]
    episode = make_episode("person", "transfer", tasks, fixed_rank=rank)
    held = {task.task_id for task in episode.holdout}
    expected = [task_id for block in CONTEXT_BLOCKS["transfer"]
                for task_id in [item for item in rank if item.startswith(f"B{block}_") and item not in held][:2]]
    assert [task.task_id for task in episode.initial] == expected
    with pytest.raises(ValueError, match="duplicate"):
        make_episode("person", "transfer", tasks, fixed_rank=rank + rank[:1])
    with pytest.raises(ValueError, match="entire"):
        make_episode("person", "transfer", tasks, fixed_rank=[])


def test_contexts_are_separate_and_diagnostic_budgets_supported(tasks):
    config = ProtocolConfig(initial_per_block=1, budgets=(14, 0, 9, 2))
    episodes = [make_episode("person", context, tasks, config) for context in CONTEXTS]
    assert all(episode.budgets == (0, 2, 9, 14) for episode in episodes)
    all_ids = [episode.acquisition_ids | {task.task_id for task in episode.holdout}
               for episode in episodes]
    assert all_ids[0].isdisjoint(all_ids[1]) and all_ids[1].isdisjoint(all_ids[2])


@pytest.mark.parametrize("kwargs", [{"holdout_per_block": 0}, {"holdout_per_block": 9},
                                    {"holdout_per_block": 2.5}, {"initial_per_block": 8},
                                    {"initial_per_block": -1}, {"budgets": ()},
                                    {"budgets": (3, 3)}, {"budgets": (1,)},
                                    {"budgets": (19,)}, {"budgets": (6.0,)},
                                    {"budgets": (True,)}, {"seed": 2.5}])
def test_invalid_protocol_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        ProtocolConfig(**kwargs)


def test_full_pool_collection_keeps_all_requested_budgets_separate_from_prediction(tasks):
    config = ProtocolConfig()
    prediction = make_episode("person", "transfer", tasks, config)
    preference = make_episode("person", "transfer", tasks, config, track="preference")
    assert config.budgets == (6, 9, 12, 15, 18)
    assert prediction.track == "prediction" and prediction.budgets == (6, 9, 12)
    assert preference.track == "preference" and preference.budgets == config.budgets
    assert not preference.holdout and len(preference.candidates) == 18
    assert len(preference.initial) == len(prediction.initial) == 4
    assert Counter(t.block for t in preference.initial) == dict.fromkeys(CONTEXT_BLOCKS["transfer"], 2)
    only_full = ProtocolConfig(budgets=(18,))
    assert only_full.budgets_for("prediction") == ()
    with pytest.raises(ValueError, match="No feasible"):
        make_episode("person", "transfer", tasks, only_full)
    assert make_episode("person", "transfer", tasks, only_full, track="preference").budgets == (18,)


def test_invalid_context_task_pool_fails(tasks):
    with pytest.raises(ValueError, match="eighteen"):
        make_episode("person", "transfer", list(tasks.values())[1:])
    altered = list(tasks.values())
    altered[0] = replace(altered[0], block=3)
    with pytest.raises(ValueError, match="block"):
        make_episode("person", "transfer", altered)
    with pytest.raises(ValueError, match="context"):
        make_episode("person", "unknown", tasks)


def test_uuid_splits_are_exact_disjoint_and_stable():
    uuids = [f"synthetic-person-{index}" for index in range(9000)]
    splits = split_uuids(uuids, 6000, 1000, 2000)
    assert [len(splits[key]) for key in ("train", "validation", "test")] == [6000, 1000, 2000]
    assert len(set().union(*map(set, splits.values()))) == 9000
    assert splits == split_uuids(reversed(uuids), 6000, 1000, 2000)
    assert splits != split_uuids(uuids, 6000, 1000, 2000, seed=2027)


@pytest.mark.parametrize("uuids,sizes", [(["a", "a"], (1, 0, 0)), ([""], (1, 0, 0)),
                                        (["a"], (1, 1, 0)), (["a"], (-1, 0, 0)),
                                        (["a"], (1.0, 0, 0))])
def test_invalid_split_inputs_fail(uuids, sizes):
    with pytest.raises(ValueError):
        split_uuids(uuids, *sizes)
