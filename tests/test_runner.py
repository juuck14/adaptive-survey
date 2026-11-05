from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_survey.metrics import score_checkpoint
from adaptive_survey.protocol import ProtocolConfig, make_episode
from adaptive_survey.runner import run_episode, run_survey, write_jsonl
from adaptive_survey.schema import BasePredictor, CONTEXT_BLOCKS, CONTEXT_TIMES, Profile, Selection, Task


def tasks_for(context="transfer"):
    return tuple(Task(f"B{b}_Q{q}", context, b, q, CONTEXT_TIMES[context],
                      tuple(tuple(float((a + q + k) % 3 == 0) if a < 3 else 0.0
                                  for k in range(13)) for a in range(4)))
                 for b in CONTEXT_BLOCKS[context] for q in range(1, 10))


class PublicPredictor(BasePredictor):
    def predict(self, state, tasks):
        counts = np.ones(4)
        for item in state.history:
            counts[item.choice] += 1
        return np.tile(counts / counts.sum(), (len(tasks), 1))


class RankedPolicy:
    def __init__(self, reverse=False):
        self.reverse = reverse
        self.name = "reverse" if reverse else "forward"

    def select(self, predictor, state, candidates, rng):
        chosen = sorted(candidates, key=lambda t: t.task_id, reverse=self.reverse)[0]
        return Selection(chosen.task_id, {t.task_id: float(t.cset) for t in candidates})


class World:
    scenario = "mnl"

    def __init__(self, tasks, overrides=None):
        self.answers = {t.task_id: t.cset % 4 for t in tasks}
        self.answers.update(overrides or {})
        self.queried = []

    def reveal(self, uuid, task_id, allowed_task_ids):
        assert task_id in allowed_task_ids
        self.queried.append((uuid, task_id))
        return self.answers[task_id]

    def answer(self, uuid, task_id):
        return self.answers[task_id]


PROFILE = Profile("respondent", (0.0,) * 9)


def test_common_readout_equal_at_zero_initial_and_full_pool():
    tasks = tasks_for()
    config = ProtocolConfig(budgets=(0, 2, 3, 9, 14))
    episode = make_episode(PROFILE, "transfer", tasks, config)
    results = [run_episode(PROFILE, "transfer", World(tasks), episode,
                           PublicPredictor(), RankedPolicy(reverse), PublicPredictor())
               for reverse in (False, True)]
    for budget in (0, 2, 14):
        left, right = [next(c for c in r.checkpoints if c.budget == budget) for r in results]
        assert left.common_probabilities == right.common_probabilities
    assert [c.budget for c in results[0].checkpoints] == list(config.budgets)
    assert len(results[0].trace) == 14
    assert len(set(o.task.task_id for o in results[0].observations)) == 14


def test_hidden_response_intervention_cannot_change_rollout_or_predictions(tmp_path):
    tasks = tasks_for()
    episode = make_episode(PROFILE, "transfer", tasks)
    overrides = {t.task_id: (t.cset + 1) % 4 for t in episode.holdout}
    worlds = [World(tasks), World(tasks, overrides)]
    results = [run_episode(PROFILE, "transfer", w, episode, PublicPredictor(), RankedPolicy()) for w in worlds]
    assert [(r.task_id, r.choice, r.scores) for r in results[0].trace] == [(r.task_id, r.choice, r.scores) for r in results[1].trace]
    assert [c.probabilities for c in results[0].checkpoints] == [c.probabilities for c in results[1].checkpoints]
    assert set(t.task_id for t in episode.holdout).isdisjoint(t for _, t in worlds[0].queried)
    # Evaluator can read changed hidden labels, while public JSON never contains them.
    score_checkpoint(results[0].checkpoints[0], worlds[0])
    path = tmp_path / "public.jsonl"
    write_jsonl(path, results[0].checkpoints)
    assert "choices" not in path.read_text()
    assert "holdout_task_ids" in path.read_text()


def test_personal_state_resets_across_contexts_and_people():
    tasks = tasks_for() + tasks_for("layover")
    profiles = {PROFILE.uuid: PROFILE, "other": Profile("other", PROFILE.features)}
    config = ProtocolConfig(budgets=(0, 2))
    episodes = [make_episode(profiles[uuid], context, tasks, config)
                for uuid, context in ((PROFILE.uuid, "transfer"), (PROFILE.uuid, "layover"), ("other", "transfer"))]
    result = run_survey(profiles, episodes, World(tasks), PublicPredictor(), RankedPolicy())
    for r in result:
        assert np.allclose(r.checkpoints[0].probabilities, 0.25)
        assert len(r.observations) == 2


def test_illegal_policy_query_and_nonfinite_scores_rejected():
    tasks = tasks_for()
    episode = make_episode(PROFILE, "transfer", tasks, ProtocolConfig(budgets=(3,)))

    class Illegal:
        name = "illegal"
        def select(self, *args):
            return Selection(episode.holdout[0].task_id, {})

    with pytest.raises(ValueError, match="hidden, repeated, or unknown"):
        run_episode(PROFILE, "transfer", World(tasks), episode, PublicPredictor(), Illegal())


def test_scenario_specific_world_never_uses_other_generator_column():
    from adaptive_survey.data import DataBundle, ResponseWorld
    tasks = tasks_for()
    mapping = {t.task_id: t for t in tasks}
    mnl = {(PROFILE.uuid, t.task_id): 0 for t in tasks}
    mmnl = {(PROFILE.uuid, t.task_id): 3 for t in tasks}
    bundle = DataBundle({PROFILE.uuid: PROFILE}, mapping, {"mnl": mnl, "mmnl": mmnl})
    episode = make_episode(PROFILE, "transfer", tasks, ProtocolConfig(budgets=(2,)))
    for scenario, expected in (("mnl", 0), ("mmnl", 3)):
        result = run_episode(PROFILE, "transfer", ResponseWorld(bundle, scenario), episode,
                             PublicPredictor(), RankedPolicy())
        assert {o.choice for o in result.observations} == {expected}
        assert {c.scenario for c in result.checkpoints} == {scenario}
