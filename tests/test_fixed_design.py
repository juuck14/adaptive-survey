from dataclasses import replace

import numpy as np
from scipy.special import softmax

from adaptive_survey.baselines import FixedEfficientDesign, UniformPredictor
from adaptive_survey.data import load_profiles, load_tasks
from adaptive_survey.demo import generate_demo
from adaptive_survey.estimation import DEFAULT_FEATURE_NAMES, design_matrix
from adaptive_survey.policies import FixedPolicy
from adaptive_survey.protocol import make_episode
from adaptive_survey.runner import run_episode


def test_fixed_design_conditions_on_initial_and_eligible_pool_before_any_answer(tmp_path):
    paths = generate_demo(tmp_path / "data", n_respondents=4)
    tasks = load_tasks(paths["tasks"])
    profiles = load_profiles(paths["profiles"])
    beta = np.zeros(len(DEFAULT_FEATURE_NAMES))
    design = FixedEfficientDesign(tasks, profiles, beta, ridge=0.3)
    profile = next(iter(profiles.values()))
    episode = make_episode(profile, "transfer", tasks)
    plan = design.plan("transfer", episode.candidates, episode.initial)
    assert plan.initial_task_ids == plan.task_ids[:4]
    assert set(plan.task_ids) == episode.acquisition_ids
    assert not set(plan.task_ids) & {task.task_id for task in episode.holdout}
    reordered = design.plan("transfer", tuple(reversed(episode.candidates)), episode.initial)
    assert reordered == plan
    # Independently reconstruct the full-feature Fisher criterion. A null-space
    # projection must leave greedy decisions unchanged, even after four seeds.
    information = {}
    for task in episode.candidates:
        total = np.zeros((len(beta), len(beta)))
        for person in profiles.values():
            x = design_matrix(task, person)
            p = softmax(x @ beta)
            total += x.T @ (np.diag(p) - np.outer(p, p)) @ x
        information[task.task_id] = total / len(profiles)
    precision = 0.3 * np.eye(len(beta))
    for task_id in plan.task_ids[:4]:
        precision += information[task_id]
    remaining = set(plan.task_ids[4:])
    for chosen in plan.task_ids[4:]:
        values = {task_id: np.linalg.slogdet(precision + information[task_id])[1] for task_id in remaining}
        assert values[chosen] >= max(values.values()) - 1e-10
        precision += information[chosen]
        remaining.remove(chosen)
    assert np.all(np.diff(plan.prefix_log_determinants) >= -1e-12)


def test_fixed_rollout_ignores_actual_answers_and_personal_features(tmp_path):
    paths = generate_demo(tmp_path / "data", n_respondents=3)
    tasks = load_tasks(paths["tasks"])
    profiles = load_profiles(paths["profiles"])
    profile = next(iter(profiles.values()))
    design = FixedEfficientDesign(tasks, profiles, np.zeros(len(DEFAULT_FEATURE_NAMES)))
    episode = make_episode(profile, "transfer", tasks, track="preference")
    plan = design.plan("transfer", episode.candidates, episode.initial)

    class NoPredictions(UniformPredictor):
        def predict(self, *args):
            raise AssertionError("Fixed collection must not call a choice predictor")

    class World:
        scenario = "mmnl"
        def __init__(self, choice):
            self.choice = choice
        def reveal(self, uuid, task_id, allowed):
            assert task_id in allowed
            return self.choice

    left = run_episode(profile, "transfer", World(0), episode, NoPredictions(), FixedPolicy(plan.task_ids))
    changed = replace(profile, features=(1.,) * 9)
    right = run_episode(changed, "transfer", World(3), episode, NoPredictions(), FixedPolicy(plan.task_ids))
    assert [row.task_id for row in left.trace] == [row.task_id for row in right.trace] == list(plan.task_ids)
    assert {row.choice for row in left.trace} == {0}
    assert {row.choice for row in right.trace} == {3}
