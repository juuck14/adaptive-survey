from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from adaptive_survey.acquisition import entropy, weighted_js_scores
from adaptive_survey.baselines import UniformPredictor
from adaptive_survey.schema import BasePredictor, Profile, Task


def task(task_id, context="transfer"):
    return Task(task_id, context, 1, 1, "2H", tuple((0.0,) * 13 for _ in range(4)))


@pytest.fixture
def state():
    return UniformPredictor().reset(Profile("person", (0.0,) * 9), "transfer")


class AnswerSignal(BasePredictor):
    """Only the answer to q changes predictions on r."""
    def predict(self, state, tasks):
        result = []
        for query in tasks:
            if query.task_id == "q":
                result.append([0.75, 0.25, 0.0, 0.0])
            elif state.history and state.history[-1].task.task_id == "q":
                result.append(np.eye(4)[state.history[-1].choice])
            else:
                result.append([0.25] * 4)
        return np.asarray(result)


def test_known_weighted_js_and_law_of_total_probability_diagnostic(state):
    result = weighted_js_scores(AnswerSignal(), state, [task("q"), task("r")], diagnostics=True)
    expected = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    assert result.scores["q"] == pytest.approx(expected)
    assert result.scores["r"] == pytest.approx(0.0)
    assert result.consistency_tv["q"] == pytest.approx(0.5)
    assert result.consistency_tv["r"] == pytest.approx(0.0)
    assert all(0 <= score <= np.log(4) for score in result.scores.values())


def test_fantasies_are_batched_and_leave_real_history_untouched(state):
    class BatchSpy(AnswerSignal):
        def predict_many(self, requests):
            self.requests = requests
            return [self.predict(fantasy, targets) for fantasy, targets in requests]

    predictor = BatchSpy()
    original = state.history
    weighted_js_scores(predictor, state, [task("r"), task("q")])
    assert state.history is original and state.history == ()
    assert len(predictor.requests) == 8
    for fantasy, targets in predictor.requests:
        assert fantasy.profile is state.profile
        assert len(fantasy.history) == 1
        assert fantasy.history[-1].task.task_id not in {target.task_id for target in targets}
        assert {target.task_id for target in targets} <= {"q", "r"}
        with pytest.raises(FrozenInstanceError):
            fantasy.history = ()


def test_candidate_only_batch_has_expected_work_and_no_holdout(state):
    class BatchSpy(UniformPredictor):
        def __init__(self):
            self.calls = 0

        def predict(self, state, tasks):
            self.calls += 1
            return super().predict(state, tasks)

        def predict_many(self, requests):
            self.batch = requests
            return [np.full((len(targets), 4), 0.25) for _, targets in requests]

    predictor = BatchSpy()
    candidates = [task(f"q{i:02d}") for i in range(14)]
    scores = weighted_js_scores(predictor, state, candidates).scores
    assert predictor.calls == 1
    assert len(predictor.batch) == 14 * 4
    assert sum(len(targets) for _, targets in predictor.batch) == 14 * 4 * 13
    assert set(scores) == {query.task_id for query in candidates}
    assert all(score == pytest.approx(0.0) for score in scores.values())


def test_one_remaining_candidate_skips_all_predictions(state):
    class NeverPredict(UniformPredictor):
        def predict(self, *args):
            raise AssertionError("There is no other candidate to score")

    result = weighted_js_scores(NeverPredict(), state, [task("q")], diagnostics=True)
    assert result.scores == {"q": 0.0}
    assert result.consistency_tv == {"q": 0.0}


@pytest.mark.parametrize("bad", [
    [0.4, 0.4, 0.4, -0.2], [0.1, 0.1, 0.1, 0.1],
    [np.nan, 0.0, 0.0, 1.0], [np.inf, 0.0, 0.0, 0.0],
])
def test_invalid_probability_predictions_fail(state, bad):
    class Invalid(UniformPredictor):
        def predict(self, state, tasks):
            return np.tile(bad, (len(tasks), 1))

    with pytest.raises(ValueError):
        weighted_js_scores(Invalid(), state, [task("q"), task("r")])


def test_invalid_fantasy_fails(state):
    class NoAppend(UniformPredictor):
        def fantasize(self, state, query, choice):
            return state

    with pytest.raises(ValueError, match="fantasy"):
        weighted_js_scores(NoAppend(), state, [task("q"), task("r")])


@pytest.mark.parametrize("candidates", [[], [task("q"), task("q")], [task("q", "layover")]])
def test_invalid_candidate_pools_fail(state, candidates):
    with pytest.raises(ValueError):
        weighted_js_scores(UniformPredictor(), state, candidates)


def test_observed_task_is_not_a_remaining_candidate(state):
    query = task("q")
    observed = state.append(query, 1)
    with pytest.raises(ValueError, match="observed"):
        weighted_js_scores(UniformPredictor(), observed, [query])


def test_entropy_handles_zero_probability_and_batched_axes():
    values = np.array([[[1.0, 0.0, 0.0, 0.0], [0.25] * 4]])
    assert entropy(values) == pytest.approx(np.array([[0.0, np.log(4)]]))


def test_identical_fantasy_branches_have_zero_js_even_if_predictor_is_incoherent(state):
    class Incoherent(UniformPredictor):
        def predict(self, current_state, tasks):
            return np.tile([0.25] * 4 if current_state.history else [1.0, 0.0, 0.0, 0.0],
                           (len(tasks), 1))

        predict_many = None

    result = weighted_js_scores(Incoherent(), state, [task("q"), task("r")], diagnostics=True)
    assert result.scores == {"q": 0.0, "r": 0.0}
    assert result.consistency_tv == {"q": 0.75, "r": 0.75}


def test_four_distinct_answers_can_reach_log_four_bound(state):
    class FourSignals(AnswerSignal):
        def predict(self, current_state, tasks):
            if current_state.history and current_state.history[-1].task.task_id == "q":
                return np.tile(np.eye(4)[current_state.history[-1].choice], (len(tasks), 1))
            return np.full((len(tasks), 4), 0.25)

    score = weighted_js_scores(FourSignals(), state, [task("q"), task("r")]).scores["q"]
    assert score == pytest.approx(np.log(4))


def test_random_conditional_distributions_obey_weighted_js_bound(state):
    rng = np.random.default_rng(13)
    for _ in range(20):
        weights = rng.dirichlet(np.ones(4))
        branches = rng.dirichlet(np.ones(4), size=4)

        class Conditional(AnswerSignal):
            def predict(self, current_state, tasks):
                values = []
                for query in tasks:
                    if query.task_id == "q":
                        values.append(weights)
                    elif current_state.history and current_state.history[-1].task.task_id == "q":
                        values.append(branches[current_state.history[-1].choice])
                    else:
                        values.append([0.25] * 4)
                return np.asarray(values)

        score = weighted_js_scores(Conditional(), state, [task("q"), task("r")]).scores["q"]
        assert 0 <= score <= float(entropy(weights)) + 1e-12
