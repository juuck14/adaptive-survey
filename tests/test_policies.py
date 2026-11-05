import numpy as np
import pytest

from adaptive_survey.baselines import PanelMAPPredictor, UniformPredictor, greedy_fisher_rankings
from adaptive_survey.policies import (EntropyPolicy, FixedPolicy, RandomPolicy,
                                      WeightedJSPolicy, make_policy)
from adaptive_survey.schema import Profile, Task


def task(task_id, active=False):
    matrix = np.zeros((4, 13), dtype=float)
    if active:
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        matrix[2, 2] = 1.0
    return Task(task_id, "transfer", 1, 1, "2H", tuple(map(tuple, matrix)))


@pytest.fixture
def state():
    return UniformPredictor().reset(Profile("person", (0.0,) * 9), "transfer")


@pytest.mark.parametrize("policy", [RandomPolicy(), EntropyPolicy(), WeightedJSPolicy()])
def test_seeded_ties_do_not_depend_on_candidate_order(state, policy):
    candidates = [task("c"), task("a"), task("b")]
    a = policy.select(UniformPredictor(), state, candidates, np.random.default_rng(17))
    b = policy.select(UniformPredictor(), state, candidates[::-1], np.random.default_rng(17))
    assert a.task_id == b.task_id
    assert a.scores == b.scores


def test_entropy_selects_most_uncertain_candidate(state):
    class ConfidentExceptOne(UniformPredictor):
        def predict(self, state, tasks):
            return np.asarray([[0.25] * 4 if query.task_id == "uncertain"
                               else [1.0, 0.0, 0.0, 0.0] for query in tasks])

    selected = EntropyPolicy().select(ConfidentExceptOne(), state,
                                      [task("certain"), task("uncertain")], np.random.default_rng(4))
    assert selected.task_id == "uncertain"


def test_fixed_ranking_uses_preplanned_order_over_remaining_tasks(state):
    ranking = {"transfer": ["observed", "second", "third"]}
    policy = FixedPolicy(ranking)
    # Snapshot the input so a caller cannot change the plan during evaluation.
    ranking["transfer"].reverse()
    selected = policy.select(UniformPredictor(), state, [task("third"), task("second")], np.random.default_rng(9))
    assert selected.task_id == "second"


def test_fixed_requires_complete_ranking(state):
    with pytest.raises(ValueError, match="cover"):
        FixedPolicy(["q"]).select(UniformPredictor(), state, [task("r")], np.random.default_rng(1))
    with pytest.raises(ValueError, match="context"):
        FixedPolicy({"layover": ["q"]}).select(UniformPredictor(), state, [task("q")], np.random.default_rng(1))


@pytest.mark.parametrize("name", ["bald", "bayesian", "eig", "js", "", "Random"])
def test_policy_factory_rejects_misleading_or_unknown_aliases(name):
    with pytest.raises(ValueError):
        make_policy(name)


def test_policy_factory_requires_fixed_plan():
    with pytest.raises(ValueError):
        make_policy("fixed")
    assert isinstance(make_policy("weighted_js"), WeightedJSPolicy)
    assert isinstance(make_policy("fixed", rankings=["q"]), FixedPolicy)


def test_panel_map_uses_only_observed_history_and_preserves_population_weights(state, tmp_path):
    from adaptive_survey.estimation import DEFAULT_FEATURE_NAMES

    source = np.zeros(len(DEFAULT_FEATURE_NAMES))
    predictor = PanelMAPPredictor(source, prior_scale=0.5)
    source[:] = 100.0
    query = task("observe", active=True)
    target = task("target", active=True)
    before = predictor.predict(state, [target])
    observed = predictor.observe(state, query, 0)
    after = predictor.predict(observed, [target])
    assert after[0, 0] > before[0, 0]
    assert state.history == ()
    assert np.array_equal(predictor.coefficients, np.zeros(len(DEFAULT_FEATURE_NAMES)))
    # UUID is metadata: the same features/history must give the same prediction.
    clone = predictor.reset(Profile("different_uuid", state.profile.features), state.context)
    clone = predictor.observe(clone, query, 0)
    assert predictor.predict(clone, [target]) == pytest.approx(after)
    checkpoint = tmp_path / "baseline.json"
    predictor.save(checkpoint)
    loaded = PanelMAPPredictor.load(checkpoint)
    assert loaded.predict(observed, [target]) == pytest.approx(after)


def test_fisher_ranking_is_preplanned_deterministic_and_covers_tasks(state):
    from adaptive_survey.estimation import DEFAULT_FEATURE_NAMES

    tasks = [task("weak"), task("strong", active=True)]
    coefficients = np.zeros(len(DEFAULT_FEATURE_NAMES))
    ranking = greedy_fisher_rankings(tasks, [state.profile], coefficients)
    reversed_ranking = greedy_fisher_rankings(tasks[::-1], [state.profile], coefficients)
    assert ranking == reversed_ranking
    mapping_ranking = greedy_fisher_rankings({query.task_id: query for query in tasks},
                                             {state.profile.uuid: state.profile}, coefficients)
    assert mapping_ranking == ranking
    assert ranking["transfer"] == ["strong", "weak"]
    assert len(ranking["transfer"]) == len(tasks)


def test_panel_population_fit_records_identification_and_training_settings(state):
    from adaptive_survey.estimation import ChoiceRecord

    query = task("q", active=True)
    records = [ChoiceRecord(Profile(f"person-{choice}", state.profile.features), query, choice)
               for choice in range(4)]
    predictor = PanelMAPPredictor.fit(records, l2=0.01, max_iter=100)
    assert predictor.fit_summary["n_observations"] == 4
    assert predictor.fit_summary["n_respondents"] == 4
    assert predictor.fit_summary["l2"] == 0.01
    assert predictor.fit_summary["population_max_iter"] == 100
    assert predictor.fit_summary["drop_unidentified"] is True
    assert predictor.fit_summary["dropped_features"]
