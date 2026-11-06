import numpy as np
import pytest

from adaptive_survey.metrics import (ScoredCheckpoint, budget_average, choice_metrics,
                                     paired_uuid_bootstrap, summarize_scores)


def score(uuid, context, policy, nll, budget=3):
    return ScoredCheckpoint(uuid, context, "mmnl", policy, budget, "common", nll, 0.5, 0.4, 4)


def test_uniform_choice_scores_and_invalid_probabilities():
    result = choice_metrics(np.full((4, 4), 0.25), [0, 1, 2, 3])
    assert result["nll"] == pytest.approx(np.log(4))
    assert result["brier"] == pytest.approx(0.75)
    assert result["accuracy"] == 0.25
    with pytest.raises(ValueError):
        choice_metrics(np.ones((2, 4)), [0, 1])
    with pytest.raises(ValueError):
        choice_metrics(np.full((1, 4), np.nan), [0])


def test_macro_gives_equal_context_weight_and_counts_unique_respondents():
    rows = [score("a", "transfer", "p", 0), score("b", "transfer", "p", 2),
            score("a", "layover", "p", 3)]
    macro = next(r for r in summarize_scores(rows) if r["context"] == "macro")
    assert macro["nll"] == pytest.approx(2)
    assert macro["n_respondents"] == 2
    assert macro["n_episodes"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_scores(rows + [rows[0]])


def test_paired_uuid_bootstrap_retains_all_contexts_per_respondent():
    rows = [score(uuid, context, policy, difference if policy == "a" else 0)
            for uuid, difference in (("one", 1), ("two", 3))
            for context in ("transfer", "layover", "stopover") for policy in ("a", "b")]
    result = paired_uuid_bootstrap(rows, "a", "b", n_bootstrap=100, seed=4)
    assert result["n_uuids"] == 2
    assert result["difference"] == 2
    assert result["ci_lower"] == 1
    assert result["ci_upper"] == 3
    with pytest.raises(ValueError, match="identical UUID"):
        paired_uuid_bootstrap(rows[:-1], "a", "b")
    # Pairing both policies does not repair a missing context for one UUID.
    with pytest.raises(ValueError, match="same context/budget grid"):
        paired_uuid_bootstrap(rows[:-2], "a", "b")


def test_budget_average_requires_the_prespecified_budgets():
    assert budget_average({6: 1, 9: 2, 12: 3}) == 2
    assert budget_average({3: 1, 5: 2, 7: 3, 9: 4}, budgets=(3, 5, 7, 9)) == 2.5
    with pytest.raises(ValueError):
        budget_average({3: 1})
