import importlib.util
from pathlib import Path

import pytest

from adaptive_survey.metrics import ScoredCheckpoint, paired_uuid_bootstrap


_PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_comparison.py"
_SPEC = importlib.util.spec_from_file_location("comparison_summary_tests", _PATH)
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def rows(offset):
    return [ScoredCheckpoint(uuid, context, "mmnl", policy, budget, "native",
                             2 + offset + (difference if policy == "weighted_js" else 0), .4, .5, 4)
            for uuid, difference in (("fictional-1", 1), ("fictional-2", 3))
            for context in ("transfer", "layover", "stopover")
            for budget in (6, 9, 12) for policy in ("fixed", "weighted_js")]


def test_seed_averaging_keeps_respondents_as_the_bootstrap_unit():
    averaged = summary.average_training_seeds([rows(0), rows(1), rows(2)])
    assert len(averaged) == len(rows(0))
    result = paired_uuid_bootstrap(averaged, "weighted_js", "fixed", reader="native",
                                   n_bootstrap=100, seed=7)
    assert result["n_uuids"] == 2  # Not two respondents times three training seeds.
    assert result["difference"] == 2
    assert result["ci_lower"] == 1
    assert result["ci_upper"] == 3
    fixed = next(row for row in averaged if row.policy == "fixed")
    assert fixed.nll == 3


def test_seed_averaging_rejects_missing_cells_and_duplicate_scores():
    with pytest.raises(ValueError, match="different"):
        summary.average_training_seeds([rows(0), rows(1)[:-1]])
    with pytest.raises(ValueError, match="Duplicate"):
        summary.average_training_seeds([rows(0) + rows(0)[:1]])


def test_report_seed_spread_is_separate_from_respondent_uncertainty():
    assert summary.statistics([1, 2, 3]) == {"mean": 2, "seed_sd": 1, "per_seed": [1., 2., 3.]}
    assert summary.statistics([2])["seed_sd"] is None
    with pytest.raises(ValueError, match="finite"):
        summary.statistics([float("nan")])
