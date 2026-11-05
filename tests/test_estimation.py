from dataclasses import replace

import numpy as np
import pytest

from adaptive_survey.estimation import (ChoiceRecord, DEFAULT_FEATURE_NAMES, coefficient_agreement,
                                       design_matrix, fit_pooled_mnl, fit_selected_vs_full)
from adaptive_survey.schema import ATTRIBUTES, CONTEXT_BLOCKS, CONTEXT_TIMES, PERSON_FIELDS, Profile, Task


def task(context="transfer", q=1, rng=None):
    rng = np.random.default_rng(10 + q) if rng is None else rng
    attributes = np.zeros((4, 13))
    for row in range(3):
        for first in range(0, 12, 2):
            level = rng.integers(3)
            if level:
                attributes[row, first + level - 1] = 1
        attributes[row, 12] = rng.integers(2)
    block = CONTEXT_BLOCKS[context][(q - 1) // 9]
    return Task(f"B{block}_Q{(q - 1) % 9 + 1}", context, block, (q - 1) % 9 + 1,
                CONTEXT_TIMES[context], tuple(map(tuple, attributes)))


def test_design_matches_public_34_column_handoff_specification():
    attributes = np.zeros((4, 13))
    attributes[0, ATTRIBUTES.index("MSA_P")] = 1
    attributes[0, ATTRIBUTES.index("ENV_3")] = 1
    t = Task("B3_Q1", "layover", 3, 1, "12H", tuple(map(tuple, attributes)))
    features = dict.fromkeys(PERSON_FIELDS, 0.0)
    features.update(GEN_Z=1, GEND_O=1, EDU_6=1)
    x = design_matrix(t, Profile("p", tuple(features[f] for f in PERSON_FIELDS)))
    assert x.shape == (4, 34)
    assert DEFAULT_FEATURE_NAMES[:3] == ("ASC_A", "ASC_B", "ASC_C")
    assert x[0, DEFAULT_FEATURE_NAMES.index("MSA_PxGENZ")] == 1
    assert x[0, DEFAULT_FEATURE_NAMES.index("MSA_PxGENDunmapped")] == 1
    assert x[0, DEFAULT_FEATURE_NAMES.index("ENV_3xEDU_6")] == 1
    assert x[3, DEFAULT_FEATURE_NAMES.index("NONE_LAYOVER")] == 1
    assert x[3].sum() == 1


def generated_records():
    rng = np.random.default_rng(42)
    tasks = [task(c, q, rng) for c in CONTEXT_BLOCKS for q in range(1, 19)]
    records = []
    beta = rng.normal(0, 0.3, len(DEFAULT_FEATURE_NAMES))
    for person in range(36):
        f = dict.fromkeys(PERSON_FIELDS, 0.0)
        f["GEN_Z"] = person % 2
        gender = (person // 2) % 3
        f["GEND_F"] = gender == 1
        f["GEND_O"] = gender == 2
        education = (person // 6) % 6
        if education:
            f[("EDU_1", "EDU_2", "EDU_3", "EDU_5", "EDU_6")[education - 1]] = 1
        p = Profile(str(person), tuple(float(f[k]) for k in PERSON_FIELDS))
        for t in tasks:
            utility = design_matrix(t, p) @ beta
            probs = np.exp(utility - utility.max())
            probs /= probs.sum()
            records.append(ChoiceRecord(p, t, int(rng.choice(4, p=probs))))
    return records


def test_fit_full_rank_design_with_uuid_clustered_standard_errors():
    rows = generated_records()
    fit = fit_pooled_mnl(rows, cluster_se=True)
    assert fit.converged and fit.rank == 34
    assert fit.log_likelihood > -len(rows) * np.log(4)
    assert fit.gradient_inf_norm < 1e-4
    assert np.isfinite(fit.coefficients).all()
    assert np.isfinite(fit.standard_errors).all()
    assert np.allclose(fit.predict(rows[0].profile, [r.task for r in rows[:4]]).sum(axis=1), 1)
    assert coefficient_agreement(fit, fit)["rmse"] == 0


def test_rank_and_optimizer_failures_are_explicit():
    p = Profile("p", (0.0,) * 9)
    rows = [ChoiceRecord(p, task(q=q), q % 4) for q in range(1, 8)]
    with pytest.raises(ValueError, match="Rank-deficient"):
        fit_pooled_mnl(rows)
    fit = fit_pooled_mnl(rows, drop_unidentified=True)
    assert fit.dropped_features and fit.rank < 34
    assert len(fit.as_dict()["coefficients"]) == fit.rank
    with pytest.raises(RuntimeError, match="optimization failed"):
        fit_pooled_mnl(rows, drop_unidentified=True, max_iter=1)


def test_population_comparison_checks_matching_reference():
    p = Profile("p", (0.0,) * 9)
    rows = [ChoiceRecord(p, task(q=q), q % 4) for q in range(1, 8)]
    report = fit_selected_vs_full(rows, rows, drop_unidentified=True)
    assert report["agreement"]["rmse"] == 0
    assert "not recovered" in report["selected"]["interpretation"]
    other = [ChoiceRecord(Profile("other", p.features), rows[0].task, rows[0].choice)]
    with pytest.raises(ValueError, match="identical UUID"):
        fit_selected_vs_full(other, rows)


def test_comparison_keeps_one_identifiable_reference_basis():
    p = Profile("p", (0.0,) * 9)
    rows = [ChoiceRecord(p, task(q=q), q % 4) for q in range(1, 8)]
    fit = fit_pooled_mnl(rows, drop_unidentified=True)
    with pytest.raises(ValueError, match="same identifiable feature basis"):
        coefficient_agreement(replace(fit, active_features=fit.active_features[:-1]), fit)
    # A one-task subset cannot identify the reference's utility specification.
    # It must fail instead of changing the feature basis and comparing aliases.
    with pytest.raises(ValueError, match="Rank-deficient"):
        fit_selected_vs_full(rows[:1], rows, drop_unidentified=True)
