"""Regression checks for the prespecified multi-seed UUID aggregation."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_recheck.py"
_SPEC = importlib.util.spec_from_file_location("adaptive_survey_recheck_summary", _SCRIPT)
aggregation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = aggregation
_SPEC.loader.exec_module(aggregation)


def make_evaluations(base: Path, n_uuids=4):
    directories = []
    for seed_index in range(3):
        directory = base / f"seed-{2026 + seed_index}" / "evaluation"
        directory.mkdir(parents=True)
        directories.append(directory)
        summary = {
            "scenario": "mmnl", "predictor": "non_graph",
            "policies": list(aggregation.POLICIES), "contexts": list(aggregation.CONTEXTS),
            "protocol": {"budgets": [3, 5, 7, 9], "holdout_per_block": 2,
                         "initial_per_block": 1, "seed": 2026},
            "split_hash": "same-split", "checkpoint_sha256": f"unique-checkpoint-{seed_index}",
            "fingerprints": {"profiles.csv": "same-profiles", "tasks.json": "same-tasks",
                             "responses.csv": "same-world"},
            "rollout_seed": 2026,
            "fixed_rankings": {c: [f"{c}-rank-{i}" for i in range(18)] for c in aggregation.CONTEXTS},
            "n_uuids": n_uuids,
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        scores = []
        with (directory / "predictions.jsonl").open("w") as stream:
            for uuid_index in range(n_uuids):
                for ci, context in enumerate(aggregation.CONTEXTS):
                    for bi, budget in enumerate(aggregation.BUDGETS):
                        for policy in aggregation.POLICIES:
                            nll = 1 + .2 * seed_index + .1 * ci + .01 * bi + .001 * uuid_index
                            accuracy = .3 + .05 * seed_index + .02 * ci + .005 * bi + .0001 * uuid_index
                            if policy == "weighted_js":
                                nll += -.02 * (seed_index + 1) + .0001 * uuid_index
                                accuracy += .01 * (seed_index + 1) - .00002 * uuid_index
                            elif policy == "fixed":
                                nll -= .01
                            elif policy == "entropy":
                                nll -= .015
                            identity = {"uuid": f"person-{uuid_index:03}", "context": context,
                                        "budget": budget, "policy": policy, "scenario": "mmnl"}
                            # The helper aggregates evaluator scores; the probabilities
                            # here are validity placeholders, with no hidden labels.
                            scores.append(dict(identity, reader="native", n_tasks=4, nll=nll, accuracy=accuracy))
                            prediction = dict(identity,
                                holdout_task_ids=[f"{context}-hold-{i}" for i in range(4)],
                                queried_task_ids=[f"{context}-q-{i}" for i in range(budget)],
                                probabilities=[[.25] * 4 for _ in range(4)])
                            stream.write(json.dumps(prediction) + "\n")
        (directory / "scores.json").write_text(json.dumps(scores))
    return directories


def test_full_production_grid_matches_analytic_seed_means_and_uuid_bootstrap(tmp_path):
    directories = make_evaluations(tmp_path, n_uuids=300)
    report = aggregation.summarize_evaluations(directories, n_bootstrap=2000)
    primary = next(r for r in report["paired_comparisons"] if r["analysis_role"] == "primary")
    js = next(r for r in report["across_seed"]["budget_average"] if r["policy"] == "weighted_js")
    assert report["n_scores_per_seed"] == 300 * 3 * 4 * 4
    assert report["seed_labels"] == ["seed-2026", "seed-2027", "seed-2028"]
    assert primary["n_uuid_clusters"] == 300
    assert primary["n_training_seeds"] == 3
    assert primary["favors_a_when"] == "negative"
    assert primary["difference_a_minus_b"] == pytest.approx(-.02505, abs=1e-12)
    assert primary["std_of_differences_across_training_seeds"] == pytest.approx(.02, abs=1e-12)
    assert js["nll"]["mean"] == pytest.approx(1.43945, abs=1e-12)
    assert js["nll"]["std_across_training_seeds"] == pytest.approx(.18, abs=1e-12)
    assert js["accuracy"]["mean"] == pytest.approx(.40946, abs=1e-12)
    # Analytically, each person's mean JS-random loss is -.04 + .0001*u.
    # Resample those 300 values, preserving all seeds/contexts/budgets inside u.
    person_differences = -.04 + .0001 * np.arange(300)
    draws = np.random.default_rng(2026).integers(0, 300, size=(2000, 300), dtype=np.int32)
    expected_ci = np.quantile(person_differences[draws].mean(axis=1), [.025, .975])
    np.testing.assert_allclose([primary["ci_lower"], primary["ci_upper"]], expected_ci, atol=1e-12)
    assert "person-000" not in json.dumps(report)


@pytest.mark.parametrize("corruption,match", [
    ("missing_score", "incomplete"),
    ("different_world", "fingerprints"),
    ("mixed_scenario", "mixed response scenarios"),
    ("duplicate_checkpoint", "distinct learner"),
    ("different_rank", "fixed_rankings"),
    ("different_holdout", "different UUIDs, contexts, or holdout"),
    ("different_predictor", "predictor"),
    ("different_random_history", "Fixed/random histories"),
])
def test_rejects_unpaired_or_changed_study_inputs(tmp_path, corruption, match):
    directories = make_evaluations(tmp_path)
    directory = directories[1]
    if corruption in ("missing_score", "mixed_scenario"):
        path = directory / "scores.json"
        rows = json.loads(path.read_text())
        if corruption == "missing_score":
            rows.pop()
        else:
            rows[0]["scenario"] = "mnl"
        path.write_text(json.dumps(rows))
    elif corruption in ("different_holdout", "different_random_history"):
        path = directory / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["uuid"] == "person-000" and row["context"] == "transfer":
                if corruption == "different_holdout":
                    row["holdout_task_ids"] = [f"other-hold-{i}" for i in range(4)]
                elif row["policy"] == "random":
                    row["queried_task_ids"][2] = "transfer-q-13"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        if corruption == "different_world":
            summary["fingerprints"]["responses.csv"] = "different-world"
        elif corruption == "duplicate_checkpoint":
            summary["checkpoint_sha256"] = "unique-checkpoint-0"
        elif corruption == "different_predictor":
            summary["predictor"] = "panel_map"
        else:
            summary["fixed_rankings"]["transfer"][:2] = list(reversed(summary["fixed_rankings"]["transfer"][:2]))
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=match):
        aggregation.summarize_evaluations(directories, expected_uuids=4, n_bootstrap=20)


def test_small_smoke_override_preserves_production_defaults_and_seed_sd(tmp_path):
    directories = make_evaluations(tmp_path)
    with pytest.raises(ValueError, match="expected 300 UUIDs"):
        aggregation.summarize_evaluations(directories, n_bootstrap=20)
    report = aggregation.summarize_evaluations(directories[:2], expected_seeds=2,
        expected_uuids=4, seed_labels=("31", "32"), n_bootstrap=20)
    assert report["seed_labels"] == ["31", "32"]
    assert report["n_training_seeds"] == 2 and report["n_uuids"] == 4
    js = next(r for r in report["across_seed"]["budget_average"] if r["policy"] == "weighted_js")
    assert js["nll"]["std_across_training_seeds"] == pytest.approx(.18 / np.sqrt(2))


def test_cli_refuses_to_overwrite_evaluator_inputs(tmp_path):
    directories = make_evaluations(tmp_path)
    output = directories[0] / "summary.json"
    original = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        aggregation.main(["--evaluations", *map(str, directories), "--expected-uuids", "4",
                          "--n-bootstrap", "20", "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_bytes() == original
