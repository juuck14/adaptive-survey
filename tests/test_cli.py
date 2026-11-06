import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from adaptive_survey.cli import main


def setup_data(tmp_path):
    data = tmp_path / "data"
    split = tmp_path / "split.json"
    main(["demo", "--output", str(data), "--respondents", "24"])
    main(["split", "--data", str(data), "--output", str(split),
          "--train", "12", "--validation", "6", "--test", "6"])
    return data, split


def train_small(data, split, tmp_path):
    checkpoint = tmp_path / "model.pt"
    main(["train", "--data", str(data), "--split", str(split), "--output", str(checkpoint),
          "--ensemble", "1", "--epochs", "2", "--episodes", "32", "--validation-episodes", "16"])
    return checkpoint


def test_default_command_compares_two_methods_in_separate_evaluation_tracks(tmp_path):
    data, split = setup_data(tmp_path)
    checkpoint = train_small(data, split, tmp_path)
    output = tmp_path / "evaluation"
    main(["evaluate", "--data", str(data), "--split", str(split), "--checkpoint", str(checkpoint),
          "--output", str(output), "--limit", "2", "--contexts", "transfer", "--bootstrap", "20"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_uuids"] == 2
    assert summary["policies"] == ["fixed", "weighted_js"]
    assert summary["protocol"]["initial_per_block"] == 2
    assert summary["protocol"]["budgets"] == [6, 9, 12, 15, 18]
    assert len(summary["budget_average"]) == 2
    assert summary["rollout_seed"] == 2026
    assert summary["fixed_prior"]["converged"]
    assert summary["fixed_design"]["information_ranks"]["transfer"] > 0
    assert summary["fixed_design"]["nested_prefixes"]
    assert summary["fixed_planning_timing"]["unique_plans"] > 0
    assert summary["fixed_planning_timing"]["training_prior_seconds"] >= 0
    assert summary["tracks"]["prediction"]["budgets"] == [6, 9, 12]
    assert summary["tracks"]["prediction"]["unavailable_budgets"] == [15, 18]
    assert summary["tracks"]["preference"]["budgets"] == [6, 9, 12, 15, 18]
    assert summary["tracks"]["prediction"]["status"] == summary["tracks"]["preference"]["status"] == "completed"
    assert summary["paired_comparisons"][0]["policy_b"] == "fixed"
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 2 * 2 * 3
    assert all(not set(p["queried_task_ids"]) & set(p["holdout_task_ids"]) for p in predictions)
    assert all("choices" not in p and "taste" not in p for p in predictions)
    collections = [json.loads(line) for line in (output / "collections.jsonl").read_text().splitlines()]
    assert len(collections) == 2 * 2 * 5
    assert all(not cp["probabilities"] and cp["track"] == "preference" for cp in collections)
    full = [cp for cp in collections if cp["budget"] == 18]
    for uid in {cp["uuid"] for cp in full}:
        paths = [cp for cp in full if cp["uuid"] == uid]
        assert len(paths) == 2
        assert set(paths[0]["queried_task_ids"]) == set(paths[1]["queried_task_ids"])
    traces = [json.loads(line) for line in (output / "traces.jsonl").read_text().splitlines()]
    for track in ("prediction", "preference"):
        for uid in {row["uuid"] for row in traces}:
            prefixes = [[(row["task_id"], row["choice"]) for row in traces
                         if row["uuid"] == uid and row["track"] == track and row["policy"] == policy and row["initial"]]
                        for policy in summary["policies"]]
            assert prefixes[0] == prefixes[1] and len(prefixes[0]) == 4
    population = tmp_path / "population.json"
    main(["estimate", "--data", str(data), "--split", str(split), "--evaluation", str(output),
          "--budgets", "18", "--output", str(population)])
    estimates = json.loads(population.read_text())
    assert estimates["track"] == "preference"
    assert len(estimates["results"]) == 2
    assert all(row["agreement"]["rmse"] == 0 for row in estimates["results"])


def test_unavailable_prediction_budgets_are_reported_without_in_sample_scores(tmp_path):
    data, split = setup_data(tmp_path)
    output = tmp_path / "full-only"
    main(["evaluate", "--data", str(data), "--split", str(split), "--predictor", "uniform",
          "--policies", "weighted_js", "--budgets", "18", "--contexts", "transfer",
          "--limit", "1", "--output", str(output)])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["tracks"]["prediction"]["status"] == "no_feasible_budgets"
    assert summary["tracks"]["prediction"]["unavailable_budgets"] == [18]
    assert summary["metrics"] == [] and summary["budget_average"] == []
    assert (output / "predictions.jsonl").read_text() == ""
    assert json.loads((output / "collections.jsonl").read_text())["budget"] == 18


def test_checkpoint_rejects_changed_history_protocol(tmp_path):
    data, split = setup_data(tmp_path)
    checkpoint = train_small(data, split, tmp_path)
    manifest = json.loads(split.read_text())
    manifest["protocol"]["seed"] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--data", str(data), "--split", str(changed), "--checkpoint", str(checkpoint),
              "--output", str(tmp_path / "should-not-exist")])
    assert error.value.code == 2
    assert not (tmp_path / "should-not-exist").exists()


def test_changed_input_fingerprint_and_output_collision_are_rejected(tmp_path):
    data, split = setup_data(tmp_path)
    before = (data / "tasks.json").read_bytes()
    with pytest.raises(SystemExit):
        main(["validate", "--data", str(data), "--output", str(data / "tasks.json")])
    assert (data / "tasks.json").read_bytes() == before
    (data / "tasks.json").write_bytes(before + b"\n")
    with pytest.raises(SystemExit):
        main(["evaluate", "--data", str(data), "--split", str(split), "--predictor", "uniform",
              "--policies", "random", "--output", str(tmp_path / "bad")])
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_generate_invalid_limit_does_not_start_full_generation(tmp_path, limit):
    data, _ = setup_data(tmp_path)
    with pytest.raises(SystemExit):
        main(["generate", "--profiles", str(data / "profiles.csv"), "--tasks", str(data / "tasks.json"),
              "--handoff", str(tmp_path / "missing-handoff"), "--output", str(tmp_path / "responses.csv"),
              "--limit", limit])
    assert not (tmp_path / "responses.csv").exists()
