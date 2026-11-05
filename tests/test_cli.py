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


def test_command_line_training_and_all_policies(tmp_path):
    data, split = setup_data(tmp_path)
    checkpoint = train_small(data, split, tmp_path)
    output = tmp_path / "evaluation"
    main(["evaluate", "--data", str(data), "--split", str(split), "--checkpoint", str(checkpoint),
          "--output", str(output), "--limit", "2", "--contexts", "transfer", "--bootstrap", "20"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_uuids"] == 2
    assert len(summary["budget_average"]) == 4
    assert summary["rollout_seed"] == 2026
    assert summary["fixed_prior"]["converged"]
    assert len(summary["fixed_rankings"]["transfer"]) == 18
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 2 * 4 * 4
    assert all(not set(p["queried_task_ids"]) & set(p["holdout_task_ids"]) for p in predictions)
    assert all("choices" not in p and "taste" not in p for p in predictions)


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
