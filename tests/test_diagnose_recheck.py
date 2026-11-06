"""Saved-rollout diagnostics retain UUID clusters and expose aggregates only."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_survey.data import ResponseWorld, load_bundle
from adaptive_survey.demo import generate_demo
from adaptive_survey.neural import ChoiceNetwork, NeuralPredictor
from adaptive_survey.policies import make_policy
from adaptive_survey.protocol import ProtocolConfig, make_episode
from adaptive_survey.runner import run_episode
from adaptive_survey.schema import CONTEXTS


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_recheck.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_recheck_test_module", _SCRIPT)
diagnose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnose)


@pytest.fixture
def completed_runs(tmp_path):
    paths = generate_demo(tmp_path / "data", n_respondents=7, seed=19)
    bundle = load_bundle(paths["profiles"], paths["tasks"], paths["responses"])
    ids = sorted(bundle.profiles)
    protocol = ProtocolConfig(initial_per_block=1, budgets=(3, 5, 7, 9))
    split = {"schema_version": 1, "uuids": {"train": ids[:2], "validation": ids[2:4], "test": ids[4:]},
             "protocol": asdict(protocol),
             "fingerprints": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    split_hash = hashlib.sha256(json.dumps(split["uuids"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    runs = []
    for seed in (41, 42):
        run = tmp_path / f"seed-{seed}"
        evaluation = run / "evaluation"
        evaluation.mkdir(parents=True)
        torch.manual_seed(seed)
        predictor = NeuralPredictor([ChoiceNetwork()], metadata={
            "training_config": {"seed": seed, "epochs": 2, "episodes_per_epoch": 4, "batch_size": 2},
            "scenario": "mmnl", "split_hash": split_hash, "fingerprints": split["fingerprints"],
            "protocol": asdict(protocol), "best_epochs": [1], "best_optimizer_steps": [2],
            "optimizer_steps_run": [4], "optimizer_steps_per_epoch": 2, "max_optimizer_steps_per_member": 4,
            "validation_respondents": 2, "validation_bank_respondents": 2,
            "validation_bank_episodes": 6, "validation_bank_respondent_context_pairs": 6,
            "validation_bank_context_counts": {context: 2 for context in CONTEXTS},
            "validation_bank_context_respondents": {context: 2 for context in CONTEXTS},
            "validation_bank_hash": "0" * 64, "validation_bank_sampling": "fixture",
        })
        checkpoint = run / "checkpoint.pt"
        predictor.save(checkpoint)
        world = ResponseWorld(bundle, "mmnl")
        rankings = {context: sorted(task.task_id for task in bundle.tasks.values() if task.context == context)
                    for context in CONTEXTS}
        with (evaluation / "predictions.jsonl").open("w") as predictions, (evaluation / "traces.jsonl").open("w") as traces:
            for policy_name in ("fixed", "random", "entropy", "weighted_js"):
                policy = make_policy(policy_name, rankings=rankings)
                for uuid in split["uuids"]["test"]:
                    for context in CONTEXTS:
                        episode = make_episode(uuid, context, bundle.tasks, protocol)
                        result = run_episode(bundle.profiles[uuid], context, world, episode, predictor,
                                             policy, seed=2026, budgets=protocol.budgets)
                        for row in result.checkpoints:
                            predictions.write(json.dumps(asdict(row)) + "\n")
                        for row in result.trace:
                            traces.write(json.dumps(asdict(row)) + "\n")
        summary = {"scenario": "mmnl", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "split_hash": split_hash, "fingerprints": split["fingerprints"], "protocol": asdict(protocol),
                   "n_uuids": 3, "contexts": list(CONTEXTS),
                   "policies": ["fixed", "random", "entropy", "weighted_js"], "rollout_seed": 2026}
        (evaluation / "summary.json").write_text(json.dumps(summary))
        runs.append(run)
    return paths["profiles"].parent, split_path, runs, bundle, split


def test_diagnose_uses_saved_H9_and_only_predicts_empty_H0(completed_runs, monkeypatch):
    data, split_path, runs, bundle, _ = completed_runs
    # A recognizable saved H9 score detects accidental recomputation of H9.
    for run in runs:
        path = run / "evaluation/predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["policy"] == "weighted_js" and row["budget"] == 9:
                probabilities = []
                for task_id in row["holdout_task_ids"]:
                    choice = bundle.responses["mmnl"][row["uuid"], task_id]
                    p = [1 / 6] * 4
                    p[choice] = .5
                    probabilities.append(p)
                row["probabilities"] = probabilities
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = NeuralPredictor.predict_many
    history_lengths = []

    def empty_prediction_only(self, requests, tasks_per_state=None):
        assert tasks_per_state is None
        history_lengths.extend(len(state.history) for state, _ in requests)
        return original(self, requests)

    monkeypatch.setattr(NeuralPredictor, "predict_many", empty_prediction_only)
    report = diagnose.diagnose_recheck(data, split_path, runs, expected_uuids=3, expected_seeds=2)
    assert history_lengths == [0] * 18
    assert report["episodes_per_seed"] == 9
    assert report["holdout_queries_per_seed"] == 36
    assert report["n_bootstrap"] == 5000
    assert report["training_seeds"] == [41, 42]
    assert report["aggregate"]["H9"]["nll"]["mean"] == pytest.approx(np.log(2))
    assert report["aggregate"]["H9"]["accuracy"]["mean"] == 1.0
    interval = report["aggregate"]["H9_minus_H0"]["nll"]
    assert interval["n_uuid_clusters"] == 3
    assert interval["difference_H9_minus_H0"] == pytest.approx(np.mean(interval["per_training_seed"]))
    assert interval["std_across_training_seeds"] == pytest.approx(np.std(interval["per_training_seed"], ddof=1))
    for row in report["per_seed"]:
        assert row["training"]["best_optimizer_steps"] == [2]
        assert row["training"]["optimizer_steps_run"] == [4]
        assert row["js_scores"]["n_values"] == 9 * sum(range(6, 13))
    text = json.dumps(report)
    assert all(uuid not in text for uuid in bundle.profiles)
    assert '"probabilities"' not in text
    assert '"features"' not in text


def test_diagnose_replays_answer_intervention_without_hidden_labels(completed_runs):
    data, split_path, runs, _, _ = completed_runs
    report = diagnose.diagnose_recheck(data, split_path, runs, expected_uuids=3, expected_seeds=2,
                                       n_bootstrap=20, check_answer_sensitivity=True,
                                       sensitivity_episodes=4)
    for row in report["per_seed"]:
        intervention = row["answer_sensitivity"]
        assert intervention["n_episodes"] == 4
        assert intervention["n_available_episodes"] == 9
        assert intervention["actual_selection_reproduced_fraction"] == 1.0
        assert 0 <= intervention["next_task_changed_fraction"] <= 1
        assert "does not establish true information gain" in intervention["interpretation"]


def test_diagnose_rejects_missing_grid_and_checkpoint_drift(completed_runs):
    data, split_path, runs, _, _ = completed_runs
    path = runs[0] / "evaluation/predictions.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    dropped, kept = False, []
    for row in rows:
        if not dropped and row["policy"] == "weighted_js" and row["budget"] == 9:
            dropped = True
            continue
        kept.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in kept))
    with pytest.raises(ValueError, match="Incomplete"):
        diagnose.diagnose_recheck(data, split_path, runs, expected_uuids=3, expected_seeds=2, n_bootstrap=5)
    path.write_text(original)
    summary_path = runs[0] / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["checkpoint_sha256"] = "incorrect-checkpoint"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="provenance"):
        diagnose.diagnose_recheck(data, split_path, runs, expected_uuids=3, expected_seeds=2, n_bootstrap=5)


def test_paired_interval_averages_training_seeds_and_keeps_contexts_together():
    differences = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                            [[2, 3, 4], [5, 6, 7], [8, 9, 10]]], dtype=float)
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    result = diagnose._paired_interval(differences, indices, .8)
    # UUID means after both seed and context averages are [2,5,8].
    assert result["difference_H9_minus_H0"] == 5
    assert result["ci_lower"] == pytest.approx(np.quantile([2, 5, 8], .1))
    assert result["ci_upper"] == pytest.approx(np.quantile([2, 5, 8], .9))
    assert result["per_training_seed"] == [4, 6]
    assert result["std_across_training_seeds"] == pytest.approx(np.sqrt(2))
