import csv
import importlib.util
import json
from pathlib import Path

import pytest

from adaptive_survey.data import RESPONSE_FIELDS, load_bundle, load_profiles
from adaptive_survey.demo import generate_demo
from adaptive_survey.schema import ALTERNATIVES


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_recheck.py"
_SPEC = importlib.util.spec_from_file_location("prepare_recheck_test_module", _SCRIPT)
recheck = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recheck)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def source_panel(tmp_path):
    paths = generate_demo(tmp_path / "source", n_respondents=8, seed=91)
    profiles = load_profiles(paths["profiles"])
    seen = list(profiles)[:4]
    rows = [row for row in read_rows(paths["responses"]) if row["uuid"] in seen]
    with paths["responses"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESPONSE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    split = {"schema_version": 1, "uuids": {"train": seen[:2], "validation": seen[2:3], "test": seen[3:]},
             "protocol": {"holdout_per_block": 2, "initial_per_block": 1, "budgets": [3, 5, 7, 9], "seed": 2026},
             "fingerprints": {path.name: recheck.file_hash(path) for path in paths.values()}}
    manifest = tmp_path / "original-split.json"
    manifest.write_text(json.dumps(split))
    return paths, manifest, split, rows


def test_recheck_preserves_original_labels_and_excludes_all_seen_ids(source_panel, tmp_path, monkeypatch):
    paths, manifest, old, old_rows = source_panel
    observed_calls = []

    def fake_generator(handoff, profiles, tasks, output, *, uuids, seed):
        observed_calls.append((tuple(uuids), seed))
        with output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESPONSE_FIELDS)
            writer.writeheader()
            for uid in sorted(uuids):
                for task in tasks.values():
                    writer.writerow({"uuid": uid, "context": task.context, "task_id": task.task_id,
                                     "block": task.block, "time_code": task.time_code,
                                     "mnl_choice": ALTERNATIVES[0], "mmnl_choice": ALTERNATIVES[3]})
        return {"seed": seed, "respondents": len(uuids), "rows": len(uuids) * len(tasks),
                "generator_sha256": "fake-generator-for-assembly-test"}

    monkeypatch.setattr(recheck, "generate_responses", fake_generator)
    out = tmp_path / "recheck"
    result = recheck.prepare_recheck(paths["profiles"].parent, manifest, "unused", out, test_size=2)
    split = json.loads((out / "split.json").read_text())
    old_ids = set().union(*map(set, old["uuids"].values()))
    assert split["uuids"]["train"] == old["uuids"]["train"]
    assert split["uuids"]["validation"] == old["uuids"]["validation"]
    assert split["protocol"] == old["protocol"]
    assert set(split["uuids"]["test"]).isdisjoint(old_ids)
    assert observed_calls == [(tuple(split["uuids"]["test"]), 2026)]
    assert result["counts"] == {"train": 2, "validation": 1, "test": 2}
    assert result["rows"] == 270
    loaded = load_bundle(out / "data/profiles.csv", out / "data/tasks.json", out / "data/responses.csv")
    assert len(loaded.profiles) == 5
    assert set(old["uuids"]["test"]).isdisjoint(loaded.profiles)
    source_profiles = load_profiles(paths["profiles"])
    assert all(profile == source_profiles[uid] for uid, profile in loaded.profiles.items())
    kept = set(old["uuids"]["train"]) | set(old["uuids"]["validation"])
    combined_old_rows = [row for row in read_rows(out / "data/responses.csv") if row["uuid"] in kept]
    assert combined_old_rows == [row for row in old_rows if row["uuid"] in kept]
    assert (out / "data/tasks.json").read_bytes() == paths["tasks"].read_bytes()
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["excluded_original"]["all_excluded"] == 4
    assert provenance["preserved"]["response_rows"] == 162
    assert provenance["fresh_test"]["generator"]["seed"] == 2026
    assert loaded.validation.profiles_without_responses == 0
    before = (out / "data/responses.csv").read_bytes()
    with pytest.raises(FileExistsError):
        recheck.prepare_recheck(paths["profiles"].parent, manifest, "unused", out, test_size=2)
    assert before == (out / "data/responses.csv").read_bytes()


def test_recheck_refuses_source_fingerprint_drift_before_creating_outputs(source_panel, tmp_path):
    paths, manifest, _, _ = source_panel
    with paths["profiles"].open("a") as stream:
        stream.write("\n")
    out = tmp_path / "mismatched"
    with pytest.raises(ValueError, match="do not match"):
        recheck.prepare_recheck(paths["profiles"].parent, manifest, "unused", out, test_size=2)
    assert not out.exists()


def test_fresh_uuid_selection_is_disjoint_reproducible_and_bounded():
    profiles = [f"artificial-{index}" for index in range(50)]
    excluded = set(profiles[:10])
    sample = recheck.choose_fresh_uuids(profiles, excluded, 12, 2026)
    assert len(sample) == len(set(sample)) == 12
    assert set(sample).isdisjoint(excluded)
    assert sample == recheck.choose_fresh_uuids(reversed(profiles), excluded, 12, 2026)
    assert sample != recheck.choose_fresh_uuids(profiles, excluded, 12, 2027)
    with pytest.raises(ValueError, match="Insufficient"):
        recheck.choose_fresh_uuids(profiles, excluded, 41, 2026)
    with pytest.raises(ValueError, match="positive"):
        recheck.choose_fresh_uuids(profiles, excluded, 0, 2026)
