"""Team contract, future-replicate exclusion, reveal boundaries and CSV export."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import torch

from adaptive_survey.cached_neural import CachedNeuralPredictor
from adaptive_survey.neural import ChoiceNetwork, NeuralPredictor
from adaptive_survey.policies import FixedPolicy, WeightedJSPolicy
from adaptive_survey.runner import run_episode
from adaptive_survey.schema import ATTRIBUTES, CONTEXTS, PERSON_FIELDS, BasePredictor, Profile, Task
from adaptive_survey.team_handoff import (TeamDesign, file_hash, load_team_samples,
                                         read_csv, run_team, write_json)

DESIGN = Path(__file__).resolve().parents[1] / "configs/team_20261112"


def task_catalog():
    _, rows = read_csv(DESIGN / "task_catalog.csv")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    return {tid: Task(tid, alts[0]["context"], int(alts[0]["task_block"]),
                      int(alts[0]["question_in_block"]), alts[0]["time_code"],
                      tuple(tuple(float(r[a]) for a in ATTRIBUTES) for r in alts))
            for tid, alts in grouped.items()}


def sample_row(uuid, context, replicate=1):
    row = dict(sample_id=context + "_0001", uuid=uuid, context=context, stratum_id="S01",
               generation="Millennial", gender="Male", education="Bachelor",
               replicate_id=str(replicate), sampling_seed=str(20261110 + replicate))
    row.update(dict.fromkeys(PERSON_FIELDS, "0"))
    return row


def csv_file(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prescribed_design_and_adaptive_candidate_scope():
    design, tasks = TeamDesign.load(DESIGN), task_catalog()
    design.validate_catalog(tasks)
    for context in CONTEXTS:
        fixed = design.episode("person", context, tasks, "fixed")
        adaptive = design.episode("person", context, tasks, "non_graph")
        assert len(fixed.candidates) == 9 and len(adaptive.candidates) == 18
        assert fixed.initial == adaptive.initial
        assert fixed.budgets == adaptive.budgets == (9,)
        assert not fixed.holdout and not adaptive.holdout
        assert tuple(t.task_id for t in fixed.candidates) == design.fixed[context]
    bad = dict(tasks)
    original = next(iter(bad))
    bad[original] = replace(bad[original], attributes=((0.0,) * 13,) * 4)
    with pytest.raises(ValueError, match="attributes differ"):
        design.validate_catalog(bad)


def test_reserved_union_covers_future_replicates(tmp_path):
    profiles, records, all_uuids = {}, [], []
    for rep in range(1, 11):
        rows = [sample_row(f"synthetic-{rep}-{c}", c, rep) for c in CONTEXTS]
        for row in rows:
            all_uuids.append(row["uuid"])
            profiles[row["uuid"]] = Profile(row["uuid"], (0.0,) * 9)
        path = tmp_path / f"sampled_respondents_rep{rep:02d}.csv"
        csv_file(path, rows)
        records.append(dict(file_name=path.name, sha256=file_hash(path), replicate_id=rep,
                            sampling_seed=20261110 + rep))
    csv_file(tmp_path / "reserved_respondent_uuids.csv", [{"uuid": u} for u in all_uuids])
    write_json(tmp_path / "sampling_manifest_10rep.json",
               {"round_files": records, "unique_uuid_all_replicates": 30})
    samples, reserved, _ = load_team_samples(tmp_path, profiles, expected_per_context=1)
    assert len(samples) == 30 and reserved == set(all_uuids)
    # Omitting only the tenth replicate must fail, even when the first is intact.
    csv_file(tmp_path / "reserved_respondent_uuids.csv", [{"uuid": u} for u in all_uuids[:-3]])
    with pytest.raises(ValueError, match="union of all ten"):
        load_team_samples(tmp_path, profiles, expected_per_context=1)


def test_step_callback_preserves_revealed_prefix_on_failure():
    design, tasks = TeamDesign.load(DESIGN), task_catalog()
    profile = Profile("synthetic", (0.0,) * 9)
    episode = design.episode(profile.uuid, "transfer", tasks, "fixed")
    saved = []

    class InterruptedWorld:
        scenario = "mnl"
        def reveal(self, uuid, task_id, allowed):
            assert task_id in allowed
            if len(saved) == 4:
                raise RuntimeError("interrupt")
            return 0

    with pytest.raises(RuntimeError, match="interrupt"):
        run_episode(profile, "transfer", InterruptedWorld(), episode, BasePredictor(),
                    FixedPolicy(design.fixed), on_step=saved.append)
    assert tuple(s.task_id for s in saved) == design.fixed["transfer"][:4]


def test_cached_probabilities_and_acquisition_match_original():
    tasks = task_catalog()
    torch.manual_seed(123)
    reference = NeuralPredictor([ChoiceNetwork(), ChoiceNetwork()])
    cached = CachedNeuralPredictor(reference, tasks)
    requests = []
    for context in CONTEXTS:
        pool = [t for t in tasks.values() if t.context == context]
        state = reference.reset(Profile("synthetic", (0.0,) * 9), context)
        requests.append((state, ()))
        for index in range(6):
            state = state.append(pool[index], index % 4)
            requests.append((state, pool[index + 1:]))
    for expected, actual in zip(reference.predict_many(requests), cached.predict_many(requests)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    from adaptive_survey.acquisition import weighted_js_scores
    state, remaining = requests[4]
    original = weighted_js_scores(reference, state, remaining).scores
    fast = weighted_js_scores(cached, state, remaining).scores
    np.testing.assert_allclose(list(original.values()), list(fast.values()), atol=1e-8)
    before = reference.predict(state, remaining)
    cached.predict(state.append(remaining[0], 3), remaining[1:])
    np.testing.assert_array_equal(reference.predict(state, remaining), before)


def test_fixed_output_passes_supplied_validator_and_is_immutable(tmp_path):
    design, tasks = TeamDesign.load(DESIGN), task_catalog()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    samples = [sample_row("synthetic-" + c, c) for c in CONTEXTS]
    sample_path = samples_dir / "sampled_respondents_rep01.csv"
    csv_file(sample_path, samples)
    csv_file(prepared / "profiles.csv", [{"uuid": r["uuid"], **dict.fromkeys(PERSON_FIELDS, "0")}
                                         for r in samples])
    task_json = [{"task_id": t.task_id, "block": t.block, "cset": t.cset, "time_code": t.time_code,
                  "alternatives": [dict(zip(ATTRIBUTES, a)) for a in t.attributes]}
                 for t in tasks.values()]
    write_json(prepared / "tasks.json", task_json)
    with sqlite3.connect(prepared / "responses.sqlite") as db:
        db.execute("CREATE TABLE responses(uuid TEXT,task TEXT,mnl INTEGER,mmnl INTEGER,PRIMARY KEY(uuid,task))")
        db.executemany("INSERT INTO responses VALUES(?,?,?,?)",
                       [(r["uuid"], t.task_id, 0, 3) for r in samples
                        for t in tasks.values() if t.context == r["context"]])
    meta = dict(protocol="team_handoff_20261112", handoff_dir=str(design.root), samples_dir=str(samples_dir),
                design_hashes=design.hashes, sample_hashes={sample_path.name: file_hash(sample_path)},
                train=["train-only"], validation=["validation-only"], response_bank_sha256="fixture",
                prepared_hashes={n: file_hash(prepared / n) for n in ("profiles.csv", "tasks.json", "responses.sqlite")})
    write_json(prepared / "prepared.json", meta)
    output = tmp_path / "run"
    result = run_team(prepared, "panel_mmnl", "fixed", 1, output, limit_per_context=1)
    assert result["rows"] == 27
    target = output / "responses_rep01_panel_mmnl_fixed.csv"
    fields, rows = read_csv(target)
    assert tuple(fields) == design.response_fields and len(fields) == 32
    assert {r["chosen_alternative"] for r in rows} == {"NONE"}
    assert json.loads((output / "rep01_panel_mmnl_fixed.verification.json").read_text())["pilot"]
    with pytest.raises(FileExistsError, match="Run already exists"):
        run_team(prepared, "panel_mmnl", "fixed", 1, output, limit_per_context=1)
