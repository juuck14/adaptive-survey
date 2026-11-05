import csv
import hashlib
import json

import pytest

from adaptive_survey.data import (ResponseWorld, load_bundle, load_profiles,
                                  load_tasks, validate_response_csv)
from adaptive_survey.demo import generate_demo
from adaptive_survey.schema import ATTRIBUTES, CONTEXTS, PERSON_FIELDS


@pytest.fixture
def panel(tmp_path):
    return generate_demo(tmp_path / "panel", n_respondents=3, seed=17)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bundle(paths, **kwargs):
    return load_bundle(paths["profiles"], paths["tasks"], paths["responses"], **kwargs)


def test_demo_contract_and_validation(panel):
    data = bundle(panel)
    assert {path.name for path in panel.values()} == {"profiles.csv", "tasks.json", "responses.csv"}
    assert len(data.profiles) == 3
    assert len(data.tasks) == 54
    assert all(len(profile.features) == 9 for profile in data.profiles.values())
    assert all(task.matrix.shape == (4, 13) for task in data.tasks.values())
    assert all(not task.matrix[3].any() for task in data.tasks.values())
    assert {context: sum(task.context == context for task in data.tasks.values())
            for context in CONTEXTS} == dict.fromkeys(CONTEXTS, 18)
    assert set(data.responses) == {"mnl", "mmnl"}
    report = data.validation
    assert report.rows == 162 and report.respondents == 3
    assert report.min_tasks_per_respondent == report.max_tasks_per_respondent == 54
    assert report.context_counts == dict.fromkeys(CONTEXTS, 54)
    assert report.block_counts == dict.fromkeys(range(1, 7), 27)
    assert report.time_code_counts == dict.fromkeys(("2H", "12H", "24H"), 54)
    assert all(sum(counts.values()) == 162 for counts in report.choice_counts.values())
    assert report.profiles_without_responses == 0
    assert report.sha256 == hashlib.sha256(panel["responses"].read_bytes()).hexdigest()
    assert report.to_dict()["scenario_counts"] == {"mnl": 162, "mmnl": 162}


def test_demo_reproducible_and_cohort_extension_preserves_existing_people(tmp_path):
    small = generate_demo(tmp_path / "small", 2, seed=23)
    repeated = generate_demo(tmp_path / "repeat", 2, seed=23)
    larger = generate_demo(tmp_path / "large", 4, seed=23)
    for key in small:
        assert small[key].read_bytes() == repeated[key].read_bytes()
    assert small["tasks"].read_bytes() == larger["tasks"].read_bytes()
    _, short = read_rows(small["responses"])
    _, full = read_rows(larger["responses"])
    assert short == full[:108]
    assert any(row["mnl_choice"] != row["mmnl_choice"] for row in full)


@pytest.mark.parametrize("existing", ["profiles.csv", "tasks.json", "responses.csv"])
def test_demo_refuses_any_existing_survey_target_before_writing(tmp_path, existing):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / existing
    sentinel.write_bytes(b"original data must survive")
    with pytest.raises(FileExistsError, match="new output directory"):
        generate_demo(target, n_respondents=2)
    assert sentinel.read_bytes() == b"original data must survive"
    assert {path.name for path in target.iterdir()} == {existing}


def test_bom_profiles_filter_and_ignored_private_columns(panel):
    fields, rows = read_rows(panel["profiles"])
    fields.append("latent_taste")
    for row in rows:
        row["latent_taste"] = "must never become a model input"
    write_rows(panel["profiles"], fields, rows)
    panel["profiles"].write_bytes(b"\xef\xbb\xbf" + panel["profiles"].read_bytes())
    profile = load_profiles(panel["profiles"], [rows[0]["uuid"]])
    assert len(profile) == 1
    assert len(next(iter(profile.values())).features) == len(PERSON_FIELDS)


@pytest.mark.parametrize("problem", ["duplicate", "missing_column", "invalid_binary", "category_conflict"])
def test_malformed_profiles_fail(panel, problem):
    fields, rows = read_rows(panel["profiles"])
    if problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "missing_column":
        fields.remove("EDU_6")
        for row in rows:
            row.pop("EDU_6")
    elif problem == "invalid_binary":
        rows[0]["GEN_Z"] = "nan"
    else:
        rows[0].update(GEND_F="1", GEND_O="1")
    write_rows(panel["profiles"], fields, rows)
    with pytest.raises(ValueError):
        load_profiles(panel["profiles"])


def test_task_list_metadata_wrapper_and_id_mapping(panel):
    records = json.loads(panel["tasks"].read_text())
    assert "context" not in records[0]
    expected = load_tasks(panel["tasks"])
    panel["tasks"].write_text(json.dumps({"metadata": {"n": 54}, "tasks": records}))
    assert load_tasks(panel["tasks"]) == expected
    mapping = {record["task_id"]: {key: value for key, value in record.items() if key != "task_id"}
               for record in records}
    panel["tasks"].write_text(json.dumps({"sp_tasks": mapping}))
    assert load_tasks(panel["tasks"]) == expected


@pytest.mark.parametrize("problem", ["duplicate", "missing", "context", "time", "id", "levels", "none", "attribute"])
def test_invalid_task_contract_fails(panel, problem):
    records = json.loads(panel["tasks"].read_text())
    if problem == "duplicate":
        records.append(dict(records[0]))
    elif problem == "missing":
        records.pop()
    elif problem == "context":
        records[0]["context"] = "stopover"
    elif problem == "time":
        records[0]["time_code"] = "24H"
    elif problem == "id":
        records[0]["task_id"] = "B2_Q1"
    elif problem == "levels":
        records[0]["alternatives"][0].update(DNS_A=1, DNS_B=1)
    elif problem == "none":
        records[0]["alternatives"][3]["WIFI_H"] = 1
    else:
        records[0]["alternatives"][0].pop(ATTRIBUTES[0])
    panel["tasks"].write_text(json.dumps(records))
    with pytest.raises(ValueError):
        load_tasks(panel["tasks"])


@pytest.mark.parametrize("field,value", [("context", "transfer"), ("block", "2"),
                                          ("time_code", "2H"), ("mnl_choice", "D"),
                                          ("mmnl_choice", "4"), ("uuid", "missing"),
                                          ("task_id", "B7_Q1")])
def test_response_contract_failures_are_found_before_filtering(panel, field, value):
    fields, rows = read_rows(panel["responses"])
    wanted = rows[0]["uuid"]
    rows[-1][field] = value
    write_rows(panel["responses"], fields, rows)
    with pytest.raises(ValueError):
        bundle(panel, uuids=[wanted])


def test_duplicate_missing_and_noncontiguous_respondents_fail(panel):
    fields, rows = read_rows(panel["responses"])
    write_rows(panel["responses"], fields, [rows[0], *rows])
    with pytest.raises(ValueError, match="Duplicate"):
        bundle(panel)
    write_rows(panel["responses"], fields, rows[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        bundle(panel)
    write_rows(panel["responses"], fields, [*rows, rows[0]])
    with pytest.raises(ValueError, match="contiguous"):
        validate_response_csv(panel["responses"], load_tasks(panel["tasks"]),
                              load_profiles(panel["profiles"]), strict_complete=False)


def test_filtered_loading_and_retention_limit(panel):
    _, rows = read_rows(panel["responses"])
    uuid = rows[0]["uuid"]
    data = bundle(panel, uuids=[uuid])
    assert set(data.profiles) == {uuid}
    assert len(data.responses["mnl"]) == 54
    assert data.validation.respondents == 3  # validation covers unselected people
    with pytest.raises(ValueError, match="retention limit"):
        bundle(panel, max_retained_rows=53)
    with pytest.raises(ValueError, match="absent from profiles"):
        bundle(panel, uuids=["unknown"])


def test_profiles_without_responses_are_not_exposed_as_available_cohort(panel):
    fields, rows = read_rows(panel["profiles"])
    extra = dict(rows[0], uuid="unused-public-profile")
    write_rows(panel["profiles"], fields, [*rows, extra])
    data = bundle(panel)
    assert "unused-public-profile" not in data.profiles
    assert data.validation.profiles_without_responses == 1
    with pytest.raises(ValueError, match="missing response"):
        bundle(panel, uuids=["unused-public-profile"])


def test_world_scenario_snapshot_and_reveal_guard(panel):
    fields, rows = read_rows(panel["responses"])
    rows[0].update(mnl_choice="A", mmnl_choice="NONE")
    write_rows(panel["responses"], fields, rows)
    data = bundle(panel)
    uuid, task_id = rows[0]["uuid"], rows[0]["task_id"]
    mnl, mmnl = ResponseWorld(data, "mnl"), ResponseWorld(data, "MMNL")
    assert mnl.reveal(uuid, task_id, {task_id}) == 0
    assert mmnl.reveal(uuid, task_id, {task_id}) == 3
    with pytest.raises(ValueError, match="held-out"):
        mmnl.reveal(uuid, task_id, {"B2_Q1"})
    with pytest.raises(ValueError, match="No response"):
        mmnl.answer("missing", task_id)
    with pytest.raises(ValueError, match="scenario"):
        ResponseWorld(data, "other")
    assert set(vars(mnl)) == {"scenario", "_answers"}
    with pytest.raises(TypeError):
        data.responses["mnl"][(uuid, task_id)] = 2


def test_explicit_partial_panel_and_ignored_response_columns(panel):
    fields, rows = read_rows(panel["responses"])
    fields.append("true_probability")
    for row in rows:
        row["true_probability"] = ".8"
    write_rows(panel["responses"], fields, rows[:10])
    report = validate_response_csv(panel["responses"], load_tasks(panel["tasks"]),
                                   load_profiles(panel["profiles"]), strict_complete=False)
    assert report.rows == 10 and report.min_tasks_per_respondent == 10
    assert report.ignored_columns == ("true_probability",)
