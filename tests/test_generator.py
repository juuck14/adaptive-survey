import csv
import json

import pytest

from adaptive_survey.data import load_profiles, load_tasks
from adaptive_survey.demo import generate_demo
from adaptive_survey.generator import generate_responses


def fake_handoff(tmp_path):
    path = tmp_path / "handoff"
    path.mkdir()
    (path / "response_generator.py").write_text('''
import numpy as np
from adaptive_survey.schema import ATTRIBUTES, PERSON_FIELDS, ALTERNATIVES
class ResponseGenerator:
    def __init__(self, mode): self.mode = mode
    def draw_person_taste(self, rng): return rng.normal(size=5)
    def sample_response(self, alternatives, context, person, rng, taste=None):
        if self.mode == 'mmnl':
            assert taste is not None
            index = int(abs(taste[0] * 10)) % 4
        else:
            assert taste is None
            index = int(rng.integers(4))
        return ALTERNATIVES[index], np.full(4, .25)
''')
    return path


def test_offline_generation_personal_taste_and_keyed_responses(tmp_path):
    data = tmp_path / "data"
    generate_demo(data, n_respondents=3)
    profiles, tasks = load_profiles(data / "profiles.csv"), load_tasks(data / "tasks.json")
    handoff = fake_handoff(tmp_path)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    meta = generate_responses(handoff, profiles, tasks, a)
    generate_responses(handoff, profiles, dict(reversed(list(tasks.items()))), b,
                       uuids=list(reversed(list(profiles))))
    assert a.read_bytes() == b.read_bytes()
    rows = list(csv.DictReader(a.open()))
    assert meta["rows"] == 162
    for uid in profiles:
        person = [row for row in rows if row["uuid"] == uid]
        assert len(person) == 54
        assert {row["context"] for row in person} == {"transfer", "layover", "stopover"}
        assert len({row["mmnl_choice"] for row in person}) == 1
    assert "public_inputs_sha256" in meta
    assert json.loads(a.with_suffix(".csv.manifest.json").read_text())["sha256"] == meta["sha256"]
    with pytest.raises(FileExistsError):
        generate_responses(handoff, profiles, tasks, a)


def test_existing_manifest_is_never_overwritten(tmp_path):
    data = tmp_path / "data"
    generate_demo(data, n_respondents=2)
    profiles, tasks = load_profiles(data / "profiles.csv"), load_tasks(data / "tasks.json")
    handoff = fake_handoff(tmp_path)
    output = tmp_path / "new.csv"
    manifest = tmp_path / "new.csv.manifest.json"
    manifest.write_text("keep")
    with pytest.raises(FileExistsError):
        generate_responses(handoff, profiles, tasks, output)
    assert manifest.read_text() == "keep"
    assert not output.exists()
