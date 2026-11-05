"""Offline integration with a separately supplied, trusted generator handoff.

The source and fitted coefficients stay outside this repository. No generator
probabilities or personal taste are returned to a policy.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .schema import ATTRIBUTES, PERSON_FIELDS, Profile, Task


def _seed(seed: int, *keys: str) -> int:
    payload = json.dumps([seed, *keys], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def generate_responses(handoff: str | Path, profiles: Mapping[str, Profile],
                       tasks: Mapping[str, Task], output: str | Path, *,
                       uuids: Sequence[str] | None = None, seed: int = 2026) -> dict:
    """Draw one MMNL taste per UUID; choices are keyed independently of order.

    This uses this package's RNG-key contract. It does not claim byte-for-byte
    equivalence to an existing CSV generated using another script or RNG scheme.
    Refuse to overwrite an existing completed or partial output.
    """
    root = Path(handoff).resolve()
    source = root / "response_generator.py" if root.is_dir() else root
    spec = importlib.util.spec_from_file_location("_survey_external_generator", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the trusted response generator handoff")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.ATTRIBUTES) != ATTRIBUTES or tuple(module.PERSON_FIELDS) != PERSON_FIELDS:
        raise ValueError("Generator input schema differs from this package")
    mnl, mmnl = module.ResponseGenerator("mnl"), module.ResponseGenerator("mmnl")
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or partial.exists() or manifest_path.exists():
        raise FileExistsError("Use a new output path; completed/partial data are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(profiles if uuids is None else uuids)
    ordered_tasks = sorted(tasks.values(), key=lambda q: (q.block, q.cset, q.task_id))
    if not ids or not ordered_tasks or len(set(ids)) != len(ids) or any(uid not in profiles for uid in ids):
        raise ValueError("UUID selection must contain unique, known profiles")
    rows = 0
    with partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("uuid", "context", "task_id", "block", "time_code", "mnl_choice", "mmnl_choice"))
        writer.writeheader()
        for uid in ids:
            person = dict(zip(PERSON_FIELDS, profiles[uid].features))
            taste = mmnl.draw_person_taste(np.random.default_rng(_seed(seed, uid, "taste")))
            for task in ordered_tasks:
                alternatives = [dict(zip(ATTRIBUTES, row)) for row in task.attributes[:3]] + [{}]
                choices = {}
                for mode, gen in (("mnl", mnl), ("mmnl", mmnl)):
                    rng = np.random.default_rng(_seed(seed, uid, task.task_id, mode, "response"))
                    choices[mode], _ = gen.sample_response(alternatives, task.context, person, rng,
                                                          taste=taste if mode == "mmnl" else None)
                writer.writerow({"uuid": uid, "context": task.context, "task_id": task.task_id,
                                 "block": task.block, "time_code": task.time_code,
                                 "mnl_choice": choices["mnl"], "mmnl_choice": choices["mmnl"]})
                rows += 1
    partial.replace(output)
    model_files = {p.name: _file_hash(p) for p in (source.parent / "models").glob("*.json")}
    metadata = {"schema_version": 1, "seed": seed, "rng_contract": "sha256-json-v1",
                "respondents": len(ids), "rows": rows, "sha256": _file_hash(output),
                "generator_sha256": _file_hash(source), "model_sha256": model_files,
                "personal_taste": "one draw per UUID, shared across all contexts",
                "provenance": "new generation; not an audit of an older CSV"}
    public_inputs = {
        "profiles": [(uid, profiles[uid].features) for uid in ids],
        "tasks": [(t.task_id, t.context, t.block, t.cset, t.time_code, t.attributes) for t in ordered_tasks],
    }
    metadata["public_inputs_sha256"] = hashlib.sha256(
        json.dumps(public_inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    return metadata
