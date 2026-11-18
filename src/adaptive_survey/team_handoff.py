"""The November 12 team protocol: prescribed cohorts, initial 3 + adaptive 6.

Inputs and outputs stay private. The learner sees public features and revealed
answers only; the response-bank connection belongs exclusively to the runner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from .data import load_profiles, load_tasks
from .protocol import Episode, stable_hash
from .runner import run_episode
from .schema import ALTERNATIVES, CONTEXTS, PERSON_FIELDS, BasePredictor, Profile


GENERATORS = {"mnl": "mnl", "panel_mmnl": "mmnl"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("Missing or duplicate CSV header")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed CSV record")
    return fields, rows


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TeamDesign:
    root: Path
    manifest: dict
    initial: dict
    fixed: dict
    base_blocks: dict
    response_fields: tuple
    manifest_fields: tuple
    hashes: dict

    @classmethod
    def load(cls, directory):
        root = Path(directory).resolve()
        manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8-sig"))
        if (manifest["question_budget"], manifest["initial_count"],
                manifest["context_pool"], manifest["prediction_holdouts"]) != (9, 3, 18, 0):
            raise ValueError("Expected the team initial-3, budget-9, no-holdout protocol")
        _, initial_rows = read_csv(root / "initial_tasks_main.csv")
        _, fixed_rows = read_csv(root / "fixed_task_order.csv")
        initial, fixed, bases = {}, {}, manifest["main_baseline_blocks"]
        for context in CONTEXTS:
            initials = sorted((r for r in initial_rows if r["context"] == context),
                              key=lambda r: int(r["initial_position"]))
            full = sorted((r for r in fixed_rows if r["context"] == context),
                          key=lambda r: int(r["question_order"]))
            initial[context] = tuple(r["task_id"] for r in initials)
            fixed[context] = tuple(r["task_id"] for r in full)
            if (len(initials) != 3 or len(full) != 9 or len(set(fixed[context])) != 9
                    or initial[context] != fixed[context][:3]
                    or {int(r["task_block"]) for r in full} != {bases[context]}):
                raise ValueError("Invalid prescribed initial/Fixed task order")
        response_fields, _ = read_csv(root / "survey_responses_template.csv")
        manifest_fields, _ = read_csv(root / "run_manifest_template.csv")
        if response_fields != manifest["response_columns"]:
            raise ValueError("Response template differs from package manifest")
        names = ("package_manifest.json", "task_catalog.csv", "initial_tasks_main.csv",
                 "fixed_task_order.csv", "survey_responses_template.csv",
                 "run_manifest_template.csv", "validate_and_convert.py")
        return cls(root, manifest, initial, fixed, bases, tuple(response_fields),
                   tuple(manifest_fields), {n: file_hash(root / n) for n in names})

    def episode(self, uuid, context, tasks, method):
        if method not in ("fixed", "non_graph"):
            raise ValueError("Implemented team methods: fixed and non_graph")
        ids = (self.fixed[context] if method == "fixed" else
               tuple(t.task_id for t in tasks.values() if t.context == context))
        if len(ids) != (9 if method == "fixed" else 18):
            raise ValueError("Team candidate pool has the wrong size")
        return Episode(uuid, context, (), tuple(tasks[t] for t in self.initial[context]),
                       tuple(tasks[t] for t in ids), (9,), track="preference")

    def validate_catalog(self, tasks):
        from .schema import ATTRIBUTES
        _, rows = read_csv(self.root / "task_catalog.csv")
        if len(rows) != 216 or len(tasks) != 54:
            raise ValueError("Expected 54 tasks and 216 alternatives")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["task_id"]].append(row)
        if grouped.keys() != tasks.keys():
            raise ValueError("Task JSON and team catalog have different IDs")
        for task_id, task in tasks.items():
            records = grouped[task_id]
            if tuple(r["alternative"] for r in records) != ALTERNATIVES:
                raise ValueError("Catalog alternative order changed")
            matrix = tuple(tuple(float(r[a]) for a in ATTRIBUTES) for r in records)
            if matrix != task.attributes or any(r["context"] != task.context for r in records):
                raise ValueError("Task JSON attributes differ from the corrected team catalog")


def load_team_samples(samples_dir, profiles, *, expected_per_context=500):
    root = Path(samples_dir)
    sampling = json.loads((root / "sampling_manifest_10rep.json").read_text(encoding="utf-8-sig"))
    _, reserved_rows = read_csv(root / "reserved_respondent_uuids.csv")
    reserved = {r["uuid"] for r in reserved_rows}
    if len(reserved) != len(reserved_rows):
        raise ValueError("Duplicate reserved UUIDs")
    all_rows, fingerprints = [], {}
    for record in sampling["round_files"]:
        path = root / record["file_name"]
        actual_hash = file_hash(path)
        if actual_hash != record["sha256"]:
            raise ValueError("A supplied replicate file changed")
        _, rows = read_csv(path)
        if (len(rows) != 3 * expected_per_context or
                len({r["uuid"] for r in rows}) != len(rows) or
                len({r["sample_id"] for r in rows}) != len(rows) or
                Counter(r["context"] for r in rows) != dict.fromkeys(CONTEXTS, expected_per_context)):
            raise ValueError("Wrong replicate size, duplicate UUID/sample ID or context assignment")
        for row in rows:
            profile = profiles.get(row["uuid"])
            if (profile is None or profile.features != tuple(float(row[k]) for k in PERSON_FIELDS)
                    or int(row["replicate_id"]) != record["replicate_id"]
                    or int(row["sampling_seed"]) != record["sampling_seed"]):
                raise ValueError("Sample/profile/replicate metadata mismatch")
        all_rows.extend(rows)
        fingerprints[path.name] = actual_hash
    if (len(fingerprints) != 10 or {r["uuid"] for r in all_rows} != reserved
            or len(reserved) != sampling["unique_uuid_all_replicates"]):
        raise ValueError("Reserved UUID list is not the union of all ten evaluation cohorts")
    return all_rows, reserved, fingerprints


def prepare_team(inputs, handoff, output, *, train_size=1200, validation_size=300, seed=20261118):
    """Exclude the union of all evaluation UUIDs, then index needed bank rows once."""
    inputs, output = Path(inputs), Path(output)
    if output.exists():
        raise FileExistsError("Preparation output exists; select a fresh directory")
    if train_size < 1 or validation_size < 1:
        raise ValueError("Training and validation cohorts must be nonempty")
    design = TeamDesign.load(handoff)
    profiles = load_profiles(inputs / "nemotron_mapped.csv")
    tasks = load_tasks(inputs / "sp_tasks.json")
    design.validate_catalog(tasks)
    sample_rows, reserved, sample_hashes = load_team_samples(inputs, profiles)
    sampling = json.loads((inputs / "sampling_manifest_10rep.json").read_text(encoding="utf-8-sig"))
    source_profile_hash = next(r["sha256"] for r in sampling["source_files"]
                               if r["file_name"] == "nemotron_mapped.csv")
    if file_hash(inputs / "nemotron_mapped.csv") != source_profile_hash:
        raise ValueError("Profiles differ from those used to construct the team sample")
    supported = {profiles[row["uuid"]].features for row in sample_rows}
    available = sorted((u for u, p in profiles.items() if u not in reserved and p.features in supported),
                       key=lambda u: (stable_hash(seed, "team-train-validation", u), u))
    if len(available) < train_size + validation_size:
        raise ValueError("Insufficient non-reserved public profiles")
    train = available[:train_size]
    validation = available[train_size:train_size + validation_size]
    learner_ids = set(train + validation)
    wanted = defaultdict(set)
    for u in learner_ids:
        wanted[u].update(tasks)
    for row in sample_rows:
        wanted[row["uuid"]].update(t.task_id for t in tasks.values() if t.context == row["context"])
    output.mkdir(parents=True)
    db = sqlite3.connect(output / "responses.sqlite")
    db.execute("CREATE TABLE responses(uuid TEXT, task TEXT, mnl INTEGER, mmnl INTEGER, PRIMARY KEY(uuid,task)) WITHOUT ROWID")
    bank = inputs / "synthetic_responses_full.csv"
    start, scanned, retained = perf_counter(), 0, 0
    try:
        with bank.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            fields = next(reader)
            indices = {name: fields.index(name) for name in ("uuid", "task_id", "context", "block", "time_code", "mnl_choice", "mmnl_choice")}
            batch = []
            for row in reader:
                scanned += 1
                uuid = row[indices["uuid"]]
                if uuid not in wanted:
                    continue
                task_id = row[indices["task_id"]]
                if task_id not in wanted[uuid]:
                    continue
                task = tasks[task_id]
                if (row[indices["context"]] != task.context or int(row[indices["block"]]) != task.block
                        or row[indices["time_code"]] != task.time_code):
                    raise ValueError("Response-bank task metadata mismatch")
                batch.append((uuid, task_id, *(ALTERNATIVES.index(row[indices[s + "_choice"]])
                                               for s in ("mnl", "mmnl"))))
                retained += 1
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO responses VALUES (?,?,?,?)", batch)
                    batch.clear()
            db.executemany("INSERT INTO responses VALUES (?,?,?,?)", batch)
        expected = sum(len(v) for v in wanted.values())
        if retained != expected:
            raise ValueError(f"Response bank missing selected records: {retained} != {expected}")
        db.commit()
    finally:
        db.close()
    with (output / "profiles.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("uuid", *PERSON_FIELDS))
        writer.writerows((u, *profiles[u].features) for u in sorted(wanted))
    import shutil
    shutil.copyfile(inputs / "sp_tasks.json", output / "tasks.json")
    shutil.copytree(design.root, output / "handoff")
    (output / "samples").mkdir()
    for name in (*sample_hashes, "sampling_manifest_10rep.json", "reserved_respondent_uuids.csv"):
        shutil.copyfile(inputs / name, output / "samples" / name)
    payload = {
        "protocol": "team_handoff_20261112", "seed": seed, "handoff_dir": str(design.root),
        "samples_dir": str(inputs.resolve()), "train": train, "validation": validation,
        "reserved_respondents": len(reserved), "evaluation_sample_records": len(sample_rows),
        "training_evaluation_overlap": len(learner_ids & reserved),
        "source_profile_sha256": source_profile_hash,
        "response_bank_sha256": file_hash(bank), "response_bank_bytes": bank.stat().st_size,
        "response_bank_scanned_rows": scanned, "indexed_rows": retained,
        "sample_hashes": sample_hashes, "design_hashes": design.hashes,
        "prepared_hashes": {n: file_hash(output / n) for n in ("responses.sqlite", "profiles.csv", "tasks.json")},
        "preparation_seconds": perf_counter() - start,
    }
    write_json(output / "prepared.json", payload)
    return {k: payload[k] for k in ("reserved_respondents", "evaluation_sample_records", "indexed_rows", "training_evaluation_overlap", "preparation_seconds")}


def read_prepared(directory):
    root = Path(directory)
    metadata = json.loads((root / "prepared.json").read_text())
    if metadata["protocol"] != "team_handoff_20261112":
        raise ValueError("Wrong preparation protocol")
    for name, digest in metadata["prepared_hashes"].items():
        if file_hash(root / name) != digest:
            raise ValueError("Prepared input changed: " + name)
    design = TeamDesign.load(root / "handoff" if (root / "handoff").is_dir() else metadata["handoff_dir"])
    if design.hashes != metadata["design_hashes"]:
        raise ValueError("Handoff design changed after preparation")
    return metadata, design, load_profiles(root / "profiles.csv"), load_tasks(root / "tasks.json")


def prepared_samples_dir(root, metadata):
    local = Path(root) / "samples"
    return local if local.is_dir() else Path(metadata["samples_dir"])


def train_team(prepared, generator, checkpoint, *, config=None):
    from .training import TrainingConfig, train_ensemble
    from .cli import _runtime
    root, checkpoint = Path(prepared), Path(checkpoint)
    if checkpoint.exists() or Path(str(checkpoint) + ".training.json").exists():
        raise FileExistsError("Checkpoint output already exists")
    metadata, design, profiles, tasks = read_prepared(root)
    scenario = GENERATORS[generator]
    learner_ids = set(metadata["train"] + metadata["validation"])
    with sqlite3.connect(root / "responses.sqlite") as db:
        responses = {(u, t): choice for u, t, choice in db.execute(f"SELECT uuid,task,{scenario} FROM responses")
                     if u in learner_ids}
    start = perf_counter()
    result = train_ensemble(profiles, tasks, responses, metadata["train"], metadata["validation"],
                            config=config or TrainingConfig(max_history=8, seed=metadata["seed"]),
                            initial_task_ids=design.initial)
    result.predictor.metadata.update({
        "team_prepared_sha256": file_hash(root / "prepared.json"), "generator_type": generator,
        "team_design_hashes": design.hashes, "training_seconds": perf_counter() - start,
        "runtime": _runtime(), "reserved_respondents": metadata["reserved_respondents"],
        "training_evaluation_overlap": 0,
    })
    result.predictor.save(checkpoint)
    write_json(str(checkpoint) + ".training.json", {"metadata": result.predictor.metadata, "history": result.history})
    return {"checkpoint": str(checkpoint), "best_epochs": result.metadata["best_epochs"],
            "training_seconds": result.predictor.metadata["training_seconds"]}


class BankWorld:
    def __init__(self, database, scenario):
        self.database, self.scenario = database, scenario

    def reveal(self, uuid, task_id, allowed_task_ids):
        if task_id not in allowed_task_ids:
            raise ValueError("Task outside allowed episode")
        row = self.database.execute(f"SELECT {self.scenario} FROM responses WHERE uuid=? AND task=?",
                                    (uuid, task_id)).fetchone()
        if row is None:
            raise ValueError("Missing response-bank entry")
        return row[0]


def validate_with_team_tool(design, path, *, expected_people=None, expected_per_context=None):
    """Use the supplied validator, preserving its stratum quotas and ID rules."""
    spec = importlib.util.spec_from_file_location("team_supplied_validator", design.root / "validate_and_convert.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    return validator.validate(path, expected_people, expected_per_context)


def run_team(prepared, generator, method, replicate, output, *, checkpoint=None,
             seed=20261118, limit_per_context=None, accelerated=False):
    """Write each revealed step; interrupted runs retain partial CSVs and status."""
    from .policies import FixedPolicy, WeightedJSPolicy
    from .neural import NeuralPredictor
    root, output = Path(prepared), Path(output)
    metadata, design, profiles, tasks = read_prepared(root)
    if generator not in GENERATORS or method not in ("fixed", "non_graph") or replicate not in range(1, 11):
        raise ValueError("Unknown team run")
    sample_file = prepared_samples_dir(root, metadata) / f"sampled_respondents_rep{replicate:02d}.csv"
    if file_hash(sample_file) != metadata["sample_hashes"][sample_file.name]:
        raise ValueError("Replicate sample changed")
    _, samples = read_csv(sample_file)
    if limit_per_context is not None:
        if limit_per_context < 1:
            raise ValueError("Pilot limit must be positive")
        counts = Counter()
        limited = []
        for row in samples:
            if counts[row["context"]] < limit_per_context:
                limited.append(row)
                counts[row["context"]] += 1
        samples = limited
    if method == "non_graph":
        if checkpoint is None:
            raise ValueError("Non-graph requires a trained checkpoint")
        predictor = NeuralPredictor.load(checkpoint)
        if (predictor.metadata.get("team_prepared_sha256") != file_hash(root / "prepared.json")
                or predictor.metadata.get("generator_type") != generator):
            raise ValueError("Checkpoint preparation or generator mismatch")
        if accelerated:
            from .cached_neural import CachedNeuralPredictor
            predictor = CachedNeuralPredictor(predictor, tasks)
        policy = WeightedJSPolicy()
    else:
        predictor = BasePredictor()
        policy = FixedPolicy(design.fixed)
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"rep{replicate:02d}_{generator}_{method}"
    destination = output / f"responses_{run_id}.csv"
    partial = destination.with_suffix(".csv.partial")
    status_path = output / f"{run_id}.run.json"
    if any(p.exists() for p in (destination, partial, status_path)):
        raise FileExistsError("Run already exists; inspect it or select a new output directory")
    record = dict.fromkeys(design.manifest_fields, "")
    record.update(schema_version=design.manifest["package_version"], run_id=run_id, replicate_id=replicate,
                  generator_type=generator, selection_method=method, sample_file=sample_file.name,
                  sampling_seed=samples[0]["sampling_seed"], task_catalog_version=design.manifest["task_catalog_version"],
                  initial_set_version=design.manifest["initial_set_version"],
                  generator_version="existing_bank_sha256:" + metadata["response_bank_sha256"],
                  policy_version="team_v1_fixed_prescribed_or_weighted_js" + ("_cached_float32" if accelerated else ""),
                  policy_seed=seed, response_source="existing_response_bank",
                  response_bank_version=metadata["response_bank_sha256"], run_status="running",
                  expected_respondents=len(samples), completed_respondents=0, response_rows=0, started_at=utc_now())
    write_json(status_path, record)
    start = perf_counter()
    db = sqlite3.connect(root / "responses.sqlite")
    world = BankWorld(db, GENERATORS[generator])
    try:
        with partial.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=design.response_fields)
            writer.writeheader()
            for index, sample in enumerate(samples, 1):
                uuid, context = sample["uuid"], sample["context"]
                if uuid in metadata["train"] or uuid in metadata["validation"]:
                    raise ValueError("Evaluation UUID leaked into learner split")
                episode = design.episode(uuid, context, tasks, method)
                def record_step(step):
                    task = tasks[step.task_id]
                    row = dict(sample)
                    row.update(schema_version=record["schema_version"], run_id=run_id, generator_type=generator,
                               selection_method=method, survey_instance_id=run_id + "__" + sample["sample_id"],
                               base_block=design.base_blocks[context], task_id=task.task_id,
                               task_block=task.block, question_in_block=task.cset, question_order=step.step,
                               phase="initial" if step.initial else "subsequent",
                               chosen_alternative=ALTERNATIVES[step.choice],
                               acquisition_score=(step.scores[task.task_id] if method == "non_graph" and not step.initial else ""),
                               selection_time_ms=1000 * step.selection_seconds if not step.initial else "")
                    writer.writerow(row)
                    stream.flush()
                    record["response_rows"] += 1
                run_episode(profiles[uuid], context, world, episode, predictor, policy,
                            seed=seed, on_step=record_step)
                record.update(completed_respondents=index, response_rows=9 * index)
                if index % 100 == 0:
                    print(json.dumps({"run_id": run_id, "respondents": index,
                                      "seconds": round(perf_counter() - start, 1)}), flush=True)
        expected_per_context = len(samples) // 3
        validate_with_team_tool(design, partial, expected_people=len(samples), expected_per_context=expected_per_context)
        partial.rename(destination)
        record.update(run_status="completed", completed_at=utc_now())
    except BaseException as error:
        record.update(run_status="failed", completed_at=utc_now(), error_message=f"{type(error).__name__}: {error}")
        raise
    finally:
        db.close()
        status_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    write_json(output / f"{run_id}.verification.json", {
        "valid": True, "pilot": limit_per_context is not None, "response_sha256": file_hash(destination),
        "prepared_sha256": file_hash(root / "prepared.json"), "design_hashes": design.hashes,
        "checkpoint_sha256": file_hash(checkpoint) if checkpoint is not None else None,
        "seconds": perf_counter() - start, "rows": record["response_rows"],
    })
    return {"run_id": run_id, "rows": record["response_rows"], "seconds": perf_counter() - start}


def collect_run_manifest(directory, handoff):
    design, directory = TeamDesign.load(handoff), Path(directory)
    records = [json.loads(p.read_text()) for p in sorted(directory.glob("*.run.json"))]
    destination = directory / "run_manifest.csv"
    if destination.exists():
        raise FileExistsError("Manifest exists")
    with destination.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=design.manifest_fields)
        writer.writeheader()
        writer.writerows(records)
    return {"runs": len(records), "completed": sum(r["run_status"] == "completed" for r in records)}
