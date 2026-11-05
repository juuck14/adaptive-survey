"""Validate imported survey files and expose one response scenario at a time.

The response validator streams UUID-contiguous CSVs.  Its working response set
contains only the current person's tasks; a temporary SQLite index detects a
UUID appearing again later without retaining the full response panel in RAM.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from .schema import (ALTERNATIVES, ATTRIBUTES, CONTEXT_BLOCKS, CONTEXT_TIMES,
                     PERSON_FIELDS, Profile, Task)

SCENARIOS = ("mnl", "mmnl")
RESPONSE_FIELDS = ("uuid", "context", "task_id", "block", "time_code",
                   "mnl_choice", "mmnl_choice")
_BLOCK_CONTEXT = {block: context for context, blocks in CONTEXT_BLOCKS.items()
                  for block in blocks}


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    respondents: int
    sha256: str
    context_counts: Mapping[str, int]
    scenario_counts: Mapping[str, int]
    min_tasks_per_respondent: int
    max_tasks_per_respondent: int
    strict_complete: bool
    block_counts: Mapping[int, int]
    time_code_counts: Mapping[str, int]
    choice_counts: Mapping[str, Mapping[str, int]]
    profiles_without_responses: int
    ignored_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DataBundle:
    profiles: Mapping[str, Profile]
    tasks: Mapping[str, Task]
    responses: Mapping[str, Mapping[tuple[str, str], int]]
    validation: ValidationReport | None = None


def _columns(fieldnames: Sequence[str] | None, required: Sequence[str]) -> None:
    if fieldnames is None:
        raise ValueError("CSV is empty or has no header")
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("CSV has duplicate column names")
    missing = set(required) - set(fieldnames)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")


def _binary(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid binary value for {label}") from exc
    if result not in (0.0, 1.0):
        raise ValueError(f"Invalid binary value for {label}")
    return result


def _integer(value: object, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}") from exc
    if not np.isfinite(number) or int(number) != number:
        raise ValueError(f"Invalid integer for {label}")
    return int(number)


def load_profiles(path: str | Path, uuids: Iterable[str] | None = None) -> dict[str, Profile]:
    """Validate every profile, retaining only the requested UUIDs if supplied."""
    wanted = None if uuids is None else set(uuids)
    profiles: dict[str, Profile] = {}
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _columns(reader.fieldnames, ("uuid", *PERSON_FIELDS))
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in ("uuid", *PERSON_FIELDS)):
                raise ValueError(f"Malformed profile row {line}")
            uuid = row["uuid"].strip()
            if not uuid or uuid in seen:
                raise ValueError(f"Empty or duplicate profile UUID at row {line}")
            seen.add(uuid)
            features = tuple(_binary(row[key], key) for key in PERSON_FIELDS)
            if sum(features[1:4]) > 1 or sum(features[4:]) > 1:
                raise ValueError(f"Mutually exclusive profile categories at row {line}")
            if wanted is None or uuid in wanted:
                profiles[uuid] = Profile(uuid, features)
    if not seen:
        raise ValueError("Profile CSV has no data rows")
    if wanted is not None and wanted - seen:
        raise ValueError("Requested UUIDs are absent from profiles")
    return profiles


def _task_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Tasks JSON must contain a list or a task mapping")
    for key in ("tasks", "sp_tasks"):
        if key in payload:
            return _task_records(payload[key])
    if payload and all(isinstance(v, dict) and "alternatives" in v for v in payload.values()):
        return [dict(value, task_id=value.get("task_id", key))
                for key, value in payload.items()]
    raise ValueError("Tasks JSON contains no recognized task records")


def load_tasks(path: str | Path, *, strict_complete: bool = True) -> dict[str, Task]:
    """Read the actual SP JSON (which derives context from block/time)."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        records = _task_records(json.load(stream))
    tasks: dict[str, Task] = {}
    block_questions: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each task must be an object")
        try:
            task_id = str(record["task_id"])
            block = _integer(record["block"], "block")
            cset = _integer(record["cset"], "cset")
            time_code = str(record["time_code"])
            context = _BLOCK_CONTEXT[block]
            alternatives = record["alternatives"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete task or unknown block") from exc
        if not task_id or task_id in tasks or (block, cset) in block_questions:
            raise ValueError("Duplicate task ID or block/question pair")
        if task_id != f"B{block}_Q{cset}" or not 1 <= cset <= 9:
            raise ValueError("Task ID does not match its block/question")
        if time_code != CONTEXT_TIMES[context] or record.get("context", context) != context:
            raise ValueError("Task context/block/time mismatch")
        if isinstance(alternatives, dict):
            if set(alternatives) != set(ALTERNATIVES):
                raise ValueError("Task alternative mapping must contain A/B/C/NONE")
            alternatives = [alternatives[label] for label in ALTERNATIVES]
        if not isinstance(alternatives, list) or len(alternatives) != 4:
            raise ValueError("Task must have ordered A/B/C/NONE alternatives")
        matrix = []
        for index, alternative in enumerate(alternatives):
            if not isinstance(alternative, dict):
                raise ValueError("Each alternative must map named attributes")
            if index < 3 and any(key not in alternative for key in ATTRIBUTES):
                raise ValueError("Airport alternative has missing attributes")
            values = tuple(_binary(alternative.get(key, 0), key) for key in ATTRIBUTES)
            if index == 3 and any(values):
                raise ValueError("NONE must have zero service attributes")
            if any(sum(values[start:start + 2]) > 1 for start in range(0, 12, 2)):
                raise ValueError("Mutually exclusive attribute levels are both active")
            matrix.append(values)
        tasks[task_id] = Task(task_id, context, block, cset, time_code, tuple(matrix))
        block_questions.add((block, cset))
    if not tasks:
        raise ValueError("No SP tasks found")
    if strict_complete and block_questions != {(b, q) for b in range(1, 7) for q in range(1, 10)}:
        raise ValueError("Expected exactly 54 tasks: six blocks of nine")
    return tasks


def _response_row(row: Mapping, tasks: Mapping[str, Task], line: int) -> tuple[str, Task, tuple[int, int]]:
    if None in row or any(row.get(key) is None for key in RESPONSE_FIELDS):
        raise ValueError(f"Malformed response row {line}")
    uuid, task_id = row["uuid"].strip(), row["task_id"].strip()
    if not uuid or task_id not in tasks:
        raise ValueError(f"Empty UUID or unknown task at response row {line}")
    task = tasks[task_id]
    if (row["context"] != task.context or
            _integer(row["block"], "block") != task.block or
            row["time_code"] != task.time_code):
        raise ValueError(f"Response context/block/time mismatch at row {line}")
    try:
        choices = tuple(ALTERNATIVES.index(row[f"{scenario}_choice"].strip())
                        for scenario in SCENARIOS)
    except ValueError as exc:
        raise ValueError(f"Invalid A/B/C/NONE choice at response row {line}") from exc
    return uuid, task, choices


def validate_response_csv(path: str | Path, tasks: Mapping[str, Task],
                          profile_ids: Iterable[str], *,
                          strict_complete: bool = True) -> ValidationReport:
    """Stream the full panel and validate UUID-contiguous groups before filtering.

    SHA-256 covers the exact source bytes.  Unknown profiles, duplicate task
    rows, and noncontiguous repeat UUIDs fail even if a later load requests only
    a small subset of people.  A partial panel is accepted only with the explicit
    ``strict_complete=False`` option.
    """
    profile_ids = set(profile_ids)
    digest = hashlib.sha256()
    rows = people = 0
    context_counts = {context: 0 for context in CONTEXT_BLOCKS}
    block_counts = {block: 0 for block in _BLOCK_CONTEXT}
    time_counts = {time_code: 0 for time_code in CONTEXT_TIMES.values()}
    choice_counts = {scenario: dict.fromkeys(ALTERNATIVES, 0) for scenario in SCENARIOS}
    min_tasks, max_tasks = len(tasks) + 1, 0
    current: str | None = None
    current_tasks: set[str] = set()
    expected = set(tasks)
    with tempfile.TemporaryDirectory(prefix="survey-validation-") as directory:
        database = sqlite3.connect(str(Path(directory) / "seen.sqlite"))
        database.execute("CREATE TABLE seen (uuid TEXT PRIMARY KEY) WITHOUT ROWID")

        def finish_group() -> None:
            nonlocal people, min_tasks, max_tasks
            if current is None:
                return
            if strict_complete and current_tasks != expected:
                raise ValueError("Incomplete respondent panel: expected all SP tasks once")
            try:
                database.execute("INSERT INTO seen VALUES (?)", (current,))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Response CSV must keep each UUID in one contiguous group") from exc
            people += 1
            min_tasks = min(min_tasks, len(current_tasks))
            max_tasks = max(max_tasks, len(current_tasks))

        try:
            with Path(path).open("rb") as source:
                def decoded_lines():
                    for raw in source:
                        digest.update(raw)
                        yield raw.decode("utf-8-sig")
                reader = csv.DictReader(decoded_lines())
                _columns(reader.fieldnames, RESPONSE_FIELDS)
                extras = tuple(key for key in reader.fieldnames if key not in RESPONSE_FIELDS)
                for line, row in enumerate(reader, 2):
                    uuid, task, choices = _response_row(row, tasks, line)
                    if uuid not in profile_ids:
                        raise ValueError(f"Response UUID is absent from profiles at row {line}")
                    if uuid != current:
                        finish_group()
                        current = uuid
                        current_tasks = set()
                    if task.task_id in current_tasks:
                        raise ValueError(f"Duplicate respondent/task at response row {line}")
                    current_tasks.add(task.task_id)
                    context_counts[task.context] += 1
                    block_counts[task.block] += 1
                    time_counts[task.time_code] += 1
                    for scenario, choice in zip(SCENARIOS, choices):
                        choice_counts[scenario][ALTERNATIVES[choice]] += 1
                    rows += 1
                finish_group()
        finally:
            database.close()
    if not rows:
        raise ValueError("Response CSV has no data rows")
    return ValidationReport(rows, people, digest.hexdigest(), context_counts,
                            {scenario: rows for scenario in SCENARIOS},
                            min_tasks, max_tasks, strict_complete,
                            block_counts, time_counts, choice_counts,
                            len(profile_ids) - people, extras)


def load_bundle(profiles_path: str | Path, tasks_path: str | Path,
                responses_path: str | Path, *, uuids: Iterable[str] | None = None,
                validate: bool = True, strict_complete: bool = True,
                max_retained_rows: int = 2_000_000) -> DataBundle:
    """Load a selected cohort after validation; never retain hidden parameters.

    All response rows are checked before a cohort filter is applied.  For large
    panels, pass explicit UUIDs; the default retention limit prevents accidental
    loading of a 21-million-row panel into Python dictionaries.  Profiles in the
    returned bundle are restricted to people with retained response records.
    """
    wanted = None if uuids is None else set(uuids)
    profiles = load_profiles(profiles_path)
    tasks = load_tasks(tasks_path, strict_complete=strict_complete)
    report = (validate_response_csv(responses_path, tasks, profiles,
                                   strict_complete=strict_complete)
              if validate else None)
    if wanted is not None and wanted - profiles.keys():
        raise ValueError("Requested UUIDs are absent from profiles")
    answers: dict[str, dict[tuple[str, str], int]] = {scenario: {} for scenario in SCENARIOS}
    retained_people: set[str] = set()
    retained_rows = 0
    with Path(responses_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _columns(reader.fieldnames, RESPONSE_FIELDS)
        for line, row in enumerate(reader, 2):
            uuid, task, choices = _response_row(row, tasks, line)
            if uuid not in profiles:
                raise ValueError(f"Response UUID is absent from profiles at row {line}")
            if wanted is not None and uuid not in wanted:
                continue
            key = (uuid, task.task_id)
            if key in answers["mnl"]:
                raise ValueError("Duplicate respondent/task in retained cohort")
            retained_rows += 1
            if retained_rows > max_retained_rows:
                raise ValueError("Response retention limit exceeded; request a UUID cohort")
            for scenario, choice in zip(SCENARIOS, choices):
                answers[scenario][key] = choice
            retained_people.add(uuid)
    if not retained_people or (wanted is not None and wanted - retained_people):
        raise ValueError("Requested cohort has missing response records")
    return DataBundle(MappingProxyType({uuid: profiles[uuid] for uuid in sorted(retained_people)}),
                      MappingProxyType(tasks),
                      MappingProxyType({key: MappingProxyType(value) for key, value in answers.items()}),
                      report)


class ResponseWorld:
    """Runner/evaluator-only response store with one immutable scenario snapshot.

    A policy receives neither this object nor its ``DataBundle``.  The allowed
    task IDs supplied by the runner are the episode's acquisition pool, so a
    held-out answer cannot be revealed through the sequential execution path.
    """
    def __init__(self, bundle: DataBundle, scenario: str):
        scenario = scenario.lower()
        if scenario not in SCENARIOS or scenario not in bundle.responses:
            raise ValueError("Unknown response scenario")
        self.scenario = scenario
        self._answers = MappingProxyType(dict(bundle.responses[scenario]))

    def reveal(self, uuid: str, task_id: str, allowed_task_ids: Iterable[str]) -> int:
        if task_id not in allowed_task_ids:
            raise ValueError("Attempt to reveal a held-out or unavailable task")
        return self.answer(uuid, task_id)

    def answer(self, uuid: str, task_id: str) -> int:
        """Evaluator-only lookup; never passed to a learner or policy."""
        try:
            return self._answers[(uuid, task_id)]
        except KeyError as exc:
            raise ValueError("No response for this UUID/task") from exc
