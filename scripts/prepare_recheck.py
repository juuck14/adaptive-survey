#!/usr/bin/env python3
"""Keep an existing train/validation panel and generate an unseen test cohort.

All inputs, generated responses, manifests, and fitted-generator files are
private runtime artifacts. This helper contains no respondent identifiers or
model parameters. Use a new output directory, normally under ``outputs/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

from adaptive_survey import __version__
from adaptive_survey.data import (RESPONSE_FIELDS, load_profiles, load_tasks,
                                  validate_response_csv)
from adaptive_survey.generator import generate_responses
from adaptive_survey.protocol import ProtocolConfig, stable_hash
from adaptive_survey.schema import PERSON_FIELDS


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def choose_fresh_uuids(profile_ids: Iterable[str], excluded: set[str],
                       test_size: int, seed: int) -> list[str]:
    if type(test_size) is not int or test_size < 1:
        raise ValueError("test_size must be positive")
    candidates = sorted((uid for uid in profile_ids if uid not in excluded),
                        key=lambda uid: (stable_hash(seed, "recheck-fresh-test", uid), uid))
    if len(candidates) < test_size:
        raise ValueError("Insufficient profiles after excluding the original respondent cohort")
    return candidates[:test_size]


def prepare_recheck(source_data: str | Path, original_split: str | Path,
                    handoff: str | Path, output_dir: str | Path, *,
                    test_size: int = 300, seed: int = 2026) -> dict:
    """Freeze preserved labels and a new respondent sample without overwriting.

    Selection uses UUIDs only. Every original split UUID and every UUID with a
    source response is excluded from the fresh test sample. The original test
    rows are omitted from the combined panel. Neither old train/validation
    labels nor old split order is regenerated or changed.
    """
    source_data, original_split, output_dir = map(Path, (source_data, original_split, output_dir))
    if output_dir.exists():
        raise FileExistsError("Output directory exists; choose a new path")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    source_paths = {name: source_data / name for name in ("profiles.csv", "tasks.json", "responses.csv")}
    source_fingerprints = {name: file_hash(path) for name, path in source_paths.items()}
    old = json.loads(original_split.read_text(encoding="utf-8"))
    if old.get("schema_version") != 1 or old.get("fingerprints") != source_fingerprints:
        raise ValueError("Source files do not match the original split manifest")
    groups = old.get("uuids")
    if not isinstance(groups, dict) or set(groups) != {"train", "validation", "test"}:
        raise ValueError("Original split needs train, validation and test UUID lists")
    if any(not isinstance(values, list) or not values for values in groups.values()):
        raise ValueError("Original split groups must be nonempty UUID lists")
    old_ids = [uid for values in groups.values() for uid in values]
    if any(not isinstance(uid, str) or not uid for uid in old_ids) or len(set(old_ids)) != len(old_ids):
        raise ValueError("Original split UUIDs must be unique nonempty strings")
    protocol = dict(old["protocol"])
    ProtocolConfig(**dict(protocol, budgets=tuple(protocol["budgets"])))
    profiles, tasks = load_profiles(source_paths["profiles.csv"]), load_tasks(source_paths["tasks.json"])
    source_report = validate_response_csv(source_paths["responses.csv"], tasks, profiles)
    retained_ids = set(groups["train"]) | set(groups["validation"])
    if set(old_ids) - profiles.keys():
        raise ValueError("An original split profile is missing from source data")
    retained_rows, source_respondents = [], set()
    with source_paths["responses.csv"].open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            uid = row["uuid"].strip()
            source_respondents.add(uid)
            if uid in retained_ids:
                retained_rows.append({name: uid if name == "uuid" else row[name] for name in RESPONSE_FIELDS})
    if set(old_ids) - source_respondents:
        raise ValueError("An original split UUID is missing response records")
    if len(retained_rows) != len(retained_ids) * len(tasks):
        raise ValueError("Retained training/validation panels are incomplete")
    excluded = set(old_ids) | source_respondents
    fresh_ids = choose_fresh_uuids(profiles, excluded, test_size, seed)
    all_ids = retained_ids | set(fresh_ids)
    selected_profiles = {uid: profiles[uid] for uid in sorted(all_ids)}

    # Every generated file is inside a new directory. Failed runs remain
    # inspectable and are never silently resumed over a partial artifact.
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = output_dir / "data"
    data_dir.mkdir()
    combined_profiles = data_dir / "profiles.csv"
    with combined_profiles.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("uuid", *PERSON_FIELDS))
        for uid, profile in selected_profiles.items():
            writer.writerow((uid, *profile.features))
    combined_tasks = data_dir / "tasks.json"
    shutil.copyfile(source_paths["tasks.json"], combined_tasks)
    fresh_responses = output_dir / "fresh-test-responses.csv"
    generation = generate_responses(handoff, selected_profiles, tasks, fresh_responses,
                                    uuids=fresh_ids, seed=seed)
    fresh_report = validate_response_csv(fresh_responses, tasks, fresh_ids)
    if fresh_report.respondents != test_size or fresh_report.rows != test_size * len(tasks):
        raise ValueError("Fresh generator output has an unexpected panel size")
    combined_responses = data_dir / "responses.csv"
    with combined_responses.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESPONSE_FIELDS)
        writer.writeheader()
        writer.writerows(retained_rows)
        with fresh_responses.open(encoding="utf-8", newline="") as generated:
            writer.writerows(csv.DictReader(generated))
    combined_report = validate_response_csv(combined_responses, tasks, selected_profiles)
    expected_people = len(retained_ids) + test_size
    if (combined_report.respondents != expected_people or
            combined_report.rows != expected_people * len(tasks) or
            combined_report.profiles_without_responses != 0):
        raise ValueError("Combined recheck dataset is incomplete")
    fingerprints = {"profiles.csv": file_hash(combined_profiles),
                    "tasks.json": file_hash(combined_tasks),
                    "responses.csv": combined_report.sha256}
    split = {"schema_version": 1, "version": __version__,
             "uuids": {"train": groups["train"], "validation": groups["validation"], "test": fresh_ids},
             "protocol": protocol, "fingerprints": fingerprints,
             "validation": combined_report.to_dict()}
    write_json(output_dir / "split.json", split)
    provenance = {
        "schema_version": 1, "version": __version__,
        "purpose": "Preserved training/validation responses and a fresh unseen respondent test cohort",
        "original_split_sha256": file_hash(original_split),
        "source_fingerprints": source_fingerprints,
        "source_validation": source_report.to_dict(),
        "preserved": {"train": len(groups["train"]), "validation": len(groups["validation"]),
                      "response_rows": len(retained_rows),
                      "ordered_split_uuid_sha256": object_hash({"train": groups["train"], "validation": groups["validation"]}),
                      "response_rows_sha256": object_hash(retained_rows)},
        "excluded_original": {"split_respondents": len(old_ids), "response_respondents": len(source_respondents),
                              "old_test_respondents": len(groups["test"]),
                              "all_excluded": len(excluded), "uuid_set_sha256": object_hash(sorted(excluded))},
        "fresh_test": {"seed": seed, "respondents": len(fresh_ids),
                       "selection": "stable_hash(seed, recheck-fresh-test, uuid); exclude all original split/response UUIDs",
                       "ordered_uuid_sha256": object_hash(fresh_ids), "generator": generation,
                       "validation": fresh_report.to_dict()},
        "combined": {"respondents": expected_people, "fingerprints": fingerprints,
                     "validation": combined_report.to_dict()},
        "scope": "Old labels are retained from the original CSV; the new test panel uses the documented generator RNG contract.",
    }
    write_json(output_dir / "provenance.json", provenance)
    return {"data": str(data_dir), "split": str(output_dir / "split.json"),
            "provenance": str(output_dir / "provenance.json"),
            "counts": {name: len(ids) for name, ids in split["uuids"].items()},
            "rows": combined_report.rows, "fingerprints": fingerprints}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--original-split", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        result = prepare_recheck(args.source_data, args.original_split, args.handoff,
                                 args.output_dir, test_size=args.test_size, seed=args.seed)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
