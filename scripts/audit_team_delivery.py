#!/usr/bin/env python3
"""Verify all delivered runs against original cohorts, bank answers and templates."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sqlite3

from adaptive_survey.team_handoff import (GENERATORS, file_hash, read_csv, read_prepared, prepared_samples_dir,
                                         validate_with_team_tool, write_json)
from adaptive_survey.schema import ALTERNATIVES


def audit(prepared, directory):
    root, directory = Path(prepared), Path(directory)
    metadata, design, profiles, tasks = read_prepared(root)
    learner_ids = set(metadata["train"] + metadata["validation"])
    summaries, trajectories = [], {}
    expected = set()
    with sqlite3.connect(root / "responses.sqlite") as db:
        answers = {(u, t): (a, b) for u, t, a, b in db.execute("SELECT * FROM responses")}
    for rep in range(1, 11):
        sample_file = prepared_samples_dir(root, metadata) / f"sampled_respondents_rep{rep:02d}.csv"
        if file_hash(sample_file) != metadata["sample_hashes"][sample_file.name]:
            raise ValueError("Source replicate changed")
        fields, samples = read_csv(sample_file)
        by_sample = {r["sample_id"]: r for r in samples}
        if len(samples) != 1500 or set(r["uuid"] for r in samples) & learner_ids:
            raise ValueError("Wrong cohort size or train/validation contamination")
        for generator in GENERATORS:
            for method in ("fixed", "non_graph"):
                run_id = f"rep{rep:02d}_{generator}_{method}"
                path = directory / f"responses_{run_id}.csv"
                expected.add(path.name)
                _, rows, _, _ = validate_with_team_tool(design, path, expected_people=1500,
                                                       expected_per_context=500)
                verification = json.loads((directory / f"{run_id}.verification.json").read_text())
                status = json.loads((directory / f"{run_id}.run.json").read_text())
                if (verification["pilot"] or not verification["valid"] or
                        verification["rows"] != 13500 or verification["response_sha256"] != file_hash(path)
                        or verification["prepared_sha256"] != file_hash(root / "prepared.json")
                        or verification["design_hashes"] != design.hashes
                        or status["run_status"] != "completed" or status["response_rows"] != 13500
                        or status["completed_respondents"] != 1500
                        or status["response_source"] != "existing_response_bank"
                        or status["response_bank_version"] != metadata["response_bank_sha256"]
                        or status["response_seed"] != ""):
                    raise ValueError("Run is incomplete, changed, a pilot or uses another response source")
                grouped = defaultdict(list)
                scores, other_block = [], 0
                for row in rows:
                    original = by_sample[row["sample_id"]]
                    if any(row[key] != original[key] for key in fields):
                        raise ValueError("A delivered profile/context/ID differs from its original sample")
                    choice = answers[row["uuid"], row["task_id"]][0 if generator == "mnl" else 1]
                    if row["chosen_alternative"] != ALTERNATIVES[choice]:
                        raise ValueError("An answer differs from the fixed response bank")
                    grouped[row["sample_id"]].append(row["task_id"])
                    if row["phase"] == "subsequent":
                        other_block += int(row["task_block"]) != int(row["base_block"])
                        if row["acquisition_score"]:
                            scores.append(float(row["acquisition_score"]))
                if set(grouped) != set(by_sample):
                    raise ValueError("Delivered cohort differs from the supplied replicate")
                trajectories[rep, generator, method] = grouped
                summaries.append({
                    "run_id": run_id, "respondents": len(grouped), "response_rows": len(rows),
                    "seconds": verification["seconds"], "sha256": verification["response_sha256"],
                    "cross_block_subsequent_fraction": other_block / 9000,
                    "mean_selected_weighted_js": sum(scores) / len(scores) if scores else None,
                })
    if {p.name for p in directory.glob("responses_*.csv")} != expected:
        raise ValueError("Unexpected or missing response CSVs")
    comparison = {}
    for generator in GENERATORS:
        changed = cross_block = total = 0
        for rep in range(1, 11):
            fixed = trajectories[rep, generator, "fixed"]
            adaptive = trajectories[rep, generator, "non_graph"]
            for sample_id in fixed:
                if fixed[sample_id][:3] != adaptive[sample_id][:3]:
                    raise ValueError("Initial questions differ across methods")
                changed += set(fixed[sample_id]) != set(adaptive[sample_id])
                total += 1
        comparison[generator] = {"compared_survey_instances": total,
                                 "different_question_set_from_fixed": changed,
                                 "different_question_set_fraction": changed / total}
    return {"protocol": metadata["protocol"], "valid": True, "runs": len(summaries),
            "respondent_survey_instances": sum(s["respondents"] for s in summaries),
            "response_rows": sum(s["response_rows"] for s in summaries),
            "train_respondents": len(metadata["train"]),
            "validation_respondents": len(metadata["validation"]),
            "reserved_evaluation_respondents": metadata["reserved_respondents"],
            "training_evaluation_overlap": 0, "comparison": comparison, "run_summaries": summaries,
            "interpretation": "Validated survey collections; coefficient recovery awaits the shared final estimator."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.prepared, args.delivery)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("valid", "runs", "response_rows", "training_evaluation_overlap")}))


if __name__ == "__main__":
    main()
