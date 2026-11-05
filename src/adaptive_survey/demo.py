"""Generate a small entirely artificial panel for execution and integration tests.

No human observations or actual Nemotron profiles are bundled.  The artificial
MMNL draw is persistent per person across all fifty-four questions and contexts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .protocol import stable_hash
from .schema import ALTERNATIVES, ATTRIBUTES, CONTEXT_BLOCKS, CONTEXT_TIMES, PERSON_FIELDS


def _rng(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(int(stable_hash(seed, *parts)[:16], 16))


def generate_demo(directory: str | Path, n_respondents: int = 300,
                  seed: int = 2026) -> dict[str, Path]:
    if not isinstance(n_respondents, int) or n_respondents < 1:
        raise ValueError("Demo needs at least one respondent")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"profiles": directory / "profiles.csv",
             "tasks": directory / "tasks.json",
             "responses": directory / "responses.csv"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Demo target contains survey files; choose a new output directory")
    task_rng = _rng(seed, "task-design")
    tasks = []
    for context, blocks in CONTEXT_BLOCKS.items():
        for block in blocks:
            for cset in range(1, 10):
                alternatives = []
                for _ in range(3):
                    vector = np.zeros(13, dtype=int)
                    for start in range(0, 12, 2):
                        level = int(task_rng.integers(3))
                        if level:
                            vector[start + level - 1] = 1
                    vector[12] = int(task_rng.integers(2))
                    alternatives.append(dict(zip(ATTRIBUTES, vector.tolist())))
                tasks.append({"task_id": f"B{block}_Q{cset}", "block": block,
                              "cset": cset, "time_code": CONTEXT_TIMES[context],
                              "alternatives": [*alternatives, {}]})
    paths["tasks"].write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    base = np.array([-.1, .45, .1, .65, .35, 1.05, .15, .8, .25, .7, -.2, .1, .4])
    asc = np.array([.65, .8, .35, 0.])
    random_axes = np.array([3, 5, 9, 11, 12])
    with paths["profiles"].open("w", encoding="utf-8", newline="") as profile_stream, \
            paths["responses"].open("w", encoding="utf-8", newline="") as response_stream:
        profiles = csv.writer(profile_stream)
        profiles.writerow(("uuid", *PERSON_FIELDS))
        responses = csv.writer(response_stream)
        responses.writerow(("uuid", "context", "task_id", "block", "time_code",
                            "mnl_choice", "mmnl_choice"))
        for person in range(n_respondents):
            uuid = f"demo_{person:07d}"
            profile_rng = _rng(seed, "profile", uuid)
            features = np.zeros(9, dtype=int)
            features[:2] = profile_rng.integers(0, 2, size=2)
            education = int(profile_rng.integers(0, 5))
            if education:
                features[3 + education] = 1
            profiles.writerow((uuid, *features.tolist()))
            beta = base.copy()
            beta[2:4] += .35 * features[0] + .2 * features[1]
            beta[10:12] += .2 * features[4:8].sum()
            person_beta = beta.copy()
            person_beta[random_axes] += _rng(seed, "persistent-taste", uuid).normal(0, .85, 5)
            for task in tasks:
                context = next(key for key, blocks in CONTEXT_BLOCKS.items() if task["block"] in blocks)
                matrix = np.array([[alt.get(key, 0) for key in ATTRIBUTES]
                                   for alt in task["alternatives"]], dtype=float)
                context_beta = np.zeros(13)
                context_beta[8:10] = {"transfer": -.15, "layover": .15, "stopover": .45}[context]
                choices = []
                for scenario, coefficients in (("mnl", beta), ("mmnl", person_beta)):
                    utility = matrix @ (coefficients + context_beta) + asc
                    probability = np.exp(utility - utility.max())
                    probability /= probability.sum()
                    choice = _rng(seed, "response", uuid, task["task_id"], scenario).choice(4, p=probability)
                    choices.append(ALTERNATIVES[int(choice)])
                responses.writerow((uuid, context, task["task_id"], task["block"], task["time_code"], *choices))
    return paths
