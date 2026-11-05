"""Answer-independent UUID splits and Context-specific survey episodes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

from .schema import CONTEXT_BLOCKS, Profile, Task


def stable_hash(seed: int, *parts: object) -> str:
    return hashlib.sha256(json.dumps([int(seed), *parts], ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ProtocolConfig:
    holdout_per_block: int = 2
    initial_per_block: int = 1
    budgets: tuple[int, ...] = (3, 5, 7, 9)
    seed: int = 2026

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in
               (self.holdout_per_block, self.initial_per_block, self.seed)):
            raise ValueError("Protocol counts and seed must be integers")
        if not 1 <= self.holdout_per_block < 9:
            raise ValueError("Holdout count must be between one and eight per block")
        if not 0 <= self.initial_per_block <= 9 - self.holdout_per_block:
            raise ValueError("Initial questions exceed the acquisition pool")
        if not self.budgets or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("Budgets must be nonempty and unique")
        available = 2 * (9 - self.holdout_per_block)
        initial = 2 * self.initial_per_block
        if any(type(b) is not int or b < 0 or b > available or (0 < b < initial)
               for b in self.budgets):
            raise ValueError("Budget must be zero or include all initial questions")


@dataclass(frozen=True)
class Episode:
    uuid: str
    context: str
    holdout: tuple[Task, ...]
    initial: tuple[Task, ...]
    candidates: tuple[Task, ...]
    budgets: tuple[int, ...]

    @property
    def acquisition_ids(self) -> frozenset[str]:
        return frozenset(task.task_id for task in self.candidates)


def split_uuids(uuids: Iterable[str], train_size: int, val_size: int,
                test_size: int, seed: int = 2026) -> dict[str, tuple[str, ...]]:
    """Rank UUIDs by a stable hash, then take exact disjoint cohort sizes."""
    ids = list(uuids)
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("UUID input must contain unique nonempty strings")
    sizes = (train_size, val_size, test_size)
    if any(type(size) is not int or size < 0 for size in sizes):
        raise ValueError("Split sizes must be nonnegative integers")
    if sum(sizes) > len(ids):
        raise ValueError("Insufficient respondents for the requested split sizes")
    ordered = sorted(ids, key=lambda uuid: (stable_hash(seed, "split", uuid), uuid))
    a, b, c = train_size, train_size + val_size, sum(sizes)
    return {"train": tuple(ordered[:a]), "validation": tuple(ordered[a:b]),
            "test": tuple(ordered[b:c])}


def make_episode(profile_or_uuid: Profile | str, context: str,
                 tasks: Mapping[str, Task] | Sequence[Task],
                 config: ProtocolConfig = ProtocolConfig(), *,
                 fixed_rank: Sequence[str] | None = None) -> Episode:
    """Select held-out tasks without inspecting any answer or model prediction.

    The default initial-task ranking uses task IDs and protocol seed only; it is
    fixed across people before removing their held-out tasks.  A predeclared
    training-only ranking may be supplied through ``fixed_rank``.
    ``candidates`` includes the initial questions, totaling fourteen by default.
    """
    uuid = profile_or_uuid.uuid if isinstance(profile_or_uuid, Profile) else profile_or_uuid
    if not isinstance(uuid, str) or not uuid or context not in CONTEXT_BLOCKS:
        raise ValueError("Episode needs a valid UUID and context")
    values = list(tasks.values()) if isinstance(tasks, Mapping) else list(tasks)
    pool = [task for task in values if task.context == context]
    if len(pool) != 18 or len({task.task_id for task in pool}) != 18:
        raise ValueError("A context must contain eighteen distinct tasks")
    blocks = CONTEXT_BLOCKS[context]
    if any(task.block not in blocks for task in pool):
        raise ValueError("Task block does not belong to the context")
    if any({task.cset for task in pool if task.block == block} != set(range(1, 10))
           for block in blocks):
        raise ValueError("Each context block must have questions one through nine")
    rank = None
    if fixed_rank is not None:
        if len(set(fixed_rank)) != len(fixed_rank):
            raise ValueError("Fixed task ranking has duplicate IDs")
        rank = {task_id: index for index, task_id in enumerate(fixed_rank)}
        if set(task.task_id for task in pool) - rank.keys():
            raise ValueError("Fixed ranking must cover the entire context pool")
    holdout: list[Task] = []
    initial: list[Task] = []
    for block in blocks:
        block_pool = [task for task in pool if task.block == block]
        held = sorted(block_pool,
                      key=lambda task: (stable_hash(config.seed, "holdout", uuid, context,
                                                    task.task_id), task.task_id))[:config.holdout_per_block]
        holdout.extend(held)
        held_ids = {task.task_id for task in held}
        remaining = [task for task in block_pool if task.task_id not in held_ids]
        remaining.sort(key=lambda task: (rank[task.task_id] if rank is not None else
                                         stable_hash(config.seed, "initial-rank", context, task.task_id),
                                         task.task_id))
        initial.extend(remaining[:config.initial_per_block])
    held_ids = {task.task_id for task in holdout}
    key = lambda task: (task.block, task.cset, task.task_id)
    return Episode(uuid, context, tuple(sorted(holdout, key=key)), tuple(initial),
                   tuple(sorted((task for task in pool if task.task_id not in held_ids), key=key)),
                   tuple(sorted(config.budgets)))
