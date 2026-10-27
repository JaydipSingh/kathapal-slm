"""
Reproducible, group-aware train/val/test splitting.

Records that share a normalized instruction (near-duplicates / paraphrases) are
grouped so that no group is split across train/val/test -- this prevents
train->test leakage that would inflate evaluation scores. Splitting is seeded
and therefore reproducible.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from kathapal.data.schema import DatasetRecord


@dataclass
class DataSplits:
    train: list[DatasetRecord]
    val: list[DatasetRecord]
    test: list[DatasetRecord]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def split_records(
    records: list[DatasetRecord],
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> DataSplits:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    # Group by normalized instruction so paraphrase clusters stay together.
    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    for r in records:
        groups[r.dedup_key()].append(r)

    group_keys = sorted(groups.keys())  # deterministic order before shuffle
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = int(round(train * n_groups))
    n_val = int(round(val * n_groups))
    # Give any rounding remainder to train; ensure test gets at least the rest.
    n_train = min(n_train, n_groups)
    n_val = min(n_val, n_groups - n_train)

    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train:n_train + n_val]
    test_keys = group_keys[n_train + n_val:]

    def collect(keys: list[str]) -> list[DatasetRecord]:
        out: list[DatasetRecord] = []
        for k in keys:
            out.extend(groups[k])
        return out

    return DataSplits(train=collect(train_keys), val=collect(val_keys), test=collect(test_keys))
