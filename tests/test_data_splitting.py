"""Tests for reproducible, group-aware splitting."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.data.loader import load_records  # noqa: E402
from kathapal.data.splitting import split_records  # noqa: E402

STARTER = ROOT / "data" / "samples" / "starter_dataset.jsonl"


def _load():
    records, _ = load_records(STARTER)
    return records


def test_split_is_reproducible():
    records = _load()
    a = split_records(records, train=0.8, val=0.1, test=0.1, seed=42)
    b = split_records(records, train=0.8, val=0.1, test=0.1, seed=42)
    assert [r.id for r in a.train] == [r.id for r in b.train]
    assert [r.id for r in a.test] == [r.id for r in b.test]


def test_split_covers_all_records_without_overlap():
    records = _load()
    s = split_records(records, train=0.8, val=0.1, test=0.1, seed=7)
    ids = {r.id for r in s.train} | {r.id for r in s.val} | {r.id for r in s.test}
    assert ids == {r.id for r in records}
    # no record appears in two splits
    total = len(s.train) + len(s.val) + len(s.test)
    assert total == len(records)


def test_split_rejects_bad_ratios():
    records = _load()
    import pytest

    with pytest.raises(ValueError):
        split_records(records, train=0.5, val=0.3, test=0.3, seed=1)


def test_group_aware_no_instruction_leak():
    """Records sharing an instruction must not split across train and test."""
    records = _load()
    # Duplicate one instruction into a paraphrase group by cloning with new id.
    clone = records[0].model_copy(update={"id": records[0].id + "_dup"})
    augmented = records + [clone]
    s = split_records(augmented, train=0.7, val=0.15, test=0.15, seed=3)
    key = records[0].dedup_key()
    in_train = any(r.dedup_key() == key for r in s.train)
    in_val = any(r.dedup_key() == key for r in s.val)
    in_test = any(r.dedup_key() == key for r in s.test)
    # the group may be in exactly one split only
    assert sum([in_train, in_val, in_test]) == 1
