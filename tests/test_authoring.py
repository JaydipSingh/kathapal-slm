"""Tests for the dataset authoring helpers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.data.authoring import (  # noqa: E402
    add_from_dict,
    build_record,
    existing_ids,
    merge_files,
    next_id,
)
from kathapal.data.loader import load_records  # noqa: E402

STARTER = ROOT / "data" / "samples" / "starter_dataset.jsonl"


def _good_payload(**overrides):
    base = dict(
        topic="animals",
        instruction="Tell a short story about a cat.",
        story_title="The Curious Cat",
        story=" ".join(["A"] * 40),
        question="What did the cat do?",
        answer="The cat explored the garden.",
        learning_goal="curiosity",
        age_min=3,
        age_max=5,
    )
    base.update(overrides)
    return base


def test_next_id_increments():
    assert next_id(["kp_0001", "kp_0012", "kp_0007"]) == "kp_0013"
    assert next_id([]) == "kp_0001"


def test_build_record_valid():
    res = build_record(rid="kp_9001", **_good_payload())
    assert res.ok
    assert res.record.id == "kp_9001"


def test_build_record_rejects_short_story():
    res = build_record(rid="kp_9002", **_good_payload(story="too short indeed"))
    assert not res.ok
    assert any("story" in e for e in res.errors)


def test_build_record_rejects_unsafe_content():
    res = build_record(
        rid="kp_9003",
        **_good_payload(story=" ".join(["A"] * 38) + " with a gun"),
    )
    assert not res.ok
    assert any("safety" in e for e in res.errors)


def test_build_record_rejects_bad_topic():
    res = build_record(rid="kp_9004", **_good_payload(topic="politics"))
    assert not res.ok


def test_add_from_dict_appends_and_autoassigns(tmp_path):
    ds = tmp_path / "ds.jsonl"
    res = add_from_dict(ds, _good_payload())
    assert res.ok
    assert res.record.id == "kp_0001"
    recs, errs = load_records(ds)
    assert len(recs) == 1 and not errs
    # second add increments the id
    res2 = add_from_dict(ds, _good_payload(topic="nature", story_title="Trees"))
    assert res2.ok and res2.record.id == "kp_0002"


def test_add_from_dict_rejects_invalid_without_writing(tmp_path):
    ds = tmp_path / "ds.jsonl"
    res = add_from_dict(ds, _good_payload(story="short"))
    assert not res.ok
    assert not ds.exists() or load_records(ds)[0] == []


def test_existing_ids_reads_starter():
    ids = existing_ids(STARTER)
    assert "kp_0001" in ids and len(ids) >= 10


def test_merge_dedups_by_id(tmp_path):
    out = tmp_path / "merged.jsonl"
    summary = merge_files([STARTER, STARTER], out)  # same file twice
    # all ids are duplicates the second time -> written == unique count
    recs, _ = load_records(out)
    ids = [r.id for r in recs]
    assert len(ids) == len(set(ids))  # no duplicates in output
    assert summary["skipped_duplicate_ids"]
