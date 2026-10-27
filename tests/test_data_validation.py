"""Tests for dataset validation and the starter dataset's integrity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.data.loader import load_records  # noqa: E402
from kathapal.data.validation import validate_dataset, validate_records  # noqa: E402

STARTER = ROOT / "data" / "samples" / "starter_dataset.jsonl"


def test_starter_dataset_loads_cleanly():
    records, errors = load_records(STARTER)
    assert errors == [], f"starter dataset has load errors: {errors}"
    assert len(records) >= 10


def test_starter_dataset_passes_validation():
    report = validate_dataset(STARTER)
    assert report.ok, f"starter dataset failed validation: {report.to_dict()}"
    assert not report.duplicate_ids
    assert not report.safety_failures
    assert not report.missing_provenance_ids


def test_validation_detects_duplicate_ids():
    records, _ = load_records(STARTER)
    dup = records[:2] + [records[0]]  # repeat the first id
    report = validate_records(dup)
    assert records[0].id in report.duplicate_ids


def test_validation_detects_safety_failure():
    records, _ = load_records(STARTER)
    bad = records[0].model_copy(deep=True)
    # Inject a violent phrase into the story to trip the rule checker.
    bad.response.story = bad.response.story + " Then someone had a gun."
    report = validate_records([bad])
    assert report.safety_failures
