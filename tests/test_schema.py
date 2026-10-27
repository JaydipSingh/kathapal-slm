"""Tests for the dataset / output schema."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.data.schema import DatasetRecord, StoryResponse  # noqa: E402


def _valid_response(**overrides):
    base = dict(
        story_title="The Kind Rabbit",
        story=" ".join(["A"] * 40),
        question="What happened?",
        answer="The rabbit shared.",
        learning_goal="sharing",
        language="en",
        age_range="3-5",
        topic="sharing",
    )
    base.update(overrides)
    return StoryResponse(**base)


def test_valid_story_response():
    r = _valid_response()
    assert r.word_count() == 40


def test_story_too_short_rejected():
    with pytest.raises(ValueError):
        _valid_response(story="only three words")


def test_disallowed_language_rejected():
    with pytest.raises(ValueError):
        _valid_response(language="fr")


def test_disallowed_topic_rejected():
    with pytest.raises(ValueError):
        _valid_response(topic="politics")


def test_bad_age_range_rejected():
    with pytest.raises(ValueError):
        _valid_response(age_range="2-9")


def test_record_metadata_must_match_response():
    with pytest.raises(ValueError):
        DatasetRecord(
            id="x1",
            language="en",
            age_min=3,
            age_max=5,
            topic="animals",            # mismatch vs response.topic below
            instruction="Tell a story.",
            response=_valid_response(topic="sharing"),
            source="original-author",
            license="CC-BY-4.0",
        )


def test_valid_record():
    rec = DatasetRecord(
        id="x1",
        language="en",
        age_min=3,
        age_max=5,
        topic="sharing",
        instruction="Tell a story about sharing.",
        response=_valid_response(),
        source="original-author",
        license="CC-BY-4.0",
    )
    assert rec.dedup_key() == "tell a story about sharing."
