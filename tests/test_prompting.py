"""Tests for prompt construction and output parsing."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.models.prompting import (  # noqa: E402
    StoryRequest,
    build_request_prompt,
    parse_story_response,
)

_VALID_JSON = (
    '{"story_title": "The Friendly Fox", '
    '"story": "A small fox woke up in the quiet woods and said hello to every '
    'animal it met that day, making many new friends before the evening came at '
    'last under the stars.", '
    '"question": "What did the fox do?", '
    '"answer": "It made new friends.", '
    '"learning_goal": "Making friends", '
    '"language": "en", "age_range": "3-5", "topic": "animals"}'
)


def test_build_prompt_valid():
    req = StoryRequest(age=4, language="en", topic="animals", learning_goal="kindness")
    system, user = build_request_prompt(req)
    assert "JSON" in system
    assert "aged 4" in user and "animals" in user


def test_build_prompt_rejects_bad_age():
    with pytest.raises(ValueError):
        build_request_prompt(StoryRequest(age=9, language="en", topic="animals"))


def test_build_prompt_rejects_bad_topic():
    with pytest.raises(ValueError):
        build_request_prompt(StoryRequest(age=4, language="en", topic="war"))


def test_parse_clean_json():
    res = parse_story_response(_VALID_JSON)
    assert res.ok
    assert res.response.story_title == "The Friendly Fox"


def test_parse_json_wrapped_in_prose():
    noisy = "Sure! Here is your story:\n```json\n" + _VALID_JSON + "\n```\nHope you like it!"
    res = parse_story_response(noisy)
    assert res.ok
    assert res.response.topic == "animals"


def test_parse_no_json_fails():
    res = parse_story_response("I'm sorry, I cannot do that.")
    assert not res.ok
    assert "no JSON" in res.error


def test_parse_schema_violation_fails():
    bad = _VALID_JSON.replace('"language": "en"', '"language": "fr"')
    res = parse_story_response(bad)
    assert not res.ok
    assert "schema" in res.error
