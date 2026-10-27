"""Tests for the generate -> parse -> validate -> fallback flow (offline, stub model)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.models.base import GenerationParams, StubModelAdapter  # noqa: E402
from kathapal.models.generation import generate_story  # noqa: E402
from kathapal.models.prompting import StoryRequest  # noqa: E402

PARAMS = GenerationParams(max_new_tokens=200, deterministic=True)


def _req(topic="animals"):
    return StoryRequest(age=4, language="en", topic=topic, learning_goal="kindness")


def test_successful_generation_no_fallback():
    adapter = StubModelAdapter()  # returns valid JSON
    outcome = generate_story(adapter, _req(), PARAMS, max_retries=1)
    assert not outcome.used_fallback
    assert outcome.response.story_title
    assert outcome.response.word_count() >= 20


def test_bad_output_triggers_fallback():
    adapter = StubModelAdapter(canned_json="I cannot produce JSON today.")
    outcome = generate_story(adapter, _req("kindness"), PARAMS, max_retries=2)
    assert outcome.used_fallback
    assert outcome.response.topic == "kindness"  # topic-specific fallback
    assert "parse" in (outcome.reason or "")


def test_unsafe_output_triggers_fallback():
    # Valid schema but contains a flagged violent word -> safety fails -> fallback.
    unsafe = (
        '{"story_title": "Scary Tale", '
        '"story": "A child found a gun in the woods and everyone was very afraid '
        'for a long time as the dark night went on and on without any comfort at '
        'all for anyone there.", '
        '"question": "What happened?", "answer": "Something scary.", '
        '"learning_goal": "none", "language": "en", "age_range": "3-5", '
        '"topic": "animals"}'
    )
    adapter = StubModelAdapter(canned_json=unsafe)
    outcome = generate_story(adapter, _req("animals"), PARAMS, max_retries=1)
    assert outcome.used_fallback
    assert "safety" in (outcome.reason or "")


def test_generation_error_triggers_fallback():
    class BrokenAdapter(StubModelAdapter):
        def generate(self, prompt, params):  # noqa: ARG002
            raise RuntimeError("model crashed")

    outcome = generate_story(BrokenAdapter(), _req("nature"), PARAMS, max_retries=1)
    assert outcome.used_fallback
    assert outcome.response.topic == "nature"
    assert "generation error" in (outcome.reason or "")
