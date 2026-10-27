"""Tests for rule-based safety checks and fallback library."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.safety.fallback import available_fallback_topics, get_fallback  # noqa: E402
from kathapal.safety.rules import check_text  # noqa: E402


def test_safe_text_passes():
    res = check_text("The little rabbit shared a carrot with a kind friend in the garden.")
    assert res.passed


def test_violent_text_flagged():
    res = check_text("The man had a gun and started to shoot.")
    assert not res.passed
    assert any(f.category == "violence" for f in res.findings)


def test_personal_info_request_flagged():
    res = check_text("Hello child, what is your address and which school do you go to?")
    assert not res.passed
    assert any(f.category == "personal_information_request" for f in res.findings)


def test_word_boundary_avoids_false_positive():
    # 'begun' contains 'gun' but must not be flagged as violence.
    res = check_text("The story had begun on a sunny morning in the meadow.")
    assert res.passed


def test_active_checks_subset():
    # Only run the 'violence' category; a personal-info phrase should then pass.
    res = check_text("what is your address", active_checks=["violence"])
    assert res.passed


def test_fallback_is_valid_and_topic_aware():
    for topic in available_fallback_topics():
        fb = get_fallback(topic)
        assert fb.topic == topic
        assert fb.word_count() >= 20


def test_fallback_default_for_unknown_topic():
    fb = get_fallback("nonexistent-topic")
    assert fb.word_count() >= 20  # returns the default, still valid
