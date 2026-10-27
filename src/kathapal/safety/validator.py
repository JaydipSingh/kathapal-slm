"""
Request and response safety validation.

A single entry point that composes the layered checks: request validation
(allowlisted topic, age bounds, language), and response validation (schema is
guaranteed by the StoryResponse type, plus rule-based content checks). Used by
the generation layer and the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass

from kathapal.data.schema import ALLOWED_LANGUAGES, ALLOWED_TOPICS, MAX_AGE, MIN_AGE, StoryResponse
from kathapal.safety.rules import check_text


@dataclass
class ValidationOutcome:
    ok: bool
    reason: str | None = None


def validate_request(*, age: int, language: str, topic: str) -> ValidationOutcome:
    if not (MIN_AGE <= age <= MAX_AGE):
        return ValidationOutcome(False, f"age {age} outside {MIN_AGE}-{MAX_AGE}")
    if language not in ALLOWED_LANGUAGES:
        return ValidationOutcome(False, f"language '{language}' not allowed")
    if topic not in ALLOWED_TOPICS:
        return ValidationOutcome(False, f"topic '{topic}' not in allowlist")
    return ValidationOutcome(True)


def validate_response(resp: StoryResponse, active_checks: list[str] | None = None) -> ValidationOutcome:
    text = f"{resp.story_title}\n{resp.story}\n{resp.question}\n{resp.answer}"
    res = check_text(text, active_checks)
    if not res.passed:
        return ValidationOutcome(False, f"safety: {res.summary()}")
    return ValidationOutcome(True)
