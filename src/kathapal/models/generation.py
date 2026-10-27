"""
Generation orchestration: request -> prompt -> model -> parse -> validate ->
safety check -> (retry) -> safe fallback.

This is the function the application/CLI calls. It is model-agnostic (takes a
``ModelAdapter``), so it is fully testable offline with a stub adapter. On any
failure (bad parse, schema failure, safety failure, or exhausted retries) it
returns a curated fallback story rather than raw/unsafe output.
"""
from __future__ import annotations

from dataclasses import dataclass

from kathapal.data.schema import StoryResponse
from kathapal.logging_config import get_logger
from kathapal.models.base import GenerationParams, ModelAdapter
from kathapal.models.prompting import (
    StoryRequest,
    build_request_prompt,
    parse_story_response,
)
from kathapal.safety.fallback import get_fallback
from kathapal.safety.rules import check_text

log = get_logger()


@dataclass
class GenerationOutcome:
    response: StoryResponse
    used_fallback: bool
    attempts: int
    reason: str | None = None  # why fallback was used, if it was


def _safety_text(resp: StoryResponse) -> str:
    return f"{resp.story_title}\n{resp.story}\n{resp.question}\n{resp.answer}"


def generate_story(
    adapter: ModelAdapter,
    req: StoryRequest,
    params: GenerationParams,
    *,
    max_retries: int = 2,
    active_safety_checks: list[str] | None = None,
) -> GenerationOutcome:
    """Generate a validated, safe story or return a curated fallback."""
    req.validate()
    system, user = build_request_prompt(req)
    prompt = adapter.apply_chat_template(system, user)

    last_reason: str | None = None
    attempts = 0
    for attempt in range(1, max_retries + 2):  # initial try + retries
        attempts = attempt
        try:
            raw = adapter.generate(prompt, params)
        except Exception as exc:  # noqa: BLE001 - must not crash the toy
            last_reason = f"generation error: {exc}"
            log.warning("generation attempt %d failed: %s", attempt, last_reason)
            continue

        parsed = parse_story_response(raw)
        if not parsed.ok or parsed.response is None:
            last_reason = f"parse/schema failure: {parsed.error}"
            log.warning("attempt %d: %s", attempt, last_reason)
            continue

        safety = check_text(_safety_text(parsed.response), active_safety_checks)
        if not safety.passed:
            last_reason = f"safety failure: {safety.summary()}"
            log.warning("attempt %d: %s", attempt, last_reason)
            continue

        # Success: parsed, schema-valid, and safe.
        return GenerationOutcome(response=parsed.response, used_fallback=False, attempts=attempt)

    # All attempts exhausted -> safe fallback.
    log.info("using fallback for topic '%s' after %d attempts (%s)", req.topic, attempts, last_reason)
    return GenerationOutcome(
        response=get_fallback(req.topic),
        used_fallback=True,
        attempts=attempts,
        reason=last_reason,
    )
