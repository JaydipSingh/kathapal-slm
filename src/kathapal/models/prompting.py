"""
Prompt construction and output parsing.

``build_request_prompt`` turns a validated request (age, language, topic,
length, optional learning goal) into system+user turns.

``parse_story_response`` extracts the first JSON object from raw model text and
validates it into a ``StoryResponse``. Models often wrap JSON in prose or code
fences, so the parser locates the outermost balanced ``{...}`` block rather than
assuming the whole output is clean JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from kathapal.data.schema import (
    ALLOWED_LANGUAGES,
    ALLOWED_TOPICS,
    MAX_AGE,
    MIN_AGE,
    StoryResponse,
)

SYSTEM_PROMPT = (
    "You are KathaPal, a gentle storyteller for young children aged 3 to 5. "
    "You write short, simple, positive stories with easy words and kind themes. "
    "You never include anything frightening, violent, sad without comfort, or "
    "otherwise unsuitable for a preschool child. "
    "You always reply with ONLY a single JSON object matching this exact schema: "
    '{"story_title": string, "story": string, "question": string, '
    '"answer": string, "learning_goal": string, "language": string, '
    '"age_range": string, "topic": string}. Do not add any text outside the JSON.'
)


@dataclass
class StoryRequest:
    age: int
    language: str
    topic: str
    story_length: str = "short"
    learning_goal: str | None = None

    def validate(self) -> None:
        if not (MIN_AGE <= self.age <= MAX_AGE):
            raise ValueError(f"age {self.age} must be within {MIN_AGE}-{MAX_AGE}")
        if self.language not in ALLOWED_LANGUAGES:
            raise ValueError(f"language '{self.language}' not allowed")
        if self.topic not in ALLOWED_TOPICS:
            raise ValueError(f"topic '{self.topic}' not in allowed set")


def build_request_prompt(req: StoryRequest) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the request."""
    req.validate()
    goal = req.learning_goal or "a gentle, positive lesson"
    user = (
        f"Write a {req.story_length} story in {req.language} for a child aged "
        f"{req.age}. Topic: {req.topic}. Learning goal: {goal}. "
        f"Reply with only the JSON object."
    )
    return SYSTEM_PROMPT, user


def build_training_text(instruction: str, response_json: str) -> tuple[str, str]:
    """
    Build (prompt_text, full_text) for one SFT example.

    ``prompt_text`` is the system+user portion the model should NOT be trained to
    reproduce; ``full_text`` additionally appends the target JSON response. The
    trainer tokenizes both and masks the prompt-length prefix in the labels so
    loss is computed only on the response tokens.
    """
    user = (
        f"{instruction} Reply with only the JSON object."
    )
    # A plain, template-agnostic format; the trainer may re-render via the
    # model's chat template, but this keeps label-masking lengths well-defined.
    prompt_text = f"System: {SYSTEM_PROMPT}\nUser: {user}\nAssistant: "
    full_text = prompt_text + response_json
    return prompt_text, full_text


def _extract_json_block(text: str) -> str | None:
    """Return the first balanced {...} block in text, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


@dataclass
class ParseResult:
    ok: bool
    response: StoryResponse | None = None
    error: str | None = None


def parse_story_response(raw: str) -> ParseResult:
    """Extract and validate a StoryResponse from raw model output."""
    block = _extract_json_block(raw)
    if block is None:
        return ParseResult(ok=False, error="no JSON object found in output")
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        return ParseResult(ok=False, error=f"invalid JSON: {exc}")
    try:
        resp = StoryResponse.model_validate(obj)
    except ValidationError as exc:
        return ParseResult(ok=False, error=f"schema validation failed: {exc.errors()!r}")
    return ParseResult(ok=True, response=resp)
