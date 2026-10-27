"""
Data and output schemas for KathaPal-SLM.

Two Pydantic models define the contract:

* ``StoryResponse`` — the structured content the model must produce (and that an
  application consumes). This is also the training *target* payload.
* ``DatasetRecord`` — one training/eval example: metadata + instruction request
  + the expected ``StoryResponse``.

Validation here is deliberately strict: preschool content must stay within tight
bounds (allowed languages, age range, bounded lengths) so that malformed or
out-of-scope data is rejected loudly rather than silently trained on.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

# Languages supported now. Architecture allows adding more (e.g. Hindi) later.
ALLOWED_LANGUAGES = {"en"}

# Allowlisted topics for the initial version (safety: constrains scope).
ALLOWED_TOPICS = {
    "animals",
    "nature",
    "friendship",
    "emotions",
    "curiosity",
    "space",
    "science",
    "kindness",
    "sharing",
    "family",
}

# Bounds appropriate for ages 3-5. Lengths are in words (approximate).
MIN_AGE = 3
MAX_AGE = 5
STORY_MIN_WORDS = 20
STORY_MAX_WORDS = 250
TITLE_MAX_CHARS = 80
QUESTION_MAX_CHARS = 160
ANSWER_MAX_CHARS = 160


class StoryLength(StrEnum):
    """Coarse story-length buckets exposed to the application/CLI."""

    short = "short"
    medium = "medium"


class ReviewStatus(StrEnum):
    """Human-review state of a dataset record (provenance/quality tracking)."""

    unreviewed = "unreviewed"
    reviewed = "reviewed"
    approved = "approved"


# ---------------------------------------------------------------------------
# Structured model output
# ---------------------------------------------------------------------------


class StoryResponse(BaseModel):
    """The validated structured response the model must generate."""

    model_config = {"extra": "forbid"}

    story_title: str = Field(..., min_length=1, max_length=TITLE_MAX_CHARS)
    story: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_CHARS)
    answer: str = Field(..., min_length=1, max_length=ANSWER_MAX_CHARS)
    learning_goal: str = Field(..., min_length=1, max_length=120)
    language: str
    age_range: str
    topic: str

    @field_validator("language")
    @classmethod
    def _lang_allowed(cls, v: str) -> str:
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"language '{v}' not in allowed set {sorted(ALLOWED_LANGUAGES)}")
        return v

    @field_validator("topic")
    @classmethod
    def _topic_allowed(cls, v: str) -> str:
        if v not in ALLOWED_TOPICS:
            raise ValueError(f"topic '{v}' not in allowed set {sorted(ALLOWED_TOPICS)}")
        return v

    @field_validator("age_range")
    @classmethod
    def _age_range_format(cls, v: str) -> str:
        # Expect "min-max" within [MIN_AGE, MAX_AGE].
        try:
            lo_s, hi_s = v.split("-")
            lo, hi = int(lo_s), int(hi_s)
        except Exception as exc:  # noqa: BLE001 - want a clear message
            raise ValueError(f"age_range '{v}' must look like '3-5'") from exc
        if not (MIN_AGE <= lo <= hi <= MAX_AGE):
            raise ValueError(f"age_range '{v}' must lie within {MIN_AGE}-{MAX_AGE}")
        return v

    @field_validator("story")
    @classmethod
    def _story_word_bounds(cls, v: str) -> str:
        n = len(v.split())
        if n < STORY_MIN_WORDS:
            raise ValueError(f"story too short ({n} words < {STORY_MIN_WORDS})")
        if n > STORY_MAX_WORDS:
            raise ValueError(f"story too long ({n} words > {STORY_MAX_WORDS})")
        return v

    def word_count(self) -> int:
        return len(self.story.split())


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------


class DatasetRecord(BaseModel):
    """One training/evaluation example."""

    model_config = {"extra": "forbid"}

    id: str = Field(..., min_length=1)
    language: str
    age_min: int = Field(..., ge=MIN_AGE, le=MAX_AGE)
    age_max: int = Field(..., ge=MIN_AGE, le=MAX_AGE)
    topic: str
    instruction: str = Field(..., min_length=1)
    response: StoryResponse
    source: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    review_status: ReviewStatus = ReviewStatus.unreviewed
    safety_tags: list[str] = Field(default_factory=list)

    @field_validator("language")
    @classmethod
    def _lang_allowed(cls, v: str) -> str:
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"language '{v}' not in allowed set {sorted(ALLOWED_LANGUAGES)}")
        return v

    @field_validator("topic")
    @classmethod
    def _topic_allowed(cls, v: str) -> str:
        if v not in ALLOWED_TOPICS:
            raise ValueError(f"topic '{v}' not in allowed set {sorted(ALLOWED_TOPICS)}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> DatasetRecord:
        if self.age_min > self.age_max:
            raise ValueError("age_min must be <= age_max")
        # Response metadata must agree with record metadata.
        if self.response.language != self.language:
            raise ValueError("response.language must match record.language")
        if self.response.topic != self.topic:
            raise ValueError("response.topic must match record.topic")
        return self

    def dedup_key(self) -> str:
        """A normalized key used to group near-duplicate instructions."""
        return " ".join(self.instruction.lower().split())
