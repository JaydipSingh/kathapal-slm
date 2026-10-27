"""
Abstract model adapter.

The training and generation code depend on this interface, not on a specific
model library. That lets us swap the base model (or substitute a stub for
offline testing) without rewriting the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationParams:
    max_new_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    deterministic: bool = False


class ModelAdapter(ABC):
    """Minimal interface the generation/training layers rely on."""

    @abstractmethod
    def apply_chat_template(self, system: str, user: str) -> str:
        """Render a prompt string from system+user turns using the model's template."""

    @abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> str:
        """Return the model's raw text completion for a prompt."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for logging / metadata."""


class StubModelAdapter(ModelAdapter):
    """
    A deterministic, dependency-free adapter for OFFLINE TESTS ONLY.

    It does not load any real model. It returns a fixed, schema-valid JSON story
    so the generation/parse/validate/fallback pipeline can be tested on machines
    without model weights (e.g. this Windows build box). The real adapter lives
    in ``models/loader.py`` and is used on the Mac.
    """

    def __init__(self, canned_json: str | None = None):
        self._canned = canned_json

    @property
    def name(self) -> str:
        return "stub-model"

    def apply_chat_template(self, system: str, user: str) -> str:
        return f"<<SYS>>{system}<</SYS>>\n{user}"

    def generate(self, prompt: str, params: GenerationParams) -> str:
        if self._canned is not None:
            return self._canned
        # A valid StoryResponse payload (>= 20 words) so parsing succeeds.
        return (
            '{"story_title": "The Friendly Fox", '
            '"story": "A small fox with a bushy tail woke up in the quiet woods. '
            'The sun was warm and the birds were singing. The fox decided to say '
            'hello to every animal it met that day, and it made many new friends '
            'before the evening came.", '
            '"question": "What did the fox decide to do?", '
            '"answer": "The fox decided to say hello and make new friends.", '
            '"learning_goal": "Making friends", '
            '"language": "en", "age_range": "3-5", "topic": "animals"}'
        )
