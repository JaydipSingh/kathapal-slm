"""
Story-quality metrics.

All metrics operate on a parsed ``StoryResponse`` (or raw text) and are pure
Python, so they are fully offline-testable. They are proxies, not ground truth:
they measure structural properties (length, repetition, topic-keyword overlap,
Q/A presence, vocabulary simplicity) rather than true 'quality', and the report
labels them as such.
"""
from __future__ import annotations

from dataclasses import dataclass

from kathapal.data.schema import StoryResponse

# A small set of topic keywords used for a coarse relevance proxy.
_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "animals": {"animal", "dog", "cat", "fox", "rabbit", "duck", "bird", "puppy", "turtle"},
    "nature": {"tree", "flower", "garden", "sun", "rain", "leaf", "seed", "forest", "river"},
    "friendship": {"friend", "together", "help", "share", "play"},
    "emotions": {"happy", "sad", "angry", "calm", "feel", "smile", "cry"},
    "curiosity": {"why", "question", "wonder", "learn", "discover", "explore"},
    "space": {"moon", "star", "sky", "night", "planet", "space"},
    "science": {"ice", "water", "melt", "heat", "float", "grow", "experiment"},
    "kindness": {"kind", "help", "share", "gentle", "care"},
    "sharing": {"share", "give", "together", "half"},
    "family": {"mama", "papa", "mom", "dad", "family", "home", "sister", "brother"},
}


@dataclass
class StoryMetrics:
    schema_valid: bool
    word_count: int
    repetition_ratio: float      # fraction of repeated bigrams (lower is better)
    topic_relevance: float       # 0..1 keyword overlap proxy
    has_question: bool
    has_answer: bool
    avg_word_length: float       # vocabulary-complexity proxy (lower ~ simpler)

    def to_dict(self) -> dict:
        return {
            "schema_valid": self.schema_valid,
            "word_count": self.word_count,
            "repetition_ratio": round(self.repetition_ratio, 4),
            "topic_relevance": round(self.topic_relevance, 4),
            "has_question": self.has_question,
            "has_answer": self.has_answer,
            "avg_word_length": round(self.avg_word_length, 2),
        }


def _repetition_ratio(text: str) -> float:
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:], strict=False))
    if not bigrams:
        return 0.0
    unique = len(set(bigrams))
    return 1.0 - (unique / len(bigrams))


def _topic_relevance(text: str, topic: str) -> float:
    keywords = _TOPIC_KEYWORDS.get(topic)
    if not keywords:
        return 0.0
    words = set(text.lower().split())
    hits = len(words & keywords)
    return min(1.0, hits / 3.0)  # 3+ keyword hits -> full relevance


def _avg_word_length(text: str) -> float:
    words = [w for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compute_story_metrics(resp: StoryResponse) -> StoryMetrics:
    return StoryMetrics(
        schema_valid=True,  # a StoryResponse is valid by construction
        word_count=resp.word_count(),
        repetition_ratio=_repetition_ratio(resp.story),
        topic_relevance=_topic_relevance(resp.story, resp.topic),
        has_question=bool(resp.question.strip().endswith("?")),
        has_answer=bool(resp.answer.strip()),
        avg_word_length=_avg_word_length(resp.story),
    )
