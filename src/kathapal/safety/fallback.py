"""
Curated, pre-approved fallback stories.

When generation fails, times out, or produces output that fails schema or safety
validation, the system returns one of these hand-written, reviewed stories
instead of a raw/unsafe model output. This guarantees the application always has
a safe, schema-valid response to show.

The fallback library is intentionally small and fully reviewed. Each entry is a
valid StoryResponse payload keyed by topic, with a generic default.
"""
from __future__ import annotations

from kathapal.data.schema import StoryResponse

# Topic -> fallback StoryResponse payload. All reviewed and safe.
_FALLBACKS: dict[str, dict] = {
    "animals": {
        "story_title": "The Happy Puppy",
        "story": (
            "Once there was a little puppy with soft brown ears. Every morning the "
            "puppy wagged its tail and ran in the green grass. It liked to chase "
            "butterflies and roll in the sunshine. When it was tired, it curled up "
            "for a cozy nap under a shady tree. The puppy was gentle and kind to "
            "everyone it met, and all the animals in the park were glad to be its "
            "friend."
        ),
        "question": "What did the little puppy like to chase?",
        "answer": "The puppy liked to chase butterflies.",
        "learning_goal": "Gentleness and friendship",
        "language": "en",
        "age_range": "3-5",
        "topic": "animals",
    },
    "kindness": {
        "story_title": "A Little Kindness",
        "story": (
            "Lily saw a friend drop all of her crayons on the floor. The crayons "
            "rolled everywhere, red and blue and green. Lily did not laugh. "
            "Instead, she kneeled down and helped pick up every single crayon, one "
            "by one. She handed them back with a warm smile. Her friend felt happy "
            "again. A little bit of kindness can turn a sad moment into a happy one, "
            "and it made both of them smile."
        ),
        "question": "How did Lily help her friend?",
        "answer": "Lily helped pick up the crayons that had fallen.",
        "learning_goal": "Helping others with kindness",
        "language": "en",
        "age_range": "3-5",
        "topic": "kindness",
    },
    "nature": {
        "story_title": "The Morning Garden",
        "story": (
            "In the garden, the sun rose slowly and warmed the green leaves. Tiny "
            "dewdrops sparkled on the petals like little jewels. A gentle bee buzzed "
            "from flower to flower, saying good morning to each one. The tall "
            "sunflowers turned their happy faces toward the light. Birds sang "
            "sweet songs in the trees. The whole garden felt calm and fresh and "
            "full of life on this bright and peaceful morning."
        ),
        "question": "What did the sunflowers turn toward?",
        "answer": "The sunflowers turned toward the light of the sun.",
        "learning_goal": "Appreciating nature",
        "language": "en",
        "age_range": "3-5",
        "topic": "nature",
    },
}

_DEFAULT_TOPIC = "animals"


def get_fallback(topic: str | None = None) -> StoryResponse:
    """
    Return a safe, schema-valid fallback story for the given topic, falling back
    to a default if the topic has no dedicated entry.
    """
    payload = _FALLBACKS.get(topic or "", _FALLBACKS[_DEFAULT_TOPIC])
    return StoryResponse.model_validate(payload)


def available_fallback_topics() -> list[str]:
    return list(_FALLBACKS)
