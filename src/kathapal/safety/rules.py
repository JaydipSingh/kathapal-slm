"""
Rule-based content safety checks.

IMPORTANT LIMITATION: these are transparent keyword/pattern heuristics. They
reduce risk but DO NOT guarantee child-appropriate content. They are one layer
in a defence-in-depth design (allowlist + schema + rules + fallback + adult
supervision), never the sole safeguard. False negatives (unsafe text that slips
through) and false positives (safe text flagged) are both expected.

Each category maps to a set of lowercase substring/word patterns. Matching is
word-boundary aware for short tokens to avoid flagging innocuous substrings
(e.g. 'gun' should not match 'begun').
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Patterns per category. Kept conservative and preschool-oriented.
_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "violence": [
        "kill", "blood", "gun", "knife", "weapon", "stab", "shoot", "fight",
        "punch", "war", "attack", "hurt badly", "die", "dead", "death",
    ],
    "frightening": [
        "monster eats", "nightmare", "demon", "ghost attacks", "terrifying",
        "haunted", "evil spirit", "curse", "devil",
    ],
    "sexual": [
        "sex", "sexual", "naked", "nude", "porn", "kiss on the lips",
    ],
    "hate": [
        "stupid people", "hate them because", "inferior race", "slur",
    ],
    "dangerous_instructions": [
        "play with fire", "drink bleach", "run into the road", "touch the stove",
        "swallow the", "sharp knife to", "climb the electric",
    ],
    "personal_information_request": [
        "what is your address", "what is your name", "where do you live",
        "your phone number", "which school", "your full name", "home address",
    ],
    "adult_themes": [
        "alcohol", "cigarette", "gambling", "divorce court", "gun shop",
    ],
}

# Short tokens that need word-boundary matching to avoid false positives.
_WORD_BOUNDARY_TOKENS = {"gun", "die", "dead", "war", "sex", "kill"}


@dataclass
class SafetyFinding:
    category: str
    matched: str


@dataclass
class SafetyResult:
    passed: bool
    findings: list[SafetyFinding] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return "passed"
        return "; ".join(f"{f.category}:'{f.matched}'" for f in self.findings)


def _matches(pattern: str, text_lower: str) -> bool:
    if pattern in _WORD_BOUNDARY_TOKENS:
        return re.search(rf"\b{re.escape(pattern)}\b", text_lower) is not None
    return pattern in text_lower


def check_text(text: str, active_checks: list[str] | None = None) -> SafetyResult:
    """
    Run rule-based checks over a block of text.

    Args:
        text: the content to check (story + question + answer concatenated).
        active_checks: which categories to apply; defaults to all categories.

    Returns a SafetyResult; ``passed`` is False if any active category matches.
    """
    text_lower = text.lower()
    categories = active_checks if active_checks is not None else list(_CATEGORY_PATTERNS)
    findings: list[SafetyFinding] = []
    for category in categories:
        for pattern in _CATEGORY_PATTERNS.get(category, []):
            if _matches(pattern, text_lower):
                findings.append(SafetyFinding(category=category, matched=pattern))
    return SafetyResult(passed=len(findings) == 0, findings=findings)


def available_categories() -> list[str]:
    return list(_CATEGORY_PATTERNS)
