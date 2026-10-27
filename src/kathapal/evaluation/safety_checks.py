"""
Evaluation-time safety checks.

Thin wrapper over the rule-based checker in ``safety.rules`` that scores a batch
of responses and reports the pass rate and per-category failures. Documented
limitation (repeated here deliberately): rule-based checks reduce but do not
guarantee safety; they are a baseline for measurement, not a certification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kathapal.data.schema import StoryResponse
from kathapal.safety.rules import check_text


@dataclass
class SafetySummary:
    total: int
    passed: int
    failures: list[tuple[str, str]] = field(default_factory=list)  # (title, reason)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "failures": self.failures,
            "note": "rule-based proxy; does NOT guarantee safety",
        }


def summarize_safety(responses: list[StoryResponse]) -> SafetySummary:
    passed = 0
    failures: list[tuple[str, str]] = []
    for r in responses:
        text = f"{r.story_title}\n{r.story}\n{r.question}\n{r.answer}"
        res = check_text(text)
        if res.passed:
            passed += 1
        else:
            failures.append((r.story_title, res.summary()))
    return SafetySummary(total=len(responses), passed=passed, failures=failures)
