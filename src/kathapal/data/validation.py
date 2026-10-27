"""
Dataset validation and reporting.

Produces a structured report covering the checks required by the project spec:
record counts (overall, by language/topic/age), missing fields / invalid JSON
(surfaced as load errors), duplicate IDs, duplicate/near-duplicate instructions,
excessive response lengths, safety-rule failures, and missing provenance.

Per the 'do not silently drop data' rule, problems are REPORTED, not removed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from kathapal.data.loader import LoadError, load_records
from kathapal.data.schema import STORY_MAX_WORDS, DatasetRecord
from kathapal.safety.rules import check_text


@dataclass
class ValidationReport:
    total_records: int = 0
    by_language: dict[str, int] = field(default_factory=dict)
    by_topic: dict[str, int] = field(default_factory=dict)
    by_age_range: dict[str, int] = field(default_factory=dict)
    load_errors: list[LoadError] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    near_duplicate_instructions: list[tuple[str, str]] = field(default_factory=list)
    excessive_length_ids: list[str] = field(default_factory=list)
    safety_failures: list[tuple[str, str]] = field(default_factory=list)
    missing_provenance_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only if there are no blocking problems."""
        return not (
            self.load_errors
            or self.duplicate_ids
            or self.excessive_length_ids
            or self.safety_failures
            or self.missing_provenance_ids
        )

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "by_language": self.by_language,
            "by_topic": self.by_topic,
            "by_age_range": self.by_age_range,
            "load_errors": [{"line": e.line_number, "error": e.error} for e in self.load_errors],
            "duplicate_ids": self.duplicate_ids,
            "near_duplicate_instructions": self.near_duplicate_instructions,
            "excessive_length_ids": self.excessive_length_ids,
            "safety_failures": self.safety_failures,
            "missing_provenance_ids": self.missing_provenance_ids,
            "ok": self.ok,
        }


def _response_text(rec: DatasetRecord) -> str:
    r = rec.response
    return f"{r.story_title}\n{r.story}\n{r.question}\n{r.answer}"


def validate_records(records: list[DatasetRecord]) -> ValidationReport:
    report = ValidationReport(total_records=len(records))

    report.by_language = dict(Counter(r.language for r in records))
    report.by_topic = dict(Counter(r.topic for r in records))
    report.by_age_range = dict(Counter(f"{r.age_min}-{r.age_max}" for r in records))

    # Duplicate IDs.
    id_counts = Counter(r.id for r in records)
    report.duplicate_ids = sorted([rid for rid, c in id_counts.items() if c > 1])

    # Near-duplicate instructions (exact match on normalized key).
    seen: dict[str, str] = {}
    for r in records:
        key = r.dedup_key()
        if key in seen:
            report.near_duplicate_instructions.append((seen[key], r.id))
        else:
            seen[key] = r.id

    # Excessive length (schema already caps, but double-check post-load).
    report.excessive_length_ids = sorted(
        [r.id for r in records if r.response.word_count() > STORY_MAX_WORDS]
    )

    # Safety-rule failures.
    for r in records:
        res = check_text(_response_text(r))
        if not res.passed:
            report.safety_failures.append((r.id, res.summary()))

    # Missing provenance / licensing.
    report.missing_provenance_ids = sorted(
        [r.id for r in records if not r.source.strip() or not r.license.strip()]
    )

    return report


def validate_dataset(path: str | Path) -> ValidationReport:
    records, errors = load_records(path)
    report = validate_records(records)
    report.load_errors = errors
    return report
