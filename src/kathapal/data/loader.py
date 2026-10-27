"""
JSONL read/write helpers for dataset records.

Reading is lenient about *IO* (reports line numbers on parse errors) but strict
about *schema*: each line is validated into a ``DatasetRecord``. Callers choose
whether to fail on the first bad record or collect all errors (used by the
validation report).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from kathapal.data.schema import DatasetRecord


@dataclass
class LoadError:
    """A single line that failed to parse or validate."""

    line_number: int
    error: str


def iter_raw_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield (line_number, raw_text) for non-empty lines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                yield i, line


def load_records(path: str | Path) -> tuple[list[DatasetRecord], list[LoadError]]:
    """
    Load all records, returning (valid_records, errors).

    Does NOT raise on bad records -- it collects them so the validation report
    can show every problem at once (per the 'do not silently drop data' rule).
    """
    records: list[DatasetRecord] = []
    errors: list[LoadError] = []
    for line_no, raw in iter_raw_lines(path):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(LoadError(line_no, f"invalid JSON: {exc}"))
            continue
        try:
            records.append(DatasetRecord.model_validate(obj))
        except ValidationError as exc:
            errors.append(LoadError(line_no, f"schema error: {exc.errors()!r}"))
    return records, errors


def write_records(path: str | Path, records: list[DatasetRecord]) -> None:
    """Write records as JSONL (one compact JSON object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
