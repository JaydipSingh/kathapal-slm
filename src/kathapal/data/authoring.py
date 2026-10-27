"""
Dataset authoring helpers: build, validate, and append new records safely.

The goal is to make growing the dataset fast and hard to get wrong: every new
record is validated against the schema AND the rule-based safety checks BEFORE
it is written, and IDs are auto-assigned to avoid collisions. Problems are
reported with clear messages rather than silently written.

Pure Python (no torch) -> fully offline-testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from kathapal.data.loader import iter_raw_lines, load_records
from kathapal.data.schema import DatasetRecord, StoryResponse
from kathapal.safety.rules import check_text


@dataclass
class AuthorResult:
    ok: bool
    record: DatasetRecord | None = None
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def next_id(existing_ids: list[str], prefix: str = "kp") -> str:
    """Return the next zero-padded id like 'kp_0013' given existing ids."""
    max_n = 0
    for rid in existing_ids:
        if rid.startswith(prefix + "_"):
            tail = rid[len(prefix) + 1:]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
    return f"{prefix}_{max_n + 1:04d}"


def build_record(
    *,
    rid: str,
    topic: str,
    instruction: str,
    story_title: str,
    story: str,
    question: str,
    answer: str,
    learning_goal: str,
    age_min: int = 3,
    age_max: int = 5,
    language: str = "en",
    source: str = "original-author",
    license: str = "CC-BY-4.0",
    review_status: str = "unreviewed",
) -> AuthorResult:
    """
    Build and fully validate a DatasetRecord. Runs schema validation AND the
    rule-based safety checks; returns errors rather than raising.
    """
    errors: list[str] = []

    # 1. Schema validation (also enforces length, allowed topic/language, etc.)
    try:
        response = StoryResponse(
            story_title=story_title,
            story=story,
            question=question,
            answer=answer,
            learning_goal=learning_goal,
            language=language,
            age_range=f"{age_min}-{age_max}",
            topic=topic,
        )
        record = DatasetRecord(
            id=rid,
            language=language,
            age_min=age_min,
            age_max=age_max,
            topic=topic,
            instruction=instruction,
            response=response,
            source=source,
            license=license,
            review_status=review_status,
        )
    except ValidationError as exc:
        for e in exc.errors():
            loc = ".".join(str(x) for x in e["loc"])
            errors.append(f"{loc}: {e['msg']}")
        return AuthorResult(ok=False, errors=errors)

    # 2. Safety check on the response content.
    text = f"{story_title}\n{story}\n{question}\n{answer}"
    safety = check_text(text)
    if not safety.passed:
        errors.append(f"safety: {safety.summary()}")
        return AuthorResult(ok=False, errors=errors)

    return AuthorResult(ok=True, record=record)


def append_record(dataset_path: str | Path, result: AuthorResult) -> None:
    """Append a validated record as a JSONL line (creates the file if needed)."""
    if not result.ok or result.record is None:
        raise ValueError("refusing to append an invalid record")
    p = Path(dataset_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(result.record.model_dump_json() + "\n")


def existing_ids(dataset_path: str | Path) -> list[str]:
    """Return the ids already present (empty if the file does not exist)."""
    p = Path(dataset_path)
    if not p.exists():
        return []
    records, _ = load_records(p)
    return [r.id for r in records]


def add_from_dict(dataset_path: str | Path, payload: dict, prefix: str = "kp") -> AuthorResult:
    """
    Convenience: build a record from a flat dict, auto-assigning an id if absent,
    validate it, and append it on success. Returns the AuthorResult either way.
    """
    ids = existing_ids(dataset_path)
    rid = payload.get("id") or next_id(ids, prefix)
    if rid in ids:
        return AuthorResult(ok=False, errors=[f"id '{rid}' already exists"])

    fields = {k: v for k, v in payload.items() if k != "id"}
    result = build_record(rid=rid, **fields)
    if result.ok:
        append_record(dataset_path, result)
    return result


def merge_files(sources: list[str | Path], out_path: str | Path) -> dict:
    """
    Merge several JSONL dataset files into one, de-duplicating by id (first wins)
    and reporting any lines that fail to load. Returns a summary.
    """
    seen: set[str] = set()
    written = 0
    skipped_dupes: list[str] = []
    bad_lines = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fout:
        for src in sources:
            for _line_no, raw in iter_raw_lines(src):
                try:
                    obj = json.loads(raw)
                    rec = DatasetRecord.model_validate(obj)
                except Exception:  # noqa: BLE001
                    bad_lines += 1
                    continue
                if rec.id in seen:
                    skipped_dupes.append(rec.id)
                    continue
                seen.add(rec.id)
                fout.write(rec.model_dump_json() + "\n")
                written += 1

    return {
        "written": written,
        "skipped_duplicate_ids": skipped_dupes,
        "bad_lines": bad_lines,
        "out_path": str(out),
    }
