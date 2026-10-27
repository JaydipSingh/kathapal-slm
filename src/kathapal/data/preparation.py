"""
Data preparation: normalize raw records into a clean processed set.

For the starter dataset the inputs are already in schema form, so preparation
is light: it validates, drops exact-duplicate IDs (reporting them), normalizes
whitespace in text fields, and writes a processed JSONL. It never silently
discards records that merely fail validation -- those are surfaced by the
validation report; preparation only de-duplicates exact ID collisions.
"""
from __future__ import annotations

from pathlib import Path

from kathapal.data.loader import load_records, write_records
from kathapal.data.schema import DatasetRecord


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _normalize_record(rec: DatasetRecord) -> DatasetRecord:
    r = rec.response
    normalized_response = r.model_copy(
        update={
            "story_title": _normalize_ws(r.story_title),
            "story": _normalize_ws(r.story),
            "question": _normalize_ws(r.question),
            "answer": _normalize_ws(r.answer),
        }
    )
    return rec.model_copy(
        update={
            "instruction": _normalize_ws(rec.instruction),
            "response": normalized_response,
        }
    )


def prepare_dataset(raw_path: str | Path, processed_path: str | Path) -> dict:
    """
    Load raw records, normalize whitespace, drop exact-duplicate IDs (keeping the
    first occurrence), and write the processed JSONL. Returns a small summary.
    """
    records, errors = load_records(raw_path)

    seen_ids: set[str] = set()
    kept: list[DatasetRecord] = []
    dropped_dup_ids: list[str] = []
    for rec in records:
        if rec.id in seen_ids:
            dropped_dup_ids.append(rec.id)
            continue
        seen_ids.add(rec.id)
        kept.append(_normalize_record(rec))

    write_records(processed_path, kept)

    return {
        "raw_valid": len(records),
        "load_errors": len(errors),
        "written": len(kept),
        "dropped_duplicate_ids": dropped_dup_ids,
        "processed_path": str(processed_path),
    }
