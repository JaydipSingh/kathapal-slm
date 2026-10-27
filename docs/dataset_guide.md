# Dataset Guide

## Format (JSONL)

One JSON object per line, validated against `DatasetRecord` (see
`src/kathapal/data/schema.py`). Required fields:

```json
{
  "id": "kp_0001",
  "language": "en",
  "age_min": 3,
  "age_max": 5,
  "topic": "sharing",
  "instruction": "Create a short story for a four-year-old about sharing.",
  "response": {
    "story_title": "The Carrot Garden",
    "story": "A short, simple story (20-250 words)...",
    "question": "What did the rabbit learn?",
    "answer": "The rabbit learned to share.",
    "learning_goal": "Sharing with friends",
    "language": "en",
    "age_range": "3-5",
    "topic": "sharing"
  },
  "source": "original-author",
  "license": "CC-BY-4.0",
  "review_status": "approved",
  "safety_tags": ["reviewed"]
}
```

Constraints enforced: allowed `language` (`en`), allowed `topic` (see schema),
age within 3–5, story 20–250 words, and response metadata must match the record.

## Adding examples

1. Append new JSONL lines to a dataset file (no code changes needed).
2. Run validation:
   ```bash
   uv run python scripts/validate_data.py --data your_dataset.jsonl
   ```
3. Fix every reported problem. Invalid records are **reported, not silently
   dropped** — correct or explicitly exclude them.

## Provenance & licensing (important)

- Record `source` and `license` for every example.
- Do **not** copy copyrighted children's books or datasets with incompatible
  licences. Prefer original, clearly-licensed, or properly-licensed content.
- Distinguish original / licensed / synthetic / human-reviewed via `source` and
  `review_status`.

## Splits

Splits are reproducible (seeded) and **group-aware**: records sharing a
normalized instruction stay in the same split, preventing paraphrase leakage
from train into test. Keep a small, separate, human-reviewed evaluation set and
do not tune hyperparameters repeatedly against the final test set.

## Scale expectations

- ~50–100 examples: smoke testing only.
- ~200–500 reviewed examples: early experimentation.
- Production quality needs substantially more, diverse, and reviewed data. The
  starter set shipped here (~12) exists only to prove the pipeline runs.
