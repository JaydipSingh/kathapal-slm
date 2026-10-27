# Evaluation Guide

## Run

```bash
# Offline demo (stub model) — verifies the harness without a download:
uv run python scripts/evaluate.py --stub

# Real model (Mac), comparing the fine-tuned model to the template baseline:
uv run python scripts/evaluate.py --config configs/generation.yaml --adapter outputs/adapters
```

Outputs `outputs/reports/evaluation_report.md` and `.json`.

## What is measured

**Story-quality proxies** (structural, not true quality):
- schema validity, word count, repetition ratio, topic-keyword relevance,
  question/answer presence, average word length (vocabulary-simplicity proxy).

**Safety** (rule-based proxy, NOT a guarantee):
- pass rate and per-category failures over the generated set.

## Arms compared

1. **kathapal-model** — the fine-tuned model (or base model) via the adapter.
2. **template-baseline** — a curated-template generator (the reviewed fallback
   library used as a generator).

On the Mac you can add a third arm by pointing the model arm at the unmodified
base model, giving the full fine-tuned vs. base vs. template comparison.

## Interpreting results — be honest

- If the **template baseline** scores as well as the fine-tuned model on these
  proxies, that is a signal the fine-tuning may not yet add enough value. For a
  narrow preschool task a good template can be competitive; the fine-tuned model
  must clearly beat it to justify its complexity.
- Proxy metrics do not capture coherence, age-appropriateness, or emotional
  tone. Use the manually-reviewed rubric (age appropriateness, coherence,
  educational usefulness, tone, factual consistency, language quality, safety)
  for those, with human reviewers.
- Report only measured values. Do not infer quality the metrics cannot measure.
