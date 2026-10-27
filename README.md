# KathaPal-SLM

A specialised Small Language Model (SLM) for an AI-powered storytelling and
early-learning toy for children aged **3–5**. Given a structured request
(age, language, topic, length, optional learning goal), the model produces a
short, age-appropriate story plus a simple comprehension question and answer, as
**validated structured output** that an application can consume.

> **Prototype / safety notice.** This is a research prototype. Model outputs can
> still fail safety checks. Rule-based filters and a small fine-tuned model
> **do not guarantee** child-appropriate content. **Adult supervision is required
> at all times during testing with children.** See `docs/safety_and_privacy.md`.

> **Commercial-use notice.** The base model is configurable. You MUST verify the
> actual licence and acceptable-use terms of any base model before commercial
> use. This project makes no licence guarantees. See `docs/model_card.md`.

## Status

Scaffolding + data/safety/eval pipeline implemented and offline-tested. Model
download and LoRA fine-tuning are run on Apple Silicon (MPS); see
`docs/training_guide.md`. Steps verified offline vs. run-on-Mac are labelled
throughout.

## Design principles

- **Stage A:** fine-tune a small, commercially *candidate* pretrained causal LM
  with LoRA — not train from scratch.
- **Layered safety:** topic allowlist, length/age constraints, schema
  validation, rule-based content checks, and a curated fallback story library —
  safety never relies on the model alone.
- **Structured, validated output:** every generation is parsed and validated
  against a strict schema; invalid output triggers a safe fallback.
- **Reproducible:** configs in YAML, seeded splits, recorded environment/metadata.
- **Honest evaluation:** the fine-tuned model is compared against both the
  unmodified base model and a simple curated-template baseline.

## Quick start (see docs for detail)

```bash
uv sync
uv run python scripts/validate_data.py --config configs/training.yaml
uv run python -m pytest -q                      # offline unit tests
uv run python scripts/train.py --config configs/training.yaml --smoke-test
uv run python scripts/generate.py --topic animals --age 4 --language en
```

## Repository layout

See `docs/architecture.md`.
