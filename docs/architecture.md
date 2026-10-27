# KathaPal-SLM Architecture

## Goal

Given a structured request, produce a short preschool story + one comprehension
question/answer as **validated structured output**, with layered safety and a
guaranteed safe fallback.

```
request (age, language, topic, length, learning_goal)
   │
   ▼
[input validation]  topic allowlist · age bounds · language allowlist
   │
   ▼
[prompt construction]  model-specific chat template
   │
   ▼
[generation]  base LM + LoRA adapter (MPS/CPU)   ──fail/invalid──┐
   │                                                             │
   ▼                                                             ▼
[output parsing + schema validation]                   [safe fallback library]
   │                                                             │
   ▼                                                             │
[rule-based safety checks] ◄─────────────────────────────────────┘
   │
   ▼
validated StoryResponse  (or documented safe fallback)
```

## Components (maps to `src/kathapal/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Load + validate YAML configs (Pydantic). |
| `logging_config.py` | Structured logging, no child PII. |
| `data/schema.py` | `DatasetRecord` + `StoryResponse` Pydantic models. |
| `data/loader.py` | Read/write JSONL. |
| `data/preparation.py` | Normalize raw → processed records. |
| `data/validation.py` | Dataset report (fields, dupes, safety, provenance). |
| `data/splitting.py` | Seeded, group-aware train/val/test splits. |
| `models/base.py` | Abstract model adapter (swap base model without touching training). |
| `models/loader.py` | Configurable HF model/tokenizer load; device selection. |
| `models/prompting.py` | Build prompts via the model's chat template. |
| `models/generation.py` | Generate → parse → validate → fallback. |
| `models/export.py` | Save adapter, tokenizer, config, model card, manifest. |
| `training/trainer.py` | LoRA SFT loop, smoke-test mode, metadata. |
| `training/callbacks.py`, `checkpointing.py` | Checkpoint/resume, early stop (if supported). |
| `evaluation/metrics.py` | Schema validity, length, repetition, relevance, Q/A presence. |
| `evaluation/evaluator.py` | Run model vs. base vs. template baseline. |
| `evaluation/safety_checks.py` | Rule-based content checks. |
| `evaluation/report.py` | Markdown + JSON report. |
| `safety/rules.py`, `validator.py`, `fallback.py` | Allowlists, content rules, fallback stories. |
| `cli.py` | Typer CLI wiring the scripts. |

## Build/run split

- **Built + offline-tested on Windows:** config, schema, data pipeline,
  prompting (template-agnostic parts), safety rules, generation parsing/fallback,
  evaluation metrics, and all unit tests that do not require model weights. The
  smoke test uses a tiny stub model so pipeline wiring is verified without a
  download.
- **Run on Apple Silicon (Mac):** base-model download, LoRA fine-tuning on MPS,
  full evaluation against the real base model. These are labelled "run on Mac".

## Key design decisions (and trade-offs)

- **Adapter pattern for the base model** (`models/base.py`): the training and
  generation code depend on an interface, not a specific model, so switching
  base models does not require rewriting the pipeline.
- **Safety is layered and model-independent**: even a perfectly-behaved model is
  wrapped by allowlist + schema + rule checks + fallback. Documented limitation:
  these reduce but do not eliminate risk.
- **Fallback-first failure mode**: any generation/parse/validation/safety failure
  returns a curated, pre-approved story rather than a raw model output.
- **Template baseline is a first-class evaluation arm**: we explicitly test
  whether the fine-tuned SLM beats a simple curated-template generator, because
  for this narrow task a template may be competitive.

## Implementation phases

1. Foundation: structure, `pyproject.toml`, configs, schema, logging, gitignore. **(this phase)**
2. Data pipeline: starter dataset, validation, seeded group-aware splits + tests.
3. Model interface: configurable loader, device selection, prompting, generation+validation.
4. LoRA fine-tuning: trainer, smoke-test, checkpoint/resume, metadata.
5. Evaluation + safety: metrics, baselines, safety checks, reports.
6. Tests + lint + docs: run Ruff + pytest offline, finalize docs/model card.
```
