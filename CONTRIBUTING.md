# Contributing

Contributions, bug reports, and dataset improvements are welcome.

## Development setup

```bash
uv sync --extra dev                 # offline work (schema/data/safety/eval + tests)
uv sync --extra train --extra dev   # add torch/transformers/peft for Mac training
```

## Running checks

```bash
python -m ruff check src scripts tests
python -m pytest tests/ -q
```

Offline unit tests require no model download. Training/generation against a real
model runs on Apple Silicon (MPS); see `docs/training_guide.md`.

## Guidelines

- Keep the safety layers intact: never weaken allowlist / schema / rule checks /
  fallback without discussion.
- Add tests for new data, prompting, safety, or evaluation logic.
- Record provenance and licence for every dataset example.
- Do not commit model weights, adapters, generated data, or secrets.

## Child-safety note

This is a prototype for a children's product. Treat every change touching
content generation or safety with extra care, and keep the documented
limitations honest.
