# Training Guide (run on Apple Silicon / Mac)

Training requires the optional `train` extras and is intended for an Apple
Silicon Mac (MPS). The pure-Python pipeline was built and unit-tested on another
machine; the steps below are the first place a real model is downloaded and
fine-tuned.

## 1. Install

```bash
uv sync --extra train --extra dev
```

This installs torch, transformers, peft, datasets, accelerate (plus pytest,
ruff). Pin the resolved versions with `uv lock` on the Mac so the environment is
reproducible there.

## 2. Verify the base model licence

Open `configs/training.yaml` → `model.model_id`. Confirm the model's licence and
acceptable-use policy permit your intended use, then set
`model.license_verified: true`. The trainer warns if this is false.

## 3. Validate the dataset

```bash
uv run python scripts/validate_data.py --config configs/training.yaml
```

Fix any reported problems (the trainer refuses to run on invalid records).

## 4. Smoke test (minutes)

```bash
uv run python scripts/train.py --config configs/training.yaml --smoke-test
```

Smoke mode uses ~8 examples, sequence length 128, and 2 steps. It verifies: the
model and tokenizer load, data tokenize, a forward pass runs, loss is finite, a
training step completes, and the adapter + metadata save. If this fails, fix it
before a full run.

## 5. Full fine-tuning

```bash
uv run python scripts/train.py --config configs/training.yaml
```

Outputs: LoRA adapter + tokenizer + `run_metadata.json` in `outputs/adapters/`,
checkpoints in `outputs/checkpoints/`.

## Troubleshooting (MPS / memory)

- **MPS out of memory:** lower `batch_size`, raise `gradient_accumulation`, or
  reduce `sequence_length`. Enable `gradient_checkpointing: true` if supported.
- **Unsupported MPS op:** the trainer uses float32 (the safe MPS dtype). If a
  specific op is unsupported, it surfaces as an error — do not silently switch
  device; report it.
- **Tokenizer has no pad token:** handled automatically (pad = eos), logged.
- **Resuming:** checkpoints are written to `outputs/checkpoints/`; pass a
  checkpoint to the Transformers trainer to resume (see `trainer.py`).

## Reproducibility

`run_metadata.json` records the seed, dataset SHA-256, config, device, and
library versions. Re-running with the same config + dataset reproduces the run.
