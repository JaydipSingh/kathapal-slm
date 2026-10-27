"""
LoRA supervised fine-tuning.

RUN ON MAC: requires the optional ``train`` extras (torch, transformers, peft,
datasets). All such imports are deferred into functions so the module imports
cleanly on the Windows build box; the offline tests exercise the pure-Python
helpers (example formatting, label-mask computation) rather than a real run.

Smoke-test mode uses a tiny sequence length, a handful of examples, and a couple
of steps to verify the whole path (load, tokenize, forward, loss finite, step,
save) without a full multi-hour run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kathapal.config import TrainingConfig
from kathapal.data.loader import load_records
from kathapal.data.splitting import split_records
from kathapal.logging_config import get_logger
from kathapal.models.prompting import build_training_text
from kathapal.training.checkpointing import (
    RunMetadata,
    adapter_dir,
    dataset_hash,
)

log = get_logger()


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]


def compute_label_mask(prompt_len: int, total_len: int) -> list[int]:
    """
    Return a labels mask template: -100 for the first ``prompt_len`` positions
    (prompt tokens, excluded from loss), then 1 for the response positions.

    Pure function so the masking logic is unit-tested without a tokenizer.
    """
    if prompt_len < 0 or total_len < prompt_len:
        raise ValueError("require 0 <= prompt_len <= total_len")
    return [-100] * prompt_len + [1] * (total_len - prompt_len)


def build_examples_text(records) -> list[tuple[str, str]]:
    """Build (prompt_text, full_text) pairs for all records (offline-testable)."""
    out = []
    for rec in records:
        prompt_text, full_text = build_training_text(
            rec.instruction, rec.response.model_dump_json()
        )
        out.append((prompt_text, full_text))
    return out


def _tokenize_example(tokenizer, prompt_text: str, full_text: str, max_len: int):
    """Tokenize one example and mask prompt tokens in the labels. RUN ON MAC."""
    full = tokenizer(full_text, truncation=True, max_length=max_len)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_len)
    input_ids = full["input_ids"]
    prompt_len = min(len(prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def train(cfg: TrainingConfig) -> dict:
    """
    Run LoRA supervised fine-tuning. RUN ON MAC.

    Returns a summary dict. Raises with actionable diagnostics on failure rather
    than silently switching method/model.
    """
    import torch  # deferred
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    from kathapal.models.loader import select_device

    # ---- data ----
    records, errors = load_records(cfg.dataset_path)
    if errors:
        raise ValueError(
            f"dataset has {len(errors)} invalid records; fix or exclude them "
            f"explicitly before training (see validate_data.py)"
        )
    if cfg.smoke_test:
        records = records[: min(8, len(records))]
        log.info("SMOKE TEST: using %d records", len(records))

    splits = split_records(
        records, train=cfg.split_train, val=cfg.split_val, test=cfg.split_test, seed=cfg.seed
    )
    log.info("Splits: %s", splits.sizes())

    # ---- model + tokenizer ----
    device = select_device(cfg.model.device_preference)
    log.info("Device: %s", device)
    tok_id = cfg.model.tokenizer_id or cfg.model.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=torch.float32,
    )

    lora = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)

    seq_len = 128 if cfg.smoke_test else cfg.sequence_length

    def make_ds(recs):
        rows = []
        for prompt_text, full_text in build_examples_text(recs):
            rows.append(_tokenize_example(tokenizer, prompt_text, full_text, seq_len))
        return Dataset.from_list(rows)

    train_ds = make_ds(splits.train)
    val_ds = make_ds(splits.val) if splits.val else None

    out_dir = Path(cfg.output_dir)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        learning_rate=cfg.learning_rate,
        num_train_epochs=1 if cfg.smoke_test else cfg.epochs,
        max_steps=2 if cfg.smoke_test else -1,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=1,
        save_steps=cfg.checkpoint_every_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=cfg.eval_every_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=[],
        seed=cfg.seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator
    )

    log.info("Starting training (smoke_test=%s)...", cfg.smoke_test)
    result = trainer.train()

    # ---- save adapter + metadata ----
    adir = adapter_dir(cfg.output_dir)
    adir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adir))
    tokenizer.save_pretrained(str(adir))

    meta = RunMetadata(
        model_id=cfg.model.model_id,
        seed=cfg.seed,
        dataset_path=cfg.dataset_path,
        dataset_sha256=dataset_hash(cfg.dataset_path),
        config=cfg.model_dump(),
        device=device,
    )
    meta.save(adir / "run_metadata.json")

    summary = {
        "train_runtime": getattr(result, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(result, "metrics", {}).get("train_loss"),
        "trainable_params": trainable,
        "total_params": total,
        "adapter_dir": str(adir),
        "smoke_test": cfg.smoke_test,
    }
    log.info("Training complete: %s", summary)
    return summary
