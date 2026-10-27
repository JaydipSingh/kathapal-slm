"""
Tests for _tokenize_example label masking (the zero/nan-loss bug).

Uses a trivial whitespace 'tokenizer' so the logic is verified offline without
torch/transformers. The key property: after masking, at least one label is NOT
-100 (otherwise the loss is 0/nan and the model cannot learn).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.training.trainer import _tokenize_example  # noqa: E402


class _WSTokenizer:
    """Minimal stand-in: splits on spaces, ids are token indices."""

    def __call__(self, text, **kwargs):
        toks = text.split()
        return {"input_ids": list(range(len(toks))), "attention_mask": [1] * len(toks)}


def _learnable(labels):
    return [x for x in labels if x != -100]


def test_mask_keeps_answer_tokens_when_fits():
    tok = _WSTokenizer()
    prompt = "a b c"            # 3 tokens
    full = "a b c d e f"        # 6 tokens -> answer is 3 tokens
    out = _tokenize_example(tok, prompt, full, max_len=100)
    assert len(out["input_ids"]) == 6
    assert len(_learnable(out["labels"])) == 3  # the 3 answer tokens


def test_long_prompt_still_leaves_answer_unmasked():
    tok = _WSTokenizer()
    prompt = " ".join(["p"] * 50)                 # 50-token prompt
    full = prompt + " " + " ".join(["x"] * 5)     # + 5 answer tokens
    out = _tokenize_example(tok, prompt, full, max_len=20)  # tight window
    # Window is 20; answer (5) must survive and be learnable.
    assert len(out["input_ids"]) == 20
    assert len(_learnable(out["labels"])) >= 5


def test_never_all_masked():
    tok = _WSTokenizer()
    prompt = " ".join(["p"] * 30)
    full = prompt + " " + " ".join(["x"] * 2)
    out = _tokenize_example(tok, prompt, full, max_len=10)
    assert _learnable(out["labels"]), "labels must not be entirely masked"


def test_lengths_consistent():
    tok = _WSTokenizer()
    prompt = "a b"
    full = "a b c d"
    out = _tokenize_example(tok, prompt, full, max_len=100)
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
