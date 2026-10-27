"""
Real Hugging Face model adapter + device selection.

RUN ON MAC: this module loads actual model weights and requires the optional
``train`` dependencies (torch, transformers). Imports of those libraries are
deferred to call time so that the rest of the package (schema, data, prompting,
generation-with-stub) imports cleanly on machines without them (e.g. the
Windows build box).

Device selection prefers MPS on Apple Silicon, then CUDA, then CPU, honoring the
config's ``device_preference``. It never assumes CUDA is available.
"""
from __future__ import annotations

from kathapal.config import ModelConfig
from kathapal.logging_config import get_logger
from kathapal.models.base import GenerationParams, ModelAdapter

log = get_logger()


def select_device(preference: str = "auto") -> str:
    """
    Choose a device string: 'mps', 'cuda', or 'cpu'.

    Requires torch; call this only on a machine with the train extras installed.
    """
    import torch  # deferred import

    if preference in {"mps", "cuda", "cpu"}:
        if preference == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS requested but unavailable; falling back to CPU")
            return "cpu"
        if preference == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return preference

    # auto
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HFModelAdapter(ModelAdapter):
    """Adapter backed by a Hugging Face causal LM. RUN ON MAC."""

    def __init__(self, cfg: ModelConfig):
        import torch  # deferred
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        if cfg.trust_remote_code:
            log.warning("trust_remote_code is ENABLED for %s; ensure you trust this model", cfg.model_id)

        self.device = select_device(cfg.device_preference)
        log.info("Loading model '%s' on device '%s'", cfg.model_id, self.device)

        tok_id = cfg.tokenizer_id or cfg.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code
        )
        # Configure a pad token deliberately if missing (document consequence:
        # we reuse EOS as PAD, which is standard for causal LMs).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            log.info("tokenizer had no pad_token; set pad_token = eos_token")

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.float32,  # float32 is the safe choice on MPS
        ).to(self.device)
        self.model.eval()
        self._torch = torch

    @property
    def name(self) -> str:
        return self.cfg.model_id

    def apply_chat_template(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback plain format if the model has no chat template.
        return f"{system}\n\n{user}\n"

    def generate(self, prompt: str, params: GenerationParams) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if params.deterministic:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(
                do_sample=True, temperature=params.temperature, top_p=params.top_p
            )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Return only the newly generated continuation.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
