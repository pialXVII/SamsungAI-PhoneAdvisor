"""Open-source LLM backend built on Hugging Face Transformers.

The model is loaded once per process and shared by the chatbot and every agent,
because a second copy of the weights would double memory for no benefit.

If the model cannot be loaded — no GPU, not enough RAM, no network to fetch
weights, or `USE_LLM=false` — `get_llm()` returns `None` and callers fall back
to deterministic templates. The system stays answerable either way; only the
prose quality changes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import config

logger = logging.getLogger(__name__)

_llm: "LocalLLM | None" = None
_load_attempted = False
_lock = threading.Lock()


def cuda_free_gb() -> float | None:
    """Free VRAM in GB, or None when there is no usable CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return None


def _resolve_device(warn: bool = False) -> str:
    """Pick the device, refusing CUDA when there is not enough free VRAM.

    The failure this prevents is mundane and easy to hit: the API server holds
    the model on the GPU, then a second process (the demo script, the test
    suite, a terminal chat) starts and tries to load its own copy. On a 6 GB
    card two copies do not fit, and the second one dies with a bare
    "CUDA error: out of memory" partway through loading. Checking first turns
    that crash into a slower-but-working CPU run.

    `warn` is only set while actually loading. Once a model is resident its own
    weights count against free VRAM, so re-running this check would report "not
    enough room for CUDA" about a model already happily running on CUDA — true
    but useless, and alarming in a /health response.
    """
    if config.LLM_DEVICE != "auto":
        return config.LLM_DEVICE

    free = cuda_free_gb()
    if free is None:
        return "cpu"

    if free < config.LLM_MIN_FREE_VRAM_GB:
        if warn:
            logger.warning(
                "Only %.1f GB VRAM free (need ~%.1f GB) — loading on CPU instead. "
                "Another process is probably holding the model; close it, or set "
                "LLM_DEVICE=cuda to force the GPU.",
                free,
                config.LLM_MIN_FREE_VRAM_GB,
            )
        return "cpu"

    return "cuda"


class LocalLLM:
    """Thin wrapper over a Transformers causal LM with a chat interface."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or config.LLM_MODEL
        self.device = device or _resolve_device(warn=True)

        logger.info("Loading %s on %s", self.model_name, self.device)

        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def _load(device: str):
            # fp16 halves memory on GPU; CPU kernels for fp16 are slow, so
            # stay in fp32 there.
            dtype = torch.float16 if device == "cuda" else torch.float32
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(device)

        try:
            self.model = _load(self.device)
        except torch.cuda.OutOfMemoryError:
            # The preflight check in _resolve_device can still lose a race with
            # another process claiming VRAM in between. Retrying on CPU beats
            # taking the whole request down.
            if self.device != "cuda":
                raise
            logger.warning("CUDA out of memory loading the model — retrying on CPU")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model = _load("cpu")

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model ready: %s", self.model_name)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run a chat exchange and return only the newly generated text."""
        import torch

        max_new_tokens = max_new_tokens or config.LLM_MAX_NEW_TOKENS
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Plain-text fallback for base models without a chat template.
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            prompt = "\n\n".join(parts) + "\n\nAssistant:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        ).to(self.device)

        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-4),
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as exc:
            # The KV cache grows with prompt length, so a long agent prompt can
            # exhaust VRAM even when the weights loaded fine. Release it and
            # report clearly; callers fall back to template answers.
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory during generation. Close any other "
                "process using the GPU, or set LLM_DEVICE=cpu in .env."
            ) from exc

        # Slice off the prompt tokens so only the completion is decoded.
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )


def get_llm() -> LocalLLM | None:
    """Return the shared LLM, or `None` when generation is unavailable.

    The load is attempted at most once per process: a failed load is cached as
    `None` so every subsequent request skips the slow retry.
    """
    global _llm, _load_attempted

    if not config.USE_LLM:
        return None

    with _lock:
        if _load_attempted:
            return _llm
        _load_attempted = True
        try:
            _llm = LocalLLM()
        except Exception as exc:
            logger.warning(
                "LLM unavailable (%s) — falling back to template answers", exc
            )
            _llm = None
        return _llm


def llm_status() -> dict:
    """Describe the LLM backend for the API's /health endpoint."""
    return {
        "enabled": config.USE_LLM,
        "model": config.LLM_MODEL,
        # Where the model actually is, not where a fresh load would put it.
        "device": _llm.device if _llm is not None else _resolve_device(),
        "loaded": _llm is not None,
        "load_attempted": _load_attempted,
        "vram_free_gb": round(cuda_free_gb() or 0.0, 2),
    }
