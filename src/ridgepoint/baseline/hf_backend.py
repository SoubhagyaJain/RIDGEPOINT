"""Blocking Hugging Face generator behind the Phase 2 streaming server."""

from __future__ import annotations

import logging
import copy
import hashlib
import os
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

LOGGER = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHT_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
CONFIG_SHA256 = "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "ridgepoint-model"


def model_source() -> str:
    explicit = os.environ.get("RIDGEPOINT_MODEL_PATH")
    if explicit:
        return explicit
    return str(DEFAULT_MODEL_PATH) if (DEFAULT_MODEL_PATH / "config.json").exists() else MODEL_ID


@dataclass(frozen=True)
class PreparedPrompt:
    token_ids: list[int]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class AbortCriteria(StoppingCriteria):
    def __init__(self, aborted: threading.Event):
        self.aborted = aborted

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.aborted.is_set()


class TokenStreamer(BaseStreamer):
    """Emit one event per generated token ID, even when decoded text is empty."""

    def __init__(self, tokenizer, emit: Callable[[dict], None], first_token: Callable[[], None]):
        self.tokenizer = tokenizer
        self.emit = emit
        self.first_token = first_token
        self.skip_prompt = True
        self.ids: list[int] = []
        self.text = ""

    def put(self, value: torch.Tensor) -> None:
        if self.skip_prompt:
            self.skip_prompt = False
            return
        ids = value.detach().cpu().reshape(-1).tolist()
        for token_id in ids:
            self.ids.append(int(token_id))
            decoded = self.tokenizer.decode(
                self.ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # The full decoded text is authoritative if a tokenizer revises an earlier chunk.
            delta = decoded[len(self.text):] if decoded.startswith(self.text) else ""
            self.text = decoded
            if len(self.ids) == 1:
                self.first_token()
            self.emit({"type": "token", "index": len(self.ids), "token_id": int(token_id),
                       "text": delta, "text_so_far": decoded})

    def end(self) -> None:
        pass


class HFBackend:
    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION,
                 max_model_len: int = 4096):
        self.model_id = model_id
        self.revision = revision
        self.max_model_len = max_model_len
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.weight_sha256 = WEIGHT_SHA256

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("V0 requires a CUDA GPU")
        source = model_source()
        if source != self.model_id:
            for filename, expected in (("config.json", CONFIG_SHA256),
                                       ("model.safetensors", WEIGHT_SHA256)):
                with (Path(source) / filename).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"{filename} SHA-256 does not match pinned model revision")
        revision = {} if source != self.model_id else {"revision": self.revision}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, **revision,
        ).to("cuda").eval()
        LOGGER.info("loaded model=%s revision=%s allocated_bytes=%s", self.model_id,
                    self.revision, torch.cuda.memory_allocated())

    def prepare(self, prompt: str | None, prompt_ids: list[int] | None,
                max_tokens: int) -> PreparedPrompt:
        if self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        ids = list(prompt_ids) if prompt_ids is not None else self.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        if not ids or any(token < 0 or token >= self.model.config.vocab_size for token in ids):
            raise ValueError("prompt tokens must be within the model vocabulary")
        if len(ids) + max_tokens > self.max_model_len:
            raise ValueError(f"prompt plus output exceeds max_model_len={self.max_model_len}")
        return PreparedPrompt(ids)

    def generate(self, prepared: PreparedPrompt, *, max_tokens: int, temperature: float,
                 top_p: float, ignore_eos: bool, aborted: threading.Event,
                 emit: Callable[[dict], None]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        enqueued_ns = time.perf_counter_ns()
        while not self.lock.acquire(timeout=0.1):
            if aborted.is_set():
                return
        try:
            if aborted.is_set():
                return
            started_ns = time.perf_counter_ns()
            first_ns: int | None = None

            def mark_first() -> None:
                nonlocal first_ns
                first_ns = time.perf_counter_ns()

            streamer = TokenStreamer(self.tokenizer, emit, mark_first)
            input_ids = torch.tensor([prepared.token_ids], dtype=torch.long, device="cuda")
            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.max_new_tokens = max_tokens
            generation_config.pad_token_id = self.tokenizer.eos_token_id
            generation_config.do_sample = temperature > 0
            generation_config.temperature = temperature if temperature > 0 else None
            generation_config.top_p = top_p if temperature > 0 else None
            generation_config.top_k = 0 if temperature > 0 else None
            if ignore_eos:
                generation_config.eos_token_id = None
            kwargs = {
                "generation_config": generation_config,
                "streamer": streamer,
                "stopping_criteria": StoppingCriteriaList([AbortCriteria(aborted)]),
                "attention_mask": torch.ones_like(input_ids),
                "use_cache": True,
            }
            with torch.inference_mode():
                self.model.generate(input_ids=input_ids, **kwargs)
            ended_ns = time.perf_counter_ns()
            emit({"type": "done", "finish_reason": "abort" if aborted.is_set() else
                  ("length" if len(streamer.ids) == max_tokens else "stop"),
                  "text": streamer.text,
                  "usage": {"prompt_tokens": prepared.token_count,
                            "completion_tokens": len(streamer.ids)},
                  "timings_ms": {
                      "queue": (started_ns - enqueued_ns) / 1e6,
                      "first_token_from_model_start":
                          (first_ns - started_ns) / 1e6 if first_ns else None,
                      "generation": (ended_ns - started_ns) / 1e6,
                  }})
        finally:
            self.lock.release()
