"""Predictable fake completion backend used to verify the load generator."""

from __future__ import annotations

import asyncio
import time
import os
from collections.abc import Callable

from ridgepoint.baseline.hf_backend import PreparedPrompt
from ridgepoint.server.app import create_app


class NullBackend:
    model_id = "ridgepoint-null"
    revision = "deterministic-v1"

    def __init__(self, first_token_delay_ms: float = 5.0, token_delay_ms: float = 2.0):
        self.first_delay = first_token_delay_ms / 1000
        self.token_delay = token_delay_ms / 1000

    def prepare(self, prompt: str | None, prompt_ids: list[int] | None,
                max_tokens: int) -> PreparedPrompt:
        ids = prompt_ids if prompt_ids is not None else list(prompt.encode("utf-8"))
        if not ids or min(ids) < 0:
            raise ValueError("prompt IDs must be nonnegative")
        return PreparedPrompt(list(ids))

    async def generate(self, prepared: PreparedPrompt, *, max_tokens: int, temperature: float,
                       top_p: float, ignore_eos: bool, aborted,
                       emit: Callable[[dict], None]) -> None:
        started = time.perf_counter_ns()
        first: int | None = None
        count = 0
        for index in range(max_tokens):
            await asyncio.sleep(self.first_delay if index == 0 else self.token_delay)
            if aborted.is_set():
                break
            if first is None:
                first = time.perf_counter_ns()
            count += 1
            emit({"type": "token", "index": count, "token_id": 42,
                  "text": "x", "text_so_far": "x" * count})
        ended = time.perf_counter_ns()
        emit({"type": "done", "finish_reason": "abort" if aborted.is_set() else "length",
              "text": "x" * count,
              "usage": {"prompt_tokens": prepared.token_count, "completion_tokens": count},
              "timings_ms": {"queue": 0.0,
                             "first_token_from_model_start":
                                 (first - started) / 1e6 if first else None,
                             "generation": (ended - started) / 1e6}})


app = create_app(NullBackend(
    first_token_delay_ms=float(os.getenv("RIDGEPOINT_NULL_FIRST_MS", "5")),
    token_delay_ms=float(os.getenv("RIDGEPOINT_NULL_TOKEN_MS", "2")),
))
