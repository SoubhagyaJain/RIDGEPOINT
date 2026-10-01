"""V1 fixed-membership batch engine behind the Phase 2 SSE contract."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch

from ridgepoint.baseline.hf_backend import HFBackend, PreparedPrompt
from ridgepoint.runtime.qwen2 import Qwen2Runtime
from ridgepoint.runtime.sampler import sample


class CapacityError(RuntimeError):
    """The GPU cannot safely reserve this request's contiguous cache."""


@dataclass
class Job:
    prepared: PreparedPrompt
    max_tokens: int
    temperature: float
    top_p: float
    ignore_eos: bool
    aborted: threading.Event
    emit: Callable[[dict], None]
    finished: asyncio.Future[None]
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    first_ns: int | None = None
    stopped: bool = False


class V1Backend(HFBackend):
    def __init__(self, *, max_batch_size: int = 4, batch_window_ms: float = 10,
                 kv_cap_mib: int = 512, free_reserve_mib: int = 512):
        super().__init__()
        if max_batch_size < 1 or batch_window_ms < 0 or kv_cap_mib < 1 or free_reserve_mib < 0:
            raise ValueError("invalid V1 engine configuration")
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.kv_cap_bytes = kv_cap_mib * 1024 * 1024
        self.free_reserve_bytes = free_reserve_mib * 1024 * 1024
        self.runtime: Qwen2Runtime | None = None
        self.pending: asyncio.Queue[Job] = asyncio.Queue()
        self.backlog: deque[Job] = deque()
        self.worker_task: asyncio.Task | None = None

    def load(self) -> None:
        super().load()
        self.runtime = Qwen2Runtime(self.model)

    def _available_kv_bytes(self) -> int:
        free, _ = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return max(0, min(self.kv_cap_bytes, free + reusable - self.free_reserve_bytes))

    def _required_bytes(self, jobs: list[Job]) -> int:
        assert self.runtime is not None
        capacity = max(job.prepared.token_count + job.max_tokens for job in jobs)
        return self.runtime.cache_bytes(len(jobs), capacity)

    def preflight(self, prepared: PreparedPrompt, max_tokens: int) -> None:
        if self.runtime is None:
            raise RuntimeError("V1 model is not loaded")
        needed = self.runtime.cache_bytes(1, prepared.token_count + max_tokens)
        if needed > self._available_kv_bytes():
            raise CapacityError("insufficient GPU headroom for contiguous KV")

    async def generate(self, prepared: PreparedPrompt, *, max_tokens: int,
                       temperature: float, top_p: float, ignore_eos: bool,
                       aborted: threading.Event, emit: Callable[[dict], None]) -> None:
        if self.runtime is None:
            raise RuntimeError("V1 model is not loaded")
        loop = asyncio.get_running_loop()
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = loop.create_task(self._worker())
        job = Job(prepared, max_tokens, temperature, top_p, ignore_eos,
                  aborted, emit, loop.create_future())
        await self.pending.put(job)
        try:
            await asyncio.shield(job.finished)
        except asyncio.CancelledError:
            aborted.set()
            raise

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _take(self) -> Job:
        return self.backlog.popleft() if self.backlog else await self.pending.get()

    async def _worker(self) -> None:
        while True:
            first = await self._take()
            jobs = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_window_ms / 1000
            while len(jobs) < self.max_batch_size:
                if self.backlog:
                    jobs.append(self.backlog.popleft())
                    continue
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    jobs.append(await asyncio.wait_for(self.pending.get(), remaining))
                except TimeoutError:
                    break
            ready = []
            for job in jobs:
                if job.aborted.is_set():
                    if not job.finished.done():
                        job.finished.set_result(None)
                else:
                    ready.append(job)
            if not ready:
                continue
            while len(ready) > 1 and self._required_bytes(ready) > self._available_kv_bytes():
                self.backlog.appendleft(ready.pop())
            if self._required_bytes(ready) > self._available_kv_bytes():
                ready[0].emit({"type": "error", "message": "insufficient GPU headroom for KV"})
                ready[0].finished.set_result(None)
                continue
            try:
                await asyncio.to_thread(self._run_batch, ready)
            except Exception as exc:
                for job in ready:
                    job.emit({"type": "error", "message": f"V1 batch failed: {exc}"})
            finally:
                for job in ready:
                    if not job.finished.done():
                        job.finished.set_result(None)

    def _emit_token(self, job: Job, token: int) -> None:
        job.tokens.append(token)
        decoded = self.tokenizer.decode(job.tokens, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        delta = decoded[len(job.text):] if decoded.startswith(job.text) else ""
        job.text = decoded
        if job.first_ns is None:
            job.first_ns = time.perf_counter_ns()
        job.emit({"type": "token", "index": len(job.tokens), "token_id": token,
                  "text": delta, "text_so_far": decoded})
        job.stopped = ((not job.ignore_eos and token == self.tokenizer.eos_token_id)
                       or len(job.tokens) >= job.max_tokens)

    def _run_batch(self, jobs: list[Job]) -> None:
        assert self.runtime is not None and self.tokenizer is not None
        started_ns = time.perf_counter_ns()
        with torch.inference_mode():
            cache, logits = self.runtime.prefill(
                [job.prepared.token_ids for job in jobs], [job.max_tokens for job in jobs]
            )
            torch.cuda.synchronize()
            prefill_ended_ns = time.perf_counter_ns()
            for row, job in enumerate(jobs):
                if not job.aborted.is_set():
                    self._emit_token(job, sample(logits[row], temperature=job.temperature,
                                                 top_p=job.top_p))
            while True:
                active_list = [not job.stopped and not job.aborted.is_set() for job in jobs]
                if not any(active_list):
                    break
                active = torch.tensor(active_list, dtype=torch.bool, device=self.runtime.device)
                inputs = torch.tensor([job.tokens[-1] if active_list[row] else 0
                                       for row, job in enumerate(jobs)],
                                      dtype=torch.long, device=self.runtime.device)
                logits = self.runtime.decode(inputs, cache, active)
                for row, job in enumerate(jobs):
                    if active_list[row] and not job.aborted.is_set():
                        self._emit_token(job, sample(logits[row], temperature=job.temperature,
                                                     top_p=job.top_p))
            torch.cuda.synchronize()
        ended_ns = time.perf_counter_ns()
        for job in jobs:
            job.emit({"type": "done", "finish_reason": "abort" if job.aborted.is_set() else
                      ("stop" if len(job.tokens) < job.max_tokens else "length"),
                      "text": job.text,
                      "usage": {"prompt_tokens": job.prepared.token_count,
                                "completion_tokens": len(job.tokens)},
                      "batch_size": len(jobs),
                      "timings_ms": {
                          "queue": (started_ns - job.enqueued_ns) / 1e6,
                          "first_token_from_model_start":
                              (job.first_ns - started_ns) / 1e6 if job.first_ns else None,
                          "generation": (ended_ns - started_ns) / 1e6,
                          "prefill": (prefill_ended_ns - started_ns) / 1e6,
                          "decode": (ended_ns - prefill_ended_ns) / 1e6,
                          "batch_size": len(jobs),
                      }})
