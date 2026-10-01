"""Validated SSE completions API. Run: uv run uvicorn ridgepoint.server.app:app."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import inspect
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ridgepoint.baseline.hf_backend import HFBackend
from ridgepoint.runtime.backend import CapacityError

LOGGER = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    prompt: str | None = None
    prompt_ids: list[int] | None = None
    max_tokens: int = Field(default=32, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    ignore_eos: bool = True
    stream: bool = True

    @model_validator(mode="after")
    def validate_prompt(self):
        if (self.prompt is None) == (self.prompt_ids is None):
            raise ValueError("provide exactly one of prompt or prompt_ids")
        if self.prompt is not None and (not self.prompt.strip() or len(self.prompt) > 100_000):
            raise ValueError("prompt must contain 1 to 100000 nonblank characters")
        if self.prompt_ids is not None and not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if not math.isfinite(self.temperature) or not math.isfinite(self.top_p):
            raise ValueError("sampling values must be finite")
        if not self.stream:
            raise ValueError("Phase 2 supports stream=true only")
        return self


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(backend=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = backend or HFBackend()
        if isinstance(active, HFBackend):
            await asyncio.to_thread(active.load)
        app.state.backend = active
        try:
            yield
        finally:
            close = getattr(active, "close", None)
            if close is not None:
                await close()

    app = FastAPI(title="Ridgepoint V0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        active = request.app.state.backend
        return {"ready": True, "model": active.model_id, "revision": active.revision,
                "weight_sha256": getattr(active, "weight_sha256", None)}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        active = request.app.state.backend
        try:
            prepared = active.prepare(body.prompt, body.prompt_ids, body.max_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        preflight = getattr(active, "preflight", None)
        if preflight is not None:
            try:
                preflight(prepared, body.max_tokens)
            except CapacityError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        request_id = uuid.uuid4().hex
        LOGGER.info("request accepted id=%s prompt_tokens=%s max_tokens=%s", request_id,
                    prepared.token_count, body.max_tokens)

        async def stream_events():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict] = asyncio.Queue()
            aborted = threading.Event()

            def emit(event: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, {"request_id": request_id, **event})

            async def work() -> None:
                try:
                    kwargs = {"max_tokens": body.max_tokens, "temperature": body.temperature,
                              "top_p": body.top_p, "ignore_eos": body.ignore_eos,
                              "aborted": aborted, "emit": emit}
                    if inspect.iscoroutinefunction(active.generate):
                        await active.generate(prepared, **kwargs)
                    else:
                        await asyncio.to_thread(active.generate, prepared, **kwargs)
                except Exception as exc:
                    LOGGER.exception("generation failed id=%s", request_id)
                    emit({"type": "error", "message": str(exc)})

            worker = asyncio.create_task(work())
            try:
                while True:
                    if await request.is_disconnected():
                        aborted.set()
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        if worker.done() and queue.empty():
                            break
                        continue
                    yield sse(event)
                    if event["type"] in {"done", "error"}:
                        break
            finally:
                aborted.set()
                if inspect.iscoroutinefunction(active.generate) and not worker.done():
                    worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
                LOGGER.info("request finished id=%s", request_id)

        return StreamingResponse(stream_events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Request-ID": request_id})

    return app


app = create_app()
