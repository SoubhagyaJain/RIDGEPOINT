# Ridgepoint

Ridgepoint is a seven-phase lab for building a single-GPU LLM inference engine from the scheduler down. **Current state: Phase 3 V1 runtime and static batching.** V0 remains the Hugging Face streaming baseline; V1 owns the Qwen2 forward pass and contiguous KV storage. Continuous batching begins in Phase 4.

The available machine is a Windows laptop with an NVIDIA RTX 4050 Laptop GPU (6 GiB). The primary model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), using BF16 at pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its config is in `configs/models/`; model weights stay outside Git.

## Set up and verify

Install [uv](https://docs.astral.sh/uv/) and an NVIDIA driver compatible with the locked CUDA PyTorch build. Python 3.12 is required. From the repository root:

```powershell
uv sync --frozen --group dev
uv run --frozen python -m ridgepoint.hardware
uv run --frozen pytest -q
uv run --frozen ruff check src bench tests
uv run --frozen python -m ridgepoint.memory_budget configs/models/qwen2.5-1.5b-instruct.config.json
uv run --frozen python -m bench.micro.run --smoke
uv run --frozen python -m bench.micro.run
```

On Linux, `make setup`, `make test`, `make gpu-check`, `make memory-budget`, and `make bench-smoke` provide shortcuts. The GPU check prints a JSON manifest and exits nonzero if CUDA BF16 matrix multiplication fails. E1 prints component measurements and writes per-sample JSON to `results/raw/`, which Git ignores.

## Run the Phase 2 baseline

The pinned model download is about 3.1 GB and defaults to `~/.cache/ridgepoint-model` (outside Git). On Windows, the downloader uses a local directory so it does not require symbolic-link privileges.

```powershell
uv run --frozen python -m ridgepoint.baseline.download
uv run --frozen uvicorn ridgepoint.server.app:app --host 127.0.0.1 --port 8000
```

In another terminal, check a real streamed answer and replay the seeded smoke trace:

```powershell
uv run --frozen python -m bench.client.demo --max-tokens 16
uv run --frozen python -m bench.workloads.generate --workload configs/workloads/chat.yaml --output results/raw/phase2-chat-smoke.jsonl --duration-s 20 --rate-rps 0.5 --warmup-s 0 --max-requests 8 --prompt-tokens 64 --output-tokens 16
uv run --frozen python -m bench.client.replay results/raw/phase2-chat-smoke.jsonl --output results/raw/v0-chat-smoke.jsonl
uv run --frozen python -m bench.analyze.metrics results/raw/v0-chat-smoke.jsonl --duration-s 20 --warmup-s 0
```

The server also has `/health` and `/ready`; `POST /v1/completions` accepts either `prompt` or exact `prompt_ids` and returns one SSE event per generated token plus a final usage/timing event. The null server uses the same API: `uv run --frozen uvicorn bench.backends.null:app --port 8001`. Use `--url http://127.0.0.1:8001/v1/completions` when replaying against it. Raw traces, requests, and manifests stay in `results/raw/`.

## Run the Phase 3 V1 engine

V1 reuses the pinned local weights but executes Qwen2 layers itself. It groups up to four queued requests after a 10 ms window and serves each group to completion. Start it from the repository root:

```powershell
uv run --frozen uvicorn ridgepoint.server.v1:app --host 127.0.0.1 --port 8002
```

In another terminal, use `uv run --frozen python -m bench.client.demo --url http://127.0.0.1:8002/v1/completions --max-tokens 16`. To disable batching, set `RIDGEPOINT_V1_CONFIG=configs/engine/v1-single.yaml` for the server process. On PowerShell use `$env:RIDGEPOINT_V1_CONFIG='configs/engine/v1-single.yaml'` before starting it. The client-facing endpoint and SSE events are the same as V0. V1 adds batch size and prefill/decode timing to the final event. A request that cannot pass the contiguous-KV headroom check receives HTTP 503 before streaming.

The Phase 3 comparison replays one fixed 24-request trace against V0 and both V1 settings. See [the Phase 3 result](results/summaries/phase3-v1.md) for exact commands, raw-run filenames, measurements, and validity limits. The Windows measurements do not support a headline throughput claim. A follow-up removed 113 host syncs per decode step from V1 and found that CPU-core placement changes every engine's speed; the reruns are in the same result file and in section 9 of the Phase 3 report.

## Read the work

- [Specification](docs/spec.md), [seven-phase plan](docs/plan.md), and [current status](docs/status.md)
- [Phase 1 teaching report](docs/learning/phase-01.md)
- [Phase 2 teaching report](docs/learning/phase-02.md) and [V0 baseline results](results/summaries/phase2-v0.md)
- [Phase 3 teaching report](docs/learning/phase-03.md) and [V1 batching results](results/summaries/phase3-v1.md)
- [Memory budget](docs/memory-budget.md) and [benchmark contract](docs/benchmarking.md)
- [Architecture](docs/architecture.md) and [E1 experiment](experiments/E1/README.md)

Numbers in the notebook are estimates until independently measured. Current GPU, V0, and V1 results are exploratory on Windows/WDDM; the Phase 3 runs had inconsistent client scheduling and no continuous clock trace. The final benchmark comparison is planned for native Linux when available and will use identical workload traces for Ridgepoint and reference backends.
