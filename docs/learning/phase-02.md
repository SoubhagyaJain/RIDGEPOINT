# Phase 2 — V0 server and trustworthy benchmark harness

## 1. What we built

V0 loads pinned Qwen2.5-1.5B-Instruct weights in BF16, runs Hugging Face `generate()` on the RTX 4050, and sends one Server-Sent Event (SSE) per generated token. A seeded trace generator, open-loop streaming client, async null backend, and raw-record analyzer form the first benchmark path. The model download goes to the user's cache, outside Git.

Exact files changed in this phase:

- Dependencies/commands: `.gitattributes`, `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`.
- V0 code: `src/ridgepoint/baseline/__init__.py`, `src/ridgepoint/baseline/download.py`, `src/ridgepoint/baseline/hf_backend.py`, `src/ridgepoint/server/__init__.py`, `src/ridgepoint/server/app.py`.
- Harness: `bench/backends/__init__.py`, `bench/backends/null.py`, `bench/client/__init__.py`, `bench/client/demo.py`, `bench/client/replay.py`, `bench/workloads/__init__.py`, `bench/workloads/corpus.txt`, `bench/workloads/generate.py`, `bench/analyze/__init__.py`, `bench/analyze/metrics.py`.
- Tests: `tests/unit/test_phase2_metrics.py`, `tests/unit/test_phase2_trace.py`, `tests/integration/test_null_stream.py`.
- Documentation/results: `docs/architecture.md`, `docs/benchmarking.md`, `docs/memory-budget.md`, `docs/report.md`, `docs/status.md`, `docs/learning/phase-02.md`, `experiments/V0-baseline/README.md`, `results/summaries/phase2-v0.md`.

The pre-existing untracked `docs/learning/phase-01-handwritten-notes.pdf` was left untouched and is not part of this phase commit.

## 2. Concepts from scratch

**Tokens are model-sized text pieces.** A tokenizer maps text into integer IDs. The server accepts text or exact IDs; the harness uses IDs so a 64-token prompt remains exactly 64 tokens across replays. The Qwen config's embedding vocabulary has 151,936 slots, while the tokenizer reports 151,643 base vocabulary entries; special IDs occupy additional slots.

**Prefill and decode happen in one `generate()` call in V0.** Prefill reads the whole prompt and builds key/value state; decode adds one output token at a time. V0 lets Hugging Face own these internals. Phase 3 will implement them ourselves. A request with a 64-token prompt and 16 output tokens incurs one prefill and 16 decode steps.

**SSE is an HTTP response that stays open.** The server writes events such as `event: token` and `data: {...}` as tokens become available, followed by `event: done` containing final usage and timing. One event per token ID lets the client timestamp every token. A token can decode to an empty visible string if it is part of a multi-byte character; the token event still counts.

**Queueing is visible even in a simple baseline.** V0 protects one GPU model with a lock. If request A is generating and B arrives, B waits. Its time to first token includes that wait, even if its own prompt is short. The server reports queue time separately from model time; the client measures actual TTFT from send to first arrival.

**Open-loop load keeps sending on schedule.** If six requests are scheduled at fixed offsets, the client sends at those offsets even when earlier requests are slow. A closed-loop client that waits for responses before sending the next request would hide overload. The generator records the difference between actual and scheduled send time, called lag.

**Goodput is about individual requests.** Suppose two requests have TTFTs of 0.5 s and 1.2 s, and both have TPOT 30 ms. With a 1 s TTFT limit, only the first request qualifies. Goodput is qualifying requests divided by the measurement window. Rejected requests are counted next to it, never silently removed from offered load.

**A manifest makes a result interpretable.** A latency number alone says little. The raw run manifest records trace hash, source hash, model revision/weight hash, package versions, OS, GPU snapshots, and config hash. Raw events let us recompute the metrics.

## 3. How the code works

`baseline.download.main()` fetches the exact Qwen revision into `~/.cache/ridgepoint-model`, avoiding Windows cache-symlink privilege problems. `HFBackend.load()` verifies local config and weight SHA-256, loads the tokenizer and BF16 model, moves it to CUDA, and calls `eval()`. `HFBackend.prepare(prompt, prompt_ids, max_tokens)` returns the exact input IDs after vocabulary and model-length checks.

`server.app.CompletionRequest` validates fields before GPU work. `create_app()` exposes `/health`, `/ready`, and `POST /v1/completions`. The endpoint creates a request ID, prepares the prompt, and starts a worker. `HFBackend.generate()` acquires the single model lock, constructs a per-call generation configuration, and calls Hugging Face `generate()` under `torch.inference_mode()`. `TokenStreamer.put()` skips the initial prompt, then emits each generated token ID and a decoded text update. The final event reports token counts, queue time, time from model start to first token, and total generation time. Disconnects signal an abort criterion checked between decode steps.

`workloads.generate.load_workload()` resolves the YAML base configuration. `generate_trace()` uses a private seeded random generator for Poisson arrival gaps and prompt/output lengths, then cuts or cycles tokenized committed text to exact prompt lengths. It writes one JSON object per request plus a byte-level SHA-256 manifest. The chat, prefill, and decode shapes are usable now; session, burst, and overload-specific behavior is planned for the phases that exercise those features.

`client.replay.replay()` gives every request a fixed send time from one monotonic origin. Each async task waits for that time and opens an HTTP stream regardless of earlier completions. `replay_one()` timestamps each token event with `perf_counter_ns`, records final usage, status and errors, and never re-tokenizes returned text. It refuses a trace whose bytes no longer match its manifest. `analyze.request_metrics()` calculates one request's TTFT, ITL gaps, TPOT, and SLO qualification; `analyze.analyze()` aggregates requests by scheduled measurement window, including explicit rejections. The async null backend uses the same server and trace contract without loading a model.

One request therefore goes: JSON validation → exact token IDs → wait for V0 lock → `generate()` prefill/decode → streamer token ID → SSE → client timestamp → raw JSONL → metric row and summary.

## 4. How to reproduce and debug

Install Python 3.12, uv, and an NVIDIA driver. From the repository root:

```powershell
uv sync --frozen --group dev
uv run --frozen python -m ridgepoint.baseline.download
uv run --frozen pytest -q
uv run --frozen ruff check src bench tests
uv run --frozen uvicorn ridgepoint.server.app:app --host 127.0.0.1 --port 8000
```

Leave the server running. In another terminal:

```powershell
uv run --frozen python -m bench.client.demo --prompt 'Explain a GPU cache in one sentence.' --max-tokens 16
uv run --frozen python -m bench.workloads.generate --workload configs/workloads/chat.yaml --output results/raw/phase2-chat-smoke.jsonl --duration-s 20 --rate-rps 0.5 --warmup-s 0 --max-requests 8 --prompt-tokens 64 --output-tokens 16
uv run --frozen python -m bench.client.replay results/raw/phase2-chat-smoke.jsonl --output results/raw/v0-chat-smoke.jsonl
uv run --frozen python -m bench.analyze.metrics results/raw/v0-chat-smoke.jsonl --duration-s 20 --warmup-s 0
```

Expected: the demo prints a request ID, visible answer pieces, then `usage` with 16 completion tokens. The trace command prints a SHA-256 and, with the pinned seed/overrides, six requests. Replay prints `errors: 0` and the same trace SHA. The analyzer prints TTFT/TPOT values, goodput, rejections, and validity flags. Exact latency numbers will vary.

For a no-GPU harness check, start `uv run --frozen uvicorn bench.backends.null:app --port 8001` in a separate terminal and replay the same trace with `--url http://127.0.0.1:8001/v1/completions`. Inspect `results/raw/*.manifest.json` and JSONL to debug hash mismatch, late sends, token-count mismatch, or server error. `GET /ready` confirms model revision and expected weight hash. Use `nvidia-smi` for current VRAM and clocks. Do not commit raw runs or weights.

## 5. Proof and its limits

`uv sync --frozen --group dev` completed. `pytest -q` passed **10 tests**: existing memory math, seeded trace behavior, per-request timing and goodput rules, open-stream counting, validation, SSE events, and null replay. `ruff check src bench tests` passed. A real Qwen request streamed 16 token events plus a final event over HTTP. The baseline loaded **3,088,346,624 allocated GPU bytes** in one direct check and left **2,033,188,864 free bytes** after load; background WDDM usage varies.

The same six-request trace ran three times against V0 with 6/6 completions and no count mismatches, errors, or rejections. Good requests were 5, 5, and 5; every trial's goodput was **0.25 req/s** at **0.3 offered req/s**. TTFT P95 was 1,018, 999, and 1,011 ms; TPOT P95 stayed near 27 ms. The exact manifest and table are in `results/summaries/phase2-v0.md`. These are **exploratory smoke observations**, not a capacity claim: six samples cannot support stable tail percentiles, and no continuous clock trace was collected.

The null backend completed the same trace without errors. In a separate async-null run, **500 streams were simultaneously open** and all 500 completed, but send-lag P99 was **16.91 ms**, above the 5 ms target. This shows the client/server path survives concurrency and also shows a measurement limit on this Windows host. vLLM was not compared because native Windows is unsupported and the installed WSL2 distribution could not start due a missing virtual disk.

## 6. Problems, fixes, and remaining limitations

- Hugging Face's normal Windows cache download failed with `WinError 1314` while creating a symlink. The downloader now uses a pinned `local_dir` outside Git; the full model download and local SHA checks succeeded.
- The first `generate()` emitted warnings for a missing attention mask and sampling settings inherited from the model's generation config. Supplying a mask and a per-call config removed them.
- The trace generator initially hashed LF text before Windows converted it to CRLF on disk. It now writes exact UTF-8 bytes and the replayer verifies their hash.
- Tests could not import the top-level `bench` package when invoked through the pytest executable. The project pytest configuration now adds the repository root to the import path.
- The first null backend used the default thread pool, so 500 slow streams queued in waves. Making null generation async let 500 streams remain open concurrently. The client still missed the 5 ms P99 send-lag target in that burst.
- The WSL2 Ubuntu instance failed to mount its missing `ext4.vhdx`, and vLLM has no supported native Windows CUDA install. The vLLM comparison remains for a working Linux environment.
- V0 serializes requests, has no bounded queue or memory admission, and leaves model runtime/KV management to Hugging Face. Those are the subjects of later phases. The small corpus and smoke trace are a reproducibility check, not representative production traffic.

## 7. Explain it back

**Interview-sized explanation:** “I built a simple real-model server before optimizing it. It streams one event per model token, records IDs and server timing, and uses one lock around Hugging Face generation. I also built an open-loop client that replays a seeded token-ID trace and timestamps every received token. The analysis counts each request that meets both latency targets, while reporting rejections separately. The first six-request baseline is reproducible but too small and too noisy to be a headline benchmark.”

Five questions:

1. Why send prompt token IDs in a comparison trace?
2. What does V0's model lock do to TTFT when requests overlap?
3. Why does an open-loop client measure overload more honestly than a client that waits between sends?
4. Why calculate TPOT for each request before checking its SLO?
5. What does a trace hash protect against, and what does it not prove?

Answer key:

1. It fixes the exact prompt length and content across replays/backends, avoiding retokenization differences where token IDs are accepted.
2. Later requests wait for earlier generations; that wait appears in client TTFT even when their own model work is short.
3. It preserves offered load while the server slows, rather than automatically reducing arrivals and hiding queue growth.
4. The SLO applies to each user's completed request; comparing only aggregate P95 values cannot count qualifying requests correctly.
5. It detects any byte change in the trace file. It does not validate server correctness, GPU stability, or the realism of the workload.

## 8. Phase gate

- [x] One client receives a real streamed Qwen answer with request ID, token events, and final usage/timing.
- [x] The harness replays the same seeded exact-token trace to V0 and null; the byte hash is checked.
- [x] Timing definitions, per-request goodput, rejection accounting, and HTTP streaming are tested.
- [x] A repeatable V0 smoke result has model, source, trace, config, software, and hardware manifest fields; limitations are explicit.
- [x] vLLM availability was checked and the Windows/WSL blocker documented.
- [x] Code, tests, results summary, status, and teaching report are committed for Phase 2.

The gate is about a working baseline and trustworthy measurement path. The 500-stream lag target and a valid headline performance comparison remain open engineering work, not silently claimed as passed.
