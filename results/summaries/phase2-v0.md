# Phase 2 V0 baseline: exploratory smoke result

**Measured on 2026-09-27.** This is a real Qwen2.5-1.5B-Instruct BF16 streaming server result on the RTX 4050 Laptop GPU. It is **not a valid headline benchmark**: only six requests per trial, Windows/WDDM, no continuous clock trace, and one trial's generator-lag P99 exceeded 5 ms. Treat the percentiles as descriptions of these six requests only.

## Exact setup

- Server: `ridgepoint.server.app` V0, Hugging Face `generate()` with SDPA, BF16, greedy sampling, `ignore_eos=true`, one serialized generation at a time, no prefix cache or admission policy.
- Model: `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; local safetensors SHA-256 verified as `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB reported by `nvidia-smi`, driver 581.42; Windows 11 build 26200; Python 3.12.10; PyTorch 2.8.0+cu128; Transformers 4.57.6; httpx 0.28.1.
- Trace: `W-chat`, seed 1234, 20 s schedule, 0.5 requested arrivals/s, capped at 8 requests, fixed 64 prompt and 16 output tokens, warmup 0. Poisson draws produced **6 requests**, or **0.3 offered requests/s**. The same exact token-ID trace was replayed in all trials. Trace SHA-256: `04d5d2ec9319b7dc630dd6d640638ef507a34b76325655df2ad9aca135ad7f39`.
- SLO: each completed request qualifies if client-observed TTFT ≤ 1,000 ms and its per-request TPOT ≤ 75 ms. Rejections are counted separately. Each trial completed 6/6, with no errors, rejections, or prompt-token-count mismatches.
- Code: raw run manifests record Git HEAD, a source-tree SHA-256 (`b0be65312d1fa516adc6bcdce5a74fc55016994c5a82af7b950befd59483bbe6`), package versions, engine/workload hashes, model ID/revision/hash, and before/after GPU snapshots. Git was dirty during measurement because Phase 2 was in progress; the source hash pins the code files used. Raw JSONL/manifests remain local in ignored `results/raw/`.

| Trial | Good requests / 6 | Goodput req/s | TTFT P95 ms | TPOT P95 ms | ITL P99 ms | Generator lag P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.25 | 1,017.55 | 27.16 | 30.81 | 11.63 |
| 2 | 5 | 0.25 | 998.81 | 26.64 | 31.68 | 0.12 |
| 3 | 5 | 0.25 | 1,011.44 | 26.66 | 30.64 | 2.79 |

Each trial's goodput was **0.25 req/s**. Input and output token cohort rates were 19.2 and 4.8 tokens/s per 20 s window. Each trial reached three simultaneously open HTTP streams. These are low-load smoke figures, not maximum capacity. The one-at-a-time V0 lock and Windows scheduling make TTFT vary when arrivals overlap; TPOT was more stable. We did not average per-run P95s to make a headline estimate.

## Harness checks and reference status

- A real HTTP client received 16 streamed token events and a final usage/timing event from V0. The server reported prompt and completion token counts, and all six replayed requests matched the trace prompt length.
- The same six-request trace completed against the null backend with no errors. The null server's configured millisecond delays were longer in observed client timing because Windows timer scheduling is coarse, so it is used only to check the harness path.
- An async null-backend stress run measured **500 simultaneously open streams** from response-header and completion timestamps and completed 500/500 with zero errors. Its generator-lag P99 was **16.91 ms**, failing the notebook's 5 ms target. The first implementation used a thread pool and took over a minute at 500 streams; changing the null backend to async removed that bottleneck. The stress run deliberately delayed first tokens 12 s, so its zero SLO goodput is expected and not a server-capacity comparison.
- vLLM was not compared. [Its CUDA installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) states that native Windows is unsupported. The local WSL2 Ubuntu instance failed before shell startup because its `ext4.vhdx` path was missing (`WSL/Service/CreateInstance/MountDisk/HCS/ERROR_PATH_NOT_FOUND`). A native Linux run remains the appropriate reference setup.

## Reproduce

From the repository root, run `uv sync --frozen --group dev`, `uv run --frozen python -m ridgepoint.baseline.download`, and start `uv run --frozen uvicorn ridgepoint.server.app:app --port 8000`. In a second terminal, use the exact trace/replay/analyze commands in `README.md`. Reusing the same trace file preserves token IDs and arrival times; changing the seed or overrides makes a new workload and must get its own manifest.
