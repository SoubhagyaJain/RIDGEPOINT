# Phase 3 V1: correctness and exploratory batching result

**Measured 2026-10-01 on a Windows RTX 4050 Laptop GPU.** This is an observed same-trace comparison, not a validated capacity result or a claim that V1 is faster. All nine final runs completed 24/24 requests without client errors, rejections, or prompt-count mismatches. V1 batches of up to four formed, but its latency and goodput were worse than V0 in these runs.

## Exact comparison

- Model: pinned `Qwen/Qwen2.5-1.5B-Instruct` BF16, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, verified safetensors SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
- Hardware/software: NVIDIA GeForce RTX 4050 Laptop GPU, driver 581.42; Windows 11 build 26200; Python 3.12.10; PyTorch 2.8.0+cu128; Transformers 4.57.6.
- Trace: `W-chat`, seed 1234, 20 s scheduled window, requested Poisson rate 2 req/s, capped at 24 requests, fixed 64 prompt and 16 output token IDs, greedy, `ignore_eos=true`, warmup 0. The resulting offered rate is **24/20 = 1.2 requests/s**. The same trace file was sent to every configuration. Trace SHA-256: `12e01503ee4f7dedbd728acde3fe319c18632efd7bbf21f3952d2475a862fada`.
- SLO: client-observed TTFT at most 1,000 ms and per-request TPOT at most 75 ms. Goodput counts qualifying completed requests divided by 20 s; rejections and errors are reported separately.
- Engines: V0 serialized Hugging Face `generate()`; V1-single used `max_batch_size=1`, window 0; V1-batch used `max_batch_size=4`, window 10 ms. Each server ran in a separate process; model weights were not duplicated on the GPU. All final manifests record source-tree SHA-256 `5e12792c63ef2f6b5ccd5ae1e6c7b42f19f00a92d199a579f5673f98d4e71bf8` and the same trace SHA.

The `Batch sizes` column counts **requests served in batches of each size**, not the number of batches. For example, `4:20` means 20 requests were served in five four-request batches.

| Engine | Trial | Complete | Good | Goodput req/s | TTFT P95 ms | TPOT P95 ms | Send lag P99 ms | Batch sizes (requests) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | 1 | 24/24 | 10 | 0.50 | 3,603.15 | 36.60 | 12.87 | 1:24 |
| V0 | 2 | 24/24 | 13 | 0.65 | 4,209.02 | 37.90 | 3.60 | 1:24 |
| V0 | 3 | 24/24 | 12 | 0.60 | 5,934.79 | 37.56 | 12.52 | 1:24 |
| V1 single | 1 | 24/24 | 0 | 0.00 | 62,431.51 | 528.87 | 10.30 | 1:24 |
| V1 single | 2 | 24/24 | 0 | 0.00 | 60,746.20 | 536.15 | 107.97 | 1:24 |
| V1 single | 3 | 24/24 | 1 | 0.05 | 13,806.85 | 76.12 | 4.57 | 1:24 |
| V1 batch | 1 | 24/24 | 0 | 0.00 | 12,262.39 | 246.91 | 120.95 | 2:4, 4:20 |
| V1 batch | 2 | 24/24 | 0 | 0.00 | 10,264.75 | 220.15 | 99.67 | 1:1, 3:3, 4:20 |
| V1 batch | 3 | 24/24 | 0 | 0.00 | 13,199.52 | 297.51 | 98.41 | 1:1, 3:3, 4:20 |

## What this proves and what it does not

V1 serves the real model, performs its own forward pass and K/V writes, and emits the established SSE events. Numerical tests use the **same loaded model weights** as the Hugging Face reference: single, equal-length batched, and mixed-length cohort prefill/decode logits meet `atol=0.05, rtol=0.01`; the fixed greedy sequence matches exactly. The HTTP integration test observes a batch of two. The runs above show that the 10 ms/four-request window formed batches as configured.

These runs do **not** establish a performance improvement. V1's Python/PyTorch path was slower than V0 here, and V1 timing varied dramatically between trials. Seven of nine final runs missed the 5 ms generator-lag P99 target. There are only 24 requests per trial, no continuous SM clock/temperature trace, and no native Linux comparison. The descriptive P95 values in this table are not stable tail-latency claims. The observed TTFT difference between V1 single and V1 batch is consistent with less serial queueing, but the invalid run conditions and temporal variation prevent a causal speedup claim. No average of invalid trial P95s is used as a headline number.

## Reproduce

From the repository root, after the locked setup and pinned model download:

```powershell
uv run --frozen python -m bench.workloads.generate --workload configs/workloads/chat.yaml --output results/raw/phase3-overlap.jsonl --duration-s 20 --rate-rps 2 --warmup-s 0 --max-requests 24 --prompt-tokens 64 --output-tokens 16
```

Start one server at a time: V0 with `uv run --frozen uvicorn ridgepoint.server.app:app --port 8000`; V1 batch with `uv run --frozen uvicorn ridgepoint.server.v1:app --port 8002`; V1 single with `$env:RIDGEPOINT_V1_CONFIG='configs/engine/v1-single.yaml'` before the V1 command. For each configuration, replay three times to separate `results/raw/phase3-final-<engine>-<trial>.jsonl` files using `uv run --frozen python -m bench.client.replay results/raw/phase3-overlap.jsonl --url <server-url>/v1/completions --output <raw-file> --engine-config <engine-config> --timeout-s 120`. Run `uv run --frozen python experiments/V1-batching/analyze.py` to verify hashes and recompute the table. Raw JSONL and manifests remain Git-ignored.
