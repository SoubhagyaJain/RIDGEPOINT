# Phase 3 V1: correctness and exploratory batching result

**Measured 2026-10-01 on a Windows RTX 4050 Laptop GPU.** This is an observed same-trace comparison, not a validated capacity result or a claim that V1 is faster. All nine final runs completed 24/24 requests without client errors, rejections, or prompt-count mismatches. V1 batches of up to four formed, but its latency and goodput were worse than V0 in these runs. A follow-up on the same day traced this to a decode-path bug and to CPU-core placement; see "Follow-up" below for the fix and the reruns.

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

## Follow-up, 2026-10-01: decode host-sync fix and CPU-core pinning

The table above is the original Phase 3 record and is unchanged. A later investigation
found two separate causes for it; `docs/learning/phase-03.md` section 9 explains them.

1. **Code.** The original decode path wrote K/V into the cache with boolean-mask
   indexing. That forced 113 GPU-to-host synchronizations per decode step. The fix
   writes with one integer slot and builds the RoPE table once per call.
2. **Host.** This laptop has four performance and four efficiency CPU cores. A process
   held on the efficiency cores ran every engine's decode step about 1.7× slower than
   the same process held on the performance cores.

All runs below replay the same trace (SHA-256 `12e01503…fada`), use the same SLOs and
analyzer, completed 24/24 with zero errors and rejections, and are still
**exploratory**: 24 requests per trial, Windows/WDDM, and most runs missed the 5 ms
send-lag P99 target. Method differences from the original table: engines were
interleaved within each trial instead of run back to back, each fresh server process
received one untimed 16-token warmup request, and `nvidia-smi` logged SM clock, power,
temperature and utilization every 500 ms.

### Same-process decode step (ms, median of per-repetition medians)

| Condition | HF forward, b=1 | V1 fixed, b=1 | V1 original, b=1 | HF forward, b=4 | V1 fixed, b=4 | V1 original, b=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unpinned, 8 reps | 60.41 | 55.36 | 92.12 | 48.90 | 49.50 | 82.20 |
| Efficiency cores (`F00`), 4 reps | 46.18 | 44.23 | — | 52.38 | 50.21 | — |
| Performance cores (`FF`), 4 reps | 27.82 | 26.61 | — | 28.81 | 26.85 | — |

Against Hugging Face in the same repetition, the original runtime (commit `c01762d`)
took 1.52× (b=1) and 1.49× (b=4) as long; the fixed runtime took 0.92× and 0.94×.
Greedy tokens matched the reference in every repetition. One original decode step
contained 113 `aten::nonzero` calls; one fixed step contains none. Raw samples:
`v1-decode-sync-20261001T053315Z.json`, `…053413Z.json`, `…053437Z.json`.

### Fixed runtime, all processes pinned to performance cores (`phase3-pin-*`)

| Engine | Trial | Complete | Good | Goodput req/s | TTFT P95 ms | TPOT P95 ms | Send lag P99 ms | Batch sizes (requests) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | 1 | 24/24 | 14 | 0.70 | 2,697.06 | 35.11 | 5.71 | 1:24 |
| V0 | 2 | 24/24 | 15 | 0.75 | 2,679.98 | 39.50 | 9.96 | 1:24 |
| V0 | 3 | 24/24 | 19 | 0.95 | 1,424.82 | 32.96 | 5.48 | 1:24 |
| V1 single | 1 | 24/24 | 21 | 1.05 | 1,109.59 | 27.58 | 5.15 | 1:24 |
| V1 single | 2 | 24/24 | 22 | 1.10 | 998.48 | 28.51 | 1.74 | 1:24 |
| V1 single | 3 | 24/24 | 21 | 1.05 | 1,078.65 | 28.40 | 1.31 | 1:24 |
| V1 batch | 1 | 24/24 | 24 | 1.20 | 821.37 | 50.90 | 4.47 | 1:9, 2:6, 3:9 |
| V1 batch | 2 | 24/24 | 24 | 1.20 | 474.37 | 31.15 | 16.20 | 1:10, 2:8, 3:6 |
| V1 batch | 3 | 24/24 | 20 | 1.00 | 829.46 | 108.87 | 10.42 | 1:10, 2:10, 4:4 |

Source-tree SHA-256 `76d638d5…3ba3` in all nine manifests. GPU trace while busy: mean
SM clock 2,599 MHz (minimum 2,130), mean power 41.0 W, maximum 60 °C.

### Original runtime, same pinned method (`phase3-pinold-*`)

| Engine | Trial | Complete | Good | Goodput req/s | TTFT P95 ms | TPOT P95 ms | Send lag P99 ms | Batch sizes (requests) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V1 single | 1 | 24/24 | 3 | 0.15 | 9,719.51 | 72.51 | 9.07 | 1:24 |
| V1 single | 2 | 24/24 | 3 | 0.15 | 8,623.88 | 64.56 | 12.60 | 1:24 |
| V1 single | 3 | 24/24 | 3 | 0.15 | 8,562.93 | 63.00 | 5.28 | 1:24 |
| V1 batch | 1 | 24/24 | 11 | 0.55 | 2,101.09 | 114.52 | 41.16 | 1:3, 2:6, 3:3, 4:12 |
| V1 batch | 2 | 24/24 | 24 | 1.20 | 927.95 | 63.28 | 8.62 | 1:6, 2:4, 3:6, 4:8 |
| V1 batch | 3 | 24/24 | 24 | 1.20 | 904.07 | 73.23 | 9.25 | 1:6, 2:4, 3:6, 4:8 |

Source-tree SHA-256 `5e12792c…1bf8`, the same bytes as the original Phase 3 runs.

### Fixed runtime, unpinned (`phase3-fix-*`)

| Engine | Trial | Complete | Good | Goodput req/s | TTFT P95 ms | TPOT P95 ms | Send lag P99 ms | Batch sizes (requests) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | 1 | 24/24 | 5 | 0.25 | 8,593.29 | 58.73 | 11.61 | 1:24 |
| V0 | 2 | 24/24 | 16 | 0.80 | 2,008.80 | 32.52 | 7.23 | 1:24 |
| V0 | 3 | 24/24 | 4 | 0.20 | 13,894.94 | 60.34 | 4.67 | 1:24 |
| V1 single | 1 | 24/24 | 22 | 1.10 | 1,103.83 | 27.08 | 11.99 | 1:24 |
| V1 single | 2 | 24/24 | 19 | 0.95 | 1,236.89 | 28.51 | 4.15 | 1:24 |
| V1 single | 3 | 24/24 | 4 | 0.20 | 6,575.62 | 50.57 | 10.24 | 1:24 |
| V1 batch | 1 | 24/24 | 24 | 1.20 | 497.33 | 28.89 | 7.47 | 1:11, 2:10, 3:3 |
| V1 batch | 2 | 24/24 | 13 | 0.65 | 1,140.56 | 102.33 | 9.70 | 1:4, 2:2, 3:6, 4:12 |
| V1 batch | 3 | 24/24 | 24 | 1.20 | 798.44 | 51.33 | 5.24 | 1:7, 2:6, 3:3, 4:8 |

Unpinned results swing by run for every engine, V0 included. In the four slowest runs
(V0 1 and 3, V1 single 3, V1 batch 2) the GPU averaged 35–49% utilization and 25–34 W
while busy; in the three fastest single-request runs it averaged 76–82% and 43–45 W.
The GPU was being fed work more slowly, not running hot or out of memory (peak 60 °C,
4,507 MiB of 6,141 MiB).

### What the follow-up supports, and what it does not

- Pinned and fixed, every V1 run had higher goodput than every V0 run (lowest V1 1.00,
  highest V0 0.95 req/s). With the same pinned method, the fix moved V1 single from
  3/24 good requests in all three trials to 21–22/24.
- This is not a validated speedup. Three trials of 24 requests cannot support tail
  percentiles, and only 3 of the 9 pinned fixed runs met the 5 ms send-lag target.
- The trace cannot separate V1 batch from V1 single or measure the fix's effect on V1
  batch. It offers 1.2 req/s, and 24/24 good requests already equals that ceiling; the
  original runtime also reached it in two pinned batch trials. A higher offered rate
  is needed.
- V1 batch TPOT P95 (31–109 ms) was worse and less stable than V1 single (about
  28 ms). The cause was not investigated.
- The 529–536 ms TPOT in the original table was never reproduced. The original
  runtime under pinning showed 63–73 ms. Unpinned core placement is the likely missing
  factor, but that is an inference, not a measurement.
- The core mapping (logical CPUs 0–7 performance, 8–11 efficiency) follows Intel's
  usual numbering for the i5-13420H and was not independently verified. The affinity
  comparison is one run of four repetitions per mask.

## Reproduce

From the repository root, after the locked setup and pinned model download:

```powershell
uv run --frozen python -m bench.workloads.generate --workload configs/workloads/chat.yaml --output results/raw/phase3-overlap.jsonl --duration-s 20 --rate-rps 2 --warmup-s 0 --max-requests 24 --prompt-tokens 64 --output-tokens 16
```

Start one server at a time: V0 with `uv run --frozen uvicorn ridgepoint.server.app:app --port 8000`; V1 batch with `uv run --frozen uvicorn ridgepoint.server.v1:app --port 8002`; V1 single with `$env:RIDGEPOINT_V1_CONFIG='configs/engine/v1-single.yaml'` before the V1 command. For each configuration, replay three times to separate `results/raw/phase3-final-<engine>-<trial>.jsonl` files using `uv run --frozen python -m bench.client.replay results/raw/phase3-overlap.jsonl --url <server-url>/v1/completions --output <raw-file> --engine-config <engine-config> --timeout-s 120`. Run `uv run --frozen python experiments/V1-batching/analyze.py` to verify hashes and recompute the table. Raw JSONL and manifests remain Git-ignored.

For the follow-up, the same replay command wrote `phase3-fix-*`, `phase3-pin-*` and `phase3-pinold-*` files; pass the prefix to the analyzer, for example `uv run --frozen python experiments/V1-batching/analyze.py phase3-pin`. Pinned runs started the server and client through `cmd /c "start /b /wait /affinity FF ..."`. The original-runtime runs used `src/ridgepoint/runtime/qwen2.py` from commit `c01762d` and cover the two V1 configurations only, so the analyzer's three-engine loop does not apply to that prefix. The same-process table comes from `experiments/V1-decode-sync/step_compare.py`.
