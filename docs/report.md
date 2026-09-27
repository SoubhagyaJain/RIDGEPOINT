# Engineering report (living document)

## Phase 1: establish ground truth

The current machine is a Windows laptop with an RTX 4050 Laptop GPU (6,140.5 MiB CUDA-visible VRAM). The BF16 CUDA check passes. Three exploratory E1 runs measured a 148.9 GB/s median 128 MiB device copy, 24.92 TFLOP/s for 4096 BF16 GEMM, and 3.76 µs per tiny PyTorch GPU add. The results are component yardsticks, not serving throughput. See `results/summaries/e1-phase1.md` for methods and run spread.

The official Qwen2.5-1.5B-Instruct config implies 1,543,714,304 architectural parameters, about 3.087 GB BF16 weight storage, and 28 KiB of KV per token. A 1 GiB KV pool would hold 37,440 token slots in complete 16-token blocks. That pool is provisional because model loading and activation peaks have not been measured.

The next result belongs to Phase 2: client-visible latency and goodput for a real baseline server on seeded traces. Comparisons will use the workload and validity contract in `docs/benchmarking.md`.

## Phase 2: real V0 and a replayable harness

V0 now streams real Qwen2.5-1.5B-Instruct tokens over SSE, with request IDs, validation, a serialized Hugging Face `generate()` worker, and final usage/timing events. A seeded token-ID trace, open-loop streaming client, and per-request analyzer make it possible to measure TTFT, ITL, TPOT, E2E, throughput, goodput, errors, and rejections from raw records. The null backend checks the harness without a GPU.

Three replays of one six-request smoke trace completed without errors or prompt-count mismatch. Each trial's goodput was 0.25 req/s at 0.3 offered req/s, with TTFT P95 spanning roughly 1.00–1.02 s and TPOT P95 near 27 ms. This is a reproducible path test, not a capacity number: the sample is small, Windows scheduling introduces lag, and there is no continuous clock trace. A null check measured 500 simultaneously open streams and completed all requests but missed the 5 ms client send-lag target (16.91 ms P99). See `results/summaries/phase2-v0.md` for exact setup and limits.

vLLM comparison is blocked on this machine: vLLM CUDA has no supported native Windows installation, and the WSL2 distribution's virtual disk is missing. A working native Linux environment is needed for the comparison called for by the notebook.
