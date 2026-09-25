# Engineering report (living document)

## Phase 1: establish ground truth

The current machine is a Windows laptop with an RTX 4050 Laptop GPU (6,140.5 MiB CUDA-visible VRAM). The BF16 CUDA check passes. Three exploratory E1 runs measured a 148.9 GB/s median 128 MiB device copy, 24.92 TFLOP/s for 4096 BF16 GEMM, and 3.76 µs per tiny PyTorch GPU add. The results are component yardsticks, not serving throughput. See `results/summaries/e1-phase1.md` for methods and run spread.

The official Qwen2.5-1.5B-Instruct config implies 1,543,714,304 architectural parameters, about 3.087 GB BF16 weight storage, and 28 KiB of KV per token. A 1 GiB KV pool would hold 37,440 token slots in complete 16-token blocks. That pool is provisional because model loading and activation peaks have not been measured.

The next result belongs to Phase 2: client-visible latency and goodput for a real baseline server on seeded traces. Comparisons will use the workload and validity contract in `docs/benchmarking.md`.
