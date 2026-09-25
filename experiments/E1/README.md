# E1 — hardware characterization

**Question:** What GPU memory-copy bandwidth, BF16 matrix throughput, and tiny-kernel latency does this RTX 4050 actually achieve?

**Hypothesis:** The notebook's 192 GB/s bandwidth is a specification estimate; measured device copy and GEMM results will depend on tensor size, power, and clocks.

**Setup:** `uv run --frozen python -m bench.micro.run`. The raw JSON includes per-operation CUDA-event times, before/after clock snapshots, OS, PyTorch/CUDA versions, and GPU.

**Result:** Three full exploratory Windows runs measured median 148.9 GB/s device-copy bandwidth, 24.92 TFLOP/s BF16 4096 GEMM, and 3.76 µs per tiny GPU add. See `results/summaries/e1-phase1.md` for all sizes, run spread, environment, and limitations. Raw JSON stays in `results/raw/`.

**Interpretation, cost, next bottleneck:** Larger GEMMs use the GPU more effectively; tiny operations show several microseconds of overhead and notable variation. The experiment cost is only a few seconds of GPU time, but it does not measure Ridgepoint serving or prove sustained thermal performance. Next, load the model in Phase 2 and measure real latency and memory pressure.
