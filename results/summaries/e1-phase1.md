# E1 Phase 1 measured summary

**Status:** exploratory Windows/WDDM component measurements, 2026-09-25. These are actual runs, not notebook estimates, but they are not valid headline serving results. We did not log a continuous clock/temperature trace or run native Linux.

## Machine and method

- NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9, 6,140.5 MiB CUDA-visible VRAM; driver 581.42.
- Windows 11 build 26200, Python 3.12.10, PyTorch 2.8.0+cu128, CUDA runtime 12.8.
- `uv run --frozen python -m bench.micro.run`, three independent process runs. Each full run uses 10 warmups and 30 timed samples per operation. CUDA events time GPU work; medians below are the median of the three run medians. Raw per-sample JSON is local under ignored `results/raw/`.
- Copy: 128 MiB source to 128 MiB destination, count both read and write traffic. BF16 square GEMM: count `2N³` arithmetic operations. Tiny-operation timing: 1,000 one-element PyTorch `add_` operations in each sample.

| Measurement | Median across runs | Run-median range | Unit |
| --- | ---: | ---: | --- |
| Device copy effective bandwidth | 148.9 | 148.8–149.0 | GB/s, decimal |
| BF16 GEMM 1024×1024 | 18.45 | 17.92–18.45 | TFLOP/s |
| BF16 GEMM 2048×2048 | 21.84 | 21.10–21.85 | TFLOP/s |
| BF16 GEMM 4096×4096 | 24.92 | 24.33–24.93 | TFLOP/s |
| Tiny add GPU elapsed per operation | 3.76 | 3.64–4.86 | µs |
| Tiny add host enqueue per operation | 3.81 | 3.66–4.90 | µs |

The GPU check separately performed BF16 16×16 matrix multiplication on CUDA and returned `16.0` for an all-ones input, with `cuda_available=true` and `bf16_supported=true`. After CUDA initialization, `mem_get_info` reported 5,073 MiB free. `pytest -q` passed 4 tests and `ruff check src bench tests` passed.

## Interpretation and limitations

The 128 MiB copy measurement is below the notebook's 192 GB/s specification estimate, as expected for a real operation on this laptop. The larger BF16 GEMM reaches higher TFLOP/s than smaller matrices, consistent with better GPU utilization; this is a measured shape effect, not evidence about Qwen serving yet. Tiny-operation times vary substantially across runs and include Python/PyTorch scheduling and the actual add kernel, so they are an overhead indicator rather than a pure CUDA launch constant.

Before/after `nvidia-smi` snapshots showed GPU temperature around 48–54 °C and SM clocks between 2,130 and 2,595 MHz. Snapshots cannot prove clock stability during a run. The runs were short and used Windows/WDDM with background desktop processes. No loaded model or KV pool was tested. Final performance claims require the full validity protocol in `docs/benchmarking.md`, ideally on native Linux.

Raw runs on this machine: `e1-20260925T083723Z.json`, `e1-20260925T083732Z.json`, and `e1-20260925T083734Z.json`. The smoke run is `e1-20260925T083717Z.json` and is excluded from the table.
