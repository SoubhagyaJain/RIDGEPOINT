# Phase 1 — environment and hardware ground truth

## 1. What we built

Phase 1 builds a reproducible Python environment, a CUDA health check, a config-derived memory calculator, and executable E1 component benchmarks. It defines workloads and measurement rules before serving code exists. No Qwen weights were downloaded.

Exact files added or changed in this phase:

- Root: `AGENTS.md`, `README.md`, `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, `.gitattributes`, and the provided `Ridgepoint-Lab-Notebook.pdf`.
- Documentation: `docs/spec.md`, `docs/plan.md`, `docs/status.md`, `docs/architecture.md`, `docs/memory-budget.md`, `docs/benchmarking.md`, `docs/report.md`, `docs/adr/0001-windows-development.md`, `docs/learning/phase-01.md`.
- Configurations: `configs/models/qwen2.5-1.5b-instruct.config.json`, `configs/engine/v0.yaml`, and `configs/workloads/{chat,prefill,decode,multiturn,burst,overload}.yaml`.
- Code and tests: `src/ridgepoint/__init__.py`, `src/ridgepoint/hardware.py`, `src/ridgepoint/memory_budget.py`, `bench/__init__.py`, `bench/micro/__init__.py`, `bench/micro/run.py`, `tests/unit/test_memory_budget.py`.
- Experiment/result structure: `experiments/E1/README.md`, `results/raw/.gitkeep`, `results/summaries/e1-phase1.md`, `profiling/FINDINGS.md`.
- Future-phase directory markers: `.gitkeep` in `src/ridgepoint/{baseline,server,engine,scheduler,kvcache,runtime,kernels,perfmodel,quant,telemetry}/`, `bench/{workloads,client,backends,analyze}/`, and `tests/{numerics,integration,stress}/`. These preserve the requested layout in a clean clone; no future-phase code is claimed.

The remaining directories in the requested layout exist for future phases; their engines, server, scheduler, and numerical tests are not yet implemented.

## 2. Concepts from scratch

**A model config is a shape recipe.** It states how wide the vectors are and how many repeated layers the model has. It does not contain learned weight values. For Qwen2.5-1.5B, 1536 hidden values split across 12 attention heads give 128 values per head.

**BF16 is a 16-bit number format.** Each stored value takes 2 bytes. Roughly 1.54 billion parameters therefore need roughly 3.09 billion bytes before runtime overhead. `GB` means billions of bytes; `GiB` means groups of 1,073,741,824 bytes. Confusing them can overpromise GPU memory.

**Grouped-query attention (GQA) saves KV memory.** Twelve query heads share only two key/value heads. For one token in one layer, we store `2 (key and value) × 2 heads × 128 values × 2 bytes = 1024 bytes`. There are 28 layers, so one token needs 28,672 bytes, or 28 KiB. A 16-token block needs 448 KiB.

**GPU memory bandwidth and compute throughput are different limits.** Copy bandwidth asks how many bytes per second move between GPU memory and GPU processors. Matrix multiplication throughput asks how many arithmetic operations per second the GPU can perform. A model's decode step often rereads weights, making bandwidth crucial; prefill usually has larger matrix operations and can use more compute.

**Kernel launch overhead is the cost of asking the GPU to do one operation.** A tiny operation can spend more time being scheduled than doing arithmetic. Our benchmark measures a one-element `add_` repeatedly, showing per-operation GPU elapsed time and host enqueue time. Those include PyTorch overhead; they are not a pure hardware launch constant.

**A service-level objective (SLO) is a latency target.** TTFT is time until the first output token. TPOT averages the gaps after the first token for one request. For a request with first token at 0.6 s and last token at 1.2 s after 11 tokens, TPOT is `(1.2 - 0.6)/10 = 60 ms`. That request meets the 1 s TTFT and 75 ms TPOT targets. Goodput counts such individually qualifying completed requests per second; rejections remain a separate number.

## 3. How the code works

`qwen2_memory(config, dtype_bytes=2, block_size=16)` receives a parsed model config and returns parameter count, BF16 weight bytes, head geometry, and KV bytes per token. It accounts for Q/K/V biases, per-layer/final RMSNorm weights, and tied embeddings. `ModelMemory.tokens_for_pool(pool_bytes)` rounds down to whole blocks. `available_pool_bytes(...)` subtracts weight and reserve estimates from currently free bytes, but does not claim an allocation will succeed.

`ridgepoint.hardware.main()` prints OS, Python, PyTorch/CUDA, GPU, driver snapshot, and live CUDA free memory. It multiplies two 16×16 BF16 matrices on the GPU, synchronizes, and checks an expected output value of 16. This verifies a real CUDA BF16 operation rather than trusting device discovery alone.

`bench.micro.run.run(smoke=False)` allocates tensors on the GPU and times three experiments with CUDA events after warmup: a large device-to-device copy, square BF16 matrix multiplies, and 200/1000 tiny `add_` launches. The copy counts one read plus one write per operation; GEMM counts approximately `2N³` FLOPs. The result contains every timing sample and a machine snapshot. `main()` writes JSON under ignored `results/raw/` and prints medians. The smoke mode uses smaller shapes and fewer repeats to catch setup errors quickly.

There is no incoming HTTP request path in Phase 1. The future path is described in `docs/architecture.md`: HTTP validation → waiting queue → scheduler → runtime/KV cache → sampler → streaming response. We build its hardware and measurement foundation first so later claims have a physical reference.

## 4. How to reproduce and debug

From the repository root with an NVIDIA driver and Python 3.12:

```powershell
uv sync --frozen --group dev
uv run --frozen python -m ridgepoint.hardware
uv run --frozen pytest -q
uv run --frozen ruff check src bench tests
uv run --frozen python -m ridgepoint.memory_budget configs/models/qwen2.5-1.5b-instruct.config.json --pool-gib 1.0
uv run --frozen python -m bench.micro.run --smoke
uv run --frozen python -m bench.micro.run
```

The hardware JSON should contain `"cuda_available": true`, `"bf16_supported": true`, and `"bf16_matmul_result": 16.0`. The memory calculator should show `kv_bytes_per_token: 28672`, `weight_bytes: 3087428608`, and `usable_tokens: 37440` for a 1 GiB pool. Tests should pass and lint should print `All checks passed!`. The E1 commands should print copy GB/s, GEMM TFLOP/s, and tiny-operation microseconds, plus the raw JSON path.

If `cuda_available` is false, inspect `nvidia-smi` and `uv run --frozen python -c "import torch; print(torch.__version__, torch.version.cuda)"`. A CPU-only PyTorch wheel or unavailable driver requires reinstalling the locked CUDA environment or fixing the driver. If an E1 result changes unexpectedly, check its raw manifest for GPU clock, temperature, competing applications, and free memory. Re-run only under comparable power and thermal conditions.

## 5. Proof: actual checks and limits

`uv sync --frozen --group dev` installed the locked environment with PyTorch 2.8.0+cu128. The GPU check printed `cuda_available=true`, `bf16_supported=true`, and `bf16_matmul_result=16.0` on the RTX 4050. `pytest -q` passed **4 tests**; `ruff check src bench tests` printed `All checks passed!`. The memory calculator returned **1,543,714,304 parameters**, **3,087,428,608 BF16 weight bytes**, **28,672 KV bytes/token**, and **37,440 usable tokens** in a hypothetical 1 GiB block pool.

One smoke E1 run and three full E1 runs completed. Across the three full runs, the median of run medians was **148.9 GB/s** for a 128 MiB device copy, **24.92 TFLOP/s** for a 4096×4096 BF16 GEMM, and **3.76 µs** for a tiny PyTorch GPU add. Full sample ranges, method, and manifests are in `results/summaries/e1-phase1.md`; raw JSON is local under ignored `results/raw/`.

The tests verify arithmetic and invalid geometry, not model output or successful model weight loading. E1 isolates GPU components; it does not prove server throughput, sustained thermals, or a final native-Linux comparison. We lack a continuous clock trace, so E1 is exploratory under the benchmark validity rules.

## 6. Problems and remaining limitations

- The first `uv sync` failed because `README.md` had not yet been created. Adding the actual README resolved the package-build requirement; the frozen sync was retried.
- The initial GPU check emitted a missing-NumPy warning, though BF16 computation passed. NumPy was added to the locked dependencies and the check then ran cleanly.
- Ruff caught closure variables later deleted by the benchmark function. Replacing `del` with reassignment preserved cleanup and made the code lint clean.
- The first `.gitignore` pattern `models/` accidentally hid the pinned `configs/models/` file. Anchoring it as `/models/` allowed the small config into Git while continuing to exclude a root model-weight cache.
- The host is Windows/WDDM. The notebook's native Linux requirement for final GPU benchmarking is unavailable on this boot. Windows E1 results are labeled exploratory.
- The starting Git repository had no commits, and `origin/main` was absent locally. Phase 1 creates the first local commit; no push was requested.
- The PDF's memory and performance figures are estimates. We recomputed Qwen geometry from the official config and will use measured GPU numbers only where actually observed.
- Phase 1 has no loaded model, so peak activation allocation and a safe full-load KV pool remain unverified. Phase 2/3 must measure them before committing an engine pool size.

## 7. Explain it back

**Interview-sized explanation:** “I started by checking the real GPU and locking a CUDA Python environment. I used the model config to calculate BF16 weight storage and KV bytes per token, then measured GPU copy, matrix multiply, and tiny-kernel costs. I keep those measurements separate from model-serving results and use them to size memory conservatively and explain later bottlenecks.”

Five questions:

1. Why can a 1.5B model need more than 3 GB of VRAM when BF16 weights are around 3 GB?
2. How does 12 query heads but 2 KV heads affect KV storage?
3. Why count both a read and a write in the copy bandwidth calculation?
4. Why can a small GEMM report much lower TFLOP/s than a larger one?
5. Why can a request pass the TPOT SLO yet still feel bad to a user?

Answer key:

1. The runtime also needs CUDA context, temporary activations, logits, KV cache, allocator overhead, and safety space.
2. Six queries share each KV head; the cache stores only two K and two V head vectors per layer, so it is smaller than a 12-KV-head cache.
3. The memory system transfers source bytes out of memory and destination bytes back into memory.
4. Small work underuses the GPU and fixed overhead is a larger fraction of elapsed time.
5. TPOT averages gaps after the first token and can hide a long individual stall; TTFT may also be too high.

## 8. Phase gate

- [x] Locked `uv` setup works cleanly on this host.
- [x] GPU BF16 check passes.
- [x] Memory copy, BF16 GEMM, and tiny-kernel benchmarks run and retain raw samples.
- [x] Weight, KV/token, block, and provisional pool calculations are reproducible from the official config.
- [x] Initial SLOs, six workload shapes, and individual-request goodput are defined.
- [x] Verification results, this report, and `docs/status.md` are final; Phase 1 commit exists.
