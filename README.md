# Ridgepoint

Ridgepoint is a seven-phase lab for building a single-GPU LLM inference engine from the scheduler down. **Current state: Phase 1 hardware characterization.** Serving and model weights begin in Phase 2; no LLM request endpoint exists yet.

The available machine is a Windows laptop with an NVIDIA RTX 4050 Laptop GPU (6 GiB). The primary planned model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), using BF16. Its pinned `config.json` is in `configs/models/`; model weights are not stored here.

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

## Read the work

- [Specification](docs/spec.md), [seven-phase plan](docs/plan.md), and [current status](docs/status.md)
- [Phase 1 teaching report](docs/learning/phase-01.md)
- [Memory budget](docs/memory-budget.md) and [benchmark contract](docs/benchmarking.md)
- [Architecture](docs/architecture.md) and [E1 experiment](experiments/E1/README.md)

Numbers in the notebook are estimates until independently measured. The Phase 1 GPU measurements are exploratory on Windows/WDDM. The final benchmark comparison is planned for native Linux when available and will use identical workload traces for Ridgepoint and reference backends.
