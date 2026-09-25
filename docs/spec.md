# Ridgepoint specification

## Goal

Build a single-GPU Qwen2.5-1.5B-Instruct inference engine that teaches and measures each layer of the system: model runtime, batching, KV memory, scheduling, overload handling, and streaming HTTP. The lab notebook is a design reference. The phase gates and teaching requirements in the user brief govern progress.

## Constraints

- One evolving `src/ridgepoint` codebase with named engine configurations.
- RTX 4050 laptop GPU with 6 GiB VRAM is the present development target. GPU numbers from Windows are exploratory; final headline runs need native Linux when available.
- BF16 model weights; model weights and raw benchmark data never enter Git.
- Code correctness precedes performance claims. Every comparison replays the same trace and reports exact hardware, model, versions, and settings.
- Allocator state must partition all blocks exactly: `free + in_use + evictable == total`; refcounts are nonnegative.
- Goodput counts individual requests meeting both client-observed TTFT and per-request TPOT thresholds. Rejected requests are reported separately.

## Definition of done

Each phase passes its gate in `docs/plan.md`, has runnable code and verification, an honest learning report, and a commit. The project is done after Phase 7 reproduces or documents the obstacle to a clean-clone result and explains the remaining performance gap.
