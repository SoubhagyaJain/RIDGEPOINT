# V0 baseline experiment

**Question:** Can a simple BF16 Hugging Face server stream real Qwen tokens and produce a replayable client-observed baseline?

**Hypothesis:** A single serialized `generate()` worker should be correct and easy to measure, but overlapping arrivals will inflate TTFT before TPOT.

**Setup:** RTX 4050 Windows host, pinned Qwen model and exact six-request seed-1234 trace, three repeats. Settings, hashes, SLOs, and limitations are in `results/summaries/phase2-v0.md`.

**Result:** All 18 real-model requests completed with exact prompt and output token counts. Each trial's goodput was 0.25 req/s at 0.3 offered req/s; TTFT P95 ranged from 999 to 1,018 ms. This is a smoke result only.

**Interpretation and cost:** The baseline demonstrates a real streaming request path and exposes queueing from the one-at-a-time model lock. It offers no batching or overload control. The small Windows runs cannot establish capacity or a vLLM performance ratio.

**Next bottleneck:** Phase 3 owns the Qwen2 forward pass and batch formation; Phase 4 changes batch membership between steps.
