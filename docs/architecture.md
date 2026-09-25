# Architecture

Phase 1 implements the measurement and configuration foundation. The request path below is the planned design for later phases, not a claim of implemented serving.

```text
client → HTTP validation/tokenization → waiting queue → step scheduler
                                            ↓              ↓
                                      admission        model runtime
                                            ↓              ↓
                                      KV block tables ← K/V writes
                                                           ↓
client ← SSE token events ← sampler ← logits (last position)
```

The server owns client-facing IDs, timestamps, and disconnect signals. The engine owns request states and the step loop. The scheduler selects work at each step based on token budget, memory, and deadlines. The KV allocator accounts for every physical block. The runtime runs Qwen2 layers and sampling. Telemetry records client-visible timing plus GPU step traces. The benchmark harness sends the same open-loop request trace to Ridgepoint and reference backends.

In Phase 1, `ridgepoint.memory_budget` reads the pinned config and computes weight and KV sizes without loading weights. `ridgepoint.hardware` tests a BF16 CUDA operation. `bench.micro.run` measures hardware components and writes raw samples plus a manifest.
