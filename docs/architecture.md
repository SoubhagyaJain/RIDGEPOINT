# Architecture

Phase 2 implements the V0 path. The scheduler, owned runtime, and paged KV portion of the lower diagram remain planned for later phases.

```text
HTTP POST → Pydantic validation → prompt token IDs → per-request worker
                                               ↓
                               serialized Hugging Face generate()
                                               ↓
                           token-ID streamer → SSE token events → client
                                               ↓
                              final usage/timing SSE event
```

The server gives every request an ID, checks prompt and output limits, and starts a worker. V0 holds a single model lock around `generate()`, so overlapping requests wait in order of lock acquisition. This simple queue is intentional baseline behavior; it has no memory-aware admission or batching. The streamer emits one token event per model token ID. The final event reports prompt/output token counts and server-side queue, first-token-from-model-start, and generation durations. The client records its own send and arrival timestamps for latency metrics.

The harness uses a pinned Qwen tokenizer and a seeded corpus to build exact-length token-ID traces. `bench.client.replay` schedules all sends from one monotonic origin, independently of completions. It stores raw request timestamps and a manifest. `bench.analyze.metrics` calculates per-request qualification and aggregates from those records. The null backend implements the same HTTP contract with async delays, so the client can be tested without a GPU.

## Planned engine path

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

In Phase 1, `ridgepoint.memory_budget` read the pinned config and computed weight and KV sizes without loading weights. `ridgepoint.hardware` tested a BF16 CUDA operation. `bench.micro.run` measured hardware components and wrote raw samples plus a manifest.
