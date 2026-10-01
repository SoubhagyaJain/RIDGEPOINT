# Architecture

Phase 3 retains the V0 path and adds an owned V1 runtime. V1 forms a short-window batch whose membership stays fixed until every request finishes. Continuous scheduling and paged KV remain later work.

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

## V1 request and cache path

```text
HTTP validation + capacity preflight → async V1 queue → 10 ms / four-request window
                                               ↓
                                    frozen batch in worker thread
                                               ↓
                         owned Qwen2 prefill → contiguous per-batch K/V
                                               ↓
                       incremental decode → sampler → SSE token events
                                               ↓
                             final usage, queue, batch and step timings
```

The V1 forward pass reads the same pinned BF16 weight tensors as V0 but does not call Hugging Face model `forward()` or `generate()` while serving. It applies RMSNorm, Q/K/V projections, RoPE, grouped-query SDPA, residuals and MLP, then computes logits only at the final needed position. The cache is `[layer, K/V, batch, KV head, position, head width]`, with logical lengths per request. Equal-length requests share prefill and decode calls. Mixed-length rows run in separate cohorts to preserve BF16 numerical agreement; they still belong to the same fixed batch and keep separate cache positions. A completed batch releases its contiguous cache.

The server keeps the V0 endpoint and SSE format. V1 adds batch size and prefill/decode timings to the final event. These are worker wall times: prefill includes cache allocation and synchronization, while decode includes sampling, text decoding, streaming, and synchronization. They are not isolated GPU kernel times. Preflight checks one request against a 512 MiB KV cap and 512 MiB free-memory reserve; the worker splits a collected batch if its predicted cache bytes exceed the current allowance. This is a capacity guard, not the Phase 5 memory admission and preemption policy.

## Later engine path

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
