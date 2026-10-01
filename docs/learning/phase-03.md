# Phase 3 — owned Qwen2 forward pass and static batching

## 1. What we built

V1 serves the pinned Qwen2.5-1.5B-Instruct BF16 weights through Ridgepoint's own layer loop. It owns prefill, incremental decode, contiguous K/V writes, greedy and temperature/top-p sampling, and a fixed-membership batch queue. The existing FastAPI endpoint still validates requests and streams one SSE event per token. A 10 ms window collects up to four requests; new arrivals wait for the next batch. This phase does not implement continuous batching or paged allocation.

Files added or changed include `src/ridgepoint/runtime/{qwen2,backend,sampler}.py`, `src/ridgepoint/server/{app,v1}.py`, `configs/engine/{v1,v1-single}.yaml`, GPU numerical and CPU sampler/cache tests, `experiments/V1-batching/`, this report, the Phase 3 result summary, and setup/architecture/status documentation. The two pre-existing handwritten-note PDFs remain untracked and untouched. Raw runs and model weights remain outside Git.

## 2. Concepts from scratch

**The weights and the forward pass are different things.** The safetensors file contains learned matrices. A forward pass specifies how input token IDs flow through embedding, normalization, attention and MLP layers to produce next-token logits. V1 loads the pinned weights with Transformers, then calls none of its model `forward()` or `generate()` methods while serving. Numerical tests call them only as a reference using the same loaded weight object, avoiding a second roughly 3 GB GPU copy.

**A Qwen2 layer has two residual branches.** The first normalizes hidden values with RMSNorm, projects query/key/value vectors, rotates query and key with RoPE, computes attention, projects it back, then adds the original hidden values. The second normalizes again, runs a gated SiLU MLP, and adds another residual. After 28 layers, a final RMSNorm and tied embedding/output matrix give logits for the last needed position. The model has 12 query heads but only two key/value heads; six query heads share each KV head.

**RoPE puts token position into Q and K.** It calculates sine/cosine rotations from the model's `rope_theta` and each token's position. V1 computes the frequencies in FP32, then casts sine/cosine to BF16 before rotating, matching the reference path. A one-position error can change every later attention result, so the cache stores a logical length for each request and decode writes at that exact position.

**Contiguous KV is a batch-owned tensor.** Its shape is `[layer, key/value, batch, KV head, position, head width]`. One Qwen token across all layers requires `2 × 28 × 2 × 128 × 2 = 28,672` bytes. A four-request batch with capacity 80 needs `4 × 80 × 28,672 = 9,175,040` bytes by architecture arithmetic. The actual tensor is allocated per batch; these formulas are sizing calculations, not observations of peak runtime memory. V1 caps predicted KV at 512 MiB and reserves 512 MiB of free/reclaimable GPU memory. It splits a collected batch that exceeds the allowance and rejects an individually unserviceable request with HTTP 503 before SSE begins. This guard is not Phase 5's block allocator or admission policy.

**Static batching freezes membership.** The worker waits at most 10 ms after its first queued request, takes up to four, and serves those requests until done. Equal-length prompts share batched prefill and decode calls. For differing lengths, V1 uses separate prefill/decode cohorts so each row keeps its own cache position and BF16 reference agreement. The cohort still retains one request lifecycle and SSE stream per client. Newly arriving requests cannot join the running batch; that is Phase 4's continuous-batching task.

**Numerical equivalence comes before throughput.** BF16 rounds after many operations, so different attention kernels can produce different logits even with the same weights. We compare entire next-token logit vectors against Transformers at `atol=0.05, rtol=0.01` and require exact greedy token IDs on fixed fixtures. The test includes one-token, short, and 64-token prompts; equal-length and mixed-length batches; and incremental cache-position updates. Stochastic sampling is exercised through HTTP but is not expected to return identical random tokens across independent runs.

## 3. How the code works

`Qwen2Runtime.prefill()` allocates the contiguous cache, embeds prompt IDs, runs the 28 owned layers, and returns final-position logits. Its layer path performs RMSNorm, Q/K/V projection, RoPE, grouped-query scaled dot-product attention, output projection, and gated MLP. It writes each layer's rotated K and V into cache slots. `decode()` reads the logical lengths, writes the new token's K/V at those positions, attends to the valid prefix, increments lengths, and returns next-token logits. `sample()` chooses argmax for temperature zero or temperature/top-p sampling otherwise.

`V1Backend` inherits pinned loading and prompt validation from V0, then creates `Qwen2Runtime` over the loaded model's tensors. The server calls `preflight()` before sending HTTP response headers. Accepted requests enter an async queue. A single worker collects a frozen group, checks its aggregate KV estimate, and runs GPU work in a thread so HTTP streams can continue. Each token is emitted to the server's SSE queue; the final event adds batch size, queue time, prefill time, decode time, completion count, and finish reason. Aborted requests are skipped between steps, although the complete disconnect/resource lifecycle remains Phase 4 work.

The V1 prefill/decode timing fields are wall-clock spans for the worker. Prefill includes cache allocation and a CUDA synchronization. Decode also includes sampling, tokenizer work, token emission, and synchronization; neither field is a pure GPU kernel measurement.

One request therefore goes: HTTP validation → exact token IDs → KV headroom check → 10 ms batch window → owned Qwen2 prefill and decode → sampler → token SSE events → final usage/timing event. V0 still uses its original Hugging Face path and remains a separate entrypoint.

## 4. How to reproduce and debug

From the repository root on a CUDA-capable machine with the locked Python 3.12 environment:

```powershell
uv sync --frozen --group dev
uv run --frozen python -m ridgepoint.baseline.download
uv run --frozen pytest -q
uv run --frozen ruff check src bench tests
uv run --frozen uvicorn ridgepoint.server.v1:app --host 127.0.0.1 --port 8002
```

In another terminal, `uv run --frozen python -m bench.client.demo --url http://127.0.0.1:8002/v1/completions --max-tokens 16` should print a request ID, text pieces, 16 completion tokens, and V1 timing fields. For the exact comparison trace and nine-run analyzer, follow `results/summaries/phase3-v1.md`. Set `$env:RIDGEPOINT_V1_CONFIG='configs/engine/v1-single.yaml'` before launching V1 to use one request per batch. Run V0 separately on port 8000; do not keep both models loaded on this 6 GiB GPU.

If logits disagree, first check the model revision/weight hash, RoPE positions, GQA key/value expansion, and causal mask. If token counts disagree, inspect the final SSE `usage`, EOS setting, and raw JSONL. If a request gets 503, inspect current GPU free memory and the exact `prompt_tokens + max_tokens` capacity. If timing shifts, inspect client send lag and GPU clocks before attributing the change to batching.

## 5. Proof and its limits

The frozen environment, 20 tests, and Ruff checks pass. GPU tests load the pinned model once and compare V1 to the Hugging Face reference on the same weight object. Single and equal-length batched prefill/decode logits meet `atol=0.05, rtol=0.01`; mixed-length requests meet it through separate cohorts; fixed greedy output IDs match. An HTTP integration test receives two token streams from one actual two-request V1 batch and checks an HTTP 503 capacity rejection. A CPU scheduler test verifies that a four-request collection splits into two batches when the KV allowance admits only two at a time. A live V1 server streamed 16 real model tokens with final batch/prefill/decode timings.

One seeded 24-request, fixed-64/16-token trace was replayed three times each against V0, V1 single, and V1 batch. All nine final runs completed 24/24 with zero client errors and rejections. The same trace and source-tree SHA-256 appear in every raw manifest. V1 batch runs served 20 requests in four-request batches in each trial, proving the window collected work. See `results/summaries/phase3-v1.md` for the complete table.

These measurements are **exploratory**. V1 did not beat V0 goodput in these runs. Its Python/PyTorch path and Windows host timing varied sharply; seven of nine runs exceeded the 5 ms client send-lag P99 target. Twenty-four requests cannot support P99 claims, and no continuous SM-clock trace was captured. The table describes the observed batching effect and failure modes, not a validated capacity, speedup, or native-Linux comparison. We cannot attribute the run-to-run slowdown to one cause from these records.

## 6. Problems, fixes, and remaining limitations

- A first owned attention path repeated K/V heads with a different SDPA dispatch and accumulated BF16 logit differences. Using Qwen2's grouped-query SDPA path for equal-length cohorts brought logits within the strict tolerance.
- Padding mixed-length prompts changed the attention kernel and exceeded the same tolerance. V1 now pre-fills and decodes different lengths as separate cohorts while retaining a common frozen batch lifecycle. This limits tensor-level batching benefit for heterogeneous prompts.
- Decode initially read a GPU cache length once per layer, forcing repeated host/device synchronization. It now computes the key span once per decode call. The final repeated measurements still varied widely; the edit is not claimed as a proven performance improvement.
- The current cache cap is predicted from model geometry and current free/reclaimable bytes. Activation peaks, allocator behavior and OS usage are not continuously profiled; the 512 MiB cap is a guard, not a proven safe pool.
- Static membership delays new arrivals until the batch finishes. The engine does not yet interleave short and long requests or reclaim paged blocks; later phases address those issues.
- vLLM still lacks a usable local reference environment because native Windows CUDA is unsupported and WSL2 cannot mount its missing virtual disk. The required valid external comparison remains open.

## 7. Explain it back

**Interview-sized explanation:** “I replaced the serving baseline's hidden Hugging Face forward pass with an explicit Qwen2 layer loop that writes contiguous KV state. I proved its prefill and decode logits against the reference using the same loaded weights, then added a short fixed batch window behind the existing streaming API. Batches of four formed on one trace, but the Windows runs were unstable and V1 missed the current goodput target, so I report the mechanism and the limitation rather than a speedup.”

Five questions:

1. Why compare against a reference using the same weight object?
2. What does RoPE need from the KV cache during incremental decode?
3. Why can padding mixed-length prompts change BF16 numerical results?
4. What does a fixed-membership batch prevent a newly arriving request from doing?
5. Why is a lower observed TTFT in the batched run not enough to claim a validated improvement?

Answer key:

1. It removes weight/version differences and avoids duplicating roughly 3 GB of weights on the GPU.
2. The logical token position determines the query/key rotation and the new K/V write slot.
3. A mask can select a different SDPA kernel and rounding path; small per-layer differences accumulate.
4. Joining the running batch. It must wait until the batch completes and the next window opens.
5. The runs had inconsistent send lag, a small sample and no continuous clock trace; other host conditions could explain part of the difference.

## 8. Phase gate

- [x] V1 serves a real streamed Qwen answer through the established API.
- [x] Owned prefill/decode and contiguous KV match reference logits at the fixed tolerance; greedy fixture IDs match.
- [x] Static batches form with a measured membership distribution on the same replay trace.
- [x] Three repeated V0/V1-single/V1-batch runs have manifests, raw records, one trace hash, and an explicit validity assessment.
- [x] CPU/GPU tests, lint, documentation, status, result summary, and Phase 3 commit are complete.

The gate demonstrates a correct owned runtime and a measured static-batching mechanism. A performance win, valid tail-latency claim, continuous batching and paged-KV admission remain later work.
