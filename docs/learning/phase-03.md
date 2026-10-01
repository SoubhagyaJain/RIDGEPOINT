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

These measurements are **exploratory**. V1 did not beat V0 goodput in these runs. Its Python/PyTorch path and Windows host timing varied sharply; seven of nine runs exceeded the 5 ms client send-lag P99 target. Twenty-four requests cannot support P99 claims, and no continuous SM-clock trace was captured. The table describes the observed batching effect and failure modes, not a validated capacity, speedup, or native-Linux comparison. We cannot attribute the run-to-run slowdown to one cause from these records. Section 9 records a later investigation that found a decode-path bug and a CPU-core placement effect, with reruns.

## 6. Problems, fixes, and remaining limitations

- A first owned attention path repeated K/V heads with a different SDPA dispatch and accumulated BF16 logit differences. Using Qwen2's grouped-query SDPA path for equal-length cohorts brought logits within the strict tolerance.
- Padding mixed-length prompts changed the attention kernel and exceeded the same tolerance. V1 now pre-fills and decodes different lengths as separate cohorts while retaining a common frozen batch lifecycle. This limits tensor-level batching benefit for heterogeneous prompts.
- Decode initially read a GPU cache length once per layer, forcing repeated host/device synchronization. It now computes the key span once per decode call. The final repeated measurements still varied widely; the edit is not claimed as a proven performance improvement. That edit missed the larger source: boolean-mask indexing in the cache write still synchronized four times per layer. Section 9 finds and removes it.
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

## 9. Follow-up — why V1 decode was slow and why runs disagreed (2026-10-01)

### 9.1 What we did

Sections 5 and 6 left two things unexplained: V1 was much slower than V0 on the same weights, and timing swung wildly between trials. This follow-up profiled one decode step, found a bug in the decode path, fixed it, added a regression test, and reran the same-trace comparison. It also found that which CPU cores the process runs on changes every engine's speed by about 1.7×. No Phase 4 work was started.

Files changed: `src/ridgepoint/runtime/qwen2.py` (the fix), `tests/numerics/test_qwen2_v1.py` (one new test, 21 tests total), `experiments/V1-batching/analyze.py` (optional run-set prefix), new `experiments/V1-decode-sync/{step_compare.py,README.md}`, `results/summaries/phase3-v1.md`, `docs/status.md`, and this report.

### 9.2 Concepts from scratch

**GPU work is queued, not executed on the spot.** When Python calls a PyTorch operation on a CUDA tensor, the CPU only places a job in the GPU's queue and moves on. The CPU can queue layer 2 while the GPU is still computing layer 1. This overlap is why a 28-layer step does not cost 28 separate round trips.

**A host sync makes the CPU stop and wait.** Anything that needs an actual number on the CPU side forces it to wait until the GPU has finished everything queued so far. The obvious cases are `.item()`, `.tolist()`, `bool(tensor)` and `print(tensor)`.

**Boolean-mask indexing is a hidden sync.** `k[active]` selects the rows where `active` is true. PyTorch cannot allocate the result until it knows how many rows that is, so it runs `nonzero` on the GPU and reads the count back to the CPU. It looks like ordinary indexing and behaves like `.item()`. Indexing with a plain Python integer or a slice has a known output shape and stays in the queue.

**A physical yardstick tells you when a number is wrong.** One decode step reads every weight once. Phase 1 measured 148.9 GB/s of copy bandwidth and 3.087 GB of BF16 weights, so `3.087 / 148.9 = 20.7 ms` is the scale to expect. Hugging Face measured about 27 ms on this GPU. A batch of four reads the same weights once, so it should cost about the same per step. V1's original 220–530 ms per token was 10–25× that scale, which points to overhead rather than to arithmetic the model needs.

**An ablation changes one thing at a time.** To learn which change matters, keep everything else fixed, measure in the same process, and interleave the variants so that slow drift in the machine hits all of them equally. Comparing a ratio inside one repetition is more trustworthy than comparing absolute times taken minutes apart.

**Performance and efficiency cores.** This laptop's i5-13420H has four fast performance cores (logical CPUs 0–7 with hyper-threading) and four slower efficiency cores (logical CPUs 8–11). Windows moves threads between them. A process's *affinity mask* restricts which logical CPUs it may use: `FF` is CPUs 0–7 and `F00` is CPUs 8–11. The GPU can only run what the CPU queues, so a slow CPU core starves a fast GPU.

### 9.3 How the investigation went

1. **Compare in one process.** Load the weights once and time a decode step for Hugging Face `generate()`, a hand-written Hugging Face forward loop, and V1. V1 took roughly twice as long as Hugging Face in most repetitions, at batch 1 and batch 4, even while absolute times drifted for all three. A steady ratio means a code cause; the drift is a separate problem.
2. **Count operations.** `torch.profiler` on one original V1 decode step showed 113 `aten::nonzero` calls. Hugging Face showed none. Four came from each of the 28 layers (`k[active, ...]`, `v[active, ...]`, `arange(batch)[active]`, `cache.lengths[active]`) and one from the capacity check.
3. **Ablate.** A one-off script kept the same math but swapped parts of the decode path. Step medians at batch 1: Hugging Face 27.0 ms; original V1 55.6 ms (2.07×); integer slot write only 26.7 ms (0.99×); plus one RoPE table per call 24.8 ms (0.92×). The integer slot write alone closed the gap. These four numbers were console output from a temporary script and were not retained as raw files.
4. **Fix and test.** See 9.4.
5. **Measure the committed code.** `step_compare.py --baseline-rev c01762d` loads the original `qwen2.py` from Git next to the fixed one in the same process. Results are in 9.5.
6. **Rerun the trace.** With the fix, unpinned results still swung by run, and V0 swung too. The GPU trace showed the slow runs at 35–49% utilization and 25–34 W: the GPU was idle much of the time, waiting to be fed.
7. **Test the CPU cores.** The same script pinned to efficiency cores gave about 46 ms per step for both engines, and pinned to performance cores gave about 27 ms. Those match the fast level and the most common slow level seen as "drift" earlier. Occasional slower repetitions (80–160 ms) are not explained by this test.
8. **Rerun pinned, both runtimes.** All nine runs were repeated with server and client pinned to performance cores, then the original runtime was run under the identical pinned method for a like-for-like "before".

### 9.4 What changed in the code

The uniform decode path only runs when every row is active and all rows have the same cache length (mixed cases recurse per row first). So one integer describes the write position for the whole batch, and the boolean masks were never needed there.

- `decode()` now reads `cache.lengths` and `active` to the host once per step with `.tolist()`, decides the branch in Python, and passes a single integer `slot` to `_layer()`.
- `_layer()` writes `cache.data[index, 0, :, :, slot, :] = k[:, :, 0, :]` instead of indexing with boolean masks. The attention call and its inputs are otherwise the same.
- `_rope_table()` builds the sine/cosine table once per `prefill()` or `decode()` call. Before, each of the 28 layers recomputed the identical table.
- A single inactive row now returns zero logits instead of attending to an unwritten slot. No caller used that case.

The new test `test_decode_layer_loop_never_syncs_host` wraps `_layer()` in `torch.cuda.set_sync_debug_mode("error")`, which raises on any synchronizing CUDA call. It fails on the original runtime at the mask-indexing line and passes on the fixed one.

A faster variant that repeats the two K/V heads by hand instead of using `enable_gqa=True` measured 0.85× of Hugging Face in the ablation. It was **not** adopted: section 6 records that an earlier K/V-repeat path failed the logit tolerance, and this variant was only checked for matching greedy tokens.

### 9.5 Proof and its limits

`pytest -q` passes **21 tests** and `ruff check src bench tests` is clean. Logit tolerance and exact greedy-token tests are unchanged and still pass.

Same-process decode step, unpinned, 8 repetitions, median of per-repetition medians (raw: `results/raw/v1-decode-sync-20261001T053315Z.json`):

| Batch | HF forward | V1 fixed | V1 original (`c01762d`) | Fixed / HF | Original / HF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.41 ms | 55.36 ms | 92.12 ms | 0.92 | 1.52 |
| 4 | 48.90 ms | 49.50 ms | 82.20 ms | 0.94 | 1.49 |

Synchronizing `aten::nonzero` calls per decode step: 113 before, 0 after. Total operator calls per step: 13,508 before, 9,543 after.

CPU-core placement, fixed runtime, 4 repetitions each:

| Affinity | HF forward b=1 | V1 b=1 | HF forward b=4 | V1 b=4 |
| --- | ---: | ---: | ---: | ---: |
| Efficiency cores (`F00`) | 46.18 ms | 44.23 ms | 52.38 ms | 50.21 ms |
| Performance cores (`FF`) | 27.82 ms | 26.61 ms | 28.81 ms | 26.85 ms |

Same 24-request trace, server and client pinned to performance cores, three trials each (full tables in `results/summaries/phase3-v1.md`):

| Engine | Good requests / 24 | Goodput req/s | TTFT P95 ms | TPOT P95 ms |
| --- | --- | --- | --- | --- |
| V0 | 14, 15, 19 | 0.70, 0.75, 0.95 | 2,697, 2,680, 1,425 | 35, 40, 33 |
| V1 single, original runtime | 3, 3, 3 | 0.15, 0.15, 0.15 | 9,720, 8,624, 8,563 | 73, 65, 63 |
| V1 single, fixed | 21, 22, 21 | 1.05, 1.10, 1.05 | 1,110, 998, 1,079 | 28, 29, 28 |
| V1 batch, original runtime | 11, 24, 24 | 0.55, 1.20, 1.20 | 2,101, 928, 904 | 115, 63, 73 |
| V1 batch, fixed | 24, 24, 20 | 1.20, 1.20, 1.00 | 821, 474, 829 | 51, 31, 109 |

What this supports: the fixed V1 is no longer slower than Hugging Face per decode step, and under the same pinned method the fix took V1 single from 3 good requests to 21–22. Pinned and fixed, every V1 run had higher goodput than every V0 run.

What it does not support:

- **A validated speedup.** Each trial has 24 requests, the host is Windows/WDDM, and only 3 of the 9 pinned fixed runs met the 5 ms send-lag P99 target. By the project's own rules these runs are exploratory.
- **A batching gain.** The trace offers 1.2 req/s, and 24 good requests out of 24 is exactly that. V1 batch is at the ceiling, so this trace cannot show how much batching helps or how much the fix helped V1 batch. The original runtime also reached the ceiling in two pinned batch trials.
- **An explanation of the original 529–536 ms TPOT.** It was never reproduced. The original runtime under pinning showed 63–73 ms.

### 9.6 What is still unexplained

- V1 batch TPOT P95 was 31–109 ms against about 28 ms for V1 single. Something stalls some batches. Not investigated.
- V0 has much higher TTFT P95 than V1 single (1.4–2.7 s against about 1.1 s) even though their decode steps are within 10%. Prefill and per-request `generate()` overhead were not profiled.
- Unpinned, the GPU clock dipped from 2,625 to 2,130 MHz in several runs. Whether that is a cause or a side effect of the GPU being idle was not determined.
- The core mapping (CPUs 0–7 performance, 8–11 efficiency) follows Intel's usual numbering and was not independently verified. The affinity test is one run per mask.
- The split of the sync cost between lost CPU/GPU overlap and the Windows driver round trip was not measured; only the total was, at roughly a quarter of a millisecond per sync.
- Mixed-length batches still decode one row at a time. That design limit from section 6 is untouched and matters for Phase 4.

### 9.7 How to reproduce

```powershell
uv run --frozen pytest -q
uv run --frozen python experiments/V1-decode-sync/step_compare.py --baseline-rev c01762d
cmd /c "start /b /wait /affinity F00 uv run --frozen python experiments/V1-decode-sync/step_compare.py --reps 4"
cmd /c "start /b /wait /affinity FF uv run --frozen python experiments/V1-decode-sync/step_compare.py --reps 4"
uv run --frozen python experiments/V1-batching/analyze.py phase3-pin
```

The first command should report 21 passed. The second prints step medians, the ratio to Hugging Face, and the per-step count of synchronizing operations for both runtimes; expect the original near 1.5–2× and the fixed runtime near or below 1.0×, with absolute times depending on the machine's state. The two pinned commands should differ by roughly 1.7× on this laptop. The last command needs the `phase3-pin-*` raw files, produced by replaying the Phase 3 trace against each server started under the `FF` affinity mask.

If a timing looks wrong, first check where the process is running (Task Manager → Details → Set affinity, or pin it explicitly), then check GPU utilization and power in `nvidia-smi`. Low utilization during a run means the CPU side is the bottleneck.

### 9.8 Explain it back

**Interview-sized explanation:** “My own forward pass was about twice as slow as Hugging Face on identical weights, which made no sense because decode is bounded by reading the weights once. I profiled one step and found 113 hidden GPU-to-CPU syncs, four per layer, from boolean-mask indexing in the KV-cache write. Because every row in that path shares one position, I replaced the masks with a single integer index, which brought the step from 1.5× Hugging Face to 0.92×, and I added a test that fails if any sync returns. Separately, I found that Windows placing the process on efficiency cores slowed every engine about 1.7×, which was the run-to-run noise. With both controlled, V1 beat the baseline in every trial, but the sample is too small to call it a validated result.”

Five questions:

1. Why does `k[active]` force the CPU to wait for the GPU when `k[:, 3]` does not?
2. The slowdown ratio stayed near 2× while absolute times drifted. What does each observation tell you?
3. Why was a GPU at 40% utilization and low power a clue about the CPU?
4. Why can't the pinned table show how much static batching helps?
5. Why was the faster hand-repeated K/V variant left out?

Answer key:

1. The size of a boolean-mask result depends on how many entries are true, so PyTorch must read that count back before continuing. An integer or slice index has a known shape.
2. A steady ratio inside one process points to the code. Drift that hits every engine, Hugging Face included, points to the machine.
3. A GPU that is the bottleneck runs near full utilization. Low utilization while work is waiting means the CPU is not queueing operations fast enough.
4. The trace offers 1.2 requests per second and V1 batch already delivers all of them within the SLO. A higher offered rate is needed to find where each engine breaks.
5. It was only checked for matching greedy tokens, and an earlier K/V-repeat path had failed the logit tolerance. Correctness comes before speed.
