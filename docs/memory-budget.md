# Phase 1 memory budget

All model geometry comes from the committed [official Qwen2.5-1.5B-Instruct config](../configs/models/qwen2.5-1.5b-instruct.config.json), downloaded from [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) on 2026-09-25 (SHA-256 `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670`). The config is 660 bytes; no weights were downloaded. `ridgepoint.memory_budget` computes these values so changes to geometry are visible.

## Config-derived arithmetic, not measurements

The model has 28 layers, hidden width 1536, 12 query heads, 2 KV heads, intermediate width 8960, and vocabulary 151,936. Head width is `1536 / 12 = 128`. Six query heads share each KV head. BF16 takes 2 bytes per value.

For one stored token, every layer keeps one key and one value for every KV head:

```text
KV bytes/token = 2 (K,V) × 28 layers × 2 KV heads × 128 values/head × 2 B
               = 28,672 B = 28 KiB
```

A 16-token KV block takes `16 × 28,672 = 458,752 B = 448 KiB`. Weight count includes embeddings, Q/K/V biases, the output and MLP projections, layer norms, and final norm. Because embeddings are tied, the LM head reuses the embedding matrix; it is counted once. The result is **1,543,714,304 parameters** and **3,087,428,608 BF16 bytes = 3.087 GB = 2.876 GiB**. This is parameter storage only; real loading can use extra allocator memory or temporary buffers.

| Hypothetical KV pool | Complete blocks | Usable tokens | Example full 1024-token sequences before watermark |
| --- | ---: | ---: | ---: |
| 1.0 GiB | 2,340 | 37,440 | 36 |
| 1.5 GiB | 3,510 | 56,160 | 54 |
| 1.8 GB (decimal) | 3,923 | 62,768 | 61 |

The table floors capacity to complete 16-token blocks. It ignores fragmentation, active decode growth, and any watermark. Actual concurrency is lower.

## Hardware observation and initial planning bound

Before PyTorch initialized CUDA, `nvidia-smi` reported **6,141 MiB total and 5,202 MiB free** on the RTX 4050, with background Windows applications present. After initializing CUDA, `torch.cuda.mem_get_info()` reported **5,073 MiB free** (5,319,426,048 bytes) and **6,140.5 MiB total**. These are measured snapshots, not guaranteed memory for a serving run. The notebook's generic 1.5–2 GB KV estimate assumes more free memory and cannot be copied directly to this host.

An initial conservative plan using the post-context snapshot is:

```text
free after context          5,073 MiB (measured snapshot)
minus BF16 weights          2,944.4 MiB (architecture estimate)
minus activation reserve      307 MiB (estimate)
minus safety reserve          256 MiB (estimate)
-------------------------------------------------
provisional upper bound     1,565.6 MiB
initial pool configuration  1,024 MiB = 1.0 GiB
```

The context-adjusted free value is measured; the activation reserve and actual model-loading footprint are not. **The 1.0 GiB pool is a planning setting, not an allocated or validated pool.** It leaves roughly 542 MiB below the provisional bound for uncertainty. WDDM/background usage can change during runs. The real pool will be set after loading model weights, measuring peak prefill allocations, and retaining a safety margin.

At 1.0 GiB of KV, the weight-plus-full-KV traffic estimate for a decode step is `3.087 GB + 1.074 GB = 4.161 GB`. Dividing by the notebook's **unmeasured** 192 GB/s specification estimate gives a 21.7 ms idealized traffic floor before kernel/host overhead. Dividing by E1's measured 148.9 GB/s device-copy median gives a 28.0 ms *copy-based yardstick*, not a model-serving measurement or guaranteed lower bound. A full 2048-position FP32 logits tensor would require `2048 × 151,936 × 4 = 1.245 GB`, so the eventual runtime must compute logits only for needed final positions.

## Reproduce

```powershell
uv run --frozen python -m ridgepoint.memory_budget configs/models/qwen2.5-1.5b-instruct.config.json --pool-gib 1.0
uv run --frozen python -m ridgepoint.hardware
```

`parameters`, `weight_bytes`, `kv_bytes_per_token`, `block_bytes`, and `usable_tokens` should match the arithmetic above. The hardware command reports current CUDA free bytes, which will vary.
