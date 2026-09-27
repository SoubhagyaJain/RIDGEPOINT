# Benchmarking contract

## Definitions

Use monotonic client timestamps. `send` is when the client starts an HTTP request; `first` is first generated token arrival; `last` is final generated token arrival. With `n_out` server-counted output tokens:

- TTFT = `first - send`.
- E2E = `last - send`.
- ITL = each consecutive token arrival gap, pooled across token gaps. Report P50 and P99.
- Per-request TPOT = `(last - first) / (n_out - 1)` for `n_out >= 2`; one-token outputs have no TPOT sample.
- Goodput = completed, nonrejected requests per measurement-window second **for which that request's TTFT ≤ 1000 ms and TPOT ≤ 75 ms**. A one-token output qualifies on TTFT alone. This differs from applying SLO thresholds to aggregate P95 values.
- Rejection rate = rejected requests / offered requests, with status and reason counts reported beside goodput. Errors/timeouts are separate from explicit rejections.
- Throughput always specifies requests/s, input tokens/s, or output tokens/s.

The P95 TTFT and P95 TPOT plots remain useful operational views, but do not define individual-request goodput. Warmup requests are excluded by scheduled send time, never by completion time. Output length is controlled using `ignore_eos` where supported; report any backend mismatch.

## Run validity and fairness

Use an open-loop seeded trace, preserving scheduled sends even when the server slows. Record generator scheduling lag; invalidate if P99 > 5 ms. Record SM clocks/temperature and invalidate if clocks fall >10% below median for >5% of the run. Invalidate on unintended errors, inadequate samples, or >10% Little's-law mismatch. Use at least three seeds/trials for headline points and at least 1,000 requests for P99 claims. Keep prompt token counts and cache state identical across backends; report discrepancies. Save config, code SHA, versions, GPU, driver, and workload seed.

Phase 1 E1 runs only measure isolated GPU operations. Windows/WDDM, short duration, and absent clock trace prevent treating them as validated serving results. Native Linux is preferred for final GPU comparisons.

## Phase 2 implementation and limits

The trace generator writes LF-delimited JSONL bytes and a SHA-256 manifest; the replayer refuses a trace whose bytes no longer match its manifest. The replayer records scheduled and actual send times, every SSE token arrival, the final server usage, request ID, response status, and errors. The analyzer calculates the definitions above from raw records. It marks clock-trace availability and sample-size limitations explicitly.

The six-request V0 smoke trace is reproducible but too small for a P95/P99 performance claim. Three repeats show spread; they do not satisfy the headline trial/sample/clock rules. The 500-stream null stress is a harness check; its deliberately delayed first token makes goodput zero by design, and its send-lag P99 on this Windows host exceeded 5 ms. These observations are recorded rather than silently discarded.
