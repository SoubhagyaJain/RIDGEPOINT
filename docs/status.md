# Status

- Current phase: 3 complete; wait for the user to say "continue" before Phase 4.
- Completed: Phase 2 V0 baseline and harness; Phase 3 owned BF16 Qwen2 forward pass, contiguous KV, static 10 ms/four-request batching, capacity preflight, preserved SSE API, numerical equivalence checks, real streamed V1 answer, and repeated same-trace comparisons.
- Measured V0 smoke: six fixed requests per trial at 0.3 offered req/s; 6/6 completions in each; goodput 0.25 req/s in all three trials. This is exploratory due small sample, Windows/WDDM, and absent continuous clock trace. See `results/summaries/phase2-v0.md`.
- Measured Phase 3: all nine final runs completed 24/24 requests on one trace; V1 batches of up to four formed. V1 did not achieve a validated goodput gain: run-to-run timing varied sharply, most send-lag checks failed, and no continuous clock trace was captured. See `results/summaries/phase3-v1.md`.
- Open limits: V1's Python/PyTorch forward pass is slower than V0 on this host; mixed-length requests are processed in separate numerical cohorts; the 512 MiB cap is a guard rather than a profiled pool. vLLM remains blocked by Windows/WSL2. Continuous batching, abort lifecycle, and owned paged allocation belong to later phases.
- Next: Phase 4 continuous batching and stream/abort lifecycle after the user says "continue".
