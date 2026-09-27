# Status

- Current phase: 2 complete; wait for the user to say "continue" before Phase 3.
- Completed: V0 BF16 Hugging Face streaming server, pinned/verified model download, validation/request IDs/timings, seeded token-ID trace generator, open-loop client, async null backend, raw-record analysis, 10 passing tests, clean lint, real streamed answer, three repeated V0 smoke runs, and 500-stream null check.
- Measured V0 smoke: six fixed requests per trial at 0.3 offered req/s; 6/6 completions in each; goodput 0.25 req/s in all three trials. This is exploratory due small sample, Windows/WDDM, and absent continuous clock trace. See `results/summaries/phase2-v0.md`.
- Open limits: 500 simultaneous null streams had send-lag P99 of 16.91 ms, above the 5 ms goal. vLLM native Windows is unsupported, and WSL2 currently fails to mount its missing virtual disk. A safe owned KV pool and valid capacity comparison remain later-phase work.
- Next: Phase 3 owned Qwen2 forward pass, contiguous KV, numerical checks, and static/short-window batching after the user says "continue".
