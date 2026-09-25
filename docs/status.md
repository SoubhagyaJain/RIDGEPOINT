# Status

- Current phase: 1 complete; waiting for the user to say "continue" before Phase 2.
- Completed: locked CUDA environment, official model config, memory calculator, passing BF16 GPU check, four unit tests, clean lint, E1 smoke and three full microbenchmark runs, memory budget, six initial workloads, SLO and measurement definitions, and teaching report.
- Phase 1 measurements: 148.9 GB/s 128 MiB device copy, 24.92 TFLOP/s BF16 4096 GEMM, 3.76 µs tiny PyTorch add (median of three run medians). See `results/summaries/e1-phase1.md`.
- Limitations: Windows/WDDM E1 is exploratory without continuous clock logging; model-loading activation peak and safe KV pool remain to be measured. Native Linux is not the current host.
- Next: Phase 2 V0 Hugging Face streaming server and repeatable benchmark harness, after user approval to continue.
