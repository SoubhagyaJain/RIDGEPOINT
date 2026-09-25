# Seven-phase plan

Day numbers are targets, not acceptance criteria. Finish one phase, commit it, and wait for "continue".

| Phase | Target days | Build | Gate |
| --- | --- | --- | --- |
| 1 | 1–3 | Hardware inspection, locked environment, GPU check, E1 microbenchmarks, memory math, SLO/workload definitions | Setup works; GPU check passes or blocker is documented; microbenchmarks run; config-derived memory math is explained |
| 2 | 4–7 | Hugging Face V0 streaming server, seeded open-loop harness, null backend, metric analysis | Streamed answer, deterministic trace replay, timing tests, baseline manifest |
| 3 | 8–12 | Qwen2 runtime, contiguous KV, static/windowed batching | Numerical correctness before claims; V1 serves; batching effect measured |
| 4 | 13–16 | Continuous batching, stream/abort lifecycle | Mixed requests; disconnect resources released; V1/V2 same-trace comparison |
| 5 | 17–21 | Paged KV, admission, preemption, overload | Allocator invariants; output equivalence; goodput and rejections under overload |
| 6 | 22–26 | Prefix cache, chunked prefill, measured cost model | Equivalence; held-out model error; reproducible gain plots |
| 7 | 27–30 | Profile-driven CUDA graphs or quantization, final checks/report | Runnable instructions; estimates versus measurements; limitations and gap explained |
