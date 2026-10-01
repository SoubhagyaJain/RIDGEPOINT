# V1 decode-step host syncs

Phase 3 follow-up. `step_compare.py` times one decode step for the Hugging Face forward
pass and the V1 runtime on the same loaded weights, in one process, with the engines
interleaved inside every repetition. It checks that greedy tokens match, counts the
synchronizing operations in one V1 decode step, and writes every sample to ignored
`results/raw/v1-decode-sync-<UTC time>.json`.

```powershell
uv run --frozen python experiments/V1-decode-sync/step_compare.py
uv run --frozen python experiments/V1-decode-sync/step_compare.py --baseline-rev c01762d
```

`--baseline-rev` loads `src/ridgepoint/runtime/qwen2.py` from that commit next to the
working-tree runtime, so the before/after comparison shares one process and one host
state. `c01762d` is the original Phase 3 runtime with boolean-mask cache writes.

To hold the process on one kind of CPU core on this 4P+4E laptop, start it through
`cmd /c "start /b /wait /affinity FF uv run --frozen python ..."` (`FF` = logical CPUs
0–7, `F00` = logical CPUs 8–11). See `docs/learning/phase-03.md` section 9 for the
findings and their limits.
