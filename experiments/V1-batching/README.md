# V1 fixed-membership batching

Phase 3 compares V0, V1 with one request per batch, and V1 with a 10 ms / four-request
window. The exact same seeded 24-request trace, SLOs, and analyzer are used for each run.
Raw JSONL and manifests remain in ignored `results/raw/`.

Generate the trace with the command in `results/summaries/phase3-v1.md`, then run each
server in a separate process. Set `RIDGEPOINT_V1_CONFIG=configs/engine/v1-single.yaml`
for the single-request mode or leave it unset for the default batch mode. Replay to each
server three times and analyze with:

```powershell
uv run --frozen python experiments/V1-batching/analyze.py
```

The script verifies common trace and source hashes before printing the table. It reads
`phase3-final-*` by default; pass another run-set prefix as the first argument, for
example `analyze.py phase3-pin` for the pinned post-fix rerun.
