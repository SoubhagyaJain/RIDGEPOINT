"""Recompute the Phase 3 exploratory table from ignored raw run records."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.analyze.metrics import analyze


def main() -> None:
    # Optional run-set prefix, e.g. `analyze.py phase3-fix` for the post-fix rerun.
    prefix = sys.argv[1] if len(sys.argv) > 1 else "phase3-final"
    raw = Path("results/raw")
    trace_hashes = set()
    source_hashes = set()
    print("| Engine | Trial | Complete | Good | Goodput req/s | TTFT P95 ms | "
          "TPOT P95 ms | Send lag P99 ms | Batch sizes (requests) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for engine in ("v0", "v1-single", "v1-batch"):
        for trial in (1, 2, 3):
            path = raw / f"{prefix}-{engine}-{trial}.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            trace_hashes.add(manifest["trace_sha256"])
            source_hashes.add(manifest["source_sha256"])
            metrics = analyze(records, duration_s=20)
            sizes = Counter(record["done"].get("batch_size", 1) for record in records
                            if record["done"] is not None)
            size_text = ", ".join(f"{size}:{count}" for size, count in sorted(sizes.items()))
            print(f"| {engine} | {trial} | {metrics['completed_requests']}/24 | "
                  f"{metrics['good_requests']} | {metrics['goodput_rps']:.2f} | "
                  f"{metrics['ttft_p95_ms']:.2f} | {metrics['tpot_p95_ms']:.2f} | "
                  f"{metrics['generator_lag_p99_ms']:.2f} | {size_text} |")
    if len(trace_hashes) != 1 or len(source_hashes) != 1:
        raise ValueError("comparison runs used different trace or source bytes")
    print(f"trace_sha256: {next(iter(trace_hashes))}")
    print(f"source_sha256: {next(iter(source_hashes))}")


if __name__ == "__main__":
    main()
