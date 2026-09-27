"""Client-visible latency, throughput, goodput, and rejection calculations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def percentile(samples: list[float], value: float) -> float | None:
    return float(np.percentile(samples, value)) if samples else None


def request_metrics(record: dict, ttft_slo_ms: float = 1000,
                    tpot_slo_ms: float = 75) -> dict:
    tokens = record["token_events"]
    done = record.get("done")
    status = record.get("status_code")
    times = [item["at_ns"] for item in tokens]
    ttft = (times[0] - record["send_ns"]) / 1e6 if times else None
    e2e = (times[-1] - record["send_ns"]) / 1e6 if times else None
    gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    tpot = (times[-1] - times[0]) / 1e6 / (len(times) - 1) if len(times) > 1 else None
    usage = done.get("usage", {}) if done else {}
    prompt_match = usage.get("prompt_tokens") == record["expected_prompt_tokens"]
    output_match = usage.get("completion_tokens") == len(tokens)
    length_match = (not record.get("ignore_eos", True)
                    or len(tokens) == record["expected_output_tokens"])
    completed = (status == 200 and record.get("error") is None and done is not None
                 and done.get("finish_reason") in {"length", "stop"}
                 and bool(times) and prompt_match and output_match and length_match)
    qualifies = (completed and ttft <= ttft_slo_ms
                 and (tpot is None or tpot <= tpot_slo_ms))
    return {"req_id": record["req_id"], "status_code": status,
            "ttft_ms": ttft, "e2e_ms": e2e, "itl_ms": gaps, "tpot_ms": tpot,
            "prompt_match": prompt_match, "output_match": output_match,
            "length_match": length_match, "completed": completed,
            "good": qualifies, "rejected": status in {400, 401, 403, 404, 422, 429},
            "output_tokens": len(tokens), "input_tokens": usage.get("prompt_tokens", 0),
            "generator_lag_ms": record["generator_lag_ms"],
            "error": record.get("error")}


def peak_open_streams(records: list[dict]) -> int:
    events = []
    for record in records:
        start = record.get("response_headers_ns")
        done = record.get("done")
        if start is not None and done is not None:
            events.extend(((start, 1), (done["at_ns"], -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def analyze(records: list[dict], *, duration_s: float, warmup_s: float = 0,
            ttft_slo_ms: float = 1000, tpot_slo_ms: float = 75) -> dict:
    if not 0 <= warmup_s < duration_s:
        raise ValueError("require 0 <= warmup_s < duration_s")
    selected = [r for r in records if warmup_s * 1000 <= r["t_offset_ms"] < duration_s * 1000]
    rows = [request_metrics(r, ttft_slo_ms, tpot_slo_ms) for r in selected]
    window = duration_s - warmup_s
    completed = [r for r in rows if r["completed"]]
    itls = [gap for row in completed for gap in row["itl_ms"]]
    rejects = Counter(str(row["status_code"]) for row in rows if row["rejected"])
    errors = sum(not row["completed"] and not row["rejected"] for row in rows)
    lag_p99 = percentile([r["generator_lag_ms"] for r in rows], 99)
    mismatches = sum(not r["prompt_match"] for r in rows if r["status_code"] == 200)
    return {
        "measurement_window_s": window, "offered_requests": len(rows),
        "peak_open_streams": peak_open_streams(selected),
        "completed_requests": len(completed), "good_requests": sum(r["good"] for r in rows),
        "rejected_requests": sum(rejects.values()), "rejections_by_status": dict(rejects),
        "errors": errors, "offered_rps": len(rows) / window,
        "completed_rps": len(completed) / window,
        "goodput_rps": sum(r["good"] for r in rows) / window,
        "rejection_rate": sum(rejects.values()) / len(rows) if rows else 0.0,
        "input_tokens_per_s": sum(r["input_tokens"] for r in completed) / window,
        "output_tokens_per_s": sum(r["output_tokens"] for r in completed) / window,
        "ttft_p50_ms": percentile([r["ttft_ms"] for r in completed], 50),
        "ttft_p95_ms": percentile([r["ttft_ms"] for r in completed], 95),
        "tpot_p95_ms": percentile([r["tpot_ms"] for r in completed if r["tpot_ms"] is not None], 95),
        "itl_p50_ms": percentile(itls, 50), "itl_p99_ms": percentile(itls, 99),
        "e2e_p95_ms": percentile([r["e2e_ms"] for r in completed], 95),
        "generator_lag_p99_ms": lag_p99,
        "prompt_count_mismatches": mismatches,
        "validity": {"generator_lag_p99_under_5_ms": lag_p99 is not None and lag_p99 <= 5,
                     "unintended_errors_zero": errors == 0,
                     "prompt_counts_match": mismatches == 0,
                     "enough_requests_for_p99": len(rows) >= 1000,
                     "clock_trace_available": False,
                     "headline_valid": False},
        "ttft_slo_ms": ttft_slo_ms, "tpot_slo_ms": tpot_slo_ms,
        "requests": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--warmup-s", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.requests.read_text(encoding="utf-8").splitlines()
               if line]
    result = analyze(records, duration_s=args.duration_s, warmup_s=args.warmup_s)
    summary = {key: value for key, value in result.items() if key != "requests"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
