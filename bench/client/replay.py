"""Replay a seeded trace on schedule and record client-observed SSE timestamps."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import httpx


def read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gpu_snapshot() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,"
             "temperature.gpu,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_sha256() -> str:
    digest = hashlib.sha256()
    paths = [*Path("src/ridgepoint").rglob("*.py"), *Path("bench").rglob("*.py")]
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


async def sleep_until(target_ns: int) -> None:
    """Keep the final wait precise on Windows while yielding until the last millisecond."""
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 20_000_000:
            await asyncio.sleep((remaining_ns - 10_000_000) / 1e9)
        elif remaining_ns > 1_000_000:
            await asyncio.sleep(0)
        else:
            while time.perf_counter_ns() < target_ns:
                pass
            return


async def replay_one(client: httpx.AsyncClient, url: str, item: dict, origin_ns: int,
                     timeout_s: float) -> dict:
    scheduled_ns = origin_ns + round(item["t_offset_ms"] * 1e6)
    await sleep_until(scheduled_ns)
    send_ns = time.perf_counter_ns()
    record = {"req_id": item["req_id"], "t_offset_ms": item["t_offset_ms"],
              "scheduled_ns": scheduled_ns, "send_ns": send_ns,
              "generator_lag_ms": max(0, send_ns - scheduled_ns) / 1e6,
              "expected_prompt_tokens": len(item["prompt_ids"]),
              "expected_output_tokens": item["max_tokens"], "token_events": [],
              "ignore_eos": item.get("ignore_eos", True),
              "status_code": None, "error": None, "done": None, "request_id": None}
    record["response_headers_ns"] = None
    body = {"prompt_ids": item["prompt_ids"], "max_tokens": item["max_tokens"],
            "temperature": item.get("temperature", 0.0), "top_p": 1.0,
            "ignore_eos": item.get("ignore_eos", True), "stream": True}
    try:
        async with client.stream("POST", url, json=body, timeout=timeout_s) as response:
            record["status_code"] = response.status_code
            record["response_headers_ns"] = time.perf_counter_ns()
            record["request_id"] = response.headers.get("x-request-id")
            if response.status_code != 200:
                record["error"] = (await response.aread()).decode("utf-8", errors="replace")
                return record
            current_event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    now_ns = time.perf_counter_ns()
                    event = json.loads(line[6:])
                    kind = current_event or event.get("type")
                    if kind == "token":
                        record["token_events"].append({"at_ns": now_ns,
                                                       "index": event["index"]})
                    elif kind == "done":
                        record["done"] = {"at_ns": now_ns, **event}
                    elif kind == "error":
                        record["error"] = event.get("message", "stream error")
                    current_event = None
            if record["done"] is None and record["error"] is None:
                record["error"] = "stream ended without done event"
    except (httpx.HTTPError, ValueError, OSError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


async def replay(trace: list[dict], url: str, timeout_s: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    # All sends use this fixed origin, independent of previous request completions.
    origin_ns = time.perf_counter_ns() + 100_000_000
    limits = httpx.Limits(max_connections=max(100, len(trace)),
                          max_keepalive_connections=100)
    async with httpx.AsyncClient(transport=transport, limits=limits) as client:
        tasks = [asyncio.create_task(replay_one(client, url, item, origin_ns, timeout_s))
                 for item in trace]
        return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--output", type=Path, default=Path("results/raw/phase2-requests.jsonl"))
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--engine-config", type=Path, default=Path("configs/engine/v0.yaml"))
    args = parser.parse_args()
    trace = read_trace(args.trace)
    trace_manifest_path = args.trace.with_suffix(".manifest.json")
    trace_manifest = (json.loads(trace_manifest_path.read_text(encoding="utf-8"))
                      if trace_manifest_path.exists() else None)
    trace_sha256 = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    if trace_manifest and trace_manifest.get("trace_sha256") != trace_sha256:
        parser.error("trace bytes do not match the generator manifest SHA-256")
    before = gpu_snapshot()
    try:
        ready = httpx.get(args.url.rsplit("/v1/completions", 1)[0] + "/ready", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        ready = None
    records = asyncio.run(replay(trace, args.url, args.timeout_s))
    after = gpu_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = {"trace_sha256": trace_sha256,
                "trace_manifest": trace_manifest, "ready": ready, "url": args.url,
                "engine_config_sha256": hashlib.sha256(args.engine_config.read_bytes()).hexdigest()
                    if args.engine_config.exists() and ready and ready.get("model") != "ridgepoint-null"
                    else None,
                "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                                  capture_output=True, text=True).stdout.strip()),
                "source_sha256": source_tree_sha256(),
                "os": platform.platform(), "python": sys.version.split()[0],
                "httpx": version("httpx"), "torch": version("torch"),
                "transformers": version("transformers"),
                "gpu_before": before, "gpu_after": after,
                "requests": len(records), "errors": sum(bool(r["error"]) for r in records)}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"requests": len(records), "errors": sum(bool(r["error"]) for r in records),
                      "trace_sha256": manifest["trace_sha256"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
