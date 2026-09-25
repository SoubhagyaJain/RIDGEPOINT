"""Measure GPU copy bandwidth, BF16 GEMM throughput, and launch overhead.

Run with ``uv run python -m bench.micro.run``. These are component measurements,
not model-serving throughput. Raw samples and a machine manifest are saved.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def gpu_ms(operation, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def smi_snapshot() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,"
             "power.draw,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(values: list[float]) -> dict:
    return {"samples": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def run(smoke: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; run `make gpu-check` first")
    device = torch.device("cuda")
    torch.cuda.init()
    manifest = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "memory_free_bytes": torch.cuda.mem_get_info(device)[0],
        "memory_total_bytes": torch.cuda.mem_get_info(device)[1],
        "smi_before": smi_snapshot(), "smoke": smoke,
    }
    repeats = 5 if smoke else 30
    warmup = 3 if smoke else 10

    # A single copy moves one read plus one write. Count both in effective traffic.
    copy_bytes = 64 * 1024**2 if smoke else 128 * 1024**2
    src = torch.empty(copy_bytes // 4, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    src.fill_(1.0)
    copy_samples = gpu_ms(lambda: dst.copy_(src), warmup=warmup, repeats=repeats)
    copy_gbps = [(2 * copy_bytes) / (ms / 1000) / 1e9 for ms in copy_samples]
    src = dst = None

    gemms = {}
    for size in ([1024, 2048] if smoke else [1024, 2048, 4096]):
        a = torch.randn((size, size), dtype=torch.bfloat16, device=device)
        b = torch.randn((size, size), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(a)
        samples = gpu_ms(lambda: torch.mm(a, b, out=out), warmup=warmup, repeats=repeats)
        gemms[str(size)] = {
            "ms": summarize(samples),
            "tflops": summarize([(2 * size**3) / (ms / 1000) / 1e12 for ms in samples]),
        }
        a = b = out = None

    # PyTorch add_ is a tiny real GPU kernel; this includes its device execution.
    tiny = torch.zeros(1, device=device)
    launches = 200 if smoke else 1000
    for _ in range(warmup):
        tiny.add_(1)
    torch.cuda.synchronize()
    launch_samples_ms = []
    host_samples_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        host_start = time.perf_counter_ns()
        for _ in range(launches):
            tiny.add_(1)
        host_samples_ms.append((time.perf_counter_ns() - host_start) / 1e6 / launches)
        end.record()
        end.synchronize()
        launch_samples_ms.append(start.elapsed_time(end) / launches)
    manifest["smi_after"] = smi_snapshot()
    result = {
        "manifest": manifest,
        "copy": {"tensor_bytes": copy_bytes, "counted_bytes_per_copy": 2 * copy_bytes,
                 "ms": summarize(copy_samples), "gb_per_s": summarize(copy_gbps)},
        "bf16_gemm": gemms,
        "tiny_add_launch": {"launches_per_sample": launches,
                            "gpu_us_per_op": summarize([ms * 1000 for ms in launch_samples_ms]),
                            "host_enqueue_us_per_op": summarize([ms * 1000 for ms in host_samples_ms])},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.smoke)
    output = args.output or Path("results/raw") / (
        f"e1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Raw samples: {output}")
    print(f"GPU copy: {result['copy']['gb_per_s']['median']:.1f} GB/s")
    for size, value in result["bf16_gemm"].items():
        print(f"BF16 GEMM {size}: {value['tflops']['median']:.2f} TFLOP/s")
    print(f"Tiny add GPU time: {result['tiny_add_launch']['gpu_us_per_op']['median']:.2f} us/op")
    print(f"Host enqueue: {result['tiny_add_launch']['host_enqueue_us_per_op']['median']:.2f} us/op")


if __name__ == "__main__":
    main()
