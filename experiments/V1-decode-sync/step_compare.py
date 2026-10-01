"""Time one decode step for Hugging Face and V1 on the same loaded weights.

Engines are interleaved inside every repetition so slow host drift hits all of them.
`--baseline-rev` also loads `qwen2.py` from an earlier commit for a before/after check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from ridgepoint.baseline.hf_backend import HFBackend
from ridgepoint.runtime.qwen2 import Qwen2Runtime

SYNC_OPS = ("aten::nonzero", "aten::item", "aten::_local_scalar_dense")


def runtime_at(rev: str):
    source = subprocess.run(["git", "show", f"{rev}:src/ridgepoint/runtime/qwen2.py"],
                            capture_output=True, text=True, check=True).stdout
    module = types.ModuleType(f"qwen2_at_{rev}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"qwen2.py@{rev}", "exec"), module.__dict__)
    return module.Qwen2Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--baseline-rev", default=None)
    args = parser.parse_args()

    backend = HFBackend()
    backend.load()
    model = backend.model
    corpus = Path("bench/workloads/corpus.txt").read_text(encoding="utf-8")
    ids = backend.tokenizer.encode(corpus, add_special_tokens=False)[:args.prompt_tokens]
    runtimes = {"v1": Qwen2Runtime(model)}
    if args.baseline_rev:
        runtimes[f"v1@{args.baseline_rev}"] = runtime_at(args.baseline_rev)(model)

    @torch.inference_mode()
    def hf_steps(batch: int) -> tuple[list[float], list[list[int]]]:
        prompt = torch.tensor([ids] * batch, device="cuda")
        out = model(input_ids=prompt, use_cache=True)
        past, token = out.past_key_values, out.logits[:, -1].argmax(-1)
        torch.cuda.synchronize()
        steps, tokens = [], [token.tolist()]
        for _ in range(args.output_tokens - 1):
            start = time.perf_counter_ns()
            out = model(input_ids=token[:, None], past_key_values=past, use_cache=True)
            past, token = out.past_key_values, out.logits[:, -1].argmax(-1)
            tokens.append(token.tolist())
            torch.cuda.synchronize()
            steps.append((time.perf_counter_ns() - start) / 1e6)
        return steps, tokens

    @torch.inference_mode()
    def v1_steps(runtime, batch: int) -> tuple[list[float], list[list[int]]]:
        cache, logits = runtime.prefill([ids] * batch, [args.output_tokens] * batch)
        token = logits.argmax(-1)
        torch.cuda.synchronize()
        active = torch.ones(batch, dtype=torch.bool, device="cuda")
        steps, tokens = [], [token.tolist()]
        for _ in range(args.output_tokens - 1):
            start = time.perf_counter_ns()
            token = runtime.decode(token, cache, active).argmax(-1)
            tokens.append(token.tolist())
            torch.cuda.synchronize()
            steps.append((time.perf_counter_ns() - start) / 1e6)
        return steps, tokens

    hf_steps(1)
    for runtime in runtimes.values():
        v1_steps(runtime, 1)
    samples: dict[str, list[list[float]]] = {}
    for batch in (1, 4):
        for _ in range(args.reps):
            steps, reference = hf_steps(batch)
            samples.setdefault(f"b{batch} hf-forward", []).append(steps)
            for name, runtime in runtimes.items():
                steps, tokens = v1_steps(runtime, batch)
                if tokens != reference:
                    raise ValueError(f"{name} greedy tokens differ from the reference")
                samples.setdefault(f"b{batch} {name}", []).append(steps)

    summary = {}
    for name, reps in samples.items():
        medians = [statistics.median(rep) for rep in reps]
        base = [statistics.median(rep) for rep in samples[f"{name.split()[0]} hf-forward"]]
        ratio = statistics.median(a / b for a, b in zip(medians, base))
        summary[name] = {"median_ms": statistics.median(medians), "min_ms": min(medians),
                         "max_ms": max(medians), "ratio_to_hf_same_rep": ratio}
        print(f"{name:<22} step median={summary[name]['median_ms']:7.2f} ms  "
              f"rep range={min(medians):7.2f}..{max(medians):7.2f}  ratio to HF={ratio:.2f}")

    ops = {}
    for name, runtime in runtimes.items():
        with torch.inference_mode():
            cache, logits = runtime.prefill([ids], [args.output_tokens])
            token = logits.argmax(-1)
            active = torch.ones(1, dtype=torch.bool, device="cuda")
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                runtime.decode(token, cache, active)
                torch.cuda.synchronize()
        counts = {event.key: event.count for event in prof.key_averages()}
        ops[name] = {"total_op_calls": sum(counts.values()),
                     **{key: counts.get(key, 0) for key in SYNC_OPS}}
        print(f"[ops per decode step] {name}: {ops[name]}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path("results/raw") / f"v1-decode-sync-{stamp}.json"
    output.write_text(json.dumps({
        "args": vars(args), "summary": summary, "ops": ops, "samples_ms": samples,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(),
    }, indent=2), encoding="utf-8")
    print(f"raw samples: {output}")


if __name__ == "__main__":
    main()
