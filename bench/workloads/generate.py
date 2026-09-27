"""Generate exact-length Qwen token-ID traces from a committed text corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from ridgepoint.baseline.hf_backend import MODEL_ID, MODEL_REVISION, model_source

CORPUS = Path(__file__).with_name("corpus.txt")


def load_workload(path: Path) -> dict:
    current = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" not in current:
        return current
    base = load_workload(path.parent / current.pop("base"))
    return {**base, **current}


def draw_length(rng: random.Random, spec: dict) -> int:
    if spec["dist"] == "lognormal":
        value = round(rng.lognormvariate(math.log(spec["median"]), spec["sigma"]))
    elif spec["dist"] == "uniform_int":
        value = rng.randint(spec["min"], spec["max"])
    else:
        raise ValueError(f"unknown distribution: {spec['dist']}")
    return max(spec["min"], min(spec["max"], value))


def token_slice(tokens: list[int], start: int, length: int) -> list[int]:
    return [tokens[(start + i) % len(tokens)] for i in range(length)]


def generate_trace(config: dict, corpus_ids: list[int], *, duration_s: float | None = None,
                   rate_rps: float | None = None, max_requests: int | None = None,
                   prompt_tokens: int | None = None, output_tokens: int | None = None) -> list[dict]:
    if not corpus_ids:
        raise ValueError("corpus must tokenize to at least one token")
    rng = random.Random(config["seed"])
    duration = duration_s if duration_s is not None else config["duration_s"]
    rate = rate_rps if rate_rps is not None else config["arrival"]["rate_rps"]
    if duration <= 0 or rate <= 0:
        raise ValueError("duration and rate must be positive")
    shared = config.get("shared_prefix")
    prefixes = []
    if shared:
        prefixes = [token_slice(corpus_ids, i * 137, shared["prefix_len"])
                    for i in range(shared["num_prefixes"])]
    rows = []
    offset_s = 0.0
    while True:
        offset_s += rng.expovariate(rate)
        if offset_s >= duration or (max_requests is not None and len(rows) >= max_requests):
            break
        p_len = prompt_tokens or draw_length(rng, config["prompt_len"])
        o_len = output_tokens or draw_length(rng, config["output_len"])
        group = None
        prefix = []
        if shared and p_len >= shared["prefix_len"] and rng.random() < shared["fraction"]:
            group = rng.randrange(len(prefixes))
            prefix = prefixes[group]
        suffix = token_slice(corpus_ids, rng.randrange(len(corpus_ids)), p_len - len(prefix))
        rows.append({"req_id": f"{config['name']}-{config['seed']}-{len(rows):06d}",
                     "t_offset_ms": round(offset_s * 1000, 3),
                     "prompt_ids": prefix + suffix, "max_tokens": o_len,
                     "prefix_group": group,
                     "temperature": config["sampling"]["temperature"],
                     "ignore_eos": config["ignore_eos"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("configs/workloads/chat.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/phase2-trace.jsonl"))
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--rate-rps", type=float)
    parser.add_argument("--warmup-s", type=float)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    args = parser.parse_args()
    config = load_workload(args.workload)
    source = model_source()
    revision = {} if source != MODEL_ID else {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **revision)
    corpus_ids = tokenizer.encode(CORPUS.read_text(encoding="utf-8"),
                                  add_special_tokens=False)
    rows = generate_trace(config, corpus_ids, duration_s=args.duration_s,
                          rate_rps=args.rate_rps, max_requests=args.max_requests,
                          prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    # write_bytes preserves identical LF bytes on Windows and Linux for a stable hash.
    args.output.write_bytes(data.encode("utf-8"))
    effective_duration = args.duration_s if args.duration_s is not None else config["duration_s"]
    effective_warmup = (args.warmup_s if args.warmup_s is not None else
                        (config["warmup_s"] if effective_duration > config["warmup_s"] else 0))
    manifest = {"workload": str(args.workload), "seed": config["seed"],
                "model": MODEL_ID, "model_revision": MODEL_REVISION,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
                "workload_config_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                "trace_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "requests": len(rows), "duration_s": effective_duration,
                "warmup_s": effective_warmup,
                "rate_rps": args.rate_rps if args.rate_rps is not None else config["arrival"]["rate_rps"],
                "prompt_tokens_override": args.prompt_tokens,
                "output_tokens_override": args.output_tokens}
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"trace": str(args.output), "manifest": str(manifest_path),
                      "requests": len(rows), "sha256": manifest["trace_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
