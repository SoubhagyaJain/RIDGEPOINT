"""Pure arithmetic for Qwen2 weight and KV memory. No model weights are downloaded."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMemory:
    parameters: int
    weight_bytes: int
    kv_bytes_per_token: int
    head_dim: int
    kv_heads: int
    block_size: int

    @property
    def block_bytes(self) -> int:
        return self.kv_bytes_per_token * self.block_size

    def tokens_for_pool(self, pool_bytes: int) -> int:
        if pool_bytes < 0:
            raise ValueError("pool_bytes must be nonnegative")
        return (pool_bytes // self.block_bytes) * self.block_size


def qwen2_memory(config: dict, *, dtype_bytes: int = 2, block_size: int = 16) -> ModelMemory:
    """Count Qwen2 parameters including QKV biases and tied/untied embeddings.

    Assumes the official Qwen2 architecture: biased Q/K/V, unbiased O/MLP,
    two RMSNorms per layer, and one final RMSNorm.
    """
    h = int(config["hidden_size"])
    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    layers = int(config["num_hidden_layers"])
    intermediate = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    if min(h, q_heads, kv_heads, layers, intermediate, vocab, dtype_bytes, block_size) <= 0:
        raise ValueError("model dimensions, dtype_bytes, and block_size must be positive")
    if h % q_heads or q_heads % kv_heads:
        raise ValueError("invalid GQA head geometry")
    head_dim = h // q_heads
    kv_width = kv_heads * head_dim
    embeddings = vocab * h
    qkv = h * h + h + 2 * (h * kv_width + kv_width)
    output = h * h
    mlp = 3 * h * intermediate
    norms = 2 * h
    parameters = layers * (qkv + output + mlp + norms) + h + embeddings
    if not config.get("tie_word_embeddings", False):
        parameters += embeddings
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return ModelMemory(parameters, parameters * dtype_bytes, kv_bytes_per_token,
                       head_dim, kv_heads, block_size)


def available_pool_bytes(
    free_bytes: int, weight_bytes: int, activation_reserve_bytes: int,
    safety_bytes: int, context_reserve_bytes: int = 0,
) -> int:
    """Conservative planning bound; measured peak activation is needed later."""
    values = (free_bytes, weight_bytes, activation_reserve_bytes, safety_bytes,
              context_reserve_bytes)
    if any(value < 0 for value in values):
        raise ValueError("memory inputs must be nonnegative")
    return max(0, free_bytes - weight_bytes - activation_reserve_bytes
               - safety_bytes - context_reserve_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--pool-gib", type=float, default=1.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    memory = qwen2_memory(config)
    pool_bytes = int(args.pool_gib * 1024**3)
    print(json.dumps({**asdict(memory), "block_bytes": memory.block_bytes,
                      "pool_bytes": pool_bytes,
                      "usable_tokens": memory.tokens_for_pool(pool_bytes)}, indent=2))


if __name__ == "__main__":
    main()
