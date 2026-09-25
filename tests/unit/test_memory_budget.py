import json
from pathlib import Path

import pytest

from ridgepoint.memory_budget import available_pool_bytes, qwen2_memory


CONFIG = Path("configs/models/qwen2.5-1.5b-instruct.config.json")


def test_official_qwen_config_geometry():
    memory = qwen2_memory(json.loads(CONFIG.read_text(encoding="utf-8")))
    assert memory.head_dim == 128
    assert memory.kv_heads == 2
    assert memory.kv_bytes_per_token == 28 * 1024
    assert memory.block_bytes == 448 * 1024
    assert 1_500_000_000 < memory.parameters < 1_600_000_000
    assert memory.weight_bytes == 2 * memory.parameters


def test_pool_rounds_down_to_whole_blocks():
    memory = qwen2_memory(json.loads(CONFIG.read_text(encoding="utf-8")))
    assert memory.tokens_for_pool(memory.block_bytes - 1) == 0
    assert memory.tokens_for_pool(memory.block_bytes) == 16
    assert memory.tokens_for_pool(2 * memory.block_bytes + 1) == 32


def test_untied_embedding_adds_head_and_invalid_geometry_rejected():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    tied = qwen2_memory(config)
    config["tie_word_embeddings"] = False
    assert qwen2_memory(config).parameters - tied.parameters == 151936 * 1536
    config["num_attention_heads"] = 11
    with pytest.raises(ValueError):
        qwen2_memory(config)


def test_available_pool_never_negative():
    assert available_pool_bytes(1000, 400, 200, 100, 50) == 250
    assert available_pool_bytes(100, 400, 200, 100) == 0
    with pytest.raises(ValueError):
        available_pool_bytes(-1, 0, 0, 0)
