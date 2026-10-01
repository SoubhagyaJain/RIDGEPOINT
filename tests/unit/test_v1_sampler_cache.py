import pytest
import torch

from ridgepoint.runtime.qwen2 import kv_bytes
from ridgepoint.runtime.sampler import sample


def test_contiguous_cache_size_uses_qwen_geometry():
    assert kv_bytes(layers=28, batch=1, kv_heads=2, capacity=1, head_dim=128) == 28_672
    assert kv_bytes(layers=28, batch=4, kv_heads=2, capacity=80, head_dim=128) == 9_175_040


def test_sampler_greedy_and_nucleus_cutoff():
    logits = torch.tensor([0.0, 5.0, 1.0])
    assert sample(logits, temperature=0, top_p=1) == 1
    generator = torch.Generator().manual_seed(7)
    assert sample(logits, temperature=1, top_p=0.1, generator=generator) == 1
    with pytest.raises(ValueError):
        sample(logits, temperature=-1, top_p=1)
