"""BF16 Qwen2 forward pass using pinned Hugging Face weights, not its forward method."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def kv_bytes(*, layers: int, batch: int, kv_heads: int, capacity: int,
             head_dim: int, element_bytes: int = 2) -> int:
    return 2 * layers * batch * kv_heads * capacity * head_dim * element_bytes


@dataclass
class ContiguousKV:
    # [layer, key_or_value, batch, kv_head, position, head_dim]
    data: torch.Tensor
    lengths: torch.Tensor  # logical token count per request

    @property
    def capacity(self) -> int:
        return self.data.shape[-2]


def _linear(x: torch.Tensor, module) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


def _norm(x: torch.Tensor, module) -> torch.Tensor:
    source_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.variance_epsilon)
    return module.weight * x.to(source_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Qwen2Runtime:
    """Runs model layers explicitly and owns every KV write.

    The Hugging Face object supplies immutable weights and config only. It is never
    called as a model in the V1 serving path.
    """

    def __init__(self, model):
        self.weights = model
        self.config = model.config
        self.layers = model.model.layers
        self.hidden_size = self.config.hidden_size
        self.query_heads = self.config.num_attention_heads
        self.kv_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.query_heads
        self.groups = self.query_heads // self.kv_heads
        self.dtype = model.model.embed_tokens.weight.dtype
        self.device = model.model.embed_tokens.weight.device
        steps = torch.arange(0, self.head_dim, 2, dtype=torch.float32,
                             device=self.device)
        self.inv_freq = 1.0 / (self.config.rope_theta ** (steps / self.head_dim))

    def cache_bytes(self, batch: int, capacity: int) -> int:
        return kv_bytes(layers=len(self.layers), batch=batch, kv_heads=self.kv_heads,
                        capacity=capacity, head_dim=self.head_dim,
                        element_bytes=torch.empty((), dtype=self.dtype).element_size())

    def _rope_table(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Match Qwen2's float32 RoPE frequency calculation and BF16 output.
        freq = positions.float().unsqueeze(-1) * self.inv_freq
        angle = torch.cat((freq, freq), dim=-1)
        return (angle.cos().to(self.dtype).unsqueeze(1),
                angle.sin().to(self.dtype).unsqueeze(1))

    def _layer(self, hidden: torch.Tensor, layer, index: int,
               cache: ContiguousKV, rope: tuple[torch.Tensor, torch.Tensor],
               mask: torch.Tensor | None, slot: int | None = None) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        cos, sin = rope
        residual = hidden
        normed = _norm(hidden, layer.input_layernorm)
        attn = layer.self_attn
        q = _linear(normed, attn.q_proj).view(batch, seq_len, self.query_heads,
                                               self.head_dim).transpose(1, 2)
        k = _linear(normed, attn.k_proj).view(batch, seq_len, self.kv_heads,
                                               self.head_dim).transpose(1, 2)
        v = _linear(normed, attn.v_proj).view(batch, seq_len, self.kv_heads,
                                               self.head_dim).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        if slot is None:
            cache.data[index, 0, :, :, :seq_len, :] = k
            cache.data[index, 1, :, :, :seq_len, :] = v
            key_len = seq_len
        else:
            # Every row of a decode call shares one slot. A plain integer index
            # stays on the GPU queue; boolean-mask indexing here would force a
            # host sync in every layer.
            cache.data[index, 0, :, :, slot, :] = k[:, :, 0, :]
            cache.data[index, 1, :, :, slot, :] = v[:, :, 0, :]
            key_len = slot + 1
        keys = cache.data[index, 0, :, :, :key_len, :]
        values = cache.data[index, 1, :, :, :key_len, :]
        use_gqa = self.groups > 1
        if not use_gqa:
            keys = keys.repeat_interleave(self.groups, dim=1)
            values = values.repeat_interleave(self.groups, dim=1)
        output = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask, dropout_p=0.0,
            is_causal=slot is None and mask is None and seq_len > 1,
            enable_gqa=use_gqa,
        )
        output = output.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        hidden = residual + _linear(output, attn.o_proj)
        residual = hidden
        normed = _norm(hidden, layer.post_attention_layernorm)
        gated = F.silu(_linear(normed, layer.mlp.gate_proj))
        up = _linear(normed, layer.mlp.up_proj)
        return residual + _linear(gated * up, layer.mlp.down_proj)

    @torch.inference_mode()
    def prefill(self, prompts: list[list[int]], max_new_tokens: list[int]
                ) -> tuple[ContiguousKV, torch.Tensor]:
        if not prompts or len(prompts) != len(max_new_tokens):
            raise ValueError("one output limit is required per nonempty prompt")
        lengths = torch.tensor([len(ids) for ids in prompts], dtype=torch.long,
                               device=self.device)
        if any(not ids for ids in prompts):
            raise ValueError("prompts must not be empty")
        capacity = max(len(ids) + count for ids, count in zip(prompts, max_new_tokens))
        batch = len(prompts)
        seq_len = max(map(len, prompts))
        ids = torch.zeros((batch, seq_len), dtype=torch.long, device=self.device)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)] = torch.tensor(prompt, device=self.device)
        data = torch.empty((len(self.layers), 2, batch, self.kv_heads, capacity,
                            self.head_dim), dtype=self.dtype, device=self.device)
        cache = ContiguousKV(data=data, lengths=lengths)
        if not bool((lengths == seq_len).all()):
            # Equal-length cohorts use one SDPA call. Different prompt lengths
            # prefill separately, preserving the reference kernel's BF16 math.
            rows = []
            for row, (prompt, count) in enumerate(zip(prompts, max_new_tokens)):
                single_cache, single_logits = self.prefill([prompt], [count])
                cache.data[:, :, row:row + 1, :, :len(prompt), :] = (
                    single_cache.data[:, :, :, :, :len(prompt), :]
                )
                rows.append(single_logits)
            return cache, torch.cat(rows, dim=0)
        positions = torch.arange(seq_len, device=self.device)[None, :].expand(batch, -1)
        query_index = positions[:, :, None]
        key_index = positions[:, None, :]
        mask = (None if bool((lengths == seq_len).all()) else
                ((key_index <= query_index) &
                 (key_index < lengths[:, None, None]))[:, None, :, :])
        rope = self._rope_table(positions)
        hidden = F.embedding(ids, self.weights.model.embed_tokens.weight)
        for index, layer in enumerate(self.layers):
            hidden = self._layer(hidden, layer, index, cache, rope, mask)
        hidden = _norm(hidden, self.weights.model.norm)
        final = hidden[torch.arange(batch, device=self.device), lengths - 1]
        logits = F.linear(final, self.weights.lm_head.weight).float()
        return cache, logits

    @torch.inference_mode()
    def decode(self, input_ids: torch.Tensor, cache: ContiguousKV,
               active: torch.Tensor) -> torch.Tensor:
        batch = cache.lengths.numel()
        if input_ids.shape != (batch,) or active.shape != (batch,):
            raise ValueError("decode input and active mask must match cache batch")
        # Read host-side state once per step; the layer loop then queues GPU work
        # without waiting on it.
        lengths = cache.lengths.tolist()
        live = active.tolist()
        if not all(live) or len(set(lengths)) > 1:
            rows = []
            for row in range(batch):
                if live[row] and batch > 1:
                    view = ContiguousKV(cache.data[:, :, row:row + 1],
                                        cache.lengths[row:row + 1])
                    rows.append(self.decode(input_ids[row:row + 1], view,
                                            active[row:row + 1]))
                else:
                    rows.append(torch.zeros((1, self.config.vocab_size),
                                            dtype=torch.float32, device=self.device))
            return torch.cat(rows, dim=0)
        slot = lengths[0]
        if slot >= cache.capacity:
            raise ValueError("KV capacity exceeded")
        rope = self._rope_table(cache.lengths[:, None])
        hidden = F.embedding(input_ids[:, None], self.weights.model.embed_tokens.weight)
        for index, layer in enumerate(self.layers):
            hidden = self._layer(hidden, layer, index, cache, rope, None, slot)
        cache.lengths += 1
        hidden = _norm(hidden, self.weights.model.norm)[:, 0, :]
        return F.linear(hidden, self.weights.lm_head.weight).float()
