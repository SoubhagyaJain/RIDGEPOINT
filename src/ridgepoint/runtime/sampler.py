"""Per-request greedy or temperature/top-p token selection."""

from __future__ import annotations

import torch


def sample(logits: torch.Tensor, *, temperature: float, top_p: float,
           generator: torch.Generator | None = None) -> int:
    if logits.ndim != 1 or not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("invalid logits or sampling settings")
    if temperature == 0:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
        remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum()
        picked = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_ids[picked].item())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
