"""V1 numerical checks against the same loaded pinned reference weights."""

import asyncio

import httpx
import pytest
import torch

from ridgepoint.baseline.hf_backend import HFBackend
from ridgepoint.runtime.backend import V1Backend
from ridgepoint.runtime.qwen2 import Qwen2Runtime
from ridgepoint.server.app import create_app


@pytest.fixture(scope="module")
def reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen2 numerics")
    backend = HFBackend()
    backend.load()
    yield backend, Qwen2Runtime(backend.model)
    del backend
    torch.cuda.empty_cache()


def _assert_logits(actual, expected):
    torch.testing.assert_close(actual, expected.float(), atol=0.05, rtol=0.01)


def _reference_step(model, ids, past=None):
    length = len(ids) if past is None else past.get_seq_length() + 1
    return model(input_ids=torch.tensor([ids], device="cuda"),
                 attention_mask=torch.ones((1, length), dtype=torch.long, device="cuda"),
                 past_key_values=past, use_cache=True)


@pytest.mark.parametrize("length", [1, 4, 64])
def test_single_prefill_and_incremental_logits(reference, length):
    backend, runtime = reference
    ids = list(range(101, 101 + length))
    with torch.inference_mode():
        expected = _reference_step(backend.model, ids)
        cache, actual = runtime.prefill([ids], [4])
        _assert_logits(actual, expected.logits[:, -1])
        token = actual.argmax(dim=-1)
        next_expected = _reference_step(backend.model, [int(token.item())],
                                        expected.past_key_values)
        next_actual = runtime.decode(token, cache, torch.tensor([True], device="cuda"))
        _assert_logits(next_actual, next_expected.logits[:, -1])
        assert cache.lengths.tolist() == [length + 1]


def test_equal_length_batch_prefill_and_decode(reference):
    backend, runtime = reference
    prompts = [list(range(101, 165)), list(range(201, 265))]
    ids = torch.tensor(prompts, device="cuda")
    with torch.inference_mode():
        expected = backend.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 use_cache=True)
        cache, actual = runtime.prefill(prompts, [4, 4])
        _assert_logits(actual, expected.logits[:, -1])
        tokens = actual.argmax(dim=-1)
        next_expected = backend.model(
            input_ids=tokens[:, None],
            attention_mask=torch.ones((2, 65), dtype=torch.long, device="cuda"),
            past_key_values=expected.past_key_values, use_cache=True,
        )
        next_actual = runtime.decode(tokens, cache, torch.tensor([True, True], device="cuda"))
        _assert_logits(next_actual, next_expected.logits[:, -1])
        assert cache.lengths.tolist() == [65, 65]


def test_mixed_length_batch_matches_individual_reference(reference):
    backend, runtime = reference
    prompts = [list(range(101, 105)), list(range(201, 208))]
    with torch.inference_mode():
        cache, actual = runtime.prefill(prompts, [4, 4])
        tokens = actual.argmax(dim=-1)
        next_actual = runtime.decode(tokens, cache, torch.tensor([True, True], device="cuda"))
        for row, ids in enumerate(prompts):
            expected = _reference_step(backend.model, ids)
            _assert_logits(actual[row:row + 1], expected.logits[:, -1])
            next_expected = _reference_step(backend.model, [int(tokens[row].item())],
                                            expected.past_key_values)
            _assert_logits(next_actual[row:row + 1], next_expected.logits[:, -1])
        assert cache.lengths.tolist() == [5, 8]


def test_greedy_token_sequence_matches_reference(reference):
    backend, runtime = reference
    prompt = [101, 102, 103, 104]
    with torch.inference_mode():
        cache, logits = runtime.prefill([prompt], [4])
        generated = []
        reference_cache = None
        for step in range(4):
            expected = _reference_step(backend.model,
                                       prompt if step == 0 else [generated[-1]],
                                       reference_cache)
            reference_cache = expected.past_key_values
            assert int(logits.argmax().item()) == int(expected.logits[:, -1].argmax().item())
            generated.append(int(logits.argmax().item()))
            if step < 3:
                logits = runtime.decode(torch.tensor([generated[-1]], device="cuda"), cache,
                                        torch.tensor([True], device="cuda"))


def test_v1_http_stream_batches_two_requests(reference):
    backend, runtime = reference

    async def scenario():
        v1 = V1Backend(max_batch_size=2, batch_window_ms=100)
        v1.model = backend.model
        v1.tokenizer = backend.tokenizer
        v1.runtime = runtime
        v1.load = lambda: None
        app = create_app(v1)
        async with app.router.lifespan_context(app):
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                       base_url="http://test")
            async with client:
                async def request(ids):
                    async with client.stream("POST", "/v1/completions", json={
                        "prompt_ids": ids, "max_tokens": 2, "ignore_eos": True,
                    }) as response:
                        assert response.status_code == 200
                        data = [line[6:] async for line in response.aiter_lines()
                                if line.startswith("data: ")]
                        return response.headers["x-request-id"], data

                import json

                results = await asyncio.gather(request([101, 102, 103, 104]),
                                               request([201, 202, 203, 204]))
                assert results[0][0] != results[1][0]
                for request_id, data in results:
                    events = [json.loads(item) for item in data]
                    assert [event["type"] for event in events] == ["token", "token", "done"]
                    assert all(event["request_id"] == request_id for event in events)
                    assert events[-1]["batch_size"] == 2
                    assert events[-1]["usage"]["completion_tokens"] == 2
                async with client.stream("POST", "/v1/completions", json={
                    "prompt_ids": [101, 102], "max_tokens": 3,
                    "temperature": 0.7, "top_p": 0.9, "ignore_eos": False,
                }) as response:
                    sampled = [json.loads(line[6:]) async for line in response.aiter_lines()
                               if line.startswith("data: ")]
                assert sampled[-1]["type"] == "done"
                assert 1 <= sampled[-1]["usage"]["completion_tokens"] <= 3
                available = v1._available_kv_bytes
                v1._available_kv_bytes = lambda: 0
                rejected = await client.post("/v1/completions", json={
                    "prompt_ids": [101, 102], "max_tokens": 3,
                })
                assert rejected.status_code == 503
                v1._available_kv_bytes = available

    asyncio.run(scenario())
