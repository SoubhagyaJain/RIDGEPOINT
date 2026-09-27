import asyncio
import json

import httpx

from bench.backends.null import NullBackend
from bench.client.replay import replay
from ridgepoint.server.app import create_app


def test_null_backend_stream_validation_and_replay():
    async def scenario():
        app = create_app(NullBackend(first_token_delay_ms=1, token_delay_ms=1))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test") as client:
                bad = await client.post("/v1/completions", json={"prompt_ids": [],
                                                                  "max_tokens": 3})
                assert bad.status_code == 422
                async with client.stream("POST", "/v1/completions",
                                         json={"prompt_ids": [1, 2], "max_tokens": 3}) as response:
                    assert response.status_code == 200
                    assert response.headers["x-request-id"]
                    lines = [line async for line in response.aiter_lines()
                             if line.startswith("data: ")]
                events = [json.loads(line[6:]) for line in lines]
                assert [event["type"] for event in events] == ["token"] * 3 + ["done"]
                assert events[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 3}
                assert all(e["request_id"] == response.headers["x-request-id"] for e in events)
            trace = [{"req_id": f"r{i}", "t_offset_ms": i * 10,
                      "prompt_ids": [1, 2], "max_tokens": 3} for i in range(2)]
            result = await replay(trace, "http://test/v1/completions", transport=transport)
            assert len(result) == 2
            assert all(r["done"] and len(r["token_events"]) == 3 for r in result)
            assert result[0]["send_ns"] < result[1]["send_ns"]

    asyncio.run(scenario())
