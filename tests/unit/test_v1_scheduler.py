import asyncio
import threading

from ridgepoint.baseline.hf_backend import PreparedPrompt
from ridgepoint.runtime.backend import V1Backend


def test_collected_batch_splits_at_kv_cap():
    class FakeRuntime:
        @staticmethod
        def cache_bytes(batch, capacity):
            return batch * capacity

    async def scenario():
        backend = V1Backend(max_batch_size=4, batch_window_ms=30)
        backend.runtime = FakeRuntime()
        backend._available_kv_bytes = lambda: 8
        batch_sizes = []
        backend._run_batch = lambda jobs: batch_sizes.append(len(jobs))
        prompt = PreparedPrompt([101, 102])
        try:
            await asyncio.gather(*(backend.generate(
                prompt, max_tokens=2, temperature=0, top_p=1, ignore_eos=True,
                aborted=threading.Event(), emit=lambda event: None,
            ) for _ in range(4)))
        finally:
            await backend.close()
        assert batch_sizes == [2, 2]

    asyncio.run(scenario())
