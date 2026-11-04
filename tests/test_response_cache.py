"""Real async scheduling/bounds tests for complete-response reuse; no model."""

import asyncio

import pytest
from starlette.responses import JSONResponse, Response

from omnivoice_triton.serving.response_cache import (
    CompleteResponseCache,
    ResponseCacheFullError,
)


def wav(size=100):
    return Response(b"WAVE" + b"\0" * (size - 4), media_type="audio/wav")


def test_one_thousand_duplicate_waiters_share_one_complete_response():
    async def run():
        cache = CompleteResponseCache(max_bytes=10000)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return wav()

        tasks = [asyncio.create_task(cache.run("same", factory)) for _ in range(1000)]
        await entered.wait()
        await asyncio.sleep(0)
        assert cache.snapshot()["waiters"] == 1000
        assert calls == 1
        release.set()
        results = await asyncio.gather(*tasks)
        assert len({r.body for r, _ in results}) == 1
        assert sum(source == "generated" for _, source in results) == 1
        assert sum(source == "coalesced" for _, source in results) == 999
        assert (await cache.run("same", factory))[1] == "cache"
        assert calls == 1
        await cache.close()

    asyncio.run(run())


def test_byte_cap_lru_expiry_and_oversize_bypass():
    async def run():
        now = [0.0]
        cache = CompleteResponseCache(max_bytes=200, ttl_s=10, clock=lambda: now[0])
        calls = 0

        async def factory(size=100):
            nonlocal calls
            calls += 1
            return wav(size)

        await cache.run("a", factory)
        await cache.run("b", factory)
        assert (await cache.run("a", factory))[1] == "cache"
        await cache.run("c", factory)
        assert cache.snapshot()["bytes"] == 200
        assert (await cache.run("b", factory))[1] == "generated"
        now[0] = 11
        assert (await cache.run("b", factory))[1] == "generated"
        for _ in range(2):
            assert (await cache.run("big", lambda: factory(300)))[1] == "generated"
        assert cache.snapshot()["bytes"] <= 200
        await cache.close()

    asyncio.run(run())


def test_cancelled_leader_does_not_cancel_other_waiters():
    async def run():
        cache = CompleteResponseCache(max_bytes=1000)
        entered, release = asyncio.Event(), asyncio.Event()

        async def factory():
            entered.set()
            await release.wait()
            return wav()

        leader = asyncio.create_task(cache.run("a", factory))
        await entered.wait()
        follower = asyncio.create_task(cache.run("a", factory))
        await asyncio.sleep(0)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        release.set()
        assert (await follower)[1] == "coalesced"
        assert (await cache.run("a", factory))[1] == "cache"
        assert cache.snapshot()["waiters"] == 0
        await cache.close()

    asyncio.run(run())


def test_waiter_and_unique_key_admission_limits_recover():
    async def run():
        cache = CompleteResponseCache(
            max_bytes=1000, max_inflight_keys=1, max_waiters=2
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def factory():
            entered.set()
            await release.wait()
            return wav()

        a = asyncio.create_task(cache.run("a", factory))
        await entered.wait()
        with pytest.raises(ResponseCacheFullError, match="key"):
            await cache.run("b", factory)
        b = asyncio.create_task(cache.run("a", factory))
        await asyncio.sleep(0)
        with pytest.raises(ResponseCacheFullError, match="waiter"):
            await cache.run("a", factory)
        release.set()
        await asyncio.gather(a, b)
        assert (await cache.run("b", factory))[1] == "generated"
        await cache.close()

    asyncio.run(run())


def test_errors_are_not_stored_and_close_awaits_detached_jobs():
    async def run():
        cache = CompleteResponseCache(max_bytes=1000)
        calls = 0

        async def failed():
            nonlocal calls
            calls += 1
            raise ValueError("inference failed")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cache.run("failed", failed)
        assert calls == 2

        async def json_result():
            return JSONResponse({"error": "busy"}, status_code=503)

        for _ in range(2):
            assert (await cache.run("json", json_result))[1] == "generated"
        entered, release = asyncio.Event(), asyncio.Event()

        async def factory():
            entered.set()
            await release.wait()
            return wav()

        caller = asyncio.create_task(cache.run("done", factory))
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        closing = asyncio.create_task(cache.close())
        await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(RuntimeError, match="shutting down"):
            await cache.run("new", factory)
        release.set()
        await closing
        assert cache.snapshot()["entries"] == cache.snapshot()["inflight_keys"] == 0

    asyncio.run(run())


def test_instant_factory_with_eager_tasks():
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager tasks need Python3.12")

    async def run():
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        loop.set_task_factory(asyncio.eager_task_factory)
        cache = CompleteResponseCache(max_bytes=1000)

        async def factory():
            return wav()

        try:
            assert (await cache.run("a", factory))[1] == "generated"
            assert (await cache.run("a", factory))[1] == "cache"
            await cache.close()
        finally:
            loop.set_task_factory(previous)

    asyncio.run(run())


@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf")])
def test_invalid_cache_lifetimes_are_rejected(ttl):
    with pytest.raises(ValueError):
        CompleteResponseCache(max_bytes=1000, ttl_s=ttl)


def test_entry_count_bounds_header_and_key_metadata():
    async def run():
        cache = CompleteResponseCache(max_bytes=100000, max_entries=2)

        async def factory():
            return wav()

        for key in ["a", "b", "c"]:
            await cache.run(key, factory)
        assert cache.snapshot()["entries"] == 2
        assert (await cache.run("a", factory))[1] == "generated"
        await cache.close()

    asyncio.run(run())


def test_shutdown_timeout_keeps_live_producer_owned_until_restart():
    async def run():
        cache = CompleteResponseCache(max_bytes=1000)
        entered, release = asyncio.Event(), asyncio.Event()

        async def factory():
            entered.set()
            await release.wait()
            return wav()

        caller = asyncio.create_task(cache.run("a", factory))
        await entered.wait()
        assert not await cache.close(timeout_s=0.01)
        assert cache.snapshot()["closed"] and cache.snapshot()["inflight_keys"] == 1
        assert not caller.done()
        release.set()
        await caller
        assert await cache.close(timeout_s=1)

    asyncio.run(run())
