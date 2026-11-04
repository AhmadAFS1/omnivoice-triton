"""Opt-in byte-bounded complete-response reuse and duplicate coalescing.

No audio transformation is performed. Each cache belongs to one loaded model/app.
The HTTP layer validates registered prompt identity and excludes uploads/save-dir
side effects. Reuse is explicit because stochastic generation becomes repeatable
for the lifetime of an entry; it does not increase novel-inference throughput.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from starlette.responses import Response

ResponseSource = Literal["generated", "coalesced", "cache"]


class ResponseCacheFullError(RuntimeError):
    """The bounded shared-response admission limit has been reached."""


@dataclass(frozen=True)
class CompleteResponse:
    body: bytes
    headers: dict[str, str]
    status_code: int
    media_type: str | None

    @classmethod
    def capture(cls, response: Response) -> CompleteResponse:
        return cls(
            bytes(response.body),
            dict(response.headers),
            response.status_code,
            response.media_type,
        )

    @property
    def cacheable(self) -> bool:
        return (
            self.status_code == 200
            and self.media_type == "audio/wav"
            and len(self.body) > 44
        )


class CompleteResponseCache:
    """Event-loop-owned shared jobs, with thread-safe health snapshots.

    A caller cancellation detaches that caller, never cancels a job needed by
    other callers. Shutdown awaits all admitted jobs before releasing the model.
    Both inflight keys and total waiters are bounded independently of GPU queues.
    TTL zero allows inflight coalescing without storing completed responses.
    """

    def __init__(
        self,
        *,
        max_bytes: int = 0,
        ttl_s: float = 600,
        max_inflight_keys: int = 64,
        max_waiters: int = 4096,
        max_entries: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            max_bytes < 0
            or ttl_s < 0
            or not math.isfinite(ttl_s)
            or max_inflight_keys < 1
            or max_waiters < 1
            or max_entries < 1
        ):
            raise ValueError("Invalid complete-response cache limits")
        self.max_bytes, self.ttl_s = int(max_bytes), float(ttl_s)
        self.max_inflight_keys, self.max_waiters = max_inflight_keys, max_waiters
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, CompleteResponse]] = OrderedDict()
        self._tasks: dict[str, asyncio.Task[CompleteResponse]] = {}
        self._bytes = self._waiters = 0
        self._closed = False
        self._hits = self._generated = self._coalesced = self._evicted = 0

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def _expire_locked(self) -> None:
        now = self._clock()
        for key, (expires, entry) in list(self._entries.items()):
            if expires <= now:
                del self._entries[key]
                self._bytes -= len(entry.body)
                self._evicted += 1

    def _store_locked(self, key: str, entry: CompleteResponse) -> None:
        if not entry.cacheable or not self.ttl_s or len(entry.body) > self.max_bytes:
            return
        self._expire_locked()
        previous = self._entries.pop(key, None)
        if previous:
            self._bytes -= len(previous[1].body)
        self._entries[key] = (self._clock() + self.ttl_s, entry)
        self._bytes += len(entry.body)
        while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
            _, (_, removed) = self._entries.popitem(last=False)
            self._bytes -= len(removed.body)
            self._evicted += 1

    async def _produce(
        self, key: str, factory: Callable[[], Awaitable[Response]]
    ) -> CompleteResponse:
        # Even an eager task factory must return control before we re-enter the
        # admission lock; never run an instant/error factory inside that lock.
        await asyncio.sleep(0)
        try:
            entry = CompleteResponse.capture(await factory())
            with self._lock:
                self._store_locked(key, entry)
            return entry
        finally:
            with self._lock:
                if self._tasks.get(key) is asyncio.current_task():
                    del self._tasks[key]

    @staticmethod
    def _observe(task: asyncio.Task) -> None:
        # A disconnected final waiter must not leave an unobserved exception.
        if not task.cancelled():
            task.exception()

    async def run(
        self, key: str, factory: Callable[[], Awaitable[Response]]
    ) -> tuple[CompleteResponse, ResponseSource]:
        if not self.enabled:
            return CompleteResponse.capture(await factory()), "generated"
        with self._lock:
            if self._closed:
                raise RuntimeError("Complete-response cache is shutting down")
            cached = self._entries.get(key)
            if cached is not None and cached[0] <= self._clock():
                del self._entries[key]
                self._bytes -= len(cached[1].body)
                self._evicted += 1
                cached = None
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached[1], "cache"
            if self._waiters >= self.max_waiters:
                raise ResponseCacheFullError("Shared response waiter limit reached")
            task = self._tasks.get(key)
            if task is None:
                if len(self._tasks) >= self.max_inflight_keys:
                    raise ResponseCacheFullError("Shared response key limit reached")
                task = asyncio.create_task(self._produce(key, factory))
                self._tasks[key] = task
                task.add_done_callback(self._observe)
                self._generated += 1
                source: ResponseSource = "generated"
            else:
                self._coalesced += 1
                source = "coalesced"
            self._waiters += 1
        try:
            return await asyncio.shield(task), source
        finally:
            with self._lock:
                self._waiters -= 1

    async def close(self, timeout_s: float = 30.0) -> bool:
        """Stop admission and drain within the worker's shutdown budget.

        False leaves producer tasks and the live model intact. The HTTP lifespan
        must fail shutdown and require a process restart, never unload CUDA while
        a producer may still be using it.
        """
        with self._lock:
            self._closed = True
            tasks = list(self._tasks.values())
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_s))
            if pending:
                return False
        with self._lock:
            self._entries.clear()
            self._bytes = 0
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return dict(
                enabled=self.enabled,
                max_bytes=self.max_bytes,
                bytes=self._bytes,
                entries=len(self._entries),
                max_entries=self.max_entries,
                ttl_s=self.ttl_s,
                inflight_keys=len(self._tasks),
                waiters=self._waiters,
                max_inflight_keys=self.max_inflight_keys,
                max_waiters=self.max_waiters,
                hits=self._hits,
                producer_jobs_started=self._generated,
                coalesced=self._coalesced,
                evictions=self._evicted,
                closed=self._closed,
            )
