"""Benchmark-only keepalive and CPU batch counters; frozen product unchanged."""

from __future__ import annotations

import json
import runpy
import sys
import threading
from pathlib import Path


def install_counters(batcher_class):
    original_execute = batcher_class._execute_batch
    original_snapshot = batcher_class.queue_snapshot
    lock = threading.Lock()
    counters = {
        "batches_started": 0,
        "members_started": 0,
        "members_succeeded": 0,
        "members_failed": 0,
    }

    def execute(self, batch):
        with lock:
            counters["batches_started"] += 1
            counters["members_started"] += len(batch)
        try:
            return original_execute(self, batch)
        finally:
            # The pinned implementation reports model outputs through these
            # futures. This observer changes no queue, output or CUDA operation.
            successes = sum(
                request.future.done()
                and not request.future.cancelled()
                and request.future.exception() is None
                for request in batch
            )
            with lock:
                counters["members_succeeded"] += successes
                counters["members_failed"] += len(batch) - successes

    def snapshot(self):
        result = original_snapshot(self)
        with lock:
            result["benchmark_raw_burst_v1"] = dict(counters)
        return result

    batcher_class._execute_batch = execute
    batcher_class.queue_snapshot = snapshot


def main():
    import asyncio
    import importlib.metadata

    import uvicorn

    original = uvicorn.run
    original_load = uvicorn.Config.load
    original_serve = uvicorn.Server.serve
    evidence = Path(sys.argv[sys.argv.index("--evidence") + 1])

    def observed_load(self):
        original_load(self)
        protocol = self.http_protocol_class
        (evidence.parent / "server-transport-runtime.json").write_text(
            json.dumps(
                {
                    "timeout_keep_alive": self.timeout_keep_alive,
                    "resolved_HTTP_protocol_class": protocol.__module__
                    + "."
                    + protocol.__qualname__,
                    "uvicorn_version": importlib.metadata.version("uvicorn"),
                },
                indent=2,
            )
            + "\n"
        )

    async def observed_serve(self, *args, **kwargs):
        loop = type(asyncio.get_running_loop())
        (evidence.parent / "server-event-loop.json").write_text(
            json.dumps(
                {
                    "actual_event_loop_class": loop.__module__
                    + "."
                    + loop.__qualname__,
                },
                indent=2,
            )
            + "\n"
        )
        return await original_serve(self, *args, **kwargs)

    def configured(*args, **kwargs):
        # seeded_server has applied its seed and imported API before this point;
        # lifespan has not constructed the batcher yet.
        from omnivoice_triton.serving.batching import GenerationBatcher

        install_counters(GenerationBatcher)
        kwargs["timeout_keep_alive"] = 600
        (evidence.parent / "server-benchmark-wrapper.json").write_text(
            json.dumps(
                {
                    "timeout_keep_alive": 600,
                    "product_default_changed": False,
                    "seed_applied_before_API_import": True,
                    "counters": "CPU-only observer around pinned _execute_batch/future completion and queue_snapshot; no model/queue/output changes",
                    "scope": "This benchmark process only",
                    "uvicorn_kwargs": kwargs,
                },
                indent=2,
            )
            + "\n"
        )
        return original(*args, **kwargs)

    uvicorn.run = configured
    uvicorn.Config.load = observed_load
    uvicorn.Server.serve = observed_serve
    runpy.run_path(
        str(Path(__file__).with_name("seeded_server.py")), run_name="__main__"
    )


if __name__ == "__main__":
    main()
