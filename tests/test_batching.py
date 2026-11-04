"""Tests for the batcher's GPU thread: task routing, faults and frame release."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omnivoice import OmniVoiceGenerationConfig

from omnivoice_triton.models.faster_runner import CudaGraphCaptureError
from omnivoice_triton.models.graph_memory import GraphMemoryAdmissionError
from omnivoice_triton.serving import (
    CudaFaultError,
    GenerationBatcher,
    GenerationBatchKey,
    PendingGeneration,
    QueueFullError,
    is_cuda_fault,
)


@pytest.fixture
def batcher_and_faults():
    faults: list[str] = []
    batcher = GenerationBatcher(SimpleNamespace(), on_cuda_fault=faults.append)
    yield batcher, faults
    batcher.close()


def test_run_on_gpu_thread_runs_on_the_batcher_thread(batcher_and_faults) -> None:
    batcher, faults = batcher_and_faults

    assert (
        batcher.run_on_gpu_thread(lambda: threading.current_thread().name, name="probe")
        == "omnivoice-batcher"
    )
    # nested calls from the batcher thread run inline instead of deadlocking
    assert (
        batcher.run_on_gpu_thread(
            lambda: batcher.run_on_gpu_thread(lambda: 7, name="inner"), name="outer"
        )
        == 7
    )
    assert faults == []


def _pending_generation(request_id: str) -> PendingGeneration:
    return PendingGeneration(
        request_id=request_id,
        mode="design",
        text="Hello.",
        language=None,
        instruct=None,
        voice_clone_prompt=None,
        speed=None,
        duration=None,
        target_tokens=10,
        conditioning_tokens=10,
        max_sequence_length=20,
        batch_key=GenerationBatchKey(
            lane="short_noref",
            num_step=16,
            guidance_scale=2.0,
            t_shift=0.1,
            layer_penalty_factor=5.0,
            position_temperature=5.0,
            class_temperature=0.0,
            denoise=True,
            postprocess_mode="off",
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
        ),
        generation_config=OmniVoiceGenerationConfig(),
    )


@pytest.mark.parametrize("kind", ["generation", "gpu_task"])
def test_queue_limit_rejects_excess_and_worker_recovers(kind: str) -> None:
    """Excess work is rejected without discarding admitted work or poisoning CUDA."""
    entered, release = threading.Event(), threading.Event()

    def blocker():
        entered.set()
        assert release.wait(5)

    model = SimpleNamespace(
        generate=lambda **kw: [np.zeros(10, dtype=np.float32) for _ in kw["text"]]
    )
    batcher = GenerationBatcher(
        model, collect_ms=0, max_pending_requests=1, max_pending_gpu_tasks=1
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(batcher.run_on_gpu_thread, blocker, name="blocker")
            assert entered.wait(2)

            def enqueue():
                if kind == "generation":
                    return batcher.submit(_pending_generation("admitted"))
                return batcher.run_on_gpu_thread(lambda: 42, name="admitted")

            admitted = pool.submit(enqueue)
            try:
                with batcher._condition:
                    assert batcher._condition.wait_for(
                        lambda: bool(
                            batcher._pending if kind == "generation" else batcher._tasks
                        ),
                        timeout=2,
                    )
                snapshot = batcher.queue_snapshot()
                key = (
                    "pending_requests" if kind == "generation" else "pending_gpu_tasks"
                )
                assert snapshot[key] == 1
                with pytest.raises(QueueFullError):
                    enqueue()
                assert batcher.cuda_fault is None
            finally:
                release.set()
            running.result(timeout=2)
            admitted.result(timeout=2)
        assert batcher.run_on_gpu_thread(lambda: "healthy", name="after") == "healthy"
        assert batcher.queue_snapshot()["pending_requests"] == 0
        assert batcher.queue_snapshot()["pending_gpu_tasks"] == 0
    finally:
        release.set()
        batcher.close()


def test_gpu_tasks_and_batches_take_turns() -> None:
    """A stream of prompt-encoding tasks must not starve queued generate requests."""
    order: list[str] = []

    def generate(**kwargs):
        order.append("batch")
        return [np.zeros(10, dtype=np.float32) for _ in kwargs["text"]]

    batcher = GenerationBatcher(SimpleNamespace(generate=generate), collect_ms=0.0)
    gate = threading.Event()

    def task(index: int):
        def run() -> None:
            order.append(f"task{index}")
            time.sleep(0.02)

        return run

    threads = [
        threading.Thread(
            target=batcher.run_on_gpu_thread,
            args=(gate.wait,),
            kwargs={"name": "blocker"},
        )
    ]
    threads[0].start()
    time.sleep(0.1)  # the blocker now holds the batcher thread
    threads += [
        threading.Thread(
            target=batcher.run_on_gpu_thread, args=(task(i),), kwargs={"name": f"t{i}"}
        )
        for i in range(4)
    ]
    threads.append(
        threading.Thread(target=batcher.submit, args=(_pending_generation("g1"),))
    )
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.2)  # everything is queued behind the blocker
    gate.set()
    for thread in threads:
        thread.join(timeout=5.0)
    batcher.close()

    assert order[0] == "batch"
    assert sorted(order[1:]) == ["task0", "task1", "task2", "task3"]


def test_async_cancellation_removes_queued_work_and_releases_capacity() -> None:
    entered, release = threading.Event(), threading.Event()
    generated: list[str] = []

    def generate(**kwargs):
        generated.extend(kwargs["text"])
        return [np.zeros(10, dtype=np.float32) for _ in kwargs["text"]]

    def blocker():
        entered.set()
        assert release.wait(5)

    batcher = GenerationBatcher(
        SimpleNamespace(generate=generate), collect_ms=0, max_pending_requests=1
    )

    async def run():
        canceled = _pending_generation("canceled")
        canceled.text = "Must never run."
        waiter = asyncio.create_task(batcher.submit_async(canceled))
        await asyncio.sleep(0)
        assert batcher.queue_snapshot()["pending_requests"] == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert canceled.future.cancelled()
        assert batcher.queue_snapshot()["pending_requests"] == 0

        replacement = asyncio.create_task(
            batcher.submit_async(_pending_generation("replacement"))
        )
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(replacement, timeout=2)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(batcher.run_on_gpu_thread, blocker, name="blocker")
            assert entered.wait(2)
            asyncio.run(run())
            running.result(timeout=2)
        assert generated == ["Hello."]
        assert batcher.cuda_fault is None
    finally:
        release.set()
        batcher.close()


def test_async_cancellation_during_inference_does_not_break_worker() -> None:
    entered, release = threading.Event(), threading.Event()

    def generate(**kwargs):
        entered.set()
        assert release.wait(5)
        return [np.zeros(10, dtype=np.float32) for _ in kwargs["text"]]

    batcher = GenerationBatcher(SimpleNamespace(generate=generate), collect_ms=0)
    pending = _pending_generation("in-flight")

    async def run():
        waiter = asyncio.create_task(batcher.submit_async(pending))
        assert await asyncio.to_thread(entered.wait, 2)
        assert pending.future.running()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        # CUDA inference is already running. Complete it safely, even though the
        # caller no longer needs its output, then verify the same worker still runs.
        assert not pending.future.cancelled()
        release.set()
        assert (
            await asyncio.to_thread(
                batcher.run_on_gpu_thread, lambda: "healthy", name="after"
            )
            == "healthy"
        )
        assert pending.future.result(timeout=1).audio.shape == (10,)

    try:
        asyncio.run(run())
        assert batcher.cuda_fault is None
    finally:
        release.set()
        batcher.close()


def test_close_reports_live_owner_after_http_waiter_is_cancelled(monkeypatch) -> None:
    """CPU-only owner lifetime contract; no CUDA/model inference is performed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    entered, release = threading.Event(), threading.Event()

    def generate(**kwargs):
        entered.set()
        assert release.wait(5)
        return [np.zeros(10, dtype=np.float32) for _ in kwargs["text"]]

    model = SimpleNamespace(generate=generate)
    batcher = GenerationBatcher(model, collect_ms=0)
    pending = _pending_generation("in-flight-at-shutdown")

    async def run():
        waiter = asyncio.create_task(batcher.submit_async(pending))
        assert await asyncio.to_thread(entered.wait, 2)
        assert pending.future.running()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not pending.future.cancelled()

        queued = batcher._enqueue(_pending_generation("queued-at-shutdown"))
        assert await asyncio.to_thread(batcher.close, 0.001) is False
        assert batcher._worker.is_alive()
        assert batcher._model is model
        with pytest.raises(RuntimeError, match="shutting down"):
            queued.result(timeout=1)
        with pytest.raises(RuntimeError, match="closed"):
            batcher._enqueue(_pending_generation("late-arrival"))

        release.set()
        assert await asyncio.to_thread(batcher.close, 2) is True
        assert not batcher._worker.is_alive()
        assert pending.future.result(timeout=1).audio.shape == (10,)
        assert batcher.close(timeout_s=0) is True

    try:
        asyncio.run(run())
    finally:
        release.set()
        assert batcher.close(timeout_s=2)


def test_failed_gpu_task_releases_its_frames(batcher_and_faults, monkeypatch) -> None:
    """A failed call's locals (GPU tensors in production) must not outlive it."""
    batcher, _ = batcher_and_faults
    # pytest's log capture keeps each record's traceback; production handlers do not
    monkeypatch.setattr(
        logging.getLogger("omnivoice_triton.serving.batching"), "disabled", True
    )

    class Payload:
        pass

    refs: list[weakref.ref[Payload]] = []

    def encode() -> None:
        payload = Payload()
        refs.append(weakref.ref(payload))
        raise RuntimeError("prompt encoder ran out of memory")

    with pytest.raises(RuntimeError, match="prompt encoder") as excinfo:
        batcher.run_on_gpu_thread(encode, name="encode")

    assert excinfo.value is not None  # the caller still holds the error...
    assert refs[0]() is None  # ...but not the failed call's frames


@pytest.mark.parametrize("kind", ["generation", "gpu_task"])
@pytest.mark.parametrize("admission_declined", [False, True])
def test_memory_admission_skips_failure_cleanup_and_worker_recovers(
    monkeypatch, kind: str, admission_declined: bool
) -> None:
    """A declined forward is backpressure; real failures retain owner-thread cleanup."""
    from omnivoice_triton.serving import batching

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cleanup_threads: list[int] = []
    exception_log_threads: list[int] = []
    monkeypatch.setattr(
        batching,
        "_cleanup_cuda_after_gpu_failure",
        lambda *args: cleanup_threads.append(threading.get_ident()),
    )
    monkeypatch.setattr(
        batching.logger,
        "exception",
        lambda *args, **kwargs: exception_log_threads.append(threading.get_ident()),
    )
    sampler_events: list[tuple[str, int, float | None]] = []

    class Sampler:
        def start(self):
            sampler_events.append(("start", threading.get_ident(), None))

        def stop(self, *, process_peak_vram_gb=None):
            sampler_events.append(("stop", threading.get_ident(), process_peak_vram_gb))

    class Payload:
        pass

    refs: list[weakref.ref[Payload]] = []
    failure_threads: list[int] = []
    error_type = GraphMemoryAdmissionError if admission_declined else RuntimeError

    def fail():
        payload = Payload()
        refs.append(weakref.ref(payload))
        failure_threads.append(threading.get_ident())
        raise error_type("declined before forward" if admission_declined else "failure")

    model = SimpleNamespace(generate=lambda **kwargs: fail())
    monitor = SimpleNamespace(create_batch_sampler=Sampler)
    faults: list[str] = []
    batcher = GenerationBatcher(
        model, collect_ms=0, gpu_metrics_monitor=monitor, on_cuda_fault=faults.append
    )
    try:
        with pytest.raises(error_type) as excinfo:
            if kind == "generation":
                batcher.submit(_pending_generation("declined"))
            else:
                batcher.run_on_gpu_thread(fail, name="declined")
        assert excinfo.value is not None
        assert refs[0]() is None
        owner = failure_threads[0]
        assert owner != threading.get_ident()
        expected = [] if admission_declined else [owner]
        assert cleanup_threads == expected
        assert exception_log_threads == expected
        assert faults == [] and batcher.cuda_fault is None
        if kind == "generation":
            assert sampler_events == [
                ("start", owner, None),
                ("stop", owner, None if admission_declined else 0.0),
            ]
        assert batcher.run_on_gpu_thread(lambda: 42, name="after-decline") == 42
        model.generate = lambda **kwargs: [
            np.zeros(10, dtype=np.float32) for _ in kwargs["text"]
        ]
        assert len(batcher.submit(_pending_generation("after-decline")).audio) == 10
    finally:
        batcher.close()


def test_cuda_fault_refuses_all_later_work(batcher_and_faults) -> None:
    batcher, faults = batcher_and_faults

    def broken() -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        batcher.run_on_gpu_thread(broken, name="broken_task")

    assert batcher.cuda_fault is not None
    assert batcher.cuda_fault.startswith("broken_task: RuntimeError: CUDA error")
    assert faults == [batcher.cuda_fault]
    with pytest.raises(CudaFaultError):
        batcher.run_on_gpu_thread(lambda: None, name="after")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("CUDA error: device-side assert triggered"), True),
        (RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"), False),
        (
            torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB."),
            False,
        ),
        (
            CudaGraphCaptureError("CUDA Graph capture failed for shape (2, 8, 491)"),
            True,
        ),
        (CudaFaultError("batch: RuntimeError: CUDA error"), True),
        (ValueError("bad input"), False),
    ],
)
def test_is_cuda_fault(error: BaseException, expected: bool) -> None:
    assert is_cuda_fault(error) is expected


def test_is_cuda_fault_follows_the_cause_chain() -> None:
    try:
        try:
            raise RuntimeError(
                "CUDA error: operation failed due to a previous error during capture"
            )
        except RuntimeError as inner:
            raise RuntimeError("Batch execution failed") from inner
    except RuntimeError as outer:
        assert is_cuda_fault(outer)
