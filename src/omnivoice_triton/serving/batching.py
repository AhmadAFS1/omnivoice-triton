"""Batching primitives for the FastAPI serving layer."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch
from omnivoice import OmniVoiceGenerationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationBatchKey:
    """Compatibility key for joining requests into the same batch."""

    lane: str
    num_step: int
    guidance_scale: float
    t_shift: float
    layer_penalty_factor: float
    position_temperature: float
    class_temperature: float
    denoise: bool
    postprocess_output: bool
    audio_chunk_duration: float
    audio_chunk_threshold: float


@dataclass(frozen=True)
class BatchedGenerationResult:
    """Per-request result returned from a merged batch execution."""

    audio: Any
    queue_wait_ms: float
    batch_exec_ms: float
    batch_requests: int
    batch_target_tokens: int
    batch_conditioning_tokens: int
    batch_max_sequence_length: int
    peak_vram_gb: float


@dataclass
class PendingGeneration:
    """Prepared API request waiting to be merged into a batch."""

    request_id: str
    mode: str
    text: str
    language: str | None
    instruct: str | None
    voice_clone_prompt: Any | None
    speed: float | None
    duration: float | None
    target_tokens: int
    conditioning_tokens: int
    max_sequence_length: int
    batch_key: GenerationBatchKey
    generation_config: OmniVoiceGenerationConfig
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: Future[BatchedGenerationResult] = field(default_factory=Future)


class ClonePromptCache:
    """Thread-safe cache for prepared voice-clone prompts."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str | None, bool], Any] = OrderedDict()
        self._inflight: dict[tuple[str, str | None, bool], threading.Event] = {}

    def get_or_create(
        self,
        *,
        ref_audio_bytes: bytes,
        ref_text: str | None,
        preprocess_prompt: bool,
        factory: Any,
    ) -> Any:
        """Return a cached prompt or create and cache it on demand."""
        if self.max_size <= 0:
            return self._to_cpu_prompt(factory())

        key = (
            hashlib.sha256(ref_audio_bytes).hexdigest(),
            ref_text,
            preprocess_prompt,
        )

        while True:
            with self._lock:
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    return cached

                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    break

            event.wait()

        try:
            prompt = self._to_cpu_prompt(factory())
        except Exception:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()
            raise

        with self._lock:
            self._entries[key] = prompt
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
            self._inflight.pop(key, None)
            event.set()

        return prompt

    def _to_cpu_prompt(self, prompt: Any) -> Any:
        ref_audio_tokens = getattr(prompt, "ref_audio_tokens", None)
        if isinstance(ref_audio_tokens, torch.Tensor):
            return type(prompt)(
                ref_audio_tokens=ref_audio_tokens.detach().cpu(),
                ref_text=getattr(prompt, "ref_text"),
                ref_rms=float(getattr(prompt, "ref_rms")),
            )
        return prompt


class GenerationBatcher:
    """Background worker that merges compatible requests into one generate call."""

    def __init__(
        self,
        model: Any,
        *,
        collect_ms: float = 10.0,
        max_batch_requests: int = 32,
        max_batch_target_tokens: int = 24000,
        max_batch_conditioning_tokens: int = 12000,
        max_batch_padding_ratio: float = 1.5,
    ) -> None:
        self._model = model
        self.collect_ms = max(0.0, collect_ms)
        self.max_batch_requests = max(1, max_batch_requests)
        self.max_batch_target_tokens = max(1, max_batch_target_tokens)
        self.max_batch_conditioning_tokens = max(1, max_batch_conditioning_tokens)
        self.max_batch_padding_ratio = max(1.0, max_batch_padding_ratio)

        self._condition = threading.Condition()
        self._pending: deque[PendingGeneration] = deque()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="omnivoice-batcher",
            daemon=True,
        )
        self._worker.start()

    def submit(self, request: PendingGeneration) -> BatchedGenerationResult:
        """Queue a request and block until its merged batch finishes."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Batcher is closed.")
            self._pending.append(request)
            self._condition.notify()

        return request.future.result()

    def close(self) -> None:
        """Stop the worker and fail any queued requests."""
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()

        error = RuntimeError("Server is shutting down.")
        for request in pending:
            if not request.future.done():
                request.future.set_exception(error)

        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()

                if self._closed and not self._pending:
                    return

                if self.collect_ms > 0:
                    self._condition.wait(timeout=self.collect_ms / 1000.0)

                batch = self._select_batch_locked()

            if not batch:
                continue

            self._execute_batch(batch)

    def _select_batch_locked(self) -> list[PendingGeneration]:
        if not self._pending:
            return []

        anchor = self._pending[0]
        selected: list[PendingGeneration] = []
        remaining: deque[PendingGeneration] = deque()

        total_target_tokens = 0
        total_conditioning_tokens = 0
        total_sequence_length = 0
        max_sequence_length = 0

        for request in self._pending:
            if request.batch_key != anchor.batch_key:
                remaining.append(request)
                continue

            next_batch_size = len(selected) + 1
            next_target_tokens = total_target_tokens + request.target_tokens
            next_conditioning_tokens = (
                total_conditioning_tokens + request.conditioning_tokens
            )
            next_total_sequence_length = (
                total_sequence_length + request.max_sequence_length
            )
            next_max_sequence_length = max(
                max_sequence_length,
                request.max_sequence_length,
            )
            padding_ratio = (
                next_batch_size * next_max_sequence_length / next_total_sequence_length
            )

            if selected and (
                next_batch_size > self.max_batch_requests
                or next_target_tokens > self.max_batch_target_tokens
                or next_conditioning_tokens > self.max_batch_conditioning_tokens
                or padding_ratio > self.max_batch_padding_ratio
            ):
                remaining.append(request)
                continue

            selected.append(request)
            total_target_tokens = next_target_tokens
            total_conditioning_tokens = next_conditioning_tokens
            total_sequence_length = next_total_sequence_length
            max_sequence_length = next_max_sequence_length

        self._pending = remaining
        return selected

    def _execute_batch(self, batch: list[PendingGeneration]) -> None:
        anchor = batch[0]
        batch_started = time.perf_counter()

        request_ids = ",".join(request.request_id for request in batch)
        logger.info(
            "Executing batch lane=%s size=%d request_ids=%s",
            anchor.batch_key.lane,
            len(batch),
            request_ids,
        )

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            generate_kwargs: dict[str, Any] = {
                "text": [request.text for request in batch],
                "generation_config": anchor.generation_config,
            }

            if any(request.language is not None for request in batch):
                generate_kwargs["language"] = [request.language for request in batch]

            if any(request.instruct is not None for request in batch):
                generate_kwargs["instruct"] = [request.instruct for request in batch]

            if all(request.voice_clone_prompt is not None for request in batch):
                generate_kwargs["voice_clone_prompt"] = [
                    request.voice_clone_prompt for request in batch
                ]

            if any(request.duration is not None for request in batch):
                generate_kwargs["duration"] = [request.duration for request in batch]
            if any(request.speed is not None for request in batch):
                generate_kwargs["speed"] = [request.speed for request in batch]

            outputs = self._model.generate(**generate_kwargs)
            batch_finished = time.perf_counter()
            batch_exec_ms = (batch_finished - batch_started) * 1000.0

            peak_vram_gb = 0.0
            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} outputs from model.generate(), got "
                    f"{len(outputs)}."
                )

            batch_target_tokens = sum(request.target_tokens for request in batch)
            batch_conditioning_tokens = sum(
                request.conditioning_tokens for request in batch
            )
            batch_max_sequence_length = max(
                request.max_sequence_length for request in batch
            )

            for request, audio in zip(batch, outputs, strict=True):
                queue_wait_ms = (batch_started - request.enqueued_at) * 1000.0
                request.future.set_result(
                    BatchedGenerationResult(
                        audio=audio,
                        queue_wait_ms=queue_wait_ms,
                        batch_exec_ms=batch_exec_ms,
                        batch_requests=len(batch),
                        batch_target_tokens=batch_target_tokens,
                        batch_conditioning_tokens=batch_conditioning_tokens,
                        batch_max_sequence_length=batch_max_sequence_length,
                        peak_vram_gb=peak_vram_gb,
                    )
                )
        except Exception as exc:
            logger.exception(
                "Batch execution failed lane=%s size=%d",
                anchor.batch_key.lane,
                len(batch),
            )
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
