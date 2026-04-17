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

from omnivoice_triton.serving.gpu_metrics import BatchGpuMetrics, GPUMetricsMonitor

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
    postprocess_mode: str
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
    replica_index: int = 0
    gpu_metrics: BatchGpuMetrics | None = None
    generation_metrics: dict[str, float] = field(default_factory=dict)


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


@dataclass(frozen=True)
class _BatchCandidate:
    """Internal scoring object for scheduler batch selection."""

    batch: list[PendingGeneration]
    padding_efficiency: float
    oldest_enqueued_at: float
    first_index: int
    exact_bucket: bool


@dataclass(frozen=True)
class ClonePromptCacheResult:
    """Prepared clone prompt plus cache-source metadata."""

    prompt: Any
    source: str


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
    ) -> ClonePromptCacheResult:
        """Return a cached prompt or create and cache it on demand."""
        if self.max_size <= 0:
            return ClonePromptCacheResult(
                prompt=self._to_cpu_prompt(factory()),
                source="disabled",
            )

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
                    return ClonePromptCacheResult(prompt=cached, source="hit")

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

        return ClonePromptCacheResult(prompt=prompt, source="miss")

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
        replica_index: int = 0,
        collect_ms: float = 10.0,
        max_batch_requests: int = 32,
        max_batch_target_tokens: int = 24000,
        max_batch_conditioning_tokens: int = 12000,
        max_batch_padding_ratio: float = 1.5,
        batch_bucket_sizes: tuple[int, ...] | None = None,
        gpu_metrics_monitor: GPUMetricsMonitor | None = None,
    ) -> None:
        self._model = model
        self.replica_index = max(0, int(replica_index))
        self.collect_ms = max(0.0, collect_ms)
        self.max_batch_requests = max(1, max_batch_requests)
        self.max_batch_target_tokens = max(1, max_batch_target_tokens)
        self.max_batch_conditioning_tokens = max(1, max_batch_conditioning_tokens)
        self.max_batch_padding_ratio = max(1.0, max_batch_padding_ratio)
        self.batch_bucket_sizes = tuple(
            sorted({size for size in batch_bucket_sizes or () if size > 0})
        )
        self._gpu_metrics_monitor = gpu_metrics_monitor
        self._scheduler_starvation_ms = max(50.0, self.collect_ms * 4.0)
        self._scheduler_batches_total = 0
        self._scheduler_non_anchor_batches = 0
        self._scheduler_starvation_batches = 0
        self._submitted_requests = 0
        self._completed_requests = 0
        self._inflight_requests = 0

        self._condition = threading.Condition()
        self._pending: deque[PendingGeneration] = deque()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name=f"omnivoice-batcher-{self.replica_index}",
            daemon=True,
        )
        self._worker.start()

    def submit(self, request: PendingGeneration) -> BatchedGenerationResult:
        """Queue a request and block until its merged batch finishes."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Batcher is closed.")
            self._pending.append(request)
            self._submitted_requests += 1
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

    def get_scheduler_metrics(self) -> dict[str, float | int]:
        """Return scheduler counters for health/debug endpoints."""
        with self._condition:
            return {
                "batches_total": self._scheduler_batches_total,
                "non_anchor_batches": self._scheduler_non_anchor_batches,
                "starvation_anchor_batches": self._scheduler_starvation_batches,
                "starvation_threshold_ms": self._scheduler_starvation_ms,
                "pending_requests": len(self._pending),
                "inflight_requests": self._inflight_requests,
                "outstanding_requests": len(self._pending) + self._inflight_requests,
                "submitted_requests": self._submitted_requests,
                "completed_requests": self._completed_requests,
            }

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
                if batch:
                    self._inflight_requests += len(batch)

            if not batch:
                continue

            try:
                self._execute_batch(batch)
            finally:
                with self._condition:
                    self._inflight_requests = max(
                        0,
                        self._inflight_requests - len(batch),
                    )
                    self._completed_requests += len(batch)

    def _select_batch_locked(self) -> list[PendingGeneration]:
        if not self._pending:
            return []

        anchor = self._pending[0]
        now = time.perf_counter()
        anchor_wait_ms = (now - anchor.enqueued_at) * 1000.0

        if anchor_wait_ms >= self._scheduler_starvation_ms:
            selected = self._select_anchor_batch_locked(anchor)
            self._scheduler_batches_total += 1
            self._scheduler_starvation_batches += 1
            logger.info(
                "Batch scheduler starvation fallback lane=%s request_id=%s wait_ms=%.2f",
                anchor.batch_key.lane,
                anchor.request_id,
                anchor_wait_ms,
            )
        else:
            groups: OrderedDict[GenerationBatchKey, list[PendingGeneration]] = OrderedDict()
            first_indices: dict[GenerationBatchKey, int] = {}
            for index, request in enumerate(self._pending):
                if request.batch_key not in groups:
                    groups[request.batch_key] = []
                    first_indices[request.batch_key] = index
                groups[request.batch_key].append(request)

            candidates = [
                candidate
                for batch_key, requests in groups.items()
                if (
                    candidate := self._build_candidate_for_requests(
                        requests,
                        first_index=first_indices[batch_key],
                    )
                )
                is not None
            ]
            if not candidates:
                return []

            best = max(candidates, key=self._candidate_sort_key)
            selected = best.batch
            self._scheduler_batches_total += 1
            if best.first_index != 0:
                self._scheduler_non_anchor_batches += 1
                logger.info(
                    "Batch scheduler promoted deeper batch lane=%s size=%d "
                    "anchor_lane=%s anchor_request_id=%s anchor_wait_ms=%.2f "
                    "padding_efficiency=%.4f",
                    selected[0].batch_key.lane,
                    len(selected),
                    anchor.batch_key.lane,
                    anchor.request_id,
                    anchor_wait_ms,
                    best.padding_efficiency,
                )

        selected_ids = {id(request) for request in selected}
        self._pending = deque(
            request for request in self._pending if id(request) not in selected_ids
        )
        return selected

    def _select_anchor_batch_locked(self, anchor: PendingGeneration) -> list[PendingGeneration]:
        requests = [request for request in self._pending if request.batch_key == anchor.batch_key]
        candidate = self._build_candidate_for_requests(requests, first_index=0)
        return candidate.batch if candidate is not None else [anchor]

    def _build_candidate_for_requests(
        self,
        requests: list[PendingGeneration],
        *,
        first_index: int,
    ) -> _BatchCandidate | None:
        if not requests:
            return None

        selected: list[PendingGeneration] = []
        total_target_tokens = 0
        total_conditioning_tokens = 0
        total_sequence_length = 0
        max_sequence_length = 0

        for request in requests:
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
                continue

            selected.append(request)
            total_target_tokens = next_target_tokens
            total_conditioning_tokens = next_conditioning_tokens
            total_sequence_length = next_total_sequence_length
            max_sequence_length = next_max_sequence_length

        if not selected:
            return None

        selected = self._apply_batch_buckets(
            selected,
            lane=requests[0].batch_key.lane,
        )
        total_sequence_length = sum(request.max_sequence_length for request in selected)
        max_sequence_length = max(request.max_sequence_length for request in selected)
        padding_efficiency = (
            total_sequence_length / (len(selected) * max_sequence_length)
            if total_sequence_length > 0 and max_sequence_length > 0
            else 1.0
        )
        return _BatchCandidate(
            batch=selected,
            padding_efficiency=padding_efficiency,
            oldest_enqueued_at=min(request.enqueued_at for request in selected),
            first_index=first_index,
            exact_bucket=(
                len(selected) in self.batch_bucket_sizes if self.batch_bucket_sizes else False
            ),
        )

    def _candidate_sort_key(self, candidate: _BatchCandidate) -> tuple[float, ...]:
        return (
            float(len(candidate.batch)),
            1.0 if candidate.exact_bucket else 0.0,
            candidate.padding_efficiency,
            -float(candidate.first_index),
            -candidate.oldest_enqueued_at,
        )

    def _apply_batch_buckets(
        self,
        selected: list[PendingGeneration],
        *,
        lane: str,
    ) -> list[PendingGeneration]:
        batch_size = len(selected)
        if batch_size <= 1 or batch_size in self.batch_bucket_sizes:
            return selected

        target_size = max(
            (size for size in self.batch_bucket_sizes if size <= batch_size),
            default=0,
        )
        if target_size <= 0 or target_size >= batch_size:
            return selected

        selected = selected[:target_size]
        logger.info(
            "Batch bucket trim lane=%s original_size=%d trimmed_size=%d",
            lane,
            batch_size,
            target_size,
        )
        return selected

    def _execute_batch(self, batch: list[PendingGeneration]) -> None:
        anchor = batch[0]
        batch_started = time.perf_counter()

        request_ids = ",".join(request.request_id for request in batch)
        logger.info(
            "Executing batch replica=%d lane=%s size=%d request_ids=%s",
            self.replica_index,
            anchor.batch_key.lane,
            len(batch),
            request_ids,
        )

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            batch_gpu_sampler = None
            if self._gpu_metrics_monitor is not None:
                batch_gpu_sampler = self._gpu_metrics_monitor.create_batch_sampler()
                if batch_gpu_sampler is not None:
                    batch_gpu_sampler.start()

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
            generation_metrics: dict[str, float] = {}
            metrics_getter = getattr(self._model, "get_runtime_stage_metrics", None)
            if callable(metrics_getter):
                raw_metrics = metrics_getter() or {}
                generation_metrics = {
                    str(name): float(value)
                    for name, value in raw_metrics.items()
                    if value is not None
                }

            peak_vram_gb = 0.0
            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            batch_gpu_metrics = None
            if batch_gpu_sampler is not None:
                batch_gpu_metrics = batch_gpu_sampler.stop(
                    process_peak_vram_gb=peak_vram_gb
                )
                batch_gpu_sampler = None

            if batch_gpu_metrics is not None:
                logger.info(
                    "Completed batch lane=%s size=%d batch_exec_ms=%.2f "
                    "replica=%d "
                    "gpu_util_avg_pct=%s gpu_util_peak_pct=%s "
                    "device_vram_used_gb_peak=%s device_vram_util_peak_pct=%s "
                    "process_peak_vram_gb=%.3f gpu_sample_count=%d",
                    anchor.batch_key.lane,
                    len(batch),
                    batch_exec_ms,
                    self.replica_index,
                    _format_optional_float(batch_gpu_metrics.gpu_util_avg_pct),
                    _format_optional_float(batch_gpu_metrics.gpu_util_peak_pct),
                    _format_optional_float(
                        batch_gpu_metrics.device_vram_used_gb_peak
                    ),
                    _format_optional_float(
                        batch_gpu_metrics.device_vram_util_peak_pct
                    ),
                    peak_vram_gb,
                    batch_gpu_metrics.sample_count,
                )
            else:
                logger.info(
                    "Completed batch lane=%s size=%d batch_exec_ms=%.2f "
                    "replica=%d "
                    "process_peak_vram_gb=%.3f gpu_metrics=unavailable",
                    anchor.batch_key.lane,
                    len(batch),
                    batch_exec_ms,
                    self.replica_index,
                    peak_vram_gb,
                )

            if generation_metrics:
                logger.info(
                    "Batch model timings lane=%s size=%d generate_total_ms=%s "
                    "replica=%d "
                    "prepare_inputs_ms=%s iterative_ms=%s chunked_ms=%s "
                    "decode_postprocess_ms=%s",
                    anchor.batch_key.lane,
                    len(batch),
                    _format_optional_float(
                        generation_metrics.get("generate_total_ms")
                    ),
                    self.replica_index,
                    _format_optional_float(
                        generation_metrics.get("prepare_inference_inputs_ms")
                    ),
                    _format_optional_float(
                        generation_metrics.get("iterative_generate_ms")
                    ),
                    _format_optional_float(
                        generation_metrics.get("chunked_generate_ms")
                    ),
                    _format_optional_float(
                        generation_metrics.get("decode_postprocess_ms")
                    ),
                )

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
                        replica_index=self.replica_index,
                        gpu_metrics=batch_gpu_metrics,
                        generation_metrics=generation_metrics,
                    )
                )
        except Exception as exc:
            try:
                peak_vram_gb = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                if "batch_gpu_sampler" in locals() and batch_gpu_sampler is not None:
                    batch_gpu_sampler.stop(process_peak_vram_gb=peak_vram_gb)
            except Exception:  # pragma: no cover - metrics should not mask failures
                logger.debug("Failed to stop GPU batch sampler after error.", exc_info=True)
            logger.exception(
                "Batch execution failed lane=%s size=%d",
                anchor.batch_key.lane,
                len(batch),
            )
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)


def _format_optional_float(value: float | None) -> str:
    """Format optional floating point values for human-readable logs."""
    if value is None:
        return "-"
    return f"{value:.2f}"
