"""Opt-in CUDA Graph retention and capture-admission policy.

Normalized row tokens bound retained entries, not total VRAM. Headroom estimates
are calibrated heuristics for the qualified inference architecture, not rigorous bounds.
No CUDA OOM is caught and retried. Fallback happens before new graph allocation.
Install graph -> memory guard -> sequence bucket -> split CFG.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass

import torch

GiB = 1024**3
MiB = 1024**2
PIN = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"


class GraphMemoryAdmissionError(RuntimeError):
    """Insufficient estimated workspace before any new graph capture begins."""


@dataclass(frozen=True)
class GraphMemoryProfile:
    qualified: bool
    reason: str = ""
    revision: str | None = None
    dtype_bytes: int = 2
    use_cache: bool = True
    layers: int = 28
    hidden: int = 1024
    intermediate: int = 3072
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    codebooks: int = 8
    audio_vocab: int = 1025

    def weight(self, shape):
        b, _c, s = shape
        return math.ceil(b * s * self.dtype_bytes / 2)

    def work_bytes(self, shape):
        b, c, s = shape
        rows = b * s
        d = self.dtype_bytes
        kv = (
            rows * self.layers * 2 * self.kv_heads * self.head_dim * d
            if self.use_cache
            else 0
        )
        logits = rows * c * self.audio_vocab * d
        scratch = (
            rows
            * (
                6 * self.hidden
                + 3 * self.intermediate
                + 2 * (self.heads + 2 * self.kv_heads) * self.head_dim
            )
            * d
        )
        # Conservative one-layer materialized scores/softmax allowance, although
        # the pinned SDPA path can use substantially less temporary storage.
        attention = 2 * b * self.heads * s * s * max(d, 4)
        return kv + logits + scratch + attention

    def capture_headroom(self, shape):
        scaled = 3 * GiB * self.weight(shape) / 8192
        return math.ceil(max(scaled, 1.5 * self.work_bytes(shape)) + 512 * MiB)

    def eager_headroom(self, shape):
        return math.ceil(self.work_bytes(shape) + 512 * MiB)

    def hit_headroom(self, shape):
        b, c, s = shape
        return 512 * MiB + b * c * s * self.audio_vocab * 4

    @classmethod
    def from_model(cls, model, *, revision=None):
        """Qualify architecture/runtime, without inferring a checkpoint revision.

        The allocation estimate depends on actual dimensions and dtype. Benchmark
        manifests separately verify checkpoint file hashes. An explicitly supplied
        revision must match the reference used to calibrate this policy.
        """
        try:
            import importlib.metadata

            config = model.llm.config
            expected = {
                "num_hidden_layers": 28,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "model_type": "qwen3",
            }
            if (revision is not None and revision != PIN) or importlib.metadata.version(
                "omnivoice"
            ) != "0.1.4":
                return cls(False, "unqualified_revision_or_library")
            if any(getattr(config, k, None) != v for k, v in expected.items()):
                return cls(False, "unqualified_dimensions")
            if (
                model.config.num_audio_codebook != 8
                or model.config.audio_vocab_size != 1025
            ):
                return cls(False, "unqualified_audio_heads")
            if getattr(config, "rope_parameters", None) != {
                "rope_type": "default",
                "rope_theta": 1000000,
            }:
                return cls(False, "unqualified_rope")
            # Transformers 5.5 exposes rope_scaling as the same default mapping
            # through a compatibility alias. Reject other legacy settings.
            legacy_rope = getattr(config, "rope_scaling", None)
            if legacy_rope is not None and legacy_rope != config.rope_parameters:
                return cls(False, "unqualified_rope_scaling")
            if getattr(config, "_attn_implementation", None) not in {"sdpa", "eager"}:
                return cls(False, "unqualified_attention")
            dtype = next(model.llm.parameters()).dtype
            if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                return cls(False, "unqualified_dtype")
            return cls(
                True,
                revision=revision,
                dtype_bytes=torch.empty((), dtype=dtype).element_size(),
                use_cache=bool(config.use_cache),
            )
        except (AttributeError, StopIteration, ImportError):
            return cls(False, "unavailable_model_contract")


class MemoryBoundGraphForward:
    MAX_SHAPE_OBSERVATIONS = 4096

    def __init__(
        self,
        graph_forward,
        *,
        max_row_tokens=0,
        capture_after_calls=0,
        profile=None,
        revision=None,
        memory_info=None,
        cleanup=None,
    ):
        if (
            not isinstance(max_row_tokens, int)
            or isinstance(max_row_tokens, bool)
            or max_row_tokens < 0
        ):
            raise ValueError(
                "max_row_tokens must be an integer>=0;0 preserves legacy behavior"
            )
        if (
            not isinstance(capture_after_calls, int)
            or isinstance(capture_after_calls, bool)
            or capture_after_calls < 0
        ):
            raise ValueError("capture_after_calls must be an integer >= 0")
        if capture_after_calls and not max_row_tokens:
            raise ValueError("Deferred capture requires a positive max_row_tokens")
        self.graph = graph_forward
        self.max_row_tokens = max_row_tokens
        self.capture_after_calls = capture_after_calls
        self._shape_calls = OrderedDict()
        self.deferred_calls = 0
        self.profile = (
            profile
            if profile is not None
            else (
                GraphMemoryProfile.from_model(graph_forward._model, revision=revision)
                if max_row_tokens
                else GraphMemoryProfile(False, "disabled")
            )
        )
        self.memory_info = memory_info or (
            lambda device: torch.cuda.mem_get_info(device=device)
        )
        self.cleanup = cleanup or self._cleanup
        self.owner_thread = None
        self.calls = 0
        self.hits = 0
        self.captures = 0
        self.evictions = 0
        self.evicted_weight = 0
        self.eager_fallbacks = Counter()
        self.admission_rejections = 0
        self.pressure_cleanups = 0
        self.capture_failures = 0
        self.clears = 0
        self.last_memory = None
        self.last_decision = None

    @property
    def _graphs(self):
        return self.graph._graphs

    @property
    def max_graphs(self):
        return self.graph.max_graphs

    @property
    def _original_forward(self):
        return self.graph._original_forward

    @staticmethod
    def _cleanup(device):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def _weight(self, key):
        if (
            type(key) is not tuple
            or len(key) != 3
            or any(type(x) is not int or x <= 0 for x in key)
        ):
            raise ValueError("Unqualified graph key; expected positive integer[B,C,S]")
        return self.profile.weight(key)

    def retained_weight(self):
        return sum(self._weight(key) for key in self._graphs)

    def _evict(self, keep=None):
        key = next((key for key in self._graphs if key != keep), None)
        if key is None:
            return False
        weight = self._weight(key)
        entry = self._graphs.pop(key)
        self._shape_calls.pop(key, None)
        del entry  # Drop every local graph/buffer reference before querying/freeing.
        self.evictions += 1
        self.evicted_weight += weight
        return True

    def _free(self, device, required):
        free, total = self.memory_info(device)
        self.last_memory = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "required_estimated_bytes": int(required),
            "device": str(device),
        }
        return int(free)

    def _ensure(self, required, device, *, keep=None, allow_cleanup=True):
        free = self._free(device, required)
        while free < required and self._evict(keep):
            free = self._free(device, required)
        if free < required and allow_cleanup:
            self.cleanup(device)
            self.pressure_cleanups += 1
            free = self._free(device, required)
        return free >= required

    def _raw(
        self,
        reason,
        ids,
        audio,
        labels,
        attention,
        documents,
        positions,
        kwargs,
        *,
        guard=False,
        clean=True,
    ):
        self.last_decision = "eager:" + reason
        self.eager_fallbacks[reason] += 1
        if guard and not self._ensure(
            self.profile.eager_headroom(tuple(ids.shape)),
            ids.device,
            allow_cleanup=clean,
        ):
            self.admission_rejections += 1
            raise GraphMemoryAdmissionError(
                "Estimated eager workspace unavailable after pre-eviction; "
                "no capture attempted"
            )
        return self._original_forward(
            ids,
            audio,
            labels=labels,
            attention_mask=attention,
            document_ids=documents,
            position_ids=positions,
            **kwargs,
        )

    def __call__(
        self,
        input_ids,
        audio_mask,
        labels=None,
        attention_mask=None,
        document_ids=None,
        position_ids=None,
        **kwargs,
    ):
        if not self.max_row_tokens:
            return self.graph(
                input_ids,
                audio_mask,
                labels=labels,
                attention_mask=attention_mask,
                document_ids=document_ids,
                position_ids=position_ids,
                **kwargs,
            )
        self.calls += 1

        def raw(reason, **options):
            return self._raw(
                reason,
                input_ids,
                audio_mask,
                labels,
                attention_mask,
                document_ids,
                position_ids,
                kwargs,
                **options,
            )

        if torch.is_autocast_enabled():
            return raw("unqualified_cuda_autocast")
        if (
            getattr(self.graph._model, "training", False)
            or labels is not None
            or torch.is_grad_enabled()
            or document_ids is not None
            or position_ids is not None
            or kwargs
        ):
            return raw("unsupported_inference_call")
        if not self.profile.qualified:
            return raw("unqualified_model:" + self.profile.reason)
        if input_ids.ndim != 3 or input_ids.shape[1] != self.profile.codebooks:
            return raw("unqualified_input_shape")
        if self.max_graphs <= 0:
            return raw("graphs_disabled")
        current = threading.get_ident()
        if self.owner_thread is None:
            self.owner_thread = current
        if current != self.owner_thread:
            raise RuntimeError("Graph memory policy must run on one GPU owner thread")
        key = self.graph._get_shape_key(input_ids)
        weight = self._weight(key)
        keep = key if weight <= self.max_row_tokens else None
        # Also normalize a cache populated before the opt-in wrapper was enabled.
        while self._graphs and (
            len(self._graphs) > self.max_graphs
            or self.retained_weight() > self.max_row_tokens
        ):
            if not self._evict(keep):
                break
        if weight > self.max_row_tokens:
            return raw("shape_exceeds_row_budget", guard=True)
        if key in self._graphs:
            if not self._ensure(
                self.profile.hit_headroom(key), input_ids.device, keep=key
            ):
                self.admission_rejections += 1
                raise GraphMemoryAdmissionError(
                    "Estimated output workspace unavailable for graph hit"
                )
            self.hits += 1
            self.last_decision = "replay"
            return self.graph(input_ids, audio_mask, attention_mask=attention_mask)
        if self.capture_after_calls:
            seen = self._shape_calls.get(key, 0)
            if seen < self.capture_after_calls:
                # A forward call is one denoising step, not one HTTP request.
                # Keep all eager workspace/owner guards while avoiding the
                # first capture's extra warmup and graph-recording forwards.
                result = raw("shape_not_hot", guard=True)
                self.deferred_calls += 1
                self._shape_calls[key] = seen + 1
                self._shape_calls.move_to_end(key)
                while len(self._shape_calls) > self.MAX_SHAPE_OBSERVATIONS:
                    self._shape_calls.popitem(last=False)
                return result
            self._shape_calls.move_to_end(key)
        while self._graphs and (
            len(self._graphs) + 1 > self.max_graphs
            or self.retained_weight() + weight > self.max_row_tokens
        ):
            self._evict()
        if not self._ensure(self.profile.capture_headroom(key), input_ids.device):
            return raw("capture_headroom", guard=True, clean=False)
        self.last_decision = "capture"
        try:
            result = self.graph(input_ids, audio_mask, attention_mask=attention_mask)
        except Exception:
            self.capture_failures += 1
            raise  # Never retry an OOM or failed stream capture as eager work.
        self.captures += 1
        assert self.retained_weight() <= self.max_row_tokens
        assert len(self._graphs) <= self.max_graphs
        return result

    def clear(self):
        """Release retained captures only from the established inference owner."""
        if (
            self.max_row_tokens
            and self.owner_thread is not None
            and threading.get_ident() != self.owner_thread
        ):
            raise RuntimeError("Graph memory policy clear requires GPU owner thread")
        self.graph.clear()
        self._shape_calls.clear()
        self.clears += 1
        self.last_decision = "clear"

    def metrics(self):
        shapes = []
        unknown = 0
        retained = 0
        if self.max_row_tokens:
            for key in tuple(self._graphs):
                try:
                    weight = self._weight(key)
                except (TypeError, ValueError, OverflowError):
                    unknown += 1
                    shapes.append(
                        {
                            "key_type": type(key).__name__,
                            "key_arity": len(key) if type(key) is tuple else None,
                            "normalized_row_tokens": None,
                        }
                    )
                else:
                    retained += weight
                    shapes.append({"shape": list(key), "normalized_row_tokens": weight})
        return {
            "enabled": bool(self.max_row_tokens),
            "max_cuda_graph_row_tokens": self.max_row_tokens,
            "capture_after_calls": self.capture_after_calls,
            "deferred_calls": self.deferred_calls,
            "observed_shape_keys": len(self._shape_calls),
            "max_shape_observations": self.MAX_SHAPE_OBSERVATIONS,
            "estimate_kind": (
                "Heuristic architecture/dtype row tokens and headroom; "
                "NOT a rigorous total VRAM bound"
            ),
            "qualification_scope": "actual_model_architecture_and_runtime",
            "profile": asdict(self.profile),
            "retained_row_tokens": retained
            if self.max_row_tokens and not unknown
            else None,
            "unqualified_graph_keys": unknown,
            "resident_graphs": len(self._graphs),
            "max_graphs": self.max_graphs,
            "calls": self.calls,
            "hits": self.hits,
            "captures": self.captures,
            "evictions": self.evictions,
            "evicted_row_tokens": self.evicted_weight,
            "eager_fallbacks": dict(self.eager_fallbacks),
            "pressure_cleanups": self.pressure_cleanups,
            "admission_rejections": self.admission_rejections,
            "capture_failures": self.capture_failures,
            "clears": self.clears,
            "last_owner_memory_snapshot": self.last_memory,
            "last_decision": self.last_decision,
            "resident_shapes": shapes,
        }
