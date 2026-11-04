"""Default-off, isolated shared-scratch CUDA graph research prototype.

NOT product integration or a general PyTorch pool-sharing guarantee. Enable only
in a root-approved finite GPU probe. CPU backends may model ownership contracts;
they do not qualify CUDA. All pool values, including strongly owned outputs, are
disposable between calls. Only fresh global clones escape this wrapper.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import threading
from collections import OrderedDict
from pathlib import Path

PIN = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
GiB = 1024**3
SOURCE_PINS = {
    "omnivoice.models.omnivoice": "c7104b1b9574fac6c84014e39636a5f1e51508902fa59cf151a713151669bd40",
    "transformers.models.qwen3.modeling_qwen3": "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab",
    "torch.cuda.graphs": "b707ce7ada16e40bed7e588da0ef46b2b31a74a20ad77f4d100a5b89f13f44b9",
}


def capture_headroom_bytes(shape):
    """Qualified FP16/no-KV estimate with a nonweakenable4GiB public floor."""
    b, c, s = shape
    rows, d = b * s, 2
    logits = rows * c * 1025 * d
    scratch = rows * (6 * 1024 + 3 * 3072 + 2 * (16 + 2 * 8) * 128) * d
    attention = 2 * b * 16 * s * s * 4
    estimated = math.ceil(
        max(3 * GiB * rows / 8192, 1.5 * (logits + scratch + attention)) + GiB / 2
    )
    return max(4 * GiB, estimated)


class SharedPoolAdmissionError(RuntimeError):
    """Rejected before capture; the finite probe must stop as incomplete."""


class SharedPoolFatalError(RuntimeError):
    """Capture/replay or ownership failed; process restart is required."""


class TorchBackend:
    """Actual Torch 2.8 operations; constructing this class does not run a model."""

    def __init__(self):
        import torch

        self.torch = torch

    def qualify(self, model, revision):
        t = self.torch
        expected = {"omnivoice": "0.1.4", "transformers": "5.5.4", "triton": "3.4.0"}
        versions = {name: importlib.metadata.version(name) for name in expected}
        if versions != expected or t.__version__ != "2.8.0+cu128":
            raise SharedPoolAdmissionError("Unqualified runtime versions")
        if revision != PIN:
            raise SharedPoolAdmissionError(
                "Explicit externally verified checkpoint pin required"
            )
        sources = {}
        for name, expected_sha in SOURCE_PINS.items():
            path = Path(inspect.getsourcefile(importlib.import_module(name)))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_sha:
                raise SharedPoolAdmissionError("Unqualified installed source: " + name)
            sources[name] = actual
        qwen = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
        if (
            getattr(qwen.apply_rotary_pos_emb, "__module__", None) != qwen.__name__
            or Path(inspect.getsourcefile(qwen.apply_rotary_pos_emb)).resolve()
            != Path(inspect.getsourcefile(qwen)).resolve()
        ):
            raise SharedPoolAdmissionError(
                "Other RoPE prototypes must not overlap this gate"
            )
        config = model.llm.config
        dimensions = {
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "model_type": "qwen3",
        }
        if any(getattr(config, k, None) != v for k, v in dimensions.items()):
            raise SharedPoolAdmissionError("Unqualified architecture")
        if config.use_cache is not False or config.rope_parameters != {
            "rope_type": "default",
            "rope_theta": 1000000,
        }:
            raise SharedPoolAdmissionError("Require no KV cache and default RoPE")
        if getattr(config, "_attn_implementation", None) != "sdpa" or any(
            getattr(layer.self_attn.forward, "__module__", None) != qwen.__name__
            for layer in model.llm.layers
        ):
            raise SharedPoolAdmissionError(
                "Require unchanged SDPA attention; no Sage or other attention patch"
            )
        if (
            model.config.num_audio_codebook != 8
            or model.config.audio_vocab_size != 1025
        ):
            raise SharedPoolAdmissionError("Unqualified audio heads")
        if next(model.llm.parameters()).dtype != t.float16:
            raise SharedPoolAdmissionError("Only pinned FP16 path is qualified")
        if t.cuda.memory.get_allocator_backend() != "native":
            raise SharedPoolAdmissionError("Only native allocator is qualified")
        if getattr(config, "output_hidden_states", False) or getattr(
            config, "output_attentions", False
        ):
            raise SharedPoolAdmissionError("Extra captured state is unsupported")
        return {
            "versions": {**versions, "torch": t.__version__},
            "source_SHA256": sources,
            "model_revision": revision,
            "allocator_backend": "native",
            "qualification": "Static restrictions only; GPU correctness unproven",
        }

    def fingerprint(self, model):
        if model.training or self.torch.is_grad_enabled():
            raise SharedPoolAdmissionError(
                "Require eval and an existing inference/no-grad context"
            )
        if self.torch.is_autocast_enabled("cuda"):
            raise SharedPoolAdmissionError("CUDA autocast is not qualified")
        if model.llm.config.use_cache is not False:
            raise SharedPoolAdmissionError("KV setting changed")
        modules = tuple(model.modules())
        if any(
            m.training
            or getattr(m, "_forward_hooks", {})
            or getattr(m, "_forward_pre_hooks", {})
            or getattr(m, "_backward_hooks", {})
            for m in modules
        ):
            raise SharedPoolAdmissionError(
                "Training or activation-retaining hooks are unsupported"
            )

        def state_item(name, tensor):
            try:
                version = tensor._version
            except RuntimeError:
                version = None
            return (
                name,
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                tensor.data_ptr(),
                version,
            )

        return (
            id(model),
            tuple(state_item(n, v) for n, v in model.named_parameters()),
            tuple(state_item(n, v) for n, v in model.named_buffers()),
            json.dumps(model.llm.config.to_dict(), sort_keys=True, default=repr),
            json.dumps(model.config.to_dict(), sort_keys=True, default=repr),
            id(
                importlib.import_module(
                    "transformers.models.qwen3.modeling_qwen3"
                ).apply_rotary_pos_emb
            ),
            tuple(
                (id(m), id(getattr(m.forward, "__func__", m.forward))) for m in modules
            ),
        )

    def signature(self, ids, audio, mask):
        t = self.torch
        if not all(
            isinstance(x, t.Tensor) and x.device.type == "cuda"
            for x in (ids, audio, mask)
        ):
            raise SharedPoolAdmissionError(
                "All inputs must be owner-produced CUDA tensors"
            )
        if ids.dtype != t.int64 or audio.dtype != t.bool or mask.dtype != t.bool:
            raise SharedPoolAdmissionError(
                "Only int64 IDs and bool masks are qualified"
            )
        if not all(x.is_contiguous() for x in (ids, audio, mask)):
            raise SharedPoolAdmissionError(
                "Only ordinary contiguous clone inputs are qualified"
            )
        b, c, s = ids.shape
        if (
            c != 8
            or b % 2
            or tuple(audio.shape) != (b, s)
            or tuple(mask.shape) != (b, 1, s, s)
        ):
            raise SharedPoolAdmissionError("Unexpected clone/CFG tensor shapes")
        if any(x.device != ids.device for x in (audio, mask)):
            raise SharedPoolAdmissionError("Inputs must use the same device")
        return tuple(
            (tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device))
            for x in (ids, audio, mask)
        )

    def owner_stream(self, ids):
        if self.torch.cuda.is_current_stream_capturing():
            raise SharedPoolAdmissionError("Nested capture is unsupported")
        stream = self.torch.cuda.current_stream(device=ids.device)
        return stream, (str(ids.device), stream.cuda_stream)

    def new_capture_stream(self, ids):
        return self.torch.cuda.Stream(device=ids.device)

    def new_pool(self):
        return self.torch.cuda.graph_pool_handle()

    def free_memory(self, ids):
        return int(self.torch.cuda.mem_get_info(device=ids.device)[0])

    def fence(self, stream):
        stream.synchronize()

    def capture(self, forward, ids, audio, mask, pool, capture_stream, owner_stream):
        t = self.torch
        # All static I/O is cloned before capture into the ordinary allocator.
        static_ids, static_audio, static_mask = ids.clone(), audio.clone(), mask.clone()
        self.fence(owner_stream)
        capture_stream.wait_stream(owner_stream)
        with t.cuda.stream(capture_stream):
            warm_output = forward(static_ids, static_audio, attention_mask=static_mask)
        owner_stream.wait_stream(capture_stream)
        self.fence(owner_stream)
        del warm_output
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(
            graph, pool=pool, stream=capture_stream, capture_error_mode="thread_local"
        ):
            output = forward(static_ids, static_audio, attention_mask=static_mask)
        return {
            "graph": graph,
            "static_input_ids": static_ids,
            "static_audio_mask": static_audio,
            "static_attn_mask": static_mask,
            "static_output": output,
        }

    def replay(self, entry, ids, audio, mask):
        entry["static_input_ids"].copy_(ids)
        entry["static_audio_mask"].copy_(audio)
        entry["static_attn_mask"].copy_(mask)
        entry["graph"].replay()

    def global_clone(self, output):
        if self.torch.cuda.is_current_stream_capturing():
            raise SharedPoolFatalError("Output clone must be outside capture")
        # .clone() is intentional: .to(float32) can alias for FP32 inputs.
        return type(output)(loss=None, logits=output.logits.clone())

    def validate_logits(self, output, shape, ids):
        if (
            tuple(output.logits.shape) != (*shape, 1025)
            or output.logits.dtype != self.torch.float16
            or output.logits.device != ids.device
        ):
            raise SharedPoolFatalError("Unexpected logits tensor contract")


class SharedPoolForward:
    """Composition around the exact private runner; default-off is passthrough.

    No private allocator/weak-reference APIs. Shapes are an explicit finite
    allowlist (up to eight), not open-ended production admission. A custom backend
    is solely a CPU-test seam; real probes must use the default TorchBackend.
    """

    def __init__(
        self,
        private_forward,
        *,
        enabled=False,
        allowed_shapes=(),
        max_graphs=8,
        model_revision=None,
        min_free_bytes=4 * GiB,
        max_single_rows=16384,
        backend=None,
    ):
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        if type(max_graphs) is not int or not 1 <= max_graphs <= 8:
            raise ValueError("Finite max_graphs must be in1..8")
        shapes = tuple(tuple(s) for s in allowed_shapes)
        if enabled and (
            not shapes or len(set(shapes)) != len(shapes) or len(shapes) > 8
        ):
            raise ValueError("Enabled probe requires1..8 unique allowed shapes")
        if any(
            len(s) != 3 or any(type(x) is not int or x <= 0 for x in s) for s in shapes
        ):
            raise ValueError("Expected positive [B,C,S] shape allowlist")
        if any(s[1] != 8 or s[0] % 2 or s[0] > 16 or s[2] > 768 for s in shapes):
            raise ValueError("Initial finite gate supports only CFG B<=16,C8,S<=768")
        if type(min_free_bytes) is not int or min_free_bytes < 4 * GiB:
            raise ValueError("Cannot weaken the4GiB capture floor")
        if type(max_single_rows) is not int or not 1 <= max_single_rows <= 16384:
            raise ValueError("Invalid finite single-shape row bound")
        self.private = private_forward
        self._model = private_forward._model
        self._original_forward = private_forward._original_forward
        self.enabled = enabled
        self.allowed_shapes = frozenset(shapes)
        self.max_graphs = max_graphs
        self.revision = model_revision
        self.min_free_bytes = min_free_bytes
        self.max_single_rows = max_single_rows
        self.backend = backend
        self._graphs = OrderedDict()
        self.pool = None
        self.capture_stream = None
        self.owner = None
        self.owner_stream = None
        self.state = None
        self.contract = None
        self.poisoned = False
        self.captures = self.hits = self.evictions = self.epochs = self.returns = 0
        self._call_lock = threading.Lock()

    @staticmethod
    def _validate_output(output):
        if set(vars(output)) != {"loss", "logits"} or output.loss is not None:
            raise SharedPoolFatalError(
                "Only pinned loss=None/logits output is supported"
            )

    def _check_owner_and_state(self, ids):
        if self.owner is not None and threading.get_ident() != self.owner[0]:
            raise SharedPoolAdmissionError("Owner thread changed before CUDA access")
        if self.backend is None:
            self.backend = TorchBackend()
        if self.contract is None:
            if getattr(self.private, "_graphs", {}):
                raise SharedPoolAdmissionError(
                    "Install before any private graph capture"
                )
            self.contract = self.backend.qualify(self._model, self.revision)
        stream, stream_key = self.backend.owner_stream(ids)
        owner = (threading.get_ident(), stream_key)
        state = self.backend.fingerprint(self._model)
        if self.owner is None:
            self.owner, self.owner_stream, self.state = owner, stream, state
        if owner != self.owner:
            raise SharedPoolAdmissionError("Owner thread/stream changed")
        if state != self.state:
            self.poisoned = True
            raise SharedPoolFatalError("Model state/config changed; retire process")

    def _evict(self):
        self.backend.fence(self.owner_stream)
        _, entry = self._graphs.popitem(last=False)
        del entry
        self.evictions += 1
        if not self._graphs:
            # Never reuse a retired handle. External results are global clones.
            self.pool = self.capture_stream = None

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
        if not self.enabled:
            return self.private(
                input_ids,
                audio_mask,
                labels=labels,
                attention_mask=attention_mask,
                document_ids=document_ids,
                position_ids=position_ids,
                **kwargs,
            )
        if not self._call_lock.acquire(blocking=False):
            raise SharedPoolAdmissionError(
                "Concurrent/reentrant call rejected before CUDA work"
            )
        try:
            return self._run(
                input_ids,
                audio_mask,
                labels,
                attention_mask,
                document_ids,
                position_ids,
                kwargs,
            )
        finally:
            self._call_lock.release()

    def _run(
        self,
        input_ids,
        audio_mask,
        labels,
        attention_mask,
        document_ids,
        position_ids,
        kwargs,
    ):
        if self.poisoned:
            raise SharedPoolFatalError("Prototype is poisoned; restart process")
        if (
            labels is not None
            or document_ids is not None
            or position_ids is not None
            or kwargs
            or attention_mask is None
        ):
            raise SharedPoolAdmissionError(
                "Unsupported training/optional argument contract"
            )
        shape = tuple(input_ids.shape)
        if (
            shape not in self.allowed_shapes
            or shape[0] * shape[2] > self.max_single_rows
        ):
            raise SharedPoolAdmissionError("Shape outside finite allowlist/row bound")
        self._check_owner_and_state(input_ids)
        signature = self.backend.signature(input_ids, audio_mask, attention_mask)
        key = shape
        if key in self._graphs:
            entry = self._graphs[key]
            if signature != entry["signature"]:
                raise SharedPoolAdmissionError(
                    "Tensor dtype/shape/stride/device signature changed"
                )
            self._graphs.move_to_end(key)
            self.hits += 1
        else:
            if len(self._graphs) >= self.max_graphs:
                self._evict()
            # Existing qualified-profile estimate, plus the public fixed floor.
            # This is an admission heuristic, not a rigorous CUDA memory bound.
            required = max(self.min_free_bytes, capture_headroom_bytes(shape))
            if self.backend.free_memory(input_ids) < required:
                raise SharedPoolAdmissionError(
                    "Insufficient public free headroom before capture"
                )
            if self.pool is None:
                self.pool = self.backend.new_pool()
                self.capture_stream = self.backend.new_capture_stream(input_ids)
                self.epochs += 1
            try:
                entry = self.backend.capture(
                    self._original_forward,
                    input_ids,
                    audio_mask,
                    attention_mask,
                    self.pool,
                    self.capture_stream,
                    self.owner_stream,
                )
                self._validate_output(entry["static_output"])
                self.backend.validate_logits(entry["static_output"], shape, input_ids)
                if self.backend.fingerprint(self._model) != self.state:
                    raise SharedPoolFatalError(
                        "Model state changed during warmup/capture"
                    )
            except BaseException as exc:
                self.poisoned = True
                raise SharedPoolFatalError("Capture failed; no retry/fallback") from exc
            entry["signature"] = signature
            self._graphs[key] = entry
            self.captures += 1
        try:
            self.backend.replay(entry, input_ids, audio_mask, attention_mask)
            result = self.backend.global_clone(entry["static_output"])
            self._validate_output(result)
            self.backend.validate_logits(result, shape, input_ids)
            if self.backend.fingerprint(self._model) != self.state:
                raise SharedPoolFatalError("Model state changed during replay")
        except BaseException as exc:
            self.poisoned = True
            raise SharedPoolFatalError("Replay/copy failed; no retry/fallback") from exc
        self.returns += 1
        return result

    def clear(self):
        if self.poisoned:
            raise SharedPoolFatalError("Do not touch a poisoned pool; restart process")
        if not self._call_lock.acquire(blocking=False):
            raise SharedPoolAdmissionError(
                "Cannot retire pool while owner is executing"
            )
        try:
            return self._clear()
        finally:
            self._call_lock.release()

    def _clear(self):
        if not self.enabled or not self._graphs:
            return
        # Called only by the live owner; cancellation of an HTTP waiter does not
        # confer ownership or authorize model/pool destruction.
        if threading.get_ident() != self.owner[0]:
            raise SharedPoolAdmissionError("Only owner may retire a pool epoch")
        stream, key = self.backend.owner_stream(
            next(iter(self._graphs.values()))["static_input_ids"]
        )
        if (threading.get_ident(), key) != self.owner:
            raise SharedPoolAdmissionError("Cannot clear from a different stream")
        self.backend.fence(stream)
        self._graphs.clear()
        self.pool = self.capture_stream = None

    def metrics(self):
        return {
            "enabled": self.enabled,
            "prototype_only": True,
            "qualification": self.contract,
            "poisoned": self.poisoned,
            "captures": self.captures,
            "hits": self.hits,
            "evictions": self.evictions,
            "pool_epochs": self.epochs,
            "fresh_global_returns": self.returns,
            "retained_shapes": [list(s) for s in self._graphs],
            "minimum_public_free_bytes": self.min_free_bytes,
            "quality_or_GPU_safety_certified": False,
        }


class SharedPoolPatch:
    """Explicit reversible attachment for a finite probe, never auto-enabled."""

    def __init__(self, model, private_forward, **options):
        if model is not private_forward._model or model.forward is not private_forward:
            raise SharedPoolAdmissionError(
                "Require the unwrapped private graph forward"
            )
        self.model = model
        self.private = private_forward
        self.forward = SharedPoolForward(private_forward, **options)
        self.active = False

    def enable(self):
        if not self.forward.enabled:
            return
        if self.active:
            if self.model.forward is not self.forward:
                raise SharedPoolFatalError("Forward ownership changed while attached")
            return
        if self.model.forward is not self.private:
            raise SharedPoolAdmissionError("Do not overwrite another forward patch")
        self.model.forward = self.forward
        self.active = True

    def disable(self):
        if not self.active:
            return
        if self.model.forward is not self.forward:
            raise SharedPoolFatalError("Cannot restore over another forward owner")
        if self.forward.poisoned:
            raise SharedPoolFatalError(
                "Keep poisoned model/pool owned until process restart"
            )
        self.forward.clear()
        self.model.forward = self.private
        self.active = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.forward.poisoned:
            # Preserve the original capture/replay failure. No CUDA cleanup or
            # eager fallback may run against an invalid context.
            return False
        self.disable()
