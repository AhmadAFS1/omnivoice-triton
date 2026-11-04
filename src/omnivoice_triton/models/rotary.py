"""Default-off, pinned-runtime rotary fusion and inference-owner lifecycle."""

import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import threading
import time
from collections import Counter
from pathlib import Path

import torch

_REGISTRY_LOCK = threading.Lock()
_ACTIVE_PATCH = None
_STARTUP_RESERVATION = None
_SOURCE_PINS = {
    "omnivoice.models.omnivoice": (
        "c7104b1b9574fac6c84014e39636a5f1e51508902fa59cf151a713151669bd40"
    ),
    "transformers.models.qwen3.modeling_qwen3": (
        "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
    ),
}


@contextlib.contextmanager
def reserve_fused_rope_startup(token):
    """Reject a second fused model before it allocates GPU model weights."""
    global _STARTUP_RESERVATION
    with _REGISTRY_LOCK:
        if _ACTIVE_PATCH is not None or _STARTUP_RESERVATION is not None:
            raise RuntimeError("Only one fused-RoPE runner may be active per process.")
        _STARTUP_RESERVATION = token
    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            if _STARTUP_RESERVATION is token:
                _STARTUP_RESERVATION = None


def _qualify(model):
    versions = {
        "torch": "2.8.0+cu128",
        "triton": "3.4.0",
        "transformers": "5.5.4",
        "omnivoice": "0.1.4",
    }
    if any(
        importlib.metadata.version(name) != expected
        for name, expected in versions.items()
    ):
        raise RuntimeError(
            "Experimental fused RoPE requires Torch2.8.0+cu128, Triton3.4.0, "
            "Transformers5.5.4 and OmniVoice0.1.4."
        )
    for name, expected in _SOURCE_PINS.items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                "Experimental fused RoPE installed source differs: " + name
            )
    module = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    original = inspect.unwrap(module.apply_rotary_pos_emb)
    if (
        original is not module.apply_rotary_pos_emb
        or inspect.getsourcefile(original) != module.__file__
    ):
        raise RuntimeError("A foreign rotary patch is already installed.")
    config, llm = model.llm.config, model.llm
    dimensions = {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    if any(getattr(config, name, None) != value for name, value in dimensions.items()):
        raise RuntimeError(
            "Experimental fused RoPE requires the pinned Qwen3 architecture."
        )
    if model.training or llm.training or config.use_cache is not False:
        raise RuntimeError(
            "Experimental fused RoPE requires eval mode and disabled LLM KV cache."
        )
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("Experimental fused RoPE startup does not support autocast.")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError(
            "Experimental fused RoPE requires the pinned SDPA attention path."
        )
    rope = getattr(config, "rope_parameters", None)
    if (
        not isinstance(rope, dict)
        or rope != {"rope_type": "default", "rope_theta": 1000000}
        or getattr(config, "rope_scaling", None) not in [None, rope]
    ):
        raise RuntimeError(
            "Experimental fused RoPE requires default rotary parameters."
        )
    _qualify_attention_forwards(llm, module)
    parameters = list(llm.parameters())
    device = parameters[0].device
    if device.type != "cuda" or any(
        p.device != device or p.dtype != torch.float16 for p in parameters
    ):
        raise RuntimeError(
            "Experimental fused RoPE requires one-device FP16 LLM parameters."
        )
    return module, device


def _qualify_attention_forwards(llm, module):
    if len(llm.layers) != 28:
        raise RuntimeError("Experimental fused RoPE requires28 Qwen3 layers.")
    for layer in llm.layers:
        attention = layer.self_attn
        if (
            type(attention) is not module.Qwen3Attention
            or getattr(attention.forward, "__func__", None)
            is not module.Qwen3Attention.forward
            or any(
                getattr(attention, name, {})
                for name in ["_forward_hooks", "_forward_pre_hooks", "_backward_hooks"]
            )
        ):
            raise RuntimeError(
                "Experimental fused RoPE requires original attention forwards "
                "without activation hooks."
            )


class _OwnedForward:
    def __init__(self, patch):
        self.patch = patch

    def __call__(self, *args, **kwargs):
        with self.patch.executing():
            with self.patch._device_context():
                return self.patch.original_forward(*args, **kwargs)

    def clear(self):
        """Keep ordinary fault recovery on the established GPU owner thread."""
        patch = self.patch
        patch.assert_owner()
        if not patch._execution_lock.acquire(blocking=False):
            raise RuntimeError("Active/reentrant fused-RoPE graphs cannot be cleared.")
        try:
            if not patch.active or patch.retiring or patch.active_calls:
                raise RuntimeError(
                    "Active/retiring fused-RoPE graphs cannot be cleared."
                )
            forward = patch.original_forward
            for _ in range(4):
                clear = getattr(forward, "clear", None)
                if callable(clear):
                    clear()
                    return
                forward = getattr(forward, "original_forward", None)
                if forward is None:
                    return
        finally:
            patch._execution_lock.release()


class FusedRoPEPatch:
    """One active pinned model per process; last four layers stay in Torch.

    Startup warms on its exclusive loading thread and fences before handoff.
    The outer forward binds the actual generation thread, including graph hits.
    A stopped/joined generation thread may hand resources back for shutdown.
    """

    def __init__(self, model, *, reservation_token=None):
        self.model, self.reservation_token = model, reservation_token
        self.active = self.ready = self.retiring = self.outer_attached = False
        self.owner_thread = None
        self.active_calls = self.forward_calls = self.fused_calls = 0
        self.fallbacks = Counter()
        self._state_lock, self._execution_lock = threading.Lock(), threading.Lock()
        self.local = threading.local()
        self.attentions = []
        self.original_forward = None
        self.forward_guard = _OwnedForward(self)
        self.warmup_summary = None
        self._retirement_thread = None
        self._disable_fenced = False
        self.rotary_installed = False

    def enable(self):
        global _ACTIVE_PATCH
        if self.active:
            raise RuntimeError("Fused RoPE is already installed.")
        module, self.device = _qualify(self.model)
        targets = [layer.self_attn for layer in self.model.llm.layers[:24]]
        if not all(callable(attention.forward) for attention in targets):
            raise RuntimeError("Invalid attention forward before installation.")
        self.module, self.original_rope = module, module.apply_rotary_pos_emb
        self.original_forward = self.model.forward
        with _REGISTRY_LOCK:
            if _ACTIVE_PATCH is not None or _STARTUP_RESERVATION not in [
                None,
                self.reservation_token,
            ]:
                raise RuntimeError(
                    "Only one fused-RoPE runner may be active per process."
                )
            _ACTIVE_PATCH = self
        self.active = True

        def rotary(q, k, cos, sin, unsqueeze_dim=1):
            if not getattr(self.local, "layer_active", False):
                return self.original_rope(q, k, cos, sin, unsqueeze_dim)
            from omnivoice_triton.kernels.rotary import (
                _apply_rotary,
                unsupported_reason,
            )

            reason = unsupported_reason(q, k, cos, sin, unsqueeze_dim)
            if not reason and q.device != self.device:
                reason = "device_not_warmed"
            if reason:
                with self._state_lock:
                    self.fallbacks[reason] += 1
                return self.original_rope(q, k, cos, sin, unsqueeze_dim)
            result = _apply_rotary(q, k, cos, sin)
            self.fused_calls += 1
            return result

        self.rotary = rotary
        try:
            for attention in targets:
                original = attention.forward

                def forward(*args, _original=original, **kwargs):
                    if not getattr(self.local, "forward_active", False):
                        raise RuntimeError(
                            "Fused attention must run through its outer owner guard."
                        )
                    self.local.layer_active = True
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        self.local.layer_active = False

                attention.forward = forward
                self.attentions.append((attention, original, forward))
            module.apply_rotary_pos_emb = rotary
            self.rotary_installed = True
            self._warmup()
            self.model.forward = self.forward_guard
            self.outer_attached = True
        except BaseException:
            self.prepare_disable()
            self.disable()
            raise

    def _synchronize(self):
        torch.cuda.synchronize(self.device)

    def _device_context(self):
        return torch.cuda.device(self.device)

    def _warmup(self):
        from omnivoice_triton.kernels.rotary import triton_rotary_pos_emb

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Fused RoPE startup warmup must be outside capture.")
        started = time.perf_counter()
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(self.device).clone()
        with self._device_context(), torch.inference_mode():
            q = torch.zeros(
                (1, 1, 16, 128), device=self.device, dtype=torch.float16
            ).transpose(1, 2)
            k = torch.zeros(
                (1, 1, 8, 128), device=self.device, dtype=torch.float16
            ).transpose(1, 2)
            cos = torch.ones((1, 1, 128), device=self.device, dtype=torch.float16)
            sin = torch.zeros_like(cos)
            triton_rotary_pos_emb(q, k, cos, sin)
            self._synchronize()
        if not torch.equal(cpu_before, torch.random.get_rng_state()) or not torch.equal(
            cuda_before, torch.cuda.get_rng_state(self.device)
        ):
            raise RuntimeError("Fused RoPE startup changed generation RNG state.")
        self.warmup_summary = {
            "device": str(self.device),
            "startup_thread_id": threading.get_ident(),
            "duration_ms": (time.perf_counter() - started) * 1000,
            "CPU_RNG_unchanged": True,
            "CUDA_RNG_unchanged": True,
            "head_shapes": [16, 8],
            "sequence_length": 1,
            "input_kind": "Deterministic zeros/ones; no model or sampling calls",
        }
        self.ready = True

    def assert_owner(self):
        if self.owner_thread is not threading.current_thread():
            raise RuntimeError("Fused RoPE requires its generation owner thread.")

    @contextlib.contextmanager
    def executing(self):
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError(
                "Concurrent/reentrant fused-RoPE forward is unsupported."
            )
        try:
            if not self.active or not self.ready or self.retiring:
                raise RuntimeError("Fused RoPE is not ready for inference.")
            if self.owner_thread is None:
                self.owner_thread = threading.current_thread()
            self.assert_owner()
            if self.model.llm.config.use_cache is not False:
                raise RuntimeError("Fused RoPE KV-cache configuration changed.")
            self.active_calls = 1
            self.forward_calls += 1
            self.local.forward_active = True
            try:
                yield
            finally:
                self.local.forward_active = False
                self.active_calls = 0
        finally:
            self._execution_lock.release()

    def prepare_disable(self):
        if not self.active:
            return
        if self.retiring:
            if (
                self._retirement_thread != threading.get_ident()
                or not self._disable_fenced
            ):
                raise RuntimeError(
                    "Fused RoPE retirement failed or is owned elsewhere; "
                    "retain resources."
                )
            return
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("Active fused-RoPE resources must be retained.")
        owner = self.owner_thread
        if (
            owner is not None
            and owner is not threading.current_thread()
            and owner.is_alive()
        ):
            self._execution_lock.release()
            raise RuntimeError("Live fused-RoPE owner must stop before unload.")
        self.retiring = True
        self.ready = False
        self._retirement_thread = threading.get_ident()
        # A failed CUDA fence deliberately retains every reference for restart.
        self._synchronize()
        self._disable_fenced = True

    def detach_outer(self):
        if self.outer_attached:
            if self.model.forward is not self.forward_guard:
                raise RuntimeError(
                    "A foreign forward replaced the fused-RoPE owner guard."
                )
            self.model.forward = self.original_forward
            self.outer_attached = False

    def disable(self):
        global _ACTIVE_PATCH
        if not self.active:
            return
        self.prepare_disable()
        self.detach_outer()
        expected_rope = self.rotary if self.rotary_installed else self.original_rope
        if self.module.apply_rotary_pos_emb is not expected_rope or any(
            attention.forward is not wrapped
            for attention, _, wrapped in self.attentions
        ):
            raise RuntimeError(
                "A foreign attention/rotary patch prevents safe restoration."
            )
        for attention, original, _ in self.attentions:
            attention.forward = original
        self.module.apply_rotary_pos_emb = self.original_rope
        with _REGISTRY_LOCK:
            if _ACTIVE_PATCH is not self:
                raise RuntimeError("Fused RoPE registry ownership changed.")
            _ACTIVE_PATCH = None
        self.active = self.ready = False
        self.attentions.clear()
        self.original_forward = self.model = self.module = self.original_rope = None
        self.owner_thread = None
        self._execution_lock.release()

    def metrics(self):
        with self._state_lock:
            return {
                "active": self.active,
                "ready": self.ready,
                "experimental": True,
                "retiring": self.retiring,
                "patch_range": [0, 24],
                "forward_calls": self.forward_calls,
                "fused_calls": self.fused_calls,
                "fallbacks": dict(self.fallbacks),
                "counter_scope": (
                    "Forward calls include graph replay; rotary counters "
                    "count Python capture/eager calls."
                ),
                "startup_warmup": self.warmup_summary,
                "scope": (
                    "Pinned runtime/source/Qwen3 architecture; "
                    "checkpoint bytes are not rehashed by this option"
                ),
            }
