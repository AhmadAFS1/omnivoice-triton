"""CPU lifecycle/layout contracts, not GPU numerical/throughput qualification."""

import contextlib
import functools
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from omnivoice_triton import create_runner
from omnivoice_triton.kernels import rotary as kernel
from omnivoice_triton.models import rotary
from omnivoice_triton.models.base_runner import BaseRunner


@pytest.fixture
def setup(monkeypatch):
    assert rotary._ACTIVE_PATCH is None
    module = SimpleNamespace(apply_rotary_pos_emb=kernel.reference_rope)

    class Attention:
        def forward(self, *args):
            return module.apply_rotary_pos_emb(*args)

    class Model:
        def __init__(self):
            self.llm = SimpleNamespace(
                config=SimpleNamespace(use_cache=False),
                layers=[SimpleNamespace(self_attn=Attention()) for _ in range(28)],
            )
            self.forward = self.original

        def original(self, *args):
            return [layer.self_attn.forward(*args) for layer in self.llm.layers]

    monkeypatch.setattr(rotary, "_qualify", lambda model: (module, torch.device("cpu")))
    monkeypatch.setattr(rotary.FusedRoPEPatch, "_synchronize", lambda self: None)
    monkeypatch.setattr(
        rotary.FusedRoPEPatch, "_device_context", lambda self: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda device: torch.zeros(8, dtype=torch.uint8)
    )
    monkeypatch.setattr(kernel, "triton_rotary_pos_emb", kernel.reference_rope)
    monkeypatch.setattr(kernel, "_apply_rotary", kernel.reference_rope)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield module, Model
    # Deliberately poisoned-fence tests restore only these CPU fake resources.
    patch = rotary._ACTIVE_PATCH
    if patch is not None:
        monkeypatch.setattr(patch, "_synchronize", lambda: None)
        patch._disable_fenced = True
        patch.disable()
    assert rotary._STARTUP_RESERVATION is None


def inputs():
    return (
        torch.zeros(1, 16, 3, 128, dtype=torch.float16),
        torch.zeros(1, 8, 3, 128, dtype=torch.float16),
        torch.ones(1, 3, 128, dtype=torch.float16),
        torch.zeros(1, 3, 128, dtype=torch.float16),
    )


@pytest.mark.parametrize("name", ["base", "faster"])
def test_unsupported_runner_rejected_before_loading(name):
    with pytest.raises(ValueError, match="triton/hybrid"):
        create_runner(
            name, device="cuda", disable_unused_kv_cache=True, fused_rope=True
        )


@pytest.mark.parametrize(
    "option",
    [dict(dtype="bf16"), dict(device="cpu"), dict(disable_unused_kv_cache=False)],
)
def test_invalid_profile_rejected_before_loading(option):
    kwargs = (
        dict(device="cuda", dtype="fp16", disable_unused_kv_cache=True, fused_rope=True)
        | option
    )
    with pytest.raises(ValueError, match="Experimental fused RoPE"):
        create_runner("triton", **kwargs)


@pytest.mark.parametrize("name", ["triton", "hybrid"])
@pytest.mark.parametrize(
    "extra", [dict(patch_range=None), dict(enable_sage_attention=True)]
)
def test_other_patch_profiles_rejected(name, extra):
    with pytest.raises(ValueError, match="default .*SDPA"):
        create_runner(
            name, device="cuda", disable_unused_kv_cache=True, fused_rope=True, **extra
        )


def test_startup_is_rng_neutral_and_does_not_bind_generation_owner(setup, monkeypatch):
    _, Model = setup
    monkeypatch.setattr(torch, "randn", lambda *a, **k: pytest.fail("RNG startup"))
    monkeypatch.setattr(torch, "rand", lambda *a, **k: pytest.fail("RNG startup"))
    before = torch.random.get_rng_state().clone()
    patch = rotary.FusedRoPEPatch(Model())
    patch.enable()
    assert patch.owner_thread is None and patch.ready
    assert torch.equal(before, torch.random.get_rng_state())
    assert patch.warmup_summary["CPU_RNG_unchanged"]
    assert patch.warmup_summary["CUDA_RNG_unchanged"]
    patch.disable()


def test_warmup_rng_change_rolls_back_all_patch_state(setup, monkeypatch):
    module, Model = setup
    model = Model()
    original, forwards = (
        module.apply_rotary_pos_emb,
        [x.self_attn.forward for x in model.llm.layers],
    )
    states = iter([torch.zeros(8, dtype=torch.uint8), torch.ones(8, dtype=torch.uint8)])
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: next(states))
    patch = rotary.FusedRoPEPatch(model)
    with pytest.raises(RuntimeError, match="RNG state"):
        patch.enable()
    assert rotary._ACTIVE_PATCH is None and not patch.active
    assert module.apply_rotary_pos_emb is original
    assert [x.self_attn.forward for x in model.llm.layers] == forwards


def test_startup_reservation_rejects_second_model_before_loading(setup):
    _, Model = setup
    token = object()
    with rotary.reserve_fused_rope_startup(token):
        with pytest.raises(RuntimeError, match="one fused-RoPE"):
            with rotary.reserve_fused_rope_startup(object()):
                pytest.fail("Second model reached GPU loading")
        patch = rotary.FusedRoPEPatch(Model(), reservation_token=token)
        patch.enable()
    with pytest.raises(RuntimeError, match="one fused-RoPE"):
        with rotary.reserve_fused_rope_startup(object()):
            pytest.fail("Second model reached GPU loading")
    patch.disable()


def test_only24_attention_paths_are_fused_and_outer_hits_keep_owner(setup, monkeypatch):
    module, Model = setup
    model = Model()
    monkeypatch.setattr(kernel, "unsupported_reason", lambda *a: "")
    monkeypatch.setattr(
        kernel,
        "_apply_rotary",
        lambda *a: tuple(x + 1 for x in kernel.reference_rope(*a)),
    )
    patch = rotary.FusedRoPEPatch(model)
    patch.enable()
    result = model.forward(*inputs())
    assert all(torch.equal(x[0], torch.ones_like(x[0])) for x in result[:24])
    assert all(torch.equal(x[0], torch.zeros_like(x[0])) for x in result[24:])
    assert patch.fused_calls == 24 and patch.forward_calls == 1
    errors = []

    def foreign():
        try:
            model.forward(*inputs())
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert len(errors) == 1 and "owner thread" in errors[0]
    patch.disable()
    assert module.apply_rotary_pos_emb is kernel.reference_rope


def test_replay_without_attention_still_checks_owner_and_reentrancy(setup):
    _, Model = setup
    model = Model()
    calls = []
    model.forward = lambda *a: calls.append(1)
    patch = rotary.FusedRoPEPatch(model)
    patch.enable()
    model.forward()
    with patch.executing(), pytest.raises(RuntimeError, match="Concurrent/reentrant"):
        model.forward()
    assert calls == [1]
    patch.disable()


def test_reentrant_clear_cannot_release_graph_buffers(setup):
    _, Model = setup
    model = Model()
    cleared = []

    class Forward:
        def __call__(self):
            model.forward.clear()

        def clear(self):
            cleared.append(True)

    model.forward = Forward()
    patch = rotary.FusedRoPEPatch(model)
    patch.enable()
    with pytest.raises(RuntimeError, match="reentrant"):
        model.forward()
    assert not cleared
    model.forward.clear()
    assert cleared == [True]
    patch.disable()


def test_original_attention_contract_rejects_foreign_forward_and_hooks():
    class Attention:
        def forward(self):
            return None

    module = SimpleNamespace(Qwen3Attention=Attention)
    llm = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=Attention()) for _ in range(28)]
    )
    rotary._qualify_attention_forwards(llm, module)
    llm.layers[0].self_attn.forward = lambda: None
    with pytest.raises(RuntimeError, match="original attention"):
        rotary._qualify_attention_forwards(llm, module)
    llm.layers[0].self_attn = Attention()
    llm.layers[-1].self_attn._forward_pre_hooks = {1: object()}
    with pytest.raises(RuntimeError, match="activation hooks"):
        rotary._qualify_attention_forwards(llm, module)


def test_transparent_foreign_rotary_wrapper_rejected_before_model_access(monkeypatch):
    def original():
        return None

    @functools.wraps(original)
    def foreign():
        return original()

    module = SimpleNamespace(apply_rotary_pos_emb=foreign, __file__=__file__)
    versions = {
        "torch": "2.8.0+cu128",
        "triton": "3.4.0",
        "transformers": "5.5.4",
        "omnivoice": "0.1.4",
    }
    monkeypatch.setattr(rotary.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(rotary, "_SOURCE_PINS", {})
    monkeypatch.setattr(rotary.importlib, "import_module", lambda name: module)
    with pytest.raises(RuntimeError, match="foreign rotary"):
        rotary._qualify(None)


def test_stopped_owner_handoff_and_live_foreign_unload_guard(setup, monkeypatch):
    _, Model = setup
    model = Model()
    model.forward = lambda: 42
    patch = rotary.FusedRoPEPatch(model)
    patch.enable()
    done, finish = threading.Event(), threading.Event()

    def worker():
        model.forward()
        done.set()
        finish.wait(5)

    thread = threading.Thread(target=worker)
    thread.start()
    assert done.wait(5)
    monkeypatch.setattr(
        patch, "_synchronize", lambda: pytest.fail("CUDA before live-owner rejection")
    )
    with pytest.raises(RuntimeError, match="Live .* owner"):
        patch.prepare_disable()
    assert patch.active and not patch.retiring
    finish.set()
    thread.join()
    monkeypatch.setattr(patch, "_synchronize", lambda: None)
    patch.disable()
    assert model.forward() == 42


def test_failed_fence_never_becomes_permission_to_free_on_retry(setup, monkeypatch):
    _, Model = setup
    patch = rotary.FusedRoPEPatch(Model())
    patch.enable()
    monkeypatch.setattr(
        patch, "_synchronize", lambda: (_ for _ in ()).throw(RuntimeError("fence"))
    )
    with pytest.raises(RuntimeError, match="fence"):
        patch.prepare_disable()
    assert patch.active and patch.model is not None and patch.retiring
    with pytest.raises(RuntimeError, match="retirement failed"):
        patch.disable()


def test_forward_fallback_is_counted_and_original_values_preserved(setup, monkeypatch):
    _, Model = setup
    model = Model()
    patch = rotary.FusedRoPEPatch(model)
    patch.enable()
    monkeypatch.setattr(kernel, "unsupported_reason", lambda *a: "future_layout")
    observed = model.forward(*inputs())
    assert all(torch.equal(x[0], torch.zeros_like(x[0])) for x in observed)
    assert patch.metrics()["fallbacks"] == {"future_layout": 24}
    patch.disable()


def test_no_model_cycle_after_disable_with_external_model_retained(setup):
    _, Model = setup
    model = Model()
    ref = weakref.ref(model)
    patch = rotary.FusedRoPEPatch(model)
    patch.enable()
    patch.disable()
    assert model.forward == model.original
    del model
    gc.collect()
    assert ref() is None and patch.model is None and patch.original_forward is None


def test_hybrid_unload_clears_graph_before_rotary_and_weights(setup, monkeypatch):
    _, Model = setup
    from omnivoice_triton.models import triton_faster_runner

    model = Model()
    events = []
    monkeypatch.setattr(
        BaseRunner, "load_model", lambda self: setattr(self, "_model", model)
    )
    monkeypatch.setattr(triton_faster_runner, "find_patchable_model", lambda m: m)
    monkeypatch.setattr(
        triton_faster_runner, "apply_triton_kernels", lambda *a, **k: None
    )
    runner = create_runner(
        "hybrid", device="cuda", disable_unused_kv_cache=True, fused_rope=True
    )
    runner.load_model()
    patch = runner._fused_rope_patch
    monkeypatch.setattr(patch, "_synchronize", lambda: events.append("fence"))
    original_disable = patch.disable

    def disable():
        events.append("rotary")
        return original_disable()

    monkeypatch.setattr(patch, "disable", disable)
    monkeypatch.setattr(runner._graph_forward, "clear", lambda: events.append("graph"))
    runner.unload_model()
    assert events == ["fence", "graph", "rotary"]
    assert runner.model is None and runner._fused_rope_patch is None
    assert model.forward == model.original


@pytest.mark.parametrize("b,s", [(1, 1), (2, 7), (16, 7)])
def test_meta_output_strides_match_torch_tensoriterator(b, s):
    q = torch.zeros(b, s, 16, 128, dtype=torch.float16).transpose(1, 2)
    k = torch.zeros(b, s, 8, 128, dtype=torch.float16).transpose(1, 2)
    c = torch.ones(1, s, 128, dtype=torch.float16)
    t = torch.zeros_like(c)
    expected = kernel.reference_rope(q, k, c, t)
    actual = kernel.output_layout(
        *[p for x in [q, k, c, t] for p in [tuple(x.shape), tuple(x.stride())]]
    )
    assert actual == tuple(tuple(x.stride()) for x in expected)
