"""CPU-only fake graph/allocator contracts; no real model or GPU inference."""

from collections import OrderedDict
from types import SimpleNamespace
from weakref import ref

import pytest
import torch

from omnivoice_triton.models.graph_memory import (
    GiB,
    GraphMemoryAdmissionError,
    GraphMemoryProfile,
    MemoryBoundGraphForward,
)


class Marker:
    pass


class Graph:
    def __init__(self, max_graphs=8):
        self._model = SimpleNamespace(training=False)
        self.max_graphs = max_graphs
        self._graphs = OrderedDict()
        self.capture_states = []
        self.raw_calls = 0
        self.replays = 0
        self.failure = None

    def clear(self):
        self._graphs.clear()

    def _get_shape_key(self, ids):
        return tuple(ids.shape)

    def _original_forward(self, ids, audio, **kwargs):
        self.raw_calls += 1
        return SimpleNamespace(logits=torch.stack([ids.double(), ids.double() + 1], -1))

    def __call__(self, ids, audio, **kwargs):
        key = self._get_shape_key(ids)
        if key not in self._graphs:
            self.capture_states.append(list(self._graphs))
            if self.failure:
                raise self.failure
            self._graphs[key] = {
                "graph": Marker(),
                "static_output": self._original_forward(ids, audio, **kwargs),
                "static_attn_mask": None,
            }
            while self.max_graphs > 0 and len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self.replays += 1
            self._graphs[key]["static_output"].logits.copy_(
                self._original_forward(ids, audio, **kwargs).logits
            )
        self._graphs.move_to_end(key)
        return self._graphs[key]["static_output"]


def data(s, b=1):
    return torch.arange(b * 8 * s).reshape(b, 8, s), torch.ones(
        (b, s), dtype=torch.bool
    )


def wrapper(g, cap, *, free=16 * GiB, profile=None, cleanup=None):
    return MemoryBoundGraphForward(
        g,
        max_row_tokens=cap,
        profile=profile or GraphMemoryProfile(True),
        memory_info=lambda device: (free, 16 * GiB),
        cleanup=cleanup or (lambda device: None),
    )


@torch.inference_mode()
def test_count_pre_eviction_occurs_before_capture_and_drops_reference():
    g = Graph(2)
    m = wrapper(g, 100)
    m(*data(10))
    old = ref(g._graphs[(1, 8, 10)]["graph"])
    m(*data(11))
    m(*data(12))
    assert old() is None
    assert g.capture_states[-1] == [(1, 8, 11)]
    assert list(g._graphs) == [(1, 8, 11), (1, 8, 12)]


@torch.inference_mode()
def test_weighted_lru_pre_eviction_preserves_current_keys_and_arithmetic():
    g = Graph()
    m = wrapper(g, 32)
    m(*data(10))
    m(*data(11))
    m(*data(10))
    ids, audio = data(12)
    expected = g._original_forward(ids, audio).logits
    out = m(ids, audio)
    assert torch.equal(out.logits, expected)
    assert list(g._graphs) == [(1, 8, 10), (1, 8, 12)]
    assert m.metrics()["retained_row_tokens"] == 22
    assert m.metrics()["evictions"] == 1


@torch.inference_mode()
@pytest.mark.parametrize("cap,captured", [(8192, False), (16384, True)])
def test_failed_real_shape_8336_rows_admitted_only_by_larger_budget(cap, captured):
    g = Graph()
    m = wrapper(g, cap)
    ids, audio = data(521, b=16)
    expected = g._original_forward(ids, audio).logits
    out = m(ids, audio)
    assert torch.equal(out.logits, expected)
    assert bool(g._graphs) == captured
    assert m.metrics()["eager_fallbacks"] == (
        {} if captured else {"shape_exceeds_row_budget": 1}
    )


@torch.inference_mode()
def test_fp32_weight_normalization_halves_retained_actual_row_capacity():
    g = Graph()
    m = wrapper(g, 30, profile=GraphMemoryProfile(True, dtype_bytes=4))
    m(*data(10))
    m(*data(11))
    assert list(g._graphs) == [(1, 8, 11)]
    assert m.metrics()["retained_row_tokens"] == 22


@torch.inference_mode()
def test_capture_headroom_falls_back_before_attempting_capture():
    g = Graph()
    m = wrapper(g, 16384, free=3 * GiB)
    ids, audio = data(512, b=16)
    assert (
        m.profile.eager_headroom(tuple(ids.shape))
        < 3 * GiB
        < m.profile.capture_headroom(tuple(ids.shape))
    )
    result = m(ids, audio)
    assert result.logits.shape[:3] == ids.shape and not g.capture_states
    assert m.metrics()["eager_fallbacks"] == {"capture_headroom": 1}


@torch.inference_mode()
def test_insufficient_eager_workspace_rejects_before_model_or_capture():
    g = Graph()
    m = wrapper(g, 16384, free=1 * GiB)
    with pytest.raises(GraphMemoryAdmissionError):
        m(*data(512, b=16))
    assert not g.capture_states and g.raw_calls == 0
    assert m.metrics()["admission_rejections"] == 1


@torch.inference_mode()
def test_pressure_cleanup_occurs_only_after_evicted_entries_are_released():
    g = Graph()
    m = wrapper(g, 8192)
    m(*data(10))
    old = ref(g._graphs[(1, 8, 10)]["graph"])
    state = {"free": 1}

    def cleanup(device):
        assert old() is None
        state["free"] = 16 * GiB

    m.memory_info = lambda device: (state["free"], 16 * GiB)
    m.cleanup = cleanup
    m(*data(512, b=16))
    assert m.metrics()["pressure_cleanups"] == 1
    assert g.capture_states[-1] == []


@torch.inference_mode()
def test_capture_failure_propagates_without_eager_retry():
    g = Graph()
    g.failure = RuntimeError("simulated capture abort")
    m = wrapper(g, 16384)
    with pytest.raises(RuntimeError, match="simulated capture abort"):
        m(*data(10))
    assert g.raw_calls == 0 and m.metrics()["capture_failures"] == 1
    assert not m.metrics()["eager_fallbacks"]


@torch.inference_mode()
def test_enabling_on_existing_cache_normalizes_budget_even_on_hit():
    g = Graph()
    g(*data(10))
    g(*data(11))
    g(*data(12))
    m = wrapper(g, 25)
    m(*data(12))
    assert m.retained_weight() <= 25
    assert (1, 8, 12) in g._graphs


@torch.inference_mode()
def test_hit_low_headroom_never_falls_back_to_larger_eager_work():
    g = Graph()
    m = wrapper(g, 100)
    m(*data(10))
    m(*data(11))
    m.memory_info = lambda device: (1, 16 * GiB)
    before = g.raw_calls
    with pytest.raises(GraphMemoryAdmissionError):
        m(*data(11))
    assert g.raw_calls == before and list(g._graphs) == [(1, 8, 11)]


def test_default_off_preserves_legacy_path_without_cuda_queries():
    g = Graph()

    def forbidden(*args):
        raise AssertionError("Disabled policy must not query memory")

    m = MemoryBoundGraphForward(
        g, max_row_tokens=0, memory_info=forbidden, cleanup=forbidden
    )
    m(*data(10))
    assert len(g._graphs) == 1
    assert m.metrics()["enabled"] is False


@torch.inference_mode()
def test_unknown_contract_is_eager_and_bucket_private_interface_is_preserved():
    g = Graph()
    m = wrapper(g, 8192, profile=GraphMemoryProfile(False, "unknown"))
    m(*data(10))
    assert not g._graphs
    assert m._graphs is g._graphs and m.max_graphs == g.max_graphs
    assert m._original_forward == g._original_forward
    assert m.metrics()["eager_fallbacks"] == {"unqualified_model:unknown": 1}


@pytest.mark.parametrize("cap", [-1, 1.5, True, float("nan")])
def test_invalid_budget_rejected_without_graph_work(cap):
    with pytest.raises(ValueError):
        wrapper(Graph(), cap)


@pytest.mark.parametrize(
    "change,qualified",
    [
        ({}, True),
        ({"revision": None}, True),
        ({"revision": "unknown"}, False),
        ({"hidden_size": 2048}, False),
        ({"rope_parameters": {"rope_type": "dynamic"}}, False),
        (
            {"rope_scaling": {"rope_type": "default", "rope_theta": 1000000}},
            True,
        ),
        ({"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, False),
        ({"dtype": torch.float64}, False),
    ],
)
def test_model_factory_requires_verified_pinned_architecture(
    monkeypatch, change, qualified
):
    import importlib.metadata

    from omnivoice_triton.models.graph_memory import PIN

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.4")
    config = {
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "model_type": "qwen3",
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
        "_attn_implementation": "sdpa",
        "rope_scaling": None,
        "use_cache": True,
    }
    config.update({k: v for k, v in change.items() if k in config})
    tensor = torch.zeros(1, dtype=change.get("dtype", torch.float16))
    model = SimpleNamespace(
        llm=SimpleNamespace(
            config=SimpleNamespace(**config), parameters=lambda: iter([tensor])
        ),
        config=SimpleNamespace(num_audio_codebook=8, audio_vocab_size=1025),
    )
    profile = GraphMemoryProfile.from_model(model, revision=change.get("revision", PIN))
    assert profile.qualified is qualified


@torch.inference_mode()
def test_cuda_autocast_bypasses_qualified_capture_estimate(monkeypatch):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
    g = Graph()
    m = wrapper(g, 8192)
    m(*data(10))
    assert not g._graphs
    assert m.metrics()["eager_fallbacks"] == {"unqualified_cuda_autocast": 1}


def test_unqualified_existing_key_does_not_break_health_snapshot():
    g = Graph()
    g._graphs[("opaque", torch.float16)] = {}
    m = wrapper(g, 8192, profile=GraphMemoryProfile(False, "unknown"))
    snapshot = m.metrics()
    assert snapshot["retained_row_tokens"] is None
    assert snapshot["unqualified_graph_keys"] == 1
    assert snapshot["resident_shapes"][0]["normalized_row_tokens"] is None


def test_runner_contract_rejects_unsupported_and_invalid_budgets():
    from omnivoice_triton import create_runner

    for name in ["base", "triton"]:
        with pytest.raises(ValueError, match="faster or hybrid"):
            create_runner(name, device="cpu", max_cuda_graph_row_tokens=16384)
    for name in ["base", "triton", "faster", "hybrid"]:
        for invalid in [-1, True, 1.5]:
            with pytest.raises(ValueError, match="integer >= 0"):
                create_runner(name, device="cpu", max_cuda_graph_row_tokens=invalid)
        runner = create_runner(name, device="cpu")
        assert runner.max_cuda_graph_row_tokens == 0


@pytest.mark.parametrize("name", ["faster", "hybrid"])
@pytest.mark.parametrize("split", [False, True])
def test_runner_installs_guard_before_split_and_releases_references(
    monkeypatch, name, split
):
    from omnivoice_triton import create_runner
    from omnivoice_triton.models import base_runner, triton_faster_runner
    from omnivoice_triton.models.base_runner import BaseRunner

    def original(*args, **kwargs):
        return None

    model = SimpleNamespace(
        training=False,
        forward=original,
        _generate_iterative=lambda *args, **kwargs: None,
        llm=SimpleNamespace(config=SimpleNamespace(use_cache=False)),
    )
    monkeypatch.setattr(
        BaseRunner, "load_model", lambda self: setattr(self, "_model", model)
    )
    monkeypatch.setattr(base_runner, "version", lambda _: "0.1.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        triton_faster_runner, "apply_triton_kernels", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(triton_faster_runner, "find_patchable_model", lambda _: None)
    runner = create_runner(
        name, device="cpu", split_cfg=split, max_cuda_graph_row_tokens=16384
    )
    runner.load_model()
    guard = runner._graph_memory
    assert guard.graph is runner._graph_forward
    assert guard._original_forward is original
    if split:
        assert runner._split_cfg_patch.original_forward is guard
    else:
        assert model.forward is guard
    snapshot = runner.get_inference_optimization_metrics()
    assert snapshot["max_cuda_graph_row_tokens"] == 16384
    assert snapshot["cuda_graph_memory"]["enabled"] is True
    assert snapshot["cuda_graph_memory"]["profile"]["revision"] is None
    with pytest.raises(ValueError, match="bypass memory admission"):
        runner.prewarm_cuda_graph_shapes([(2, 8, 459)])
    assert runner.prewarm_cuda_graph_shapes([])["warmed_shapes"] == []
    runner.unload_model()
    assert runner._graph_forward is None
    assert runner._graph_memory is None
    assert runner._split_cfg_patch is None
    assert model.forward is original
    assert not guard._graphs
    assert runner.get_inference_optimization_metrics()["cuda_graph_memory"] is None


def test_disabled_runner_keeps_direct_graph_forward(monkeypatch):
    from omnivoice_triton.models.base_runner import BaseRunner
    from omnivoice_triton.models.faster_runner import FasterRunner

    model = SimpleNamespace(training=False, forward=lambda *args, **kwargs: None)
    monkeypatch.setattr(
        BaseRunner, "load_model", lambda self: setattr(self, "_model", model)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner = FasterRunner(device="cpu")
    runner.load_model()
    assert runner._graph_memory is None
    assert model.forward is runner._graph_forward
    runner.unload_model()


@pytest.mark.parametrize("split", [False, True])
def test_guarded_unload_releases_wrapper_before_allocator_cleanup(monkeypatch, split):
    from omnivoice_triton.models import base_runner
    from omnivoice_triton.models.base_runner import BaseRunner
    from omnivoice_triton.models.faster_runner import FasterRunner

    model = SimpleNamespace(
        training=False,
        forward=lambda *args, **kwargs: None,
        _generate_iterative=lambda *args, **kwargs: None,
    )
    original = model.forward
    monkeypatch.setattr(
        BaseRunner, "load_model", lambda self: setattr(self, "_model", model)
    )
    monkeypatch.setattr(base_runner, "version", lambda _: "0.1.4")
    runner = FasterRunner(
        device="cpu", split_cfg=split, max_cuda_graph_row_tokens=16384
    )
    runner.load_model()
    guard_ref = ref(runner._graph_memory)
    graph_ref = ref(runner._graph_forward)

    def cleanup():
        assert guard_ref() is None
        assert graph_ref() is None
        assert model.forward is original

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", cleanup)
    runner.unload_model()


@torch.inference_mode()
def test_clear_delegates_only_from_inference_owner():
    from concurrent.futures import ThreadPoolExecutor

    g = Graph()
    m = wrapper(g, 8192)
    m(*data(10))
    captured = ref(g._graphs[(1, 8, 10)]["graph"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="owner thread"):
            executor.submit(m.clear).result()
    assert captured() is not None
    m.clear()
    assert captured() is None
    assert not g._graphs and m.metrics()["clears"] == 1


def test_health_does_not_format_unknown_graph_keys_or_call_cuda(monkeypatch):
    class Opaque:
        def __repr__(self):
            raise AssertionError("Health must not render opaque objects")

        def __len__(self):
            raise AssertionError("Health must not inspect opaque objects")

    def forbidden(*args, **kwargs):
        raise AssertionError("Health must not call CUDA")

    g = Graph()
    g._graphs[Opaque()] = {}
    m = wrapper(g, 8192, profile=GraphMemoryProfile(False, "unknown"))
    m.memory_info = forbidden
    m.cleanup = forbidden
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    snapshot = m.metrics()
    assert snapshot["unqualified_graph_keys"] == 1
    assert snapshot["resident_shapes"] == [
        {"key_type": "Opaque", "key_arity": None, "normalized_row_tokens": None}
    ]
