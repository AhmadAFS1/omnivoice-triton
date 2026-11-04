"""CPU contracts for explicit inference optimizations; no model weights loaded."""

from types import SimpleNamespace

import pytest
import torch

from omnivoice_triton import create_runner
from omnivoice_triton.models.base_runner import BaseRunner
from omnivoice_triton.models.faster_runner import FasterRunner


@pytest.mark.parametrize(
    "checkpoint_cache,disable_cache,expected",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_kv_option_preserves_checkpoint_default_and_is_applied_at_load(
    monkeypatch, checkpoint_cache, disable_cache, expected
):
    from omnivoice import OmniVoice

    model = SimpleNamespace(
        llm=SimpleNamespace(config=SimpleNamespace(use_cache=checkpoint_cache))
    )
    monkeypatch.setattr(OmniVoice, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observed = []
    monkeypatch.setattr(
        BaseRunner,
        "_install_generation_metrics",
        lambda self, m: observed.append(m.llm.config.use_cache),
    )
    runner = BaseRunner(device="cpu", disable_unused_kv_cache=disable_cache)
    runner.load_model()
    try:
        assert observed == [expected]
        settings = runner.get_inference_optimization_metrics()
        assert settings["llm_use_cache"] is expected
        assert settings["disable_unused_kv_cache"] is disable_cache
        assert settings["split_cfg_enabled"] is False
    finally:
        runner.unload_model()


@pytest.mark.parametrize("name", ["base", "triton"])
def test_split_cfg_rejects_runners_without_graph_wrapper(name):
    with pytest.raises(ValueError, match="faster or hybrid"):
        create_runner(name, device="cpu", split_cfg=True)


def test_split_cfg_installs_outside_graph_wrapper_and_restores_on_unload(monkeypatch):
    from omnivoice_triton.models import base_runner

    model = SimpleNamespace(
        training=False,
        forward=lambda *args, **kwargs: None,
        _generate_iterative=lambda *args, **kwargs: None,
        llm=SimpleNamespace(config=SimpleNamespace(use_cache=True)),
    )
    monkeypatch.setattr(
        BaseRunner, "load_model", lambda self: setattr(self, "_model", model)
    )
    monkeypatch.setattr(base_runner, "version", lambda _: "0.1.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner = FasterRunner(device="cpu", split_cfg=True)
    runner.load_model()
    assert runner._split_cfg_patch.original_forward is runner._graph_forward
    assert runner.model.forward is runner._split_cfg_patch.forward
    assert runner.get_inference_optimization_metrics()["split_cfg_enabled"] is True
    with pytest.raises(ValueError, match="warm through real generation"):
        runner.prewarm_cuda_graph_shapes([(2, 8, 459)])
    runner.unload_model()
    assert runner._split_cfg_patch is None
    assert runner.get_inference_optimization_metrics()["split_cfg_enabled"] is False


def test_split_cfg_rejects_unqualified_library_interface(monkeypatch):
    from omnivoice_triton.models import base_runner

    monkeypatch.setattr(base_runner, "version", lambda _: "0.2.1")
    runner = FasterRunner(device="cpu", split_cfg=True)
    with pytest.raises(RuntimeError, match="pinned OmniVoice 0.1.4"):
        runner._install_split_cfg()
