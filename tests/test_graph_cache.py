"""CPU-only graph ownership contract using fake CUDA streams and capture."""

from contextlib import nullcontext
from types import SimpleNamespace
from weakref import ref

import torch

from omnivoice_triton.models.faster_runner import _CUDAGraphForward


def test_graph_eviction_releases_entry_without_flushing_global_allocator(monkeypatch):
    class Stream:
        def wait_stream(self, other):
            pass

    class Graph:
        pass

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def forbid_global_flush():
        raise AssertionError("LRU eviction must not flush the global CUDA allocator")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbid_global_flush)
    model = SimpleNamespace(
        forward=lambda ids, mask, **kwargs: SimpleNamespace(logits=ids.float()),
    )
    forward = _CUDAGraphForward(model, max_graphs=1)
    first = forward._capture(torch.ones(1, 2, 3), torch.ones(1, 3))
    evicted_graph = ref(first["graph"])
    del first

    latest = forward._capture(torch.ones(1, 2, 4), torch.ones(1, 4))
    assert evicted_graph() is None
    assert list(forward._graphs) == [(1, 2, 4)]
    assert forward._graphs[(1, 2, 4)] is latest
