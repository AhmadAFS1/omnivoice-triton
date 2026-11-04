"""Experimental removal of padded unconditional classifier-free-guidance work.

This optimization is experimental and disabled by default. Install only after loading
an inference model. It wraps the existing forward/graph implementation and the
UNCHANGED installed iterative method. No generation steps, temperatures, sample
order, prompt/audio tokens, or output codec operations are changed.

The real conditional/unconditional attention submatrices are identical. Different
GEMM/attention batch shapes can change floating-point rounding, so real-model
seeded token, waveform and word checks remain mandatory before qualification.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


@dataclass(frozen=True)
class GuidanceLayout:
    batch_size: int
    target_lengths: tuple[int, ...]

    @property
    def max_target_length(self):
        return max(self.target_lengths)


class SplitLogits:
    """Minimal lazy tensor interface used by OmniVoice0.1.4 iterative inference.

    The caller reads per-item conditional target slices and unconditional target
    prefixes only. Unused full-sequence padding logits are never materialized.
    Unsupported reads fail loudly rather than return semantically false padding.
    """

    def __init__(self, conditional: torch.Tensor, unconditional: torch.Tensor):
        if conditional.ndim != 4 or unconditional.ndim != 4:
            raise ValueError("Expected [B,C,S,V] logits")
        if conditional.shape[:2] != unconditional.shape[:2]:
            raise ValueError("Conditional and unconditional batches must match")
        if conditional.shape[-1] != unconditional.shape[-1]:
            raise ValueError("Conditional and unconditional vocabularies must match")
        self.conditional = conditional
        self.unconditional = unconditional
        b, c, s, v = conditional.shape
        self.shape = (2 * b, c, s, v)
        self.dtype = conditional.dtype
        self.device = conditional.device
        self.ndim = 4

    def to(self, *args, **kwargs):
        return SplitLogits(
            self.conditional.to(*args, **kwargs), self.unconditional.to(*args, **kwargs)
        )

    def float(self):
        return self.to(torch.float32)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 4:
            raise NotImplementedError("Only four-axis target slices are supported")
        rows, codebooks, sequence, vocab = key
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise NotImplementedError("Expected a contiguous batch slice")
        batch = self.conditional.shape[0]
        start, stop, step = rows.indices(2 * batch)
        if stop <= batch:
            return self.conditional[
                (slice(start, stop, step), codebooks, sequence, vocab)
            ]
        if start >= batch:
            if not isinstance(sequence, slice):
                raise NotImplementedError("Expected a target-length sequence slice")
            seq_start, seq_stop, seq_step = sequence.indices(self.shape[2])
            if seq_stop > self.unconditional.shape[2]:
                raise NotImplementedError("Unconditional padding logits were omitted")
            return self.unconditional[
                (
                    slice(start - batch, stop - batch, step),
                    codebooks,
                    slice(seq_start, seq_stop, seq_step),
                    vocab,
                )
            ]
        raise NotImplementedError("Do not mix guidance branches in one lazy slice")


class SplitGuidanceForward:
    """Call existing inference forward separately for conditional and real U length."""

    def __init__(self, original_forward, *, model):
        self.original_forward = original_forward
        self.model = model
        self.local = threading.local()
        self.split_calls = 0
        self.fallback_calls = 0
        self.original_row_tokens = 0
        self.executed_row_tokens = 0

    @contextmanager
    def layout(self, task):
        previous = getattr(self.local, "layout", None)
        current = GuidanceLayout(
            int(task.batch_size), tuple(map(int, task.target_lens))
        )
        if len(current.target_lengths) != current.batch_size:
            raise ValueError("Target lengths do not match the current task")
        self.local.layout = current
        try:
            yield
        finally:
            self.local.layout = previous

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
        layout = getattr(self.local, "layout", None)
        seq_len = input_ids.shape[-1]
        eligible = (
            layout is not None
            and not self.model.training
            and labels is None
            and document_ids is None
            and position_ids is None
            and not kwargs
            and attention_mask is not None
            and attention_mask.ndim == 4
            and input_ids.shape[0] == 2 * layout.batch_size
            and 0 < layout.max_target_length < seq_len
        )
        if not eligible:
            self.fallback_calls += 1
            return self.original_forward(
                input_ids,
                audio_mask,
                labels=labels,
                attention_mask=attention_mask,
                document_ids=document_ids,
                position_ids=position_ids,
                **kwargs,
            )
        batch = layout.batch_size
        target = layout.max_target_length
        conditional = self.original_forward(
            input_ids[:batch], audio_mask[:batch], attention_mask=attention_mask[:batch]
        )
        unconditional = self.original_forward(
            input_ids[batch:, :, :target],
            audio_mask[batch:, :target],
            attention_mask=attention_mask[batch:, :, :target, :target],
        )
        self.split_calls += 1
        self.original_row_tokens += 2 * batch * seq_len
        self.executed_row_tokens += batch * (seq_len + target)
        return SimpleNamespace(
            logits=SplitLogits(conditional.logits, unconditional.logits)
        )


class SplitGuidancePatch:
    """Opt-in, reversible patch using thread-local task layout metadata."""

    def __init__(self, model: Any):
        self.model = model
        self.original_forward = model.forward
        self.original_iterative = model._generate_iterative
        self.forward = SplitGuidanceForward(self.original_forward, model=model)
        original = self.original_iterative
        split = self.forward

        def iterative(task, generation_config):
            with split.layout(task):
                return original(task, generation_config)

        self.iterative = iterative
        self.active = False

    def enable(self):
        if not self.active:
            self.model.forward = self.forward
            self.model._generate_iterative = self.iterative
            self.active = True

    def disable(self):
        if self.active:
            self.model.forward = self.original_forward
            self.model._generate_iterative = self.original_iterative
            self.active = False

    def metrics(self):
        f = self.forward
        return {
            "split_calls": f.split_calls,
            "fallback_calls": f.fallback_calls,
            "original_transformer_row_tokens": f.original_row_tokens,
            "executed_transformer_row_tokens": f.executed_row_tokens,
            "row_token_reduction_fraction": (
                1 - f.executed_row_tokens / f.original_row_tokens
                if f.original_row_tokens
                else None
            ),
        }
