"""CPU-only mask/slice/restore contracts, not real OmniVoice/GPU qualification."""

from types import SimpleNamespace

import torch

from omnivoice_triton.models.split_cfg import SplitGuidancePatch, SplitLogits


class FakeNoncausalModel(torch.nn.Module):
    def __init__(self, *, codebooks=3, vocabulary=7):
        super().__init__()
        self.codebooks = codebooks
        self.vocabulary = vocabulary
        self.calls = []
        self.eval()

    def forward(
        self,
        input_ids,
        audio_mask,
        labels=None,
        attention_mask=None,
        document_ids=None,
        position_ids=None,
    ):
        self.calls.append(tuple(input_ids.shape))
        x = input_ids[:, 0, :].to(torch.float64) / 20
        hidden = torch.stack([x, x.sin(), x.cos(), audio_mask.double()], dim=-1)
        scores = hidden @ hidden.transpose(-1, -2)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, 0], -torch.inf)
        hidden = scores.softmax(-1) @ hidden
        weights = (
            torch.arange(
                4 * self.codebooks * self.vocabulary, dtype=torch.float64
            ).reshape(4, -1)
            / 100
        )
        y = hidden @ weights
        logits = y.reshape(
            input_ids.shape[0], input_ids.shape[-1], self.codebooks, self.vocabulary
        ).permute(0, 2, 1, 3)
        return SimpleNamespace(logits=logits)

    def _generate_iterative(self, task, config):
        return self(task.ids, task.audio_mask, attention_mask=task.attention_mask)


def task(conditional_lengths, target_lengths):
    batch = len(target_lengths)
    length = max(conditional_lengths)
    ids = torch.arange(2 * batch * 3 * length).reshape(2 * batch, 3, length) % 31
    mask = torch.ones((2 * batch, length), dtype=torch.bool)
    attention = torch.zeros((2 * batch, 1, length, length), dtype=torch.bool)
    for i, (cond, target) in enumerate(zip(conditional_lengths, target_lengths)):
        attention[i, :, :cond, :cond] = True
        # Padded conditional queries do not matter; keep their softmax finite.
        for p in range(cond, length):
            attention[i, :, p, p] = True
        attention[batch + i, :, :target, :target] = True
        for p in range(target, length):
            attention[batch + i, :, p, p] = True
    return SimpleNamespace(
        batch_size=batch,
        target_lens=target_lengths,
        ids=ids,
        audio_mask=mask,
        attention_mask=attention,
    ), conditional_lengths


def test_split_cfg_preserves_real_attention_slices_and_restores():
    torch.manual_seed(17)
    checked = []
    for conditional, targets in [
        ([10], [4]),
        ([10, 8], [4, 3]),
        ([15, 9, 13], [2, 4, 3]),
    ]:
        model = FakeNoncausalModel()
        data, _ = task(conditional, targets)
        baseline = model._generate_iterative(data, None).logits.float()
        patch = SplitGuidancePatch(model)
        patch.enable()
        optimized = model._generate_iterative(data, None).logits.to(torch.float32)
        assert isinstance(optimized, SplitLogits)
        batch = len(targets)
        diffs = []
        for i, (cond, target) in enumerate(zip(conditional, targets)):
            for row, seq in [
                (i, slice(cond - target, cond)),
                (batch + i, slice(0, target)),
            ]:
                expected = baseline[row : row + 1, :, seq, :]
                actual = optimized[row : row + 1, :, seq, :]
                assert expected.shape == actual.shape
                diff = (expected - actual).abs().max().item()
                diffs.append(diff)
                assert torch.equal(expected, actual), diff
        assert model.calls[-2:] == [
            (batch, 3, max(conditional)),
            (batch, 3, max(targets)),
        ]
        patch.disable()
        assert not isinstance(model._generate_iterative(data, None).logits, SplitLogits)
        assert torch.equal(
            model._generate_iterative(data, None).logits.float(), baseline
        )
        checked.append(
            {
                "conditional_lengths": conditional,
                "targets": targets,
                "max_used_logit_difference": max(diffs),
                **patch.metrics(),
            }
        )
    print(
        {
            "measurement_kind": (
                "CPU-only fake noncausal attention contract; no GPU/model inference"
            ),
            "cases": checked,
            "status": "passed",
        }
    )
