#!/usr/bin/env python3
"""Finite isolated GPU tensor/stride/raw-bit RoPE gate; default is CPU plan."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-release", type=Path)
    parser.add_argument("--observed-shapes", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    observed_shapes = json.loads(args.observed_shapes.read_text())
    observed_sequences = observed_shapes["sequence_lengths"]
    assert observed_sequences and len(set(observed_sequences)) == len(
        observed_sequences
    )
    assert all(
        isinstance(value, int) and 1 <= value <= 4096 for value in observed_sequences
    )
    plan = {
        "measurement_kind": "CPU-only plan until explicit execute; tensor component, not TTS capacity",
        "input_batches": [1, 2, 16],
        "actual_TTS_CFG_batches": {"B1": 2, "B8": 16},
        "sequence_lengths": [1, 457],
        "QK_layouts": ["BSHD-transpose", "BHSD-contiguous", "BSHD-gapped-heads"],
        "cosine_layouts": ["shared-batch1", "per-batch", "per-batch-noncontiguous"],
        "values": ["normal", "large-small-signed-zero-subnormal"],
        "tensor_pairs": 108,
        "additional_observed_sequence_pairs": len(observed_sequences),
        "observed_shapes_provenance": observed_shapes,
        "component_timings": "Excluded: raw-bit gate may coexist with approved CPU-only WAV transfer",
        "checks": [
            "raw FP16 bits",
            "exact output stride",
            "no input/output alias",
            "CUDA Graph updated-input replay",
        ],
        "release_scope": "isolated tensor gate only; no model or default change",
        "bound_s": 300,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    release = json.loads(args.gpu_release.read_text())
    assert release["gpu_released"] and release["fused_rope_product_tensor_gate"]
    assert (
        0
        <= time.time() - datetime.datetime.fromisoformat(release["at"]).timestamp()
        < 600
    )

    import torch
    import transformers.models.qwen3.modeling_qwen3 as qwen
    from omnivoice_triton.kernels import rotary as product_kernel
    from omnivoice_triton.kernels.rotary import triton_rotary_pos_emb as fused_rope
    from omnivoice_triton.models.rotary import FusedRoPEPatch
    from run_split_cfg_probe import hardware, seed, smi

    assert not smi("compute-apps=pid,process_name,used_memory")
    assert "GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33" in smi(
        "gpu=name,uuid,memory.total"
    )
    assert torch.__version__ == "2.8.0+cu128"
    assert importlib.metadata.version("triton") == "3.4.0"
    assert importlib.metadata.version("transformers") == "5.5.4"
    assert (
        hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest()
        == "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    seed(20261104)
    metadata = {
        "plan": plan,
        "hardware": hardware(),
        "release": release,
        "prototype_sha256": hashlib.sha256(
            Path(product_kernel.__file__).read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "Triton_enable_fp_fusion": False,
        "TF32_original": original_tf32,
        "TF32_effective": torch.backends.cuda.matmul.allow_tf32,
        "FP16_reduced_precision_reduction_unchanged": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }
    write(args.output / "metadata.json", metadata)
    startup = FusedRoPEPatch(None)
    startup.device = torch.device("cuda:0")
    startup._warmup()
    assert startup.ready and startup.owner_thread is None and not startup.active
    write(
        args.output / "actual-product-startup-warmup.json",
        {
            "scope": "Actual product warmup helper without model weights; full model qualification/lifecycle remains a separate gate",
            "ready_for_handoff": startup.ready,
            "generation_owner_unbound": startup.owner_thread is None,
            "warmup": startup.warmup_summary,
        },
    )
    deadline = time.monotonic() + 280
    records, status, timings = [], "failed", []
    sweep_rows = []

    def tensor_fixture(batch, sequence, layout, cosine_layout, magnitude):
        def value(heads):
            if layout == "BHSD-contiguous":
                return torch.randn(
                    batch, heads, sequence, 128, device="cuda", dtype=torch.float16
                )
            actual_heads = heads * 2 if layout == "BSHD-gapped-heads" else heads
            result = torch.randn(
                batch, sequence, actual_heads, 128, device="cuda", dtype=torch.float16
            )
            if actual_heads != heads:
                result = result[:, :, ::2]
            return result.transpose(1, 2)

        q, k = value(16), value(8)
        cosine_batch = 1 if cosine_layout == "shared-batch1" else batch
        angles = torch.randn(
            cosine_batch, sequence, 128, device="cuda", dtype=torch.float32
        )
        cos, sin = angles.cos().half(), angles.sin().half()
        if cosine_layout == "per-batch-noncontiguous":
            cos = cos.transpose(0, 1).contiguous().transpose(0, 1)
            sin = sin.transpose(0, 1).contiguous().transpose(0, 1)
        if magnitude == "large-small-signed-zero-subnormal":
            choices = torch.tensor(
                [
                    -65504,
                    -32752,
                    -256,
                    -2,
                    -1,
                    -(2**-24),
                    -0.0,
                    0.0,
                    2**-24,
                    1,
                    2,
                    256,
                    32752,
                    65504,
                ],
                device="cuda",
                dtype=torch.float16,
            )
            for destination in [q, k]:
                logical = choices[
                    torch.arange(destination.numel(), device="cuda") % choices.numel()
                ]
                destination.copy_(logical.reshape(destination.shape))
            # Magnitudes0.5 keep every result finite even at maximum half input.
            cos.copy_(
                torch.where(
                    torch.arange(cos.numel(), device="cuda").reshape(cos.shape) % 2
                    == 0,
                    0.5,
                    -0.5,
                )
            )
            sin.copy_(
                torch.where(
                    torch.arange(sin.numel(), device="cuda").reshape(sin.shape) % 3
                    == 0,
                    -0.5,
                    0.5,
                )
            )
        return q, k, cos, sin

    def compare(expected, observed, label):
        comparisons = []
        for index, (reference, actual) in enumerate(
            zip(expected, observed, strict=True)
        ):
            differences = int(
                torch.count_nonzero(
                    reference.view(torch.int16) != actual.view(torch.int16)
                ).item()
            )
            record = {
                "tensor": index,
                "shape": list(reference.shape),
                "reference_stride": list(reference.stride()),
                "observed_stride": list(actual.stride()),
                "raw_uint16_mismatches": differences,
                "finite": bool(torch.isfinite(reference).all().item())
                and bool(torch.isfinite(actual).all().item()),
            }
            comparisons.append(record)
            if (
                differences
                or reference.stride() != actual.stride()
                or not record["finite"]
            ):
                write(
                    args.output / "failure.json",
                    {"label": label, "comparisons": comparisons},
                )
                torch.save(
                    {
                        "expected": [x.cpu() for x in expected],
                        "observed": [x.cpu() for x in observed],
                    },
                    args.output / "failed-tensors.pt",
                )
                raise AssertionError("Raw bits, layout or finite-value gate failed")
        return comparisons

    try:
        with torch.inference_mode():
            for batch in plan["input_batches"]:
                for sequence in plan["sequence_lengths"]:
                    for layout in plan["QK_layouts"]:
                        for cosine_layout in plan["cosine_layouts"]:
                            for magnitude in plan["values"]:
                                assert time.monotonic() < deadline
                                values = tensor_fixture(
                                    batch, sequence, layout, cosine_layout, magnitude
                                )
                                expected = qwen.apply_rotary_pos_emb(*values)
                                observed = fused_rope(*values)
                                label = {
                                    "batch": batch,
                                    "sequence": sequence,
                                    "layout": layout,
                                    "cosine_layout": cosine_layout,
                                    "magnitude": magnitude,
                                }
                                comparisons = compare(expected, observed, label)
                                assert all(
                                    actual.data_ptr() != value.data_ptr()
                                    for actual, value in zip(
                                        observed, values[:2], strict=True
                                    )
                                )
                                records.append(
                                    {
                                        **label,
                                        "inputs": [
                                            {
                                                "shape": list(x.shape),
                                                "stride": list(x.stride()),
                                            }
                                            for x in values
                                        ],
                                        "comparisons": comparisons,
                                    }
                                )
                                write(args.output / "tensor-comparisons.json", records)
            assert len(records) == plan["tensor_pairs"]
            for sequence in observed_sequences:
                assert time.monotonic() < deadline
                values = tensor_fixture(
                    2, sequence, "BSHD-transpose", "shared-batch1", "normal"
                )
                expected = qwen.apply_rotary_pos_emb(*values)
                observed = fused_rope(*values)
                sweep_rows.append(
                    {
                        "batch": 2,
                        "sequence": sequence,
                        "comparisons": compare(
                            expected, observed, "observed-shape-sweep"
                        ),
                    }
                )
                write(args.output / "observed-shape-sweep.json", sweep_rows)
            assert len(sweep_rows) == plan["additional_observed_sequence_pairs"]
            values = tensor_fixture(
                16, 457, "BSHD-transpose", "shared-batch1", "normal"
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused_rope(*values)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = fused_rope(*values)
            replay_rows = []
            prior_result, prior_snapshot = None, None
            for repetition in range(4):
                assert time.monotonic() < deadline
                for value in values[:2]:
                    value.copy_(torch.randn_like(value))
                graph.replay()
                observed = tuple(x.clone() for x in static_output)
                expected = qwen.apply_rotary_pos_emb(*values)
                replay_rows.append(
                    {
                        "rep": repetition,
                        "comparisons": compare(expected, observed, "graph-replay"),
                    }
                )
                if prior_result is not None:
                    assert all(
                        torch.equal(a.view(torch.int16), b.view(torch.int16))
                        for a, b in zip(prior_result, prior_snapshot, strict=True)
                    )
                prior_result, prior_snapshot = (
                    observed,
                    tuple(x.clone() for x in observed),
                )
            write(args.output / "graph-replay.json", replay_rows)
        status = "completed"
    finally:
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        write(
            args.output / "outcome.json",
            {
                "status": status,
                "at": datetime.datetime.now(datetime.UTC).isoformat(),
                "pairs_checked": len(records),
                "observed_shape_pairs_checked": len(sweep_rows),
                "component_measurements": len(timings),
                "GPU_idle_requires_parent_confirmation": True,
            },
        )


if __name__ == "__main__":
    main()
