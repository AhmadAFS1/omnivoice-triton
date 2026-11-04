#!/usr/bin/env python3
"""Matched-seed, full-logit/token/WAV gate after the isolated RoPE tensor gate.

This is a prototype-only component experiment. It changes no serving defaults.
All logits are copied to CPU and hashed during parity calls; those calls are
excluded from latency measurements. Timed calls retain every complete WAV.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class ProductPatch:
    """Exercise actual product installation on an already loaded paired runner.

    This avoids reloading weights between paired tensors. Constructor/API startup
    is a separate required gate; no serving app's immutable options are changed.
    """

    def __init__(self, runner):
        self.runner = runner
        self.patch = None

    def __enter__(self):
        self.runner.fused_rope = True
        with self.runner._fused_rope_startup():
            self.runner._install_fused_rope()
        self.patch = self.runner._fused_rope_patch
        assert self.patch.active and self.patch.ready
        return self.patch

    def __exit__(self, *_):
        # ExitStack clears/fences captured graphs before this exit is invoked.
        self.patch.prepare_disable()
        self.patch.detach_outer()
        self.patch.disable()
        self.runner._fused_rope_patch = None
        self.runner.fused_rope = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--tensor-gate", type=Path)
    parser.add_argument("--gpu-release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timed-repetitions", choices=[1, 2], type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--eager-b1-only", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.parity_only:
        args.timed_repetitions = 0
    corpus = json.loads(args.corpus.read_text())
    corpus = corpus["cases"] if isinstance(corpus, dict) else corpus
    cases = [
        next(x for x in corpus if x["id"] == key)
        for key in ["en-short", "es-short", "ar-short"]
    ]
    batches = [1] if args.eager_b1_only else [1, 8]
    if args.eager_b1_only:
        cases = cases[:1]
    references = json.loads(args.references.read_text())
    references = (
        references["references"] if isinstance(references, dict) else references
    )
    by_id = {x.get("id", x.get("reference_id")): x for x in references}
    for case in cases:
        reference = by_id[case["reference_id"]]
        path = Path(reference.get("path", reference.get("file")))
        if not path.is_absolute():
            path = args.references.parent / path
        assert path.is_file() and sha(path) == reference["sha256"]
    plan = {
        "measurement_kind": "CPU-only plan until explicit execute; full-model component, not HTTP capacity",
        "cases": cases,
        "batch_sizes": batches,
        "runner_mode": "triton" if args.eager_b1_only else "hybrid",
        "implementation": "2D runtime-S/stride RoPE; first24 layers only",
        "arms": ["reference", "fused-rope-first24"],
        "steps": 16,
        "KV_cache": False,
        "split": False,
        "codec_batch": 1,
        "graphs": 0 if args.eager_b1_only else 8,
        "graph_memory_row_tokens": 0 if args.eager_b1_only else 16384,
        "speed": 1,
        "postprocess": "off",
        "patch_range": [0, 24],
        "timed_repetitions": args.timed_repetitions,
        "parity_gate": "Every full-logit FP16 bit via SHA256 on every denoising step; exact input tensors, final tokens, float32 WAV bits and PCM16 WAV SHA",
        "timing_note": (
            "PARITY ONLY. Concurrent approved CPU/SSH WAV transfer; "
            "all warm/timing claims excluded."
            if args.parity_only
            else "Separate warmed model calls; not p95 or HTTP capacity"
        ),
        "tensor_prerequisite": "Current identical prototype SHA must have completed108 isolated tensor pairs and updated-input graph replay",
        "bound_s": 300,
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    release = json.loads(args.gpu_release.read_text())
    assert release["gpu_released"] and release["fused_rope_product_model_gate"]
    assert (
        0
        <= time.time() - datetime.datetime.fromisoformat(release["at"]).timestamp()
        < 600
    )
    tensor_outcome = json.loads((args.tensor_gate / "outcome.json").read_text())
    tensor_metadata = json.loads((args.tensor_gate / "metadata.json").read_text())
    assert (
        tensor_outcome["status"] == "completed"
        and tensor_outcome["pairs_checked"] == 108
    )
    assert tensor_outcome["observed_shape_pairs_checked"] == 203
    assert (
        sha(args.model / "config.json")
        == "5e359117e13b420c5e0c925d4aba650d624767131f1d1746928f8b850d5dc372"
    )
    deadline = time.monotonic() + 280

    import numpy as np
    import omnivoice_triton
    import soundfile as sf
    import torch
    from omnivoice import OmniVoiceGenerationConfig
    from omnivoice_triton import create_runner
    from omnivoice_triton.kernels import rotary as product_kernel
    from run_split_cfg_probe import hardware, seed, smi

    assert not smi("compute-apps=pid,process_name,used_memory")
    assert "GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33" in smi(
        "gpu=name,uuid,memory.total"
    )
    assert (
        torch.__version__ == "2.8.0+cu128"
        and torch.backends.cuda.matmul.allow_tf32 is False
    )
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = Path(omnivoice_triton.__file__).resolve().parents[2]
    write(
        args.output / "metadata.json",
        {
            "hardware": hardware(),
            "plan": plan,
            "release": release,
            "source_ref": args.source_ref,
            "source_files": {
                str(p.relative_to(source_root)): sha(p)
                for p in sorted((source_root / "src").rglob("*.py"))
            },
            "script_sha256": sha(Path(__file__)),
            "prototype_sha256": sha(Path(product_kernel.__file__)),
            "tensor_gate": str(args.tensor_gate),
            "tensor_metadata_sha256": sha(args.tensor_gate / "metadata.json"),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "FP16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "TF32": torch.backends.cuda.matmul.allow_tf32,
        },
    )
    graph_options = (
        {}
        if args.eager_b1_only
        else {"max_cuda_graphs": 8, "max_cuda_graph_row_tokens": 16384}
    )
    assert tensor_metadata["prototype_sha256"] == sha(Path(product_kernel.__file__))
    runner = create_runner(
        "triton" if args.eager_b1_only else "hybrid",
        device="cuda",
        model_id=str(args.model),
        dtype="fp16",
        disable_unused_kv_cache=True,
        audio_decode_batch_size=1,
        **graph_options,
    )
    write(
        args.output / "actual-runner.json",
        {
            "class": type(runner).__name__,
            "mode": "triton" if args.eager_b1_only else "hybrid",
            "graph_options": graph_options,
        },
    )
    runner.load_model()
    model = runner.model
    model.eval()
    config = OmniVoiceGenerationConfig(
        num_step=16,
        class_temperature=0.0,
        position_temperature=5.0,
        denoise=True,
        postprocess_output=False,
    )
    config.postprocess_mode = "off"
    original_iterative = model._generate_iterative
    tokens = []

    def iterative(*pos, **kw):
        result = original_iterative(*pos, **kw)
        tokens[:] = result
        return result

    model._generate_iterative = iterative
    records, saved, status = [], [], "failed"

    def tensor_identity(value):
        raw = value.detach().cpu().contiguous().numpy()
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }

    def wave_identity(value):
        raw = np.ascontiguousarray(value)
        return {
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }

    def clear_graphs():
        torch.cuda.synchronize()
        memory = getattr(runner, "_graph_memory", None)
        if memory is not None:
            memory.clear()
        torch.cuda.empty_cache()

    def generate(case, prompt, batch, call_seed):
        assert time.monotonic() < deadline, "Finite model-gate budget exhausted"
        seed(call_seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        waves = model.generate(
            text=[case["text"]] * batch,
            language=[case["api_language"]] * batch,
            voice_clone_prompt=[prompt] * batch,
            speed=[1.0] * batch,
            generation_config=config,
        )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        return waves, [x.detach().cpu().numpy().copy() for x in tokens], elapsed

    def save(case, batch, arm, category, repetition, waves, output_tokens):
        stem = f"{case['id']}-B{batch}-{arm}-{category}-r{repetition}"
        wave_hashes = []
        np.savez_compressed(
            args.output / (stem + "-tokens.npz"),
            **{f"item{i}": x for i, x in enumerate(output_tokens)},
        )
        np.savez_compressed(
            args.output / (stem + "-float-audio.npz"),
            **{f"item{i}": x for i, x in enumerate(waves)},
        )
        for index, wave in enumerate(waves):
            path = args.output / (stem + f"-{index:03d}.wav")
            sf.write(
                path,
                np.asarray(wave, dtype=np.float32),
                int(model.sampling_rate),
                subtype="PCM_16",
            )
            wave_hashes.append(sha(path))
            saved.append(
                {
                    "id": stem + f"-{index:03d}",
                    "path": path.name,
                    "sha256": wave_hashes[-1],
                    "language": case["language"],
                    "omnilingual_language": case["omnilingual_language"],
                    "expected": case["text"],
                    "reference_id": case["reference_id"],
                    "arm": arm,
                    "phase": category,
                }
            )
        write(args.output / "quality-manifest.json", saved)
        return wave_hashes

    try:
        with torch.inference_mode():
            for case_index, case in enumerate(cases):
                reference = by_id[case["reference_id"]]
                path = Path(reference.get("path", reference.get("file")))
                if not path.is_absolute():
                    path = args.references.parent / path
                prompt = model.create_voice_clone_prompt(
                    ref_audio=str(path),
                    ref_text=reference["text"],
                    preprocess_prompt=True,
                )
                for batch in batches:
                    pair, per_arm = {}, {}
                    call_seed = 20261104 + case_index * 100 + batch
                    for arm in ["reference", "fused-rope-first24"]:
                        clear_graphs()
                        patch = ProductPatch(runner) if arm != "reference" else None
                        with contextlib.ExitStack() as stack:
                            if patch is not None:
                                stack.enter_context(patch)
                            # Release captured math on every exit before the
                            # temporary global function is restored.
                            stack.callback(clear_graphs)
                            underlying_forward, logits = model.forward, []

                            def spy(
                                *pos, _forward=underlying_forward, _rows=logits, **kw
                            ):
                                output = _forward(*pos, **kw)
                                names = [
                                    "input_ids",
                                    "audio_mask",
                                    "labels",
                                    "attention_mask",
                                    "document_ids",
                                    "position_ids",
                                ]
                                supplied = dict(zip(names, pos)) | kw
                                _rows.append(
                                    {
                                        "step": len(_rows),
                                        "logits": tensor_identity(output.logits),
                                        "inputs": {
                                            name: tensor_identity(value)
                                            for name, value in supplied.items()
                                            if isinstance(value, torch.Tensor)
                                        },
                                    }
                                )
                                return output

                            model.forward = spy
                            try:
                                waves, output_tokens, _ = generate(
                                    case, prompt, batch, call_seed
                                )
                            finally:
                                model.forward = underlying_forward
                            assert len(logits) == 16
                            pcm_hashes = save(
                                case, batch, arm, "parity", 0, waves, output_tokens
                            )
                            pair[arm] = {
                                "logits": logits,
                                "tokens": [wave_identity(x) for x in output_tokens],
                                "float_waves": [wave_identity(x) for x in waves],
                                "PCM16_WAV_SHA256": pcm_hashes,
                            }
                            write(
                                args.output
                                / f"{case['id']}-B{batch}-{arm}-full-logit-hashes.json",
                                logits,
                            )
                            if arm != "reference":
                                exact = pair[arm] == pair["reference"]
                                write(
                                    args.output / f"{case['id']}-B{batch}-parity.json",
                                    {
                                        "every_full_logit_bit_and_input_token_wave_equal": exact,
                                        "patched_python_calls": patch.fused_calls,
                                        "product_metrics": patch.metrics(),
                                    },
                                )
                                assert patch.fused_calls > 0 and exact, (
                                    "Full logits/tokens/float-WAV gate failed; retain without promotion"
                                )
                            timings = []
                            for repetition in range(args.timed_repetitions):
                                measured_waves, measured_tokens, elapsed = generate(
                                    case, prompt, batch, call_seed
                                )
                                assert [
                                    wave_identity(x) for x in measured_tokens
                                ] == pair[arm]["tokens"]
                                assert [
                                    wave_identity(x) for x in measured_waves
                                ] == pair[arm]["float_waves"]
                                measured_pcm = save(
                                    case,
                                    batch,
                                    arm,
                                    "timed",
                                    repetition,
                                    measured_waves,
                                    measured_tokens,
                                )
                                assert measured_pcm == pair[arm]["PCM16_WAV_SHA256"]
                                timings.append(elapsed)
                            per_arm[arm] = {
                                "full_model_ms": timings,
                                "optimizations": runner.get_inference_optimization_metrics(),
                            }
                    records.append(
                        {
                            "case_id": case["id"],
                            "batch": batch,
                            "seed": call_seed,
                            "arms": per_arm,
                            "exact": True,
                        }
                    )
                    write(args.output / "measurements.json", records)
        status = "completed"
    finally:
        model._generate_iterative = original_iterative
        runner.unload_model()
        write(
            args.output / "outcome.json",
            {
                "status": status,
                "at": datetime.datetime.now(datetime.UTC).isoformat(),
                "pairs_checked": len(records),
                "saved_WAVs": len(saved),
                "model_unloaded": True,
                "GPU_idle_requires_parent_confirmation": True,
            },
        )


if __name__ == "__main__":
    main()
