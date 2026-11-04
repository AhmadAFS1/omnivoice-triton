#!/usr/bin/env python3
"""Exclusive-GPU real-model split-CFG parity and full-model latency experiment.

This is NOT HTTP service capacity. Timings include generation and codec output,
but exclude WAV serialization, network, gateway and MuseTalk. All paired outputs
are saved as complete WAVs for independent blind ASR after timing is finished.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from split_cfg import SplitGuidancePatch

DEFAULT_INPUTS = Path(__file__).parent / "fixtures"
LANG_IDS = {"en": "eng_Latn", "es": "spa_Latn", "ar": "arb_Arab"}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def smi(fields):
    return subprocess.check_output(
        ["nvidia-smi", "--query-" + fields, "--format=csv,noheader"], text=True
    ).strip()


def hardware():
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_gpu_nvidia_smi": smi(
            "gpu=name,uuid,memory.total,driver_version,power.limit"
        ),
        "co_resident_processes_at_start": smi(
            "compute-apps=pid,process_name,used_memory"
        ),
        "torch_visible_gpu": [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                "multiprocessor_count": torch.cuda.get_device_properties(
                    i
                ).multi_processor_count,
            }
            for i in range(torch.cuda.device_count())
        ],
        "cuda_runtime": torch.version.cuda,
        "cpu_logical": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchaudio",
                "triton",
                "transformers",
                "omnivoice",
                "numpy",
                "soundfile",
            ]
        },
        "measurement_kind": "Real GPU TTS model inference; synchronized full-model latency, not HTTP capacity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_INPUTS / "corpus.json")
    parser.add_argument(
        "--references", type=Path, default=DEFAULT_INPUTS / "references"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids", default="en-short,es-short,ar-short")
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--graphs", type=int, default=8)
    parser.add_argument("--fallback-reference", default="en")
    args = parser.parse_args()
    assert importlib.metadata.version("omnivoice") == "0.1.4", (
        "Prototype is qualified only against the pinned0.1.4 iterative interface"
    )
    assert torch.cuda.is_available(), (
        "CUDA model inference is required; CPU contract tests are separate"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    config_data = dict(vars(args))
    config_data = {
        k: str(v) if isinstance(v, Path) else v for k, v in config_data.items()
    }
    write(
        args.output / "metadata.json",
        {
            "hardware": hardware(),
            "arguments": config_data,
            "source_ref": args.source_ref,
            "model_revision": args.model_revision,
            "profile": "hybrid FP16,16steps unless explicitly overridden,speed1,denoise on,postprocess off,serial codec",
            "seed_note": "Seed reset before each paired invocation; original per-row Gumbel/random sampling order is unchanged",
            "strict_gate": "Final token arrays and saved PCM16 WAV bytes must match in every paired fixture; discrepancies remain review flags",
            "co_load_requirement": "Run on an idle exclusive GPU; never overlap with ASR or other timed inference",
            "prototype_sha256": sha(Path(__file__)),
            "patch_sha256": sha(Path(__file__).with_name("split_cfg.py")),
        },
    )
    from omnivoice import OmniVoiceGenerationConfig
    from omnivoice_triton import create_runner

    runner = create_runner(
        "hybrid",
        model_id=str(args.model),
        device="cuda",
        dtype="fp16",
        max_cuda_graphs=args.graphs,
        audio_decode_batch_size=1,
    )
    runner.load_model()
    model = runner.model
    model.eval()
    original_iterative = model._generate_iterative
    original_predictor = model._predict_tokens_with_scoring
    token_capture = None
    prediction_capture = None

    def capture_iterative(task, config):
        outputs = original_iterative(task, config)
        if token_capture is not None:
            token_capture.extend(x.detach().cpu().clone() for x in outputs)
        return outputs

    def capture_predictor(*pos, **kw):
        predictions, confidence = original_predictor(*pos, **kw)
        if prediction_capture is not None:
            prediction_capture.append(
                (predictions.detach().cpu().clone(), confidence.detach().cpu().clone())
            )
        return predictions, confidence

    model._generate_iterative = capture_iterative
    model._predict_tokens_with_scoring = capture_predictor
    patch = SplitGuidancePatch(model)
    references = {
        r.get("lang", r["language"]): r
        for r in json.loads((args.references / "manifest.json").read_text())
    }
    prompts = {}
    corpus = {row["id"]: row for row in json.loads(args.corpus.read_text())}
    summaries = []
    manifests = {"baseline": [], "split-cfg": []}
    config = OmniVoiceGenerationConfig(
        num_step=args.steps,
        class_temperature=0.0,
        position_temperature=5.0,
        postprocess_output=False,
    )
    config.postprocess_mode = "off"
    for arm in manifests:
        (args.output / arm / "audio").mkdir(parents=True)
    try:
        with torch.inference_mode():
            for case_index, case_id in enumerate(args.case_ids.split(",")):
                case = corpus[case_id]
                lang = case["language"]
                ref_lang = lang if lang in references else args.fallback_reference
                ref = references[ref_lang]
                if ref_lang not in prompts:
                    path = args.references / ref["file"]
                    assert sha(path) == ref["sha256"]
                    prompts[ref_lang] = model.create_voice_clone_prompt(
                        ref_audio=str(path),
                        ref_text=ref["text"],
                        preprocess_prompt=True,
                    )
                for batch in map(int, args.batch_sizes.split(",")):
                    identity = f"{case_id}-b{batch}"
                    directory = args.output / identity
                    directory.mkdir()
                    kwargs = {
                        "text": [case["text"]] * batch,
                        "language": [case.get("api_language", lang)] * batch,
                        "voice_clone_prompt": [prompts[ref_lang]] * batch,
                        "generation_config": config,
                        "speed": [1.0] * batch,
                    }
                    # Discard prior shapes so retained graph memory cannot silently
                    # crowd later batch sizes; both arms warm their actual shapes.
                    patch.disable()
                    graph = getattr(runner, "_graph_forward", None)
                    if graph is not None:
                        graph.clear()
                    torch.cuda.empty_cache()
                    for enabled in [False, True]:
                        patch.enable() if enabled else patch.disable()
                        seed(20261103 + case_index)
                        model.generate(**kwargs)
                        torch.cuda.synchronize()
                    records = {}
                    for arm, enabled in [("baseline", False), ("split-cfg", True)]:
                        patch.enable() if enabled else patch.disable()
                        token_capture = []
                        prediction_capture = []
                        seed(20261103 + case_index)
                        audio = model.generate(**kwargs)
                        torch.cuda.synchronize()
                        if len(audio) != batch:
                            raise RuntimeError(
                                "Model did not return every requested waveform"
                            )
                        current_tokens = token_capture
                        current_predictions = prediction_capture
                        token_capture = None
                        prediction_capture = None
                        records[arm] = {
                            "tokens": current_tokens,
                            "predictions": current_predictions,
                            "audio": audio,
                        }
                        np.savez_compressed(
                            directory / (arm + "-tokens.npz"),
                            **{
                                f"item{i}": x.numpy()
                                for i, x in enumerate(current_tokens)
                            },
                        )
                        np.savez_compressed(
                            directory / (arm + "-trace.npz"),
                            **{
                                f"{kind}{i}": pair[j].numpy()
                                for i, pair in enumerate(current_predictions)
                                for kind, j in [("prediction", 0), ("confidence", 1)]
                            },
                        )
                        for i, wave in enumerate(audio):
                            path = (
                                args.output / arm / "audio" / f"{identity}-{i:03d}.wav"
                            )
                            sf.write(
                                path,
                                np.asarray(wave, dtype=np.float32),
                                int(model.sampling_rate),
                                subtype="PCM_16",
                            )
                            samples = np.asarray(wave, dtype=np.float32)
                            finite = bool(np.isfinite(samples).all())
                            peak = (
                                float(np.max(np.abs(samples))) if samples.size else 0.0
                            )
                            rms = (
                                float(np.sqrt(np.mean(samples**2)))
                                if samples.size
                                else 0.0
                            )
                            signal_valid = bool(
                                samples.size and finite and peak > 0.001 and rms > 1e-5
                            )
                            manifests[arm].append(
                                {
                                    "signal_valid": signal_valid,
                                    "signal": {
                                        "finite": finite,
                                        "peak": peak,
                                        "rms": rms,
                                        "sample_count": samples.size,
                                    },
                                    "id": f"{identity}-{i:03d}",
                                    "path": str(path.relative_to(args.output / arm)),
                                    "sha256": sha(path),
                                    "language": lang,
                                    "expected": case["text"],
                                    "text": case["text"],
                                    "omnilingual_language": case.get(
                                        "omnilingual_language", LANG_IDS.get(lang)
                                    ),
                                    "reference_language": ref_lang,
                                    "case_id": case_id,
                                    "run": arm,
                                    "phase": "fixed-seed-split-cfg",
                                    "profile": config_data,
                                }
                            )
                        write(
                            args.output / arm / "quality-manifest.json", manifests[arm]
                        )
                    base, candidate = records["baseline"], records["split-cfg"]
                    equal_tokens = len(base["tokens"]) == len(
                        candidate["tokens"]
                    ) and all(
                        torch.equal(a, b)
                        for a, b in zip(base["tokens"], candidate["tokens"])
                    )
                    equal_calls = len(base["predictions"]) == len(
                        candidate["predictions"]
                    )
                    equal_predictions = equal_calls and all(
                        torch.equal(a[0], b[0])
                        for a, b in zip(base["predictions"], candidate["predictions"])
                    )
                    equal_confidence = equal_calls and all(
                        torch.equal(a[1], b[1])
                        for a, b in zip(base["predictions"], candidate["predictions"])
                    )
                    max_confidence_difference = (
                        max(
                            (a[1] - b[1]).abs().max().item()
                            for a, b in zip(
                                base["predictions"], candidate["predictions"]
                            )
                        )
                        if equal_calls and base["predictions"]
                        else None
                    )
                    b_manifest = [
                        x
                        for x in manifests["baseline"]
                        if x["id"].startswith(identity + "-")
                    ]
                    c_manifest = [
                        x
                        for x in manifests["split-cfg"]
                        if x["id"].startswith(identity + "-")
                    ]
                    equal_wav = all(
                        a["sha256"] == b["sha256"]
                        for a, b in zip(b_manifest, c_manifest)
                    )
                    timings = {"baseline": [], "split-cfg": []}
                    for repetition in range(args.repetitions):
                        order = [("baseline", False), ("split-cfg", True)]
                        if repetition % 2:
                            order.reverse()
                        for arm, enabled in order:
                            patch.enable() if enabled else patch.disable()
                            seed(20261103 + case_index)
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter()
                            model.generate(**kwargs)
                            torch.cuda.synchronize()
                            row = {
                                "ms": (time.perf_counter() - started) * 1000,
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            }
                            timings[arm].append(row)
                    row = {
                        "id": identity,
                        "case": case,
                        "batch_size": batch,
                        "reference": ref,
                        "final_tokens_equal": equal_tokens,
                        "per_step_predictions_equal": equal_predictions,
                        "per_step_confidence_equal": equal_confidence,
                        "max_confidence_difference": max_confidence_difference,
                        "saved_wav_bytes_equal": equal_wav,
                        "baseline_predictor_calls": len(base["predictions"]),
                        "split_predictor_calls": len(candidate["predictions"]),
                        "timings": timings,
                        "split_forward_metrics_cumulative": patch.metrics(),
                    }
                    row["median_full_model_ms"] = {
                        arm: float(np.median([x["ms"] for x in values]))
                        for arm, values in timings.items()
                    }
                    row["observed_speed_ratio"] = (
                        row["median_full_model_ms"]["baseline"]
                        / row["median_full_model_ms"]["split-cfg"]
                    )
                    valid_signals = all(
                        x["signal_valid"] for x in b_manifest + c_manifest
                    )
                    row["all_signals_valid"] = valid_signals
                    row["strict_parity_gate"] = (
                        "pass"
                        if equal_tokens and equal_wav and valid_signals
                        else "review_required"
                    )
                    write(directory / "result.json", row)
                    summaries.append(row)
                    write(args.output / "summary.json", summaries)
                    for arm, manifest in manifests.items():
                        write(args.output / arm / "quality-manifest.json", manifest)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in [
                                    "id",
                                    "final_tokens_equal",
                                    "per_step_predictions_equal",
                                    "per_step_confidence_equal",
                                    "saved_wav_bytes_equal",
                                    "median_full_model_ms",
                                    "observed_speed_ratio",
                                    "strict_parity_gate",
                                ]
                            }
                        ),
                        flush=True,
                    )
            write(
                args.output / "outcome.json",
                {
                    "status": "completed",
                    "all_cases_strict_parity": all(
                        x["strict_parity_gate"] == "pass" for x in summaries
                    ),
                    "qualifies_http_1000_in_two_seconds": False,
                    "note": "This is a model-stage parity/performance probe; no HTTP service or multilingual quality certification",
                },
            )
    except BaseException as exc:
        write(
            args.output / "outcome.json",
            {
                "status": "failed",
                "error": repr(exc),
                "completed_cases": len(summaries),
                "note": "Partial evidence retained; no quality/capacity claim",
            },
        )
        for arm, manifest in manifests.items():
            write(args.output / arm / "quality-manifest.json", manifest)
        raise
    finally:
        token_capture = None
        prediction_capture = None
        patch.disable()
        model._generate_iterative = original_iterative
        model._predict_tokens_with_scoring = original_predictor
        runner.unload_model()


if __name__ == "__main__":
    main()
