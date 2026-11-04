#!/usr/bin/env python3
"""CPU-only paired audit of the retained RTX 4070 Ti reference-prefix run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent
VARIANTS = ("full", "prefix10", "prefix17")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def lines(path, rows):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


def stats(values):
    a = np.asarray(values, dtype=float)
    assert a.size and np.isfinite(a).all()
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def captures(path):
    """Associate logged actual capture shapes with sequential request IDs.

    Hits without a capture are deliberately not assigned an inferred shape.
    Admission headers remain labeled estimates, including known rounding gaps.
    """
    found = {}
    owner = None
    for line in path.read_text().splitlines():
        match = re.search(r"Executing batch .*size=1 request_ids=([a-z0-9]+)", line)
        if match:
            owner = match.group(1)
            found[owner] = []
        match = re.search(
            r"CUDA Graph captured for shape \((\d+), (\d+), (\d+)\)", line
        )
        if match and owner:
            found[owner].append([int(x) for x in match.groups()])
        if "Completed batch" in line:
            owner = None
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=TASK / "retained-59071593/results"
    )
    parser.add_argument(
        "--output", type=Path, default=TASK / "quality/reference-prefix-analysis-v1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    references = {
        x["id"]: x
        for x in read(
            TASK / "quality/reference-prefix-plans-v1/cpu-local-reference-paths.json"
        )
    }
    reference_audit = []
    for ref in references.values():
        path = Path(ref["path"])
        assert digest(path) == ref["sha256"], ref["id"]
        info = sf.info(path)
        assert abs(info.duration - ref["duration_s"]) < 1 / info.samplerate
        reference_audit.append(
            {
                "id": ref["id"],
                "path": str(path),
                "sha256": ref["sha256"],
                "frames": info.frames,
                "sample_rate": info.samplerate,
                "duration_s": info.duration,
            }
        )
    assert len(reference_audit) == 80

    provenance = {
        "analysis_kind": "Local CPU-only hash/decode/paired statistical audit"
    }
    data = {}
    audio_audit = []
    for variant in VARIANTS:
        run = args.results / f"reference-prefix-v1-{variant}"
        metadata = read(run / "metadata.json")
        hardware = metadata["hardware"]
        device = hardware["torch_cuda_devices"][0]
        assert device["name"] == "NVIDIA GeForce RTX 4070 Ti"
        assert device["visible_vram_bytes"] == 12481134592
        assert metadata["source_ref"] == "ce48b253e66b951d87ea42919f945828c62e96a4"
        assert metadata["requested_args"]["response_cache_max_bytes"] == 0
        if variant == "full":
            provenance.update(
                GPU=device["name"],
                visible_vram_bytes=device["visible_vram_bytes"],
                physical_vram_MiB=12282,
                physical_vram_evidence="Fresh nvidia-smi memory.total in each metadata.json",
                visible_vram_evidence="Fresh torch.cuda device properties in each metadata.json",
                driver="610.57.04",
                CUDA_runtime=hardware["torch_cuda_runtime"],
                packages=hardware["packages"],
                date="2026-11-04",
                source_ref=metadata["source_ref"],
                source_tree_sha256=metadata["source"]["source_tree_sha256"],
                profile=metadata["profile"],
                startup_seed=20261103,
                graph_limit=8,
                graph_row_token_budget=16384,
                unused_KV_disabled=True,
                split_cfg=False,
                codec_batch_size=1,
                response_cache_bytes=0,
                concurrent_requests=1,
                co_resident_load="Dedicated TTS lease; no ASR/setup/transfers during timing; telemetry and health probes active",
                limitations=[
                    "One ordered sequential pass per arm, startup seed reset per process; not repeated randomized trials.",
                    "Target/conditioning/max-sequence headers are admission estimates, not a trace of all actual model tensors.",
                    "Actual reference audio token counts come from registration responses.",
                    "Output duration is measured from decoded WAVs; it is not an ASR or voice-quality score.",
                    "Startup/registration are outside per-request completion_ms. Graph captures during requests are included.",
                    "Reference lexical gates passed; generated speech word and speaker/perceptual checks remain pending.",
                    "This does not measure concurrent inference capacity or certify a default change.",
                ],
            )
        registration = {
            x["reference_id"]: x
            for x in (
                json.loads(line)
                for line in (run / "registrations.jsonl").read_text().splitlines()
            )
        }
        captured = captures(run / "server.log")
        rows = [
            json.loads(line)
            for line in (run / "requests.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 132
        lookup = {}
        for row in rows:
            case = row["case"]
            key = case["comparison_original_case_id"]
            assert (
                key not in lookup
                and row["status"] == 200
                and not row["validation_only"]
            )
            assert case["worker_options"] == {
                "num_step": 16,
                "denoise": True,
                "preprocess_prompt": True,
                "postprocess_mode": "off",
                "speed": 1.0,
            }
            path = run / row["path"]
            assert digest(path) == row["sha256"]
            wave, rate = sf.read(path, dtype="float32", always_2d=True)
            finite = bool(np.isfinite(wave).all())
            peak = float(np.abs(wave).max())
            rms = float(np.sqrt(np.mean(wave**2)))
            duration = len(wave) / rate
            assert finite and wave.size and row["audio"]["signal_valid"]
            for actual, expected in (
                (duration, row["audio"]["duration_s"]),
                (peak, row["audio"]["peak"]),
                (rms, row["audio"]["rms"]),
            ):
                assert abs(actual - expected) < 1e-7, (variant, key, actual, expected)
            ref = references[case["reference_id"]]
            reg = registration[case["reference_id"]]
            assert (
                row["reference_identity"]["sha256"]
                == ref["sha256"]
                == case["conditioning_reference_sha256"]
            )
            assert row["reference_identity"]["text"] == ref["text"]
            # The pinned OmniVoice preparation appends a terminal ASCII period
            # to some transcripts (including the 10-word unpunctuated prefix).
            # Retain both representations and reject any other transformation.
            prepared_ref_text = reg["body"]["ref_text"]
            assert prepared_ref_text in (ref["text"], ref["text"] + ".")
            headers = row["headers"]
            result = {
                "id": row["id"],
                "comparison_original_case_id": key,
                "variant": variant,
                "run": run.name,
                "path": str(path.resolve()),
                "sha256": row["sha256"],
                "language": case["language"],
                "omnilingual_language": case["omnilingual_language"],
                "voice_id": case["voice_id"],
                "voice_profile": case["voice_profile"],
                "reference_language": case["reference_language"],
                "unchanged_reference_control": case["unchanged_reference_control"],
                "reference_id": case["reference_id"],
                "reference_path": ref["path"],
                "reference_sha256": ref["sha256"],
                "reference_duration_s": ref["duration_s"],
                "reference_text_forwarded": ref["text"],
                "reference_text_after_worker_preparation": prepared_ref_text,
                "reference_audio_tokens_actual": reg["body"]["prompt_audio_tokens"],
                "expected": case["expected"],
                "source": case["source"],
                "complete_ms": row["complete_ms"],
                "server_latency_ms": float(headers["x-omnivoice-latency-ms"]),
                "iterative_ms": float(headers["x-omnivoice-iterative-generate-ms"]),
                "decode_ms": float(headers["x-omnivoice-audio-decode-ms"]),
                "target_tokens_header_estimate": int(
                    headers["x-omnivoice-batch-target-tokens"]
                ),
                "conditioning_tokens_header_estimate": int(
                    headers["x-omnivoice-batch-conditioning-tokens"]
                ),
                "max_sequence_length_header_estimate": int(
                    headers["x-omnivoice-batch-max-sequence-length"]
                ),
                "new_graph_capture_shapes_actual": captured.get(
                    headers["x-omnivoice-request-id"], []
                ),
                "output_duration_s": duration,
                "rms": rms,
                "peak": peak,
                "frames": len(wave),
                "sample_rate": rate,
                "finite": finite,
                "signal_valid": True,
                "normalized_worker_options": row["normalized_worker_options"],
            }
            lookup[key] = result
            audio_audit.append(result)
        data[variant] = lookup
        provenance[variant] = {
            "metadata_path": str((run / "metadata.json").resolve()),
            "metadata_sha256": digest(run / "metadata.json"),
            "requests_sha256": digest(run / "requests.jsonl"),
            "quality_manifest_sha256": digest(run / "quality-manifest.json"),
            "health_after_sha256": digest(run / "health-after.json"),
            "reference_manifest_sha256": digest(run / "reference-manifest.json"),
            "plan_sha256": metadata["plan_sha256"],
            "protocol_sha256": metadata["protocol_sha256"],
            "hardware": hardware,
        }
    assert all(set(data[x]) == set(data["full"]) for x in VARIANTS)
    summary = {
        "provenance": provenance,
        "verified_generated_WAVs": 396,
        "verified_reference_WAVs": 80,
        "groups": {},
        "paired": {},
    }
    paired = []
    for variant in VARIANTS:
        summary["groups"][variant] = {}
        for group, predicate in {
            "all132": lambda row: True,
            "changed_English110": lambda row: not row["unchanged_reference_control"],
            "unchanged_native22": lambda row: row["unchanged_reference_control"],
            "Aoede_all104": lambda row: row["voice_id"] == "Aoede",
        }.items():
            rows = [row for row in data[variant].values() if predicate(row)]
            summary["groups"][variant][group] = {
                "n": len(rows),
                "latency_ms": stats([x["complete_ms"] for x in rows]),
                "server_latency_ms": stats([x["server_latency_ms"] for x in rows]),
                "iterative_ms": stats([x["iterative_ms"] for x in rows]),
                "decode_ms": stats([x["decode_ms"] for x in rows]),
                "output_duration_s": stats([x["output_duration_s"] for x in rows]),
                "reference_duration_s": stats(
                    [x["reference_duration_s"] for x in rows]
                ),
                "reference_audio_tokens_actual": stats(
                    [x["reference_audio_tokens_actual"] for x in rows]
                ),
                "target_tokens_header_estimate": stats(
                    [x["target_tokens_header_estimate"] for x in rows]
                ),
                "conditioning_tokens_header_estimate": stats(
                    [x["conditioning_tokens_header_estimate"] for x in rows]
                ),
                "full_WAV_within_500ms": sum(x["complete_ms"] <= 500 for x in rows),
                "RMS_below_0_005_diagnostic_only": sum(x["rms"] < 0.005 for x in rows),
            }
    for key, before in data["full"].items():
        entry = {
            "comparison_original_case_id": key,
            "language": before["language"],
            "voice_id": before["voice_id"],
            "expected": before["expected"],
            "unchanged_reference_control": before["unchanged_reference_control"],
            "arms": {variant: data[variant][key] for variant in VARIANTS},
        }
        entry["comparisons"] = {}
        for variant in VARIANTS[1:]:
            after = data[variant][key]
            assert before["expected"] == after["expected"]
            assert (
                before["normalized_worker_options"]
                == after["normalized_worker_options"]
            )
            assert (
                before["unchanged_reference_control"]
                == after["unchanged_reference_control"]
            )
            if before["unchanged_reference_control"]:
                assert before["reference_sha256"] == after["reference_sha256"]
            else:
                assert before["reference_sha256"] != after["reference_sha256"]
            entry["comparisons"][variant] = {
                "latency_ratio_after_over_full": after["complete_ms"]
                / before["complete_ms"],
                "latency_delta_ms": after["complete_ms"] - before["complete_ms"],
                "body_sha_equal": before["sha256"] == after["sha256"],
                "output_duration_ratio": after["output_duration_s"]
                / before["output_duration_s"],
                "output_duration_delta_s": after["output_duration_s"]
                - before["output_duration_s"],
                "reference_audio_tokens_delta_actual": after[
                    "reference_audio_tokens_actual"
                ]
                - before["reference_audio_tokens_actual"],
                "target_tokens_delta_header_estimate": after[
                    "target_tokens_header_estimate"
                ]
                - before["target_tokens_header_estimate"],
                "conditioning_tokens_delta_header_estimate": after[
                    "conditioning_tokens_header_estimate"
                ]
                - before["conditioning_tokens_header_estimate"],
                "reference_duration_delta_s": after["reference_duration_s"]
                - before["reference_duration_s"],
            }
        paired.append(entry)
    for variant in VARIANTS[1:]:
        summary["paired"][variant] = {}
        for control in (False, True):
            rows = [
                x["comparisons"][variant]
                for x in paired
                if x["unchanged_reference_control"] == control
            ]
            summary["paired"][variant][
                "unchanged_native22" if control else "changed_English110"
            ] = {
                "n": len(rows),
                **{
                    field: stats([x[field] for x in rows])
                    for field in (
                        "latency_ratio_after_over_full",
                        "latency_delta_ms",
                        "output_duration_ratio",
                        "output_duration_delta_s",
                        "reference_audio_tokens_delta_actual",
                        "target_tokens_delta_header_estimate",
                        "conditioning_tokens_delta_header_estimate",
                        "reference_duration_delta_s",
                    )
                },
                "body_sha_equal": sum(x["body_sha_equal"] for x in rows),
                "latency_faster": sum(x["latency_delta_ms"] < 0 for x in rows),
                "output_duration_equal": sum(
                    x["output_duration_delta_s"] == 0 for x in rows
                ),
            }
    dump(args.output / "summary.json", summary)
    lines(args.output / "paired132.jsonl", paired)
    lines(args.output / "verified-audio396.jsonl", audio_audit)
    dump(args.output / "reference-audit80.json", reference_audit)
    report = [
        "# RTX 4070 Ti reference-prefix A/B — sequential complete-WAV results",
        "",
        (
            "Measured 2026-11-04 on NVIDIA GeForce RTX 4070 Ti: physical 12,282 MiB "
            "(`nvidia-smi`), Torch-visible 12,481,134,592 bytes, driver 610.57.04, CUDA 12.8. "
            "The linked per-arm metadata records the fresh device evidence, package pins and idle preflight. "
            "This report is a local CPU-only audit of retained GPU results; no ASR or speaker model ran here."
        ),
        "",
        (
            "All three arms used source `ce48b253e66b951d87ea42919f945828c62e96a4`, "
            "16 steps, speed 1, postprocessing off, hybrid FP16, unused KV off, split CFG off, "
            "graph limit 8 / row-token budget 16,384, codec batch 1 and response cache disabled. "
            "Each fresh process reset startup seed 20261103. Telemetry and health probes ran; "
            "ASR, setup and transfers were absent during timing."
        ),
        "",
        (
            "396/396 offers returned HTTP 200 and readable, finite, nonempty audio. "
            "The independent protocol verifiers passed and no fatal CUDA/context fault was logged. "
            "All 396 generated WAV hashes and decoded duration/RMS/peak match the retained manifests; "
            "all 80 reference WAV hashes and durations were verified locally."
        ),
        "",
        (
            "132 paired cases per arm cover all 104 Aoede language routes plus all 8 voices on "
            "Arabic, English, Spanish and Hindi. 110 cases replace an English reference; "
            "22 keep their native-language reference unchanged. Prefix10/prefix17 mean the first "
            "10/17 English words, with matching shortened transcripts, not 10/17 seconds."
        ),
        "",
        "| RTX 4070 Ti cohort | Arm | Requests | Complete-WAV p50 / p95 ms | Within 500 ms | Median output seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in ("changed_English110", "unchanged_native22", "Aoede_all104"):
        for variant in VARIANTS:
            row = summary["groups"][variant][group]
            report.append(
                f"| {group} | {variant} | {row['n']} | "
                f"{row['latency_ms']['p50']:.1f} / {row['latency_ms']['p95']:.1f} | "
                f"{row['full_WAV_within_500ms']} / {row['n']} | "
                f"{row['output_duration_s']['p50']:.2f} |"
            )
    report += [
        "",
        "Paired effects (each case compared with the same full-reference case):",
        "",
        "| RTX 4070 Ti cohort | Variant | Median latency ratio | Median reference-token delta (actual) | Median target-token delta (header estimate) | Identical WAV bodies |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS[1:]:
        for group in ("changed_English110", "unchanged_native22"):
            row = summary["paired"][variant][group]
            report.append(
                f"| {group} | {variant} | {row['latency_ratio_after_over_full']['p50']:.3f} | "
                f"{row['reference_audio_tokens_delta_actual']['p50']:.0f} | "
                f"{row['target_tokens_delta_header_estimate']['p50']:.0f} | "
                f"{row['body_sha_equal']} / {row['n']} |"
            )
    report += [
        "",
        (
            "The changed-reference latency reduction is accompanied by different generated audio "
            "and changed duration/target estimates. It cannot be described as mathematically equivalent "
            "inference. All 22 unchanged-reference controls remain byte-identical in both prefix arms; "
            "their latency ratios near 1 provide an internal timing control."
        ),
        "",
        (
            "Registration returns actual reference audio-token counts. Admission headers are "
            "estimates for target/conditioning/sequence length and must not be presented as exact "
            "sampler tensor lengths. New graph captures are separately traced from server logs; "
            "cache-hit shapes are left unobserved rather than inferred. Per-request latency includes "
            "on-demand graph captures and full HTTP body receipt, and excludes worker startup/reference "
            "registration. The cohorts were run in one fixed order without repeat/randomized trials."
        ),
        (
            "The pinned worker appends a terminal ASCII period to the unpunctuated 10-word "
            "transcript and to the Urdu reference. Forwarded and prepared transcripts are both "
            "retained in the paired artifact; no other transcript transformation was observed."
        ),
        "",
        (
            "Reference lexical checks passed in the two Meta judges and cached Mac Whisper before "
            "this run. Generated speech ASR, speaker similarity and perceptual review remain pending. "
            "Nonempty audio and a lower latency do not certify words or voice identity. No default "
            "promotion and no concurrent-capacity claim follow from this experiment."
        ),
        "",
        (
            "Artifacts: `summary.json` contains source/hardware/pins and every denominator; "
            "`paired132.jsonl` exposes absolute local WAV/reference paths, exact texts, SHA hashes "
            "and paired deltas for downstream ASR/speaker checks. `verified-audio396.jsonl` and "
            "`reference-audit80.json` retain the CPU audit evidence. Original run files remain unchanged."
        ),
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    source_copy = args.output / "analyze-source.py"
    source_copy.write_bytes(Path(__file__).read_bytes())
    artifacts = [
        {
            "path": str(path.relative_to(args.output)),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.output.iterdir())
        if path.is_file()
    ]
    dump(args.output / "artifacts-sha256.json", artifacts)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "WAVs_verified": len(audio_audit),
                "summary_sha256": digest(args.output / "summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
