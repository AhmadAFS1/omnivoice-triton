#!/usr/bin/env python3
"""CPU-only independent SHA/decoded-signal/coverage audit of retained V6 runs."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2]
RETAINED = TASK / "retained-59071593"
SOURCE_REF = "377de39b5716266aea64adb05aaf1e674e5adffd"
TREE_SHA = "bf1637509899516dbe66390127e250db7829f724ed6643592bb1b857f40fe991"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = path.read_text()
    return (
        json.loads(raw)
        if raw.lstrip().startswith(("[", "{")) and path.suffix != ".jsonl"
        else [json.loads(line) for line in raw.splitlines() if line]
    )


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def latency(rows):
    values = [r["complete_ms"] for r in rows if r.get("status") == 200]
    return {
        "HTTP200": len(values),
        "p50_ms": float(np.percentile(values, 50)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
        "p99_ms": float(np.percentile(values, 99)) if values else None,
        "max_ms": max(values) if values else None,
        "within500ms": sum(x <= 500 for x in values),
        "kind": "sequential per-request complete HTTP body; not a simultaneous cohort metric",
    }


def main():
    final = read(RETAINED / "MODERN_VOICE_DIAGNOSTIC_OUTCOME.json")
    assert final["status"] == "completed" and len(final["outcomes"]) == 23
    grid = ["modern-v6-voice-gate104", "modern-v6-voice-primary300"] + [
        r["id"] for r in final["outcomes"]
    ]
    independent = read(
        RETAINED / "quality/voice-controls/modern-v6-independent-v1/outcomes.json"
    )
    extras = [r["id"] for r in independent]
    assert len(extras) == 13
    all_requests, all_audio, sources, run_records = [], [], [], []
    signal_mismatches = []
    for label in grid + extras:
        directory = RETAINED / "results" / label
        metadata = read(directory / "metadata.json")
        requested = metadata["requested_args"]
        assert requested["source_ref"] == SOURCE_REF
        assert metadata["source"]["source_tree_sha256"] == TREE_SHA
        assert (
            metadata["protocol_sha256"]
            == "1a077901bd91b8b7c16bd2424b52519f092935db2de4f74d068e288b1eb89302"
        )
        assert (
            requested["max_cuda_graphs"] == 2
            and requested["response_cache_max_bytes"] == 0
        )
        assert "RTX 4070 Ti" in metadata["hardware"]["nvidia_smi_gpu"]
        rows = read(directory / "requests.jsonl")
        quality = read(directory / "quality-manifest.json")
        by_id = {r["id"]: r for r in rows}
        assert len(by_id) == len(rows)
        sources.extend(
            {"file": str(path), "sha256": sha(path)}
            for path in [
                directory / "metadata.json",
                directory / "requests.jsonl",
                directory / "quality-manifest.json",
                directory / "summary.json",
                directory / "outcome.json",
            ]
        )
        for row in rows:
            row["audit_run"] = label
            row["audit_is_grid"] = label in grid
        all_requests.extend(rows)
        verified = []
        for item in quality:
            request = by_id[item["id"]]
            assert request["status"] == 200
            path = directory / item["path"]
            digest = sha(path)
            assert digest == item["sha256"], path
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            finite = bool(np.isfinite(samples).all())
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
            signal = bool(samples.size and finite and peak > 0.001 and rms > 1e-5)
            measured = {
                "duration_s": len(samples) / rate,
                "sample_rate": rate,
                "channels": samples.shape[1],
                "samples_per_channel": len(samples),
                "finite": finite,
                "peak": peak,
                "rms": rms,
                "signal_valid": signal,
                "clipped_sample_fraction": float(np.mean(np.abs(samples) >= 0.999))
                if samples.size
                else 0.0,
            }
            claimed = item["audio"]
            if (
                any(
                    abs(measured[key] - claimed[key]) > 1e-7
                    for key in ("duration_s", "peak", "rms")
                )
                or signal != claimed["signal_valid"]
                or finite != claimed["finite"]
            ):
                signal_mismatches.append(
                    {
                        "run": label,
                        "id": item["id"],
                        "measured": measured,
                        "claimed": claimed,
                    }
                )
            record = {
                "run": label,
                "id": item["id"],
                "case_id": item["case"]["id"],
                "is_grid": label in grid,
                "language": item["language"],
                "omnilingual_language": item.get("omnilingual_language"),
                "voice_id": item["case"].get("voice_id"),
                "mode": item["case"]["mode"],
                "config_type": item["case"].get("config_type"),
                "voice_profile": item.get("voice_profile"),
                "reference_id": item["case"].get("reference_id"),
                "reference_language": item.get("reference_language"),
                "generation_options": item.get("generation_options"),
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "audio": measured,
                "low_level_review_rms_lt_0.005": rms < 0.005,
                "quality_certified": False,
            }
            verified.append(record)
        all_audio.extend(verified)
        run_records.append(
            {
                "run": label,
                "is_grid": label in grid,
                "rows": len(rows),
                "HTTP_status": dict(
                    collections.Counter(str(r.get("status")) for r in rows)
                ),
                "audio": len(verified),
                "signal_valid": sum(r["audio"]["signal_valid"] for r in verified),
                "latency": latency(rows),
                "hardware": metadata["hardware"],
                "profile": metadata["profile"],
                "requested_args": requested,
                "runtime_validation": metadata["runtime_validation"],
            }
        )
    assert len(all_requests) == 3182 and len(all_audio) == 3154
    assert not signal_mismatches, signal_mismatches[:3]
    grid_rows = [r for r in all_requests if r["audit_is_grid"]]
    grid_audio = [r for r in all_audio if r["is_grid"]]
    expected = read(
        TASK / "quality/voice-controls/all104-all8voices-and-config-smoke.json"
    )
    assert len(grid_rows) == 2704 and len({r["case"]["id"] for r in grid_rows}) == 2704
    assert {r["case"]["id"] for r in grid_rows} == {r["id"] for r in expected}
    validation = [r for r in all_requests if r.get("validation_only")]
    assert len(validation) == 28 and all(
        r["status"] == r["expected_http_status"] == 400 for r in validation
    )
    assert all(r["status"] == 200 for r in all_requests if not r.get("validation_only"))
    route_records = []
    for language in sorted({r["language"] for r in grid_audio}):
        chosen = [r for r in grid_audio if r["language"] == language]
        assert len(chosen) == 26
        route_records.append(
            {
                "language": language,
                "expected_cells": 26,
                "observed_cells": len(chosen),
                "signal_valid": sum(r["audio"]["signal_valid"] for r in chosen),
                "invalid_cases": [
                    r["case_id"] for r in chosen if not r["audio"]["signal_valid"]
                ],
                "low_level_review": sum(
                    r["low_level_review_rms_lt_0.005"] for r in chosen
                ),
            }
        )
    configuration_records = []
    for config in sorted({r["config_type"] for r in grid_audio}):
        chosen = [r for r in grid_audio if r["config_type"] == config]
        requests = [r for r in grid_rows if r["case"]["config_type"] == config]
        configuration_records.append(
            {
                "config_type": config,
                "cases": len(chosen),
                "signal_valid": sum(r["audio"]["signal_valid"] for r in chosen),
                "low_level_review": sum(
                    r["low_level_review_rms_lt_0.005"] for r in chosen
                ),
                "latency": latency(requests),
            }
        )
    with (HERE / "verified-audio.jsonl").open("w") as out:
        out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in all_audio)
    write("per-route-coverage.json", route_records)
    write("per-configuration.json", configuration_records)
    write("per-run.json", run_records)
    invalid = [r for r in all_audio if not r["audio"]["signal_valid"]]
    write("invalid-audio.json", invalid)
    frontier = {
        str(step): next(
            r for r in run_records if r["run"] == f"candidate-step-frontier-{step}"
        )
        for step in (16, 8, 4)
    }
    summary = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "measurement_kind": "CPU-only independent rehash+decoded PCM audit of retained GPU results; no new model/GPU/API calls",
        "audit_hardware": {
            "platform": platform.platform(),
            "CPU": "Apple M5 Pro",
            "numpy": np.__version__,
            "soundfile": sf.__version__,
        },
        "inference_hardware": "NVIDIA GeForce RTX 4070 Ti; physical 12,282 MiB via nvidia-smi; visible 12,481,134,592 bytes; driver 610.57.04; CUDA 12.8; per-run full evidence in per-run.json",
        "source_ref": SOURCE_REF,
        "source_tree_sha256": TREE_SHA,
        "audit_source_sha256": sha(Path(__file__)),
        "verified_WAVs": len(all_audio),
        "verified_WAV_bytes": sum(r["bytes"] for r in all_audio),
        "declared_vs_independent_signal_mismatches": signal_mismatches,
        "grid": {
            "cases": 2704,
            "HTTP200": 2704,
            "signal_valid": sum(r["audio"]["signal_valid"] for r in grid_audio),
            "invalid_audio": sum(not r["audio"]["signal_valid"] for r in grid_audio),
            "languages": 104,
            "voices": 8,
            "clone_cases": 2496,
            "auto_cases": 104,
            "design_cases": 104,
            "routes_with_all26cells": len(route_records),
            "low_level_review_rms_lt_0.005": sum(
                r["low_level_review_rms_lt_0.005"] for r in grid_audio
            ),
        },
        "matrix": {
            "cases": 130,
            "signal_valid": next(
                r["signal_valid"]
                for r in run_records
                if r["run"] == "modern-v6-voice-matrix130"
            ),
        },
        "validation": {"cases": 28, "expected400": 28, "generation_denominator": 0},
        "targeted_repeats": {
            "cases": 8,
            "invalid_audio": sum(
                not r["audio"]["signal_valid"]
                for r in all_audio
                if r["run"].startswith("modern-v6-diagnostic-") and not r["is_grid"]
            ),
        },
        "frontier": {
            step: {
                "cases": 104,
                "signal_valid": record["signal_valid"],
                "latency": record["latency"],
            }
            for step, record in frontier.items()
        },
        "sources": sources,
        "quality_certified": False,
        "caveats": [
            "HTTP200 and weak-signal validity are not word/native/voice-identity quality judgments; ASR/native review pending.",
            "Low-level RMS<0.005 is a diagnostic review band, not a validated failure boundary; quiet styles can be intentional.",
            "Four grid empty outputs and matrix/targeted-repeat failures remain actual retained results, not discarded or retried away.",
            "Original404 gate failed; remaining2300 were explicitly authorized diagnostic expansion, not a passed quality gate.",
            "One sample per voice/route/preset and startup-seeded fresh-worker phases; counts are observed coverage, not population failure probabilities.",
            "Phase05 parent bookkeeping includes temporary transport-lease suspension; actual V5 per-request/child metrics remain separate.",
        ],
    }
    write("summary.json", summary)
    lines = [
        "# NVIDIA GeForce RTX 4070 Ti voice/control audit",
        "",
        summary["inference_hardware"],
        "",
        "CPU-only audit on Apple M5 Pro. All 3,154 retained generation WAVs were independently rehashed and decoded; signal/duration/peak/RMS matched the original harness records. No new model or API calls were made.",
        "",
        "The grid of 2,704 cases covers 104 routes × eight voices × three presets (2,496 clone cases), plus 104 auto and 104 design cases. All 2,704 returned HTTP 200; 2,700 passed the weak signal check. Four default-preset outputs were zero-frame, 44-byte WAVs. This diagnostic expansion preserved the original failed gate of 404 cases and does not promote any configuration's word/voice quality.",
        "",
        "| Configuration | Cases | Valid signal | Sequential HTTP200 p50 / p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for r in configuration_records:
        lines.append(
            f"| {r['config_type']} | {r['cases']} | {r['signal_valid']} | {r['latency']['p50_ms']:.1f} / {r['latency']['p95_ms']:.1f} |"
        )
    lines += [
        "",
        "Four grid failures: Somali/Aoede, Turkmen/Aoede, Turkmen/Charon and Finnish/Iapetus; all speed 1.3/postfull. The controlled Turkmen repeat returned a nearly silent 1.6-second WAV with postoff and an empty WAV with postfull at matched startup seed. The off result barely passed the weak signal floor and is not word-quality-cleared. Somali's four fresh-start repeats were valid-signal, so the original long-run RNG position was not reproduced.",
        "",
        "The 130-case matrix returned 130 HTTP 200 responses and 129 valid signals; the Russian 64-step/CFG 0/speed 0.5/denoise false/preprocess false/postfull case was empty. All 28 validation requests correctly returned 400 and stay outside generation/ASR quality denominators.",
        "",
        "| Steps | Requests / valid signal | Sequential p50 / p95 ms | Under 500 ms |",
        "|---|---:|---:|---:|",
    ]
    for step, record in frontier.items():
        l = record["latency"]
        lines.append(
            f"| {step} | 104 / {record['signal_valid']} | {l['p50_ms']:.1f} / {l['p95_ms']:.1f} | {l['within500ms']}/104 |"
        )
    lines += [
        "",
        "All step-count results require ASR/native review before promotion; these sequential latency values do not measure 1,000 simultaneous novel requests. Dataset/source/proxy/reference caveats remain unchanged. `verified-audio.jsonl` lists every retained hash/path, `per-route-coverage.json` checks all 26 cells per route, and `invalid-audio.json` preserves all six observed invalid WAV records across grid/matrix/repeats.",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "verified_WAVs": len(all_audio),
                "grid": summary["grid"],
                "invalid_audio_all": len(invalid),
                "output": str(HERE),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
