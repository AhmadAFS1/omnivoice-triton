#!/usr/bin/env python3
"""Finite multilingual full-WAV load experiment with an explicit JSON workload.

Hardware is unverified until execution. Every run independently captures the
actual GPU/VRAM/runtime. Expected words are retained for later blind ASR only.
This runner never rents a GPU, contacts a transcription service or streams audio.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from runtime import (
    append_json,
    collect_hardware,
    percentile,
    source_identity,
    utc_now,
    wait_ready,
    write_json,
)

HERE = Path(__file__).resolve().parent


def configure_file_limits(connections):
    """Give only this fresh harness/server enough descriptors for its socket cap."""
    import resource

    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    minimum = max(4096, connections * 2 + 1024)
    soft, hard = before
    if hard != resource.RLIM_INFINITY and hard < minimum:
        raise RuntimeError(
            f"RLIMIT_NOFILE hard limit{hard} cannot support{connections} connections; "
            f"need at least{minimum}"
        )
    if soft != resource.RLIM_INFINITY and soft < minimum:
        target = max(16384, minimum)
        if hard != resource.RLIM_INFINITY:
            target = min(target, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    after = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert after[0] == resource.RLIM_INFINITY or after[0] >= minimum
    return {
        "resource": "RLIMIT_NOFILE",
        "before_soft_hard": list(before),
        "after_soft_hard": list(after),
        "minimum_required": minimum,
        "configured_http_connection_cap": connections,
        "scope": "Only this harness process and its newly spawned descendants",
        "server_inherits_limit": True,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_defaults(steps=16):
    return {
        "num_step": steps,
        "guidance_scale": 2.0,
        "t_shift": 0.1,
        "layer_penalty_factor": 5.0,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
        "speed": 1.0,
        "duration": None,
        "denoise": True,
        "preprocess_prompt": True,
        "postprocess_mode": "off",
        "insert_pauses": False,
    }


def generation_options(case, steps=16):
    """Normalize every sent worker field; unknown or misplaced overrides fail."""
    values = generation_defaults(steps)
    assert not set(case) & values.keys(), "Put generation controls in worker_options"
    supplied = case.get("worker_options", {})
    assert isinstance(supplied, dict) and supplied.keys() <= values.keys(), (
        "Unknown worker_options would be silently ignored"
    )
    values.update(supplied)
    validation = case.get("validation_only", False)
    assert isinstance(validation, bool)
    for key, value in values.items():
        assert not isinstance(value, float) or math.isfinite(value), (
            "Use literal form strings for nonfinite validation cases, never JSONNaN/Inf"
        )
        if validation:
            continue
        if key in {"denoise", "preprocess_prompt", "insert_pauses"}:
            assert isinstance(value, bool), f"{key} must be boolean"
        elif key == "postprocess_mode":
            assert value in {"full", "light", "off"}
        elif key == "num_step":
            assert isinstance(value, int) and not isinstance(value, bool) and value > 0
        elif value is None:
            assert key in {"speed", "duration"}, f"{key} cannot be unset"
        else:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), key
            if key != "duration":
                assert value > 0 if key == "speed" else value >= 0, key
            values[key] = float(value)
    if "reference_preprocess_prompt" in case:
        wanted = case["reference_preprocess_prompt"]
        assert isinstance(wanted, bool)
        if "preprocess_prompt" in supplied:
            assert wanted == values["preprocess_prompt"], (
                "Reference preprocessing conflicts with posted worker option"
            )
        values["preprocess_prompt"] = wanted
    if validation:
        assert case.get("expected_http_status") == 400, (
            "Boundary validation needs an explicit expected400 status"
        )
    else:
        assert case.get("expected_http_status", 200) == 200
    return values


def form_values(values):
    return {
        key: ("true" if value else "false") if isinstance(value, bool) else str(value)
        for key, value in values.items()
        if value is not None
    }


def reference_identity(case, references, steps=16):
    reference = references[case["reference_id"]]
    return {
        "sha256": reference["sha256"],
        "text": reference["text"],
        "preprocess_prompt": generation_options(case, steps)["preprocess_prompt"],
    }


def registration_key(case, references, steps=16):
    return json.dumps(
        reference_identity(case, references, steps),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def reference_mode(case, phase):
    transport = case.get("request_transport")
    assert transport in {
        None,
        "registered",
        "upload",
        "upload_or_separately_registered_preprocess_variant",
    }
    mode = (
        transport
        if transport in {"registered", "upload"}
        else phase.get("reference_mode", "registered")
    )
    assert mode in {"registered", "upload", "cold_upload"}
    return "upload" if mode == "cold_upload" else mode


def load_inputs(
    corpus_path: Path, plan_path: Path, references_path: Path | None, default_steps=16
):
    cases = json.loads(corpus_path.read_text())
    if isinstance(cases, dict):
        cases = cases["cases"]
    assert cases, "Empty corpus"
    by_id = {}
    for case in cases:
        assert case["id"] not in by_id, "Duplicate corpus case ID"
        assert case.get("text", "").strip(), "Empty text"
        assert case.get("language") and case.get("api_language")
        assert case.get("mode") in {"clone", "design", "auto"}
        generation_options(case, default_steps)
        if case["mode"] == "clone":
            assert case.get("reference_id"), "Clone case has no explicit reference"
            assert case.get("reference_language"), "Reference language must be explicit"
            assert case.get("voice_profile"), "Native/cross-language profile required"
        by_id[case["id"]] = case
    references = {}
    if references_path:
        raw = json.loads(references_path.read_text())
        if isinstance(raw, dict):
            raw = raw.get("references", list(raw.values()))
        for reference in raw:
            key = reference.get("id", reference.get("reference_id"))
            assert key and key not in references, "Duplicate/missing reference ID"
            path = Path(reference.get("path", reference.get("file")))
            if not path.is_absolute():
                path = references_path.parent / path
            assert path.exists() and digest(path) == reference["sha256"]
            assert reference.get("text"), "Known reference transcript required"
            references[key] = {**reference, "path": str(path.resolve())}
    for case in cases:
        if case["mode"] == "clone":
            assert case["reference_id"] in references, (
                f"Missing reference for {case['id']}; no silent voice fallback"
            )
    plan = json.loads(plan_path.read_text())
    phases = plan["phases"] if isinstance(plan, dict) else plan
    names = set()
    exposed_references = set()
    for phase in phases:
        assert phase["id"] not in names, "Duplicate phase ID"
        names.add(phase["id"])
        assert phase["kind"] in {"sequential", "burst", "staggered", "closed_loop"}
        selected = expand_cases(phase, by_id)
        assert selected, f"No selected cases in {phase['id']}"
        validation = {case.get("validation_only", False) for case in selected}
        assert len(validation) == 1, (
            "Do not mix API-boundary and generation denominators"
        )
        if True in validation:
            assert phase.get("validation_only") is True
            assert phase["kind"] == "sequential", (
                "Boundary checks must be finite sequential"
            )
        else:
            assert not phase.get("validation_only", False)
        if phase["kind"] == "staggered":
            assert phase.get("arrival_rate_rps", 0) > 0
        if phase["kind"] == "closed_loop":
            assert phase.get("concurrency", 0) > 0
        if phase.get("content_profile") == "unique":
            signatures = [
                effective_signature(case, references, default_steps)
                for case in selected
            ]
            assert len(set(signatures)) == len(signatures), (
                f"{phase['id']} repeats effective input but claims unique inference"
            )
        if phase.get("reference_mode") == "cold_upload":
            assert all(case["mode"] == "clone" for case in selected)
            assert all(reference_mode(case, phase) == "upload" for case in selected)
            keys = [
                registration_key(case, references, default_steps) for case in selected
            ]
            assert len(set(keys)) == len(keys), (
                "Cold-upload phase repeats a reference cache key; label it warm upload"
            )
            assert not set(keys) & exposed_references, (
                "Cold-upload phase reuses a previously exposed reference"
            )
        exposed_references.update(
            registration_key(case, references, default_steps)
            for case in selected
            if case["mode"] == "clone"
        )
    return by_id, references, phases


def effective_signature(case, references, steps=16):
    return json.dumps(
        {
            "mode": case["mode"],
            "text": case["text"],
            "language": case["api_language"],
            "instruct": case.get("instruct") or None,
            "generation": generation_options(case, steps),
            "reference": reference_identity(case, references, steps)
            if case["mode"] == "clone"
            else None,
        },
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def expand_cases(phase, by_id):
    ids = phase.get("case_ids", "all")
    selected = list(by_id.values()) if ids == "all" else [by_id[key] for key in ids]
    if phase.get("lengths"):
        selected = [case for case in selected if case.get("length") in phase["lengths"]]
    if phase.get("languages"):
        selected = [case for case in selected if case["language"] in phase["languages"]]
    if phase.get("shuffle_seed") is not None:
        random.Random(phase["shuffle_seed"]).shuffle(selected)
    count = phase.get("requests", len(selected))
    assert count > 0 and selected
    return [selected[index % len(selected)] for index in range(count)]


def build_command(args):
    command = [
        sys.executable,
        "-m",
        "omnivoice_triton.cli.api_server",
        "--model",
        args.model,
        "--runner",
        args.runner,
        "--device",
        "cuda",
        "--dtype",
        "fp16",
        "--no-asr",
        "--ip",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--batch-collect-ms",
        str(args.collect_ms),
        "--max-batch-requests",
        str(args.max_batch_requests),
        "--max-pending-requests",
        str(args.max_pending_requests),
        "--max-pending-gpu-tasks",
        str(args.max_pending_gpu_tasks),
        "--max-batch-target-tokens",
        str(args.max_batch_target_tokens),
        "--max-batch-conditioning-tokens",
        str(args.max_batch_conditioning_tokens),
        "--batch-bucket-sizes",
        args.batch_bucket_sizes,
        "--max-cuda-graphs",
        str(args.max_cuda_graphs),
    ]
    if args.audio_decode_batch_size:
        command.extend(["--audio-decode-batch-size", str(args.audio_decode_batch_size)])
    if args.response_cache_max_bytes is not None:
        command.extend(
            ["--response-cache-max-bytes", str(args.response_cache_max_bytes)]
        )
    if args.response_cache_ttl is not None:
        command.extend(["--response-cache-ttl", str(args.response_cache_ttl)])
    extra = json.loads(args.server_extra_flags_json)
    assert isinstance(extra, list) and all(isinstance(item, str) for item in extra)
    protected = {part for part in command if part.startswith("--")}
    assert not any(item in protected for item in extra), (
        "Duplicate protected server flag"
    )
    command.extend(extra)
    if args.startup_seed is not None:
        command = [
            sys.executable,
            str(HERE / "seeded_server.py"),
            "--seed",
            str(args.startup_seed),
            "--evidence",
            str(args.output / "server-seed.json"),
            *command[3:],
        ]
    return command


def validate_runtime(args, health):
    expected = {
        "runner": args.runner,
        "device": "cuda",
        "dtype": "fp16",
        "batch_collect_ms": args.collect_ms,
        "max_batch_requests": args.max_batch_requests,
        "max_batch_target_tokens": args.max_batch_target_tokens,
        "max_batch_conditioning_tokens": args.max_batch_conditioning_tokens,
        "max_cuda_graphs": args.max_cuda_graphs,
    }
    if args.audio_decode_batch_size:
        expected["audio_decode_batch_size"] = args.audio_decode_batch_size
    expected.update(json.loads(args.expected_health_json))
    if args.response_cache_max_bytes is not None:
        expected["response_cache.max_bytes"] = args.response_cache_max_bytes
    if args.response_cache_ttl is not None:
        expected["response_cache.ttl_s"] = args.response_cache_ttl
    for key, value in expected.items():
        actual = health
        for part in key.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        assert actual == value, f"Requested {key}={value}, actual={actual}"
    assert health["batcher_queue"]["max_pending_requests"] == args.max_pending_requests
    assert (
        health["batcher_queue"]["max_pending_gpu_tasks"] == args.max_pending_gpu_tasks
    )
    assert health["load_asr"] is False
    assert health["batch_bucket_sizes"] == [
        int(x) for x in args.batch_bucket_sizes.split(",")
    ]
    evidence = None
    if args.startup_seed is not None:
        evidence = json.loads((args.output / "server-seed.json").read_text())
        assert all(
            evidence[key] == args.startup_seed
            for key in ["requested_seed", "torch_initial_seed", "cuda_initial_seed"]
        )
        assert evidence["applied_before_module_import"] is True
    return {"actual_health_checked": expected, "startup_seed_evidence": evidence}


def retain_wave_batch(output, label, captured):
    """CPU/file I/O after the network cohort, called in a worker thread."""
    records = []
    for identity, (body, row, case) in captured.items():
        started = time.perf_counter()
        path = output / "audio" / f"{identity}.wav"
        row.update(
            path=str(path.relative_to(output)), sha256=hashlib.sha256(body).hexdigest()
        )
        try:
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(body)
            temporary.replace(path)
            samples, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
            finite = bool(np.isfinite(samples).all())
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
            row["audio"] = {
                "duration_s": len(samples) / sr,
                "sample_rate": sr,
                "channels": samples.shape[1],
                "finite": finite,
                "peak": peak,
                "rms": rms,
                "asr_readable": bool(samples.size and finite),
                "signal_valid": samples.size > 0
                and finite
                and peak > 0.001
                and rms > 1e-5,
            }
        except Exception as error:
            row["audio"] = {
                "asr_readable": False,
                "signal_valid": False,
                "retention_or_decode_error": repr(error),
            }
            if not path.exists():
                row["retention_error"] = repr(error)
        finished = time.perf_counter()
        row["client_processing_ms"] = (finished - started) * 1000
        row["deferred_retention_delay_ms"] = (
            finished - row["network_finished_perf"]
        ) * 1000
        row["retained_at"] = utc_now()
        records.append(
            {
                "id": identity,
                "path": row["path"],
                "sha256": row["sha256"],
                "language": case["language"],
                "omnilingual_language": case.get("omnilingual_language"),
                "expected": case["text"],
                "text": case["text"],
                "phase": row["phase"],
                "run": label,
                "voice_profile": case.get("voice_profile", "design"),
                "reference_language": case.get("reference_language"),
                "translation_status": case.get("translation_status", "unverified"),
                "audio": row["audio"],
                "asr_mapping_is_proxy": case.get("asr_mapping_is_proxy"),
                "asr_mapping_note": case.get("asr_mapping_note"),
                "source": case.get("source"),
                "case": case,
                "validation_only": case.get("validation_only", False),
                "expected_http_status": case.get("expected_http_status", 200),
                "generation_options": row.get("normalized_worker_options"),
                "reference_identity": row.get("reference_identity"),
                "request": {"text": case["text"], "phase": row["phase"]},
            }
        )
    return records


def phase_summary(rows, phase, wall_s, offered_count):
    if phase.get("validation_only"):
        return {
            "phase": phase,
            "measurement_kind": "API-boundary validation only",
            "attempts": len(rows),
            "offered_count": offered_count,
            "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
            "http200": sum(row.get("status") == 200 for row in rows),
            "expected_validation_status_matches": sum(
                row.get("status") == row["expected_http_status"] for row in rows
            ),
            "validation_mismatches": [
                {
                    "id": row["id"],
                    "case_id": row["case"]["id"],
                    "actual": row.get("status"),
                    "expected": row["expected_http_status"],
                }
                for row in rows
                if row.get("status") != row["expected_http_status"]
            ],
            "generation_offered_count": 0,
            "throughput_eligible": False,
            "all_offers_completed_200_within_2s": False,
            "phase_wall_s": wall_s,
            "http200_per_phase_wall_s": None,
            "quality": "Boundary cases are excluded from throughput and ASR quality",
        }
    successful = [row for row in rows if row.get("status") == 200]
    raw = [row["complete_ms"] for row in successful]
    cohort = [row["completion_from_barrier_ms"] for row in successful]
    return {
        "phase": phase,
        "attempts": len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "http200": len(successful),
        "response_source_counts": dict(
            Counter(
                row.get("headers", {}).get(
                    "x-omnivoice-response-source", "unreported_legacy"
                )
                for row in successful
            )
        ),
        "inference_executed_counts": dict(
            Counter(
                row.get("headers", {}).get(
                    "x-omnivoice-inference-executed", "unreported_legacy"
                )
                for row in successful
            )
        ),
        "unique_origin_request_ids": len(
            {
                row.get("headers", {}).get("x-omnivoice-origin-request-id")
                for row in successful
                if row.get("headers", {}).get("x-omnivoice-origin-request-id")
            }
        ),
        "unique_wav_sha256": len({row["sha256"] for row in successful}),
        "signal_valid": sum(
            row.get("audio", {}).get("signal_valid", False) for row in successful
        ),
        "full_wav_within_500ms": sum(value <= 500 for value in raw),
        "all_offers_completed_200_within_2s": len(successful) == offered_count
        and all(value <= 2000 for value in cohort),
        "http200_completed_from_barrier_within_2s": sum(
            value <= 2000 for value in cohort
        ),
        "http200_p50_ms": percentile(raw, 50),
        "http200_p95_ms": percentile(raw, 95),
        "http200_p99_ms": percentile(raw, 99),
        "http200_max_ms": max(raw) if raw else None,
        "http200_cohort_max_ms": max(cohort) if cohort else None,
        "http200_cohort_p95_ms": percentile(cohort, 95),
        "phase_wall_s": wall_s,
        "http200_per_phase_wall_s": len(successful) / wall_s if wall_s else None,
        "dispatch_spread_ms": max(row["dispatched_from_barrier_ms"] for row in rows)
        - min(row["dispatched_from_barrier_ms"] for row in rows),
        "new_tcp_connections": sum(
            "connection.connect_tcp.started" in row.get("trace", {}) for row in rows
        ),
        "unique_effective_inputs": len({row["effective_input_sha256"] for row in rows}),
        "quality": (
            "Blind ASR and listening review pending; HTTP200 is not word validity"
        ),
        "latency_denominator": "HTTP200 only; all rejections/errors remain explicit",
    }


async def execute(args, process):
    url = f"http://127.0.0.1:{args.port}"
    health = await wait_ready(url, process, args.startup_timeout)
    write_json(args.output / "health-before.json", health)
    checks = validate_runtime(args, health)
    metadata = json.loads((args.output / "metadata.json").read_text())
    metadata["runtime_validation"] = checks
    write_json(args.output / "metadata.json", metadata)
    by_id, references, phases = load_inputs(
        args.corpus, args.plan, args.references, args.steps
    )
    summaries, quality, prompt_ids = {}, [], {}
    current_phase = "setup"
    probe_stop = asyncio.Event()

    async def probes():
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while not probe_stop.is_set():
                started = time.perf_counter()
                row = {"at": utc_now(), "phase": current_phase}
                try:
                    response = await client.get(url + "/health")
                    row.update(status=response.status_code, body=response.json())
                except (httpx.HTTPError, ValueError) as error:
                    row.update(status=None, error=repr(error))
                row["complete_ms"] = (time.perf_counter() - started) * 1000
                append_json(args.output / "health-probes.jsonl", row)
                try:
                    await asyncio.wait_for(probe_stop.wait(), timeout=0.5)
                except TimeoutError:
                    pass

    def save_quality():
        ordered = sorted(quality, key=lambda row: row["id"])
        write_json(args.output / "quality-manifest.json", ordered)
        write_json(
            args.output / "quality-asr-ready.json",
            [
                row
                for row in ordered
                if row.get("omnilingual_language")
                and row["audio"].get("asr_readable")
                and not row.get("validation_only", False)
            ],
        )
        write_json(
            args.output / "quality-unmapped.json",
            [row for row in ordered if not row.get("omnilingual_language")],
        )
        write_json(
            args.output / "quality-invalid-audio.json",
            [row for row in ordered if not row["audio"].get("asr_readable")],
        )
        write_json(
            args.output / "quality-validation-only-audio.json",
            [row for row in ordered if row.get("validation_only", False)],
        )

    limits = httpx.Limits(
        max_connections=args.connections,
        max_keepalive_connections=args.connections,
        keepalive_expiry=args.keepalive_expiry,
    )
    async with httpx.AsyncClient(
        timeout=args.request_timeout, limits=limits, trust_env=False
    ) as client:
        language_response = await client.get(url + "/languages")
        language_response.raise_for_status()
        language_body = language_response.json()
        supported_ids = {row["id"] for row in language_body["languages"]}
        unsupported = [
            case for case in by_id.values() if case["api_language"] not in supported_ids
        ]
        write_json(
            args.output / "language-support.json",
            {
                "actual_server_response": language_body,
                "corpus_languages": sorted(
                    {case["language"] for case in by_id.values()}
                ),
                "unsupported_cases": unsupported,
                "check": "Exact API code membership, no silent upstream fallback",
            },
        )
        assert not unsupported, (
            "Corpus contains unsupported API language IDs; "
            "inspect language-support.json"
        )
        probe_task = asyncio.create_task(probes())

        async def register(case):
            key = registration_key(case, references, args.steps)
            if key in prompt_ids:
                return
            reference_id = case["reference_id"]
            reference = references[reference_id]
            identity = reference_identity(case, references, args.steps)
            path = Path(reference["path"])
            started = time.perf_counter()
            response = await client.post(
                url + "/clone-prompts",
                data={
                    "ref_text": reference["text"],
                    "preprocess_prompt": form_values(
                        {"v": identity["preprocess_prompt"]}
                    )["v"],
                },
                files={"ref_audio": (path.name, path.read_bytes(), "audio/wav")},
            )
            row = {
                "reference_id": reference_id,
                "status": response.status_code,
                "complete_ms": (time.perf_counter() - started) * 1000,
                "reference": reference,
                "registration_identity": identity,
                "reference_preprocess_prompt": identity["preprocess_prompt"],
                "body": response.json(),
            }
            append_json(args.output / "registrations.jsonl", row)
            response.raise_for_status()
            prompt_ids[key] = response.json()["prompt_id"]

        async def request(phase, index, case, barrier, barrier_unix, offset_s):
            scheduled = barrier + offset_s
            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
            started = time.perf_counter()
            identity = f"{phase['id']}-{index:05d}"
            options = generation_options(case, args.steps)
            params = {
                "mode": case["mode"],
                "text": case["text"],
                "language": case["api_language"],
                **form_values(options),
            }
            if case.get("instruct"):
                params["instruct"] = case["instruct"]
            files = None
            transport = None
            reference_spec = None
            if case["mode"] == "clone":
                key = case["reference_id"]
                transport = reference_mode(case, phase)
                reference_spec = reference_identity(case, references, args.steps)
                if transport == "registered":
                    params["prompt_id"] = prompt_ids[
                        registration_key(case, references, args.steps)
                    ]
                else:
                    reference = references[key]
                    path = Path(reference["path"])
                    params.update(ref_text=reference["text"])
                    files = {"ref_audio": (path.name, path.read_bytes(), "audio/wav")}
            signature = effective_signature(case, references, args.steps)
            row = {
                "id": identity,
                "phase": phase["id"],
                "arm": args.label,
                "case": case,
                "parameters": params,
                "normalized_worker_options": options,
                "reference_identity": reference_spec,
                "reference_transport": transport,
                "reference_preprocessing_applied_at": "registration"
                if transport == "registered"
                else "generation_upload"
                if transport == "upload"
                else "not_used",
                "validation_only": case.get("validation_only", False),
                "expected_http_status": case.get("expected_http_status", 200),
                "started_at": utc_now(),
                "scheduled_offset_ms": offset_s * 1000,
                "dispatched_from_barrier_ms": (started - barrier) * 1000,
                "dispatch_lateness_ms": max(0, started - scheduled) * 1000,
                "effective_input_sha256": hashlib.sha256(
                    signature.encode()
                ).hexdigest(),
                "trace": {},
            }

            async def trace(name, info):
                # Record timing only, never request headers or transport credentials.
                if name.endswith(("started", "complete", "failed")):
                    row["trace"][name] = (time.perf_counter() - started) * 1000

            network_finished = None
            try:
                response = await client.post(
                    url + "/generate",
                    data=params,
                    files=files,
                    extensions={"trace": trace},
                )
                network_finished = time.perf_counter()
                row.update(
                    status=response.status_code,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower().startswith("x-omnivoice-")
                    },
                )
                if response.status_code == 200:
                    # Keep immutable complete bodies in RAM. Retention/inspection must
                    # not delay reads of other responses in the timed cohort.
                    captured_audio[identity] = (response.content, row, case)
                    row["response_bytes"] = len(response.content)
                else:
                    row["error_body"] = response.text[:1500]
            except asyncio.CancelledError:
                row.update(status=None, error="Phase deadline/client task cancellation")
                raise
            except Exception as error:
                row.setdefault("status", None)
                row["error"] = repr(error)
            finally:
                finished = time.perf_counter()
                body_finished = network_finished or finished
                row.update(
                    complete_ms=(body_finished - started) * 1000,
                    completion_from_barrier_ms=(body_finished - barrier) * 1000,
                    client_processing_ms=(finished - body_finished) * 1000,
                    completed_at=utc_now(),
                    phase_barrier_unix=barrier_unix,
                )
                row["network_finished_perf"] = body_finished
                completed_rows[identity] = row
            return row

        try:
            for phase in phases:
                current_phase = phase["id"]
                selected = expand_cases(phase, by_id)
                for case in selected:
                    if (
                        case["mode"] == "clone"
                        and reference_mode(case, phase) == "registered"
                    ):
                        await register(case)
                preconnect = phase.get("preconnect", 0)
                if preconnect:
                    started = time.perf_counter()
                    responses = await asyncio.gather(
                        *[client.get(url + "/health") for _ in range(preconnect)]
                    )
                    append_json(
                        args.output / "connection-warmups.jsonl",
                        {
                            "phase": phase["id"],
                            "requests": preconnect,
                            "http200": sum(
                                response.status_code == 200 for response in responses
                            ),
                            "wall_s": time.perf_counter() - started,
                            "note": (
                                "Preconnect request count is not proven "
                                "connection count"
                            ),
                        },
                    )
                completed_rows = {}
                captured_audio = {}
                barrier = time.perf_counter() + 0.05
                barrier_unix = time.time() + 0.05
                offsets = []
                offset = 0.0
                rng = random.Random(phase.get("arrival_seed", 20261103))
                for index in range(len(selected)):
                    offsets.append(offset)
                    if phase["kind"] == "staggered":
                        rate = phase["arrival_rate_rps"]
                        offset += (
                            rng.expovariate(rate)
                            if phase.get("arrival_distribution") == "poisson"
                            else 1 / rate
                        )
                tasks = []
                try:
                    if phase["kind"] == "sequential":

                        async def sequential():
                            for index, case in enumerate(selected):
                                await request(
                                    phase, index, case, barrier, barrier_unix, 0
                                )

                        await asyncio.wait_for(
                            sequential(),
                            timeout=phase.get("deadline_s", args.phase_timeout),
                        )
                    elif phase["kind"] == "closed_loop":
                        semaphore = asyncio.Semaphore(phase["concurrency"])

                        async def limited(index, case):
                            async with semaphore:
                                return await request(
                                    phase, index, case, barrier, barrier_unix, 0
                                )

                        tasks = [
                            asyncio.create_task(limited(index, case))
                            for index, case in enumerate(selected)
                        ]
                        await asyncio.wait_for(
                            asyncio.gather(*tasks),
                            timeout=phase.get("deadline_s", args.phase_timeout),
                        )
                    else:
                        tasks = [
                            asyncio.create_task(
                                request(
                                    phase,
                                    index,
                                    case,
                                    barrier,
                                    barrier_unix,
                                    offsets[index],
                                )
                            )
                            for index, case in enumerate(selected)
                        ]
                        await asyncio.wait_for(
                            asyncio.gather(*tasks),
                            timeout=phase.get("deadline_s", args.phase_timeout),
                        )
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    rows = list(completed_rows.values())
                    network_wall = max(
                        (row["completion_from_barrier_ms"] / 1000 for row in rows),
                        default=0.001,
                    )
                    current_phase = "retention:" + phase["id"]
                    retention_started = time.perf_counter()
                    quality.extend(
                        await asyncio.to_thread(
                            retain_wave_batch, args.output, args.label, captured_audio
                        )
                    )
                    for row in rows:
                        append_json(args.output / "requests.jsonl", row)
                    retention_wall = time.perf_counter() - retention_started
                    summaries[phase["id"]] = (
                        phase_summary(
                            rows,
                            phase,
                            max(0.001, network_wall),
                            len(selected),
                        )
                        if rows
                        else {"attempts": 0, "phase": phase}
                    )
                    summaries[phase["id"]]["retention_wall_s"] = retention_wall
                    summaries[phase["id"]]["total_wall_including_retention_s"] = (
                        time.perf_counter() - barrier
                    )
                    summaries[phase["id"]]["offered_count"] = len(selected)
                    summaries[phase["id"]]["not_dispatched_count"] = len(
                        selected
                    ) - len(rows)
                    write_json(args.output / "summary.json", summaries)
                    save_quality()
                    print(json.dumps(summaries[phase["id"]]), flush=True)
                if any(
                    row.get("status")
                    not in (
                        {row["expected_http_status"]}
                        if row.get("validation_only")
                        else {200, 429}
                    )
                    or row.get("error")
                    for row in completed_rows.values()
                ):
                    raise RuntimeError("Unexpected outcome; stop and retain evidence")
        finally:
            probe_stop.set()
            await probe_task
            save_quality()
            try:
                write_json(
                    args.output / "health-after.json",
                    (await client.get(url + "/health")).json(),
                )
            except Exception as error:
                write_json(args.output / "health-after.json", {"error": repr(error)})


def main():
    if sys.argv[1:] == ["--collect-hardware"]:
        print(json.dumps(collect_hardware()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["source", "corpus", "plan", "output"]:
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runner", default="hybrid")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--collect-ms", type=float, default=5.0)
    parser.add_argument("--max-batch-requests", type=int, default=8)
    parser.add_argument("--max-pending-requests", type=int, default=16)
    parser.add_argument("--max-pending-gpu-tasks", type=int, default=4)
    parser.add_argument("--max-batch-target-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-conditioning-tokens", type=int, default=4096)
    parser.add_argument("--batch-bucket-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--max-cuda-graphs", type=int, default=8)
    parser.add_argument("--audio-decode-batch-size", type=int, default=1)
    parser.add_argument("--server-extra-flags-json", default="[]")
    parser.add_argument("--expected-health-json", default="{}")
    parser.add_argument("--response-cache-max-bytes", type=int, default=None)
    parser.add_argument("--response-cache-ttl", type=float, default=None)
    parser.add_argument("--startup-seed", type=int)
    parser.add_argument("--connections", type=int, default=128)
    parser.add_argument("--keepalive-expiry", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--phase-timeout", type=float, default=600)
    args = parser.parse_args()
    file_limits = configure_file_limits(args.connections)
    for name in ["source", "corpus", "plan", "output", "references"]:
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    cases, refs, phases = load_inputs(
        args.corpus, args.plan, args.references, args.steps
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    write_json(args.output / "corpus.json", list(cases.values()))
    write_json(args.output / "plan.json", phases)
    write_json(args.output / "reference-manifest.json", list(refs.values()))
    command = build_command(args)
    hardware = json.loads(
        subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--collect-hardware"],
            text=True,
            timeout=60,
        )
    )
    write_json(
        args.output / "metadata.json",
        {
            "started_at": utc_now(),
            "label": args.label,
            "source_ref": args.source_ref,
            "source": source_identity(args.source),
            "hardware": hardware,
            "file_descriptor_limits": file_limits,
            "command": command,
            "protocol_sha256": digest(Path(__file__).resolve()),
            "runtime_sha256": digest(HERE / "runtime.py"),
            "seed_wrapper_sha256": digest(HERE / "seeded_server.py"),
            "corpus_sha256": digest(args.corpus),
            "plan_sha256": digest(args.plan),
            "requested_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "profile": {
                "generation_defaults": generation_defaults(args.steps),
                "per_case_options": "Every worker_options value is posted and logged",
                "identity_schema": "V5 worker fields + text/mode/language/instruct "
                "+ reference WAV SHA/transcript/preprocess; random UUIDs excluded",
                "return": "complete WAV",
            },
            "interpretation": "All language/mode/reference mappings are explicit. "
            "Cached voices do not imply cached generated audio; "
            "repeated input is labeled. "
            "Only all-HTTP200 completion from the common barrier "
            "measures a whole burst. "
            "No ASR GPU inference runs during TTS measurement.",
        },
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LINGUA_") and key != "WORKER_ID"
    }
    environment.update(
        PYTHONPATH=str(args.source / "src"),
        TOKENIZERS_PARALLELISM="false",
        OMNIVOICE_LOG_LEVEL="INFO",
    )
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=args.source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        outcome = {"status": "failed", "server_pid": process.pid}
        try:
            limit_path = Path(f"/proc/{process.pid}/limits")
            if limit_path.exists():
                process_limits = limit_path.read_text()
                (args.output / "server-process-limits.txt").write_text(process_limits)
                max_open = next(
                    line.split()
                    for line in process_limits.splitlines()
                    if line.startswith("Max open files")
                )
                assert (
                    max_open[3] == "unlimited"
                    or int(max_open[3]) >= file_limits["minimum_required"]
                ), "Child server did not inherit sufficient file descriptors"
            asyncio.run(execute(args, process))
            outcome["status"] = "completed"
        except BaseException as error:
            outcome["error"] = repr(error)
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            outcome.update(finished_at=utc_now(), server_exit=process.returncode)
            write_json(args.output / "outcome.json", outcome)


if __name__ == "__main__":
    main()
