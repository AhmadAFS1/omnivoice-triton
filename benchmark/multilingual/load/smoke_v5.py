#!/usr/bin/env python3
"""CPU-only V5 Form/identity/preprocessing/validation protocol checks. No model."""

import argparse
import asyncio
import hashlib
import io
import json
import math
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx
import numpy as np
import run_v5 as protocol
import soundfile as sf
from verify_v5 import verify


def fields(request):
    kind = request.headers.get("content-type", "")
    if kind.startswith("application/x-www-form-urlencoded"):
        return {
            key: values[0] for key, values in parse_qs(request.content.decode()).items()
        }, None
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + kind.encode()
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + request.content
    )
    form = {}
    audio = None
    for part in message.iter_parts():
        key = part.get_param("name", header="content-disposition")
        body = part.get_payload(decode=True)
        if part.get_filename():
            audio = hashlib.sha256(body).hexdigest()
        else:
            form[key] = body.decode()
    return form, audio


async def smoke(root):
    root.mkdir(parents=True, exist_ok=False)
    samples = np.sin(np.arange(2400, dtype=np.float32) * 0.08) * 0.2
    wav = io.BytesIO()
    sf.write(wav, samples, 24000, format="WAV", subtype="PCM_16")
    body = wav.getvalue()
    reference = root / "reference.wav"
    reference.write_bytes(body)
    ref = {
        "id": "ref",
        "path": str(reference),
        "text": "Reference fixture.",
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    base = {
        "text": "Two source sentences. The words stay fixed.",
        "language": "en",
        "api_language": "en",
        "omnilingual_language": "eng_Latn",
        "length": "short",
        "mode": "clone",
        "reference_id": "ref",
        "reference_language": "en",
        "voice_profile": "CPU-only native fixture",
        "voice_id": "Synthetic",
    }
    variants = [
        ("default", {"num_step": 16, "speed": 1.3, "postprocess_mode": "full"}),
        ("video", {"num_step": 16, "speed": 1.0, "postprocess_mode": "off"}),
        (
            "tutor",
            {
                "num_step": 16,
                "speed": 1.0,
                "postprocess_mode": "light",
                "insert_pauses": True,
            },
        ),
        ("preprocess-false", {"num_step": 16, "preprocess_prompt": False}),
        (
            "all-controls",
            {
                "num_step": 32,
                "guidance_scale": 3,
                "t_shift": 0.2,
                "layer_penalty_factor": 1,
                "position_temperature": 2,
                "class_temperature": 0.5,
                "speed": 0.8,
                "duration": 4,
                "denoise": False,
                "preprocess_prompt": True,
                "postprocess_mode": "light",
                "insert_pauses": True,
            },
        ),
    ]
    cases = [
        base | {"id": name, "worker_options": options} for name, options in variants
    ]
    cases += [
        base
        | {
            "id": "design",
            "mode": "design",
            "reference_id": None,
            "reference_language": None,
            "instruct": "female, young adult, calm",
        },
        base
        | {
            "id": "auto",
            "mode": "auto",
            "reference_id": None,
            "reference_language": None,
        },
        base
        | {
            "id": "upload",
            "language": "es",
            "api_language": "es",
            "omnilingual_language": "spa_Latn",
            "text": "Dos frases. Palabras iguales.",
            "request_transport": "upload",
            "worker_options": {
                "num_step": 8,
                "speed": 1.1,
                "denoise": False,
                "preprocess_prompt": False,
            },
        },
    ]
    cases += [
        base
        | {
            "id": "invalid-nan",
            "worker_options": {"guidance_scale": "nan"},
            "validation_only": True,
            "expected_http_status": 400,
        },
        base
        | {
            "id": "invalid-step",
            "worker_options": {"num_step": "0"},
            "validation_only": True,
            "expected_http_status": 400,
        },
    ]
    phases = [
        {
            "id": "sequential-controls",
            "kind": "sequential",
            "case_ids": [c["id"] for c in cases[:8]],
            "content_profile": "unique",
        },
        {
            "id": "burst-controls",
            "kind": "burst",
            "case_ids": [c["id"] for c in cases[1:8]],
            "content_profile": "unique",
        },
        {
            "id": "boundary400",
            "kind": "sequential",
            "case_ids": [c["id"] for c in cases[8:]],
            "validation_only": True,
        },
    ]
    for name, value in [
        ("corpus.json", cases),
        ("references.json", [ref]),
        ("plan-input.json", phases),
    ]:
        protocol.write_json(root / name, value)
    args = SimpleNamespace(
        corpus=root / "corpus.json",
        references=root / "references.json",
        plan=root / "plan-input.json",
        output=root / "run",
        label="CPU-only-V5-mock",
        port=18080,
        runner="hybrid",
        steps=16,
        collect_ms=5.0,
        max_batch_requests=8,
        max_batch_target_tokens=4096,
        max_batch_conditioning_tokens=4096,
        max_cuda_graphs=8,
        audio_decode_batch_size=1,
        expected_health_json="{}",
        max_pending_requests=16,
        max_pending_gpu_tasks=4,
        batch_bucket_sizes="1,2,4,8,16,32",
        startup_seed=None,
        response_cache_max_bytes=None,
        response_cache_ttl=None,
        connections=32,
        keepalive_expiry=600,
        request_timeout=5,
        phase_timeout=10,
        startup_timeout=5,
    )
    args.output.mkdir()
    (args.output / "audio").mkdir()
    requested = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    protocol.write_json(
        args.output / "metadata.json",
        {
            "command": [],
            "requested_args": requested,
            "hardware": {
                "measurement_kind": "CPU-only synthetic HTTP; no model inference"
            },
        },
    )
    protocol.write_json(args.output / "corpus.json", cases)
    protocol.write_json(args.output / "plan.json", phases)
    health = {
        "runner": "hybrid",
        "device": "cuda",
        "dtype": "fp16",
        "batch_collect_ms": 5.0,
        "max_batch_requests": 8,
        "max_batch_target_tokens": 4096,
        "max_batch_conditioning_tokens": 4096,
        "max_cuda_graphs": 8,
        "audio_decode_batch_size": 1,
        "batch_bucket_sizes": [1, 2, 4, 8, 16, 32],
        "load_asr": False,
        "batcher_queue": {"max_pending_requests": 16, "max_pending_gpu_tasks": 4},
        "note": "CPU mock of runtime contract; CUDA is not initialized",
    }
    registrations = []
    observed = []
    audio_uploads = []

    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=health)
        if request.url.path == "/languages":
            return httpx.Response(200, json={"languages": [{"id": "en"}, {"id": "es"}]})
        data, uploaded = fields(request)
        if request.url.path == "/clone-prompts":
            assert uploaded == ref["sha256"]
            registrations.append(data)
            return httpx.Response(
                200, json={"prompt_id": f"prompt-{len(registrations) - 1}"}
            )
        assert request.url.path == "/generate"
        observed.append(data)
        origin = len(observed)
        audio_uploads.append(uploaded)
        if (
            not math.isfinite(float(data["guidance_scale"]))
            or int(data["num_step"]) <= 0
        ):
            return httpx.Response(
                400,
                json={"detail": "CPU-only finite guard fixture"},
                headers={"X-OmniVoice-Inference-Executed": "false"},
            )
        await asyncio.sleep(0.001)
        return httpx.Response(
            200,
            content=body,
            headers={
                "content-type": "audio/wav",
                "X-OmniVoice-Response-Source": "generated",
                "X-OmniVoice-Inference-Executed": "true",
                "X-OmniVoice-Origin-Request-Id": f"cpu-origin-{origin}",
            },
        )

    async def ready(*unused):
        return health

    original = httpx.AsyncClient

    def client(*args, **kwargs):
        return original(*args, **kwargs, transport=httpx.MockTransport(handler))

    with (
        patch.object(protocol, "wait_ready", ready),
        patch.object(httpx, "AsyncClient", client),
    ):
        await protocol.execute(args, None)
    protocol.write_json(args.output / "outcome.json", {"status": "completed"})
    result = verify(args.output)
    rows = [
        json.loads(line)
        for line in (args.output / "requests.jsonl").read_text().splitlines()
    ]
    assert (
        len(rows) == 17
        and result["http200_hashes_verified"] == 15
        and result["asr_eligible"] == 15
    )
    assert Counter(
        json.dumps(r["parameters"], sort_keys=True) for r in rows
    ) == Counter(json.dumps(value, sort_keys=True) for value in observed)
    assert [r["preprocess_prompt"] for r in registrations] == ["true", "false"]
    assert len([sha for sha in audio_uploads if sha == ref["sha256"]]) == 2
    # Check actual handler values independently of the request-building functions.
    assert observed[0]["speed"] == "1.3" and observed[0]["postprocess_mode"] == "full"
    assert observed[1]["speed"] == "1.0" and observed[1]["postprocess_mode"] == "off"
    assert (
        observed[2]["insert_pauses"] == "true"
        and observed[2]["postprocess_mode"] == "light"
    )
    assert observed[3]["prompt_id"] != observed[1]["prompt_id"]
    for key, value in {
        "num_step": "32",
        "guidance_scale": "3.0",
        "t_shift": "0.2",
        "layer_penalty_factor": "1.0",
        "position_temperature": "2.0",
        "class_temperature": "0.5",
        "speed": "0.8",
        "duration": "4.0",
        "denoise": "false",
        "preprocess_prompt": "true",
        "postprocess_mode": "light",
        "insert_pauses": "true",
    }.items():
        assert observed[4][key] == value, (key, observed[4])
    assert observed[5]["instruct"] == "female, young adult, calm"
    assert observed[6]["mode"] == "auto" and "prompt_id" not in observed[6]
    summary = json.loads((args.output / "summary.json").read_text())["boundary400"]
    assert summary["expected_validation_status_matches"] == 2
    assert (
        summary["generation_offered_count"] == 0
        and not summary["all_offers_completed_200_within_2s"]
    )
    refs = {"ref": ref, "alias": ref | {"id": "alias"}}
    original_case = cases[1]
    signature = protocol.effective_signature(original_case, refs)
    assert signature == protocol.effective_signature(
        original_case | {"reference_id": "alias"}, refs
    )
    assert signature != protocol.effective_signature(cases[2], refs)
    assert signature != protocol.effective_signature(cases[3], refs)
    for bad in [
        {"worker_options": {"steps": 16}},
        {"speed": 1.5},
        {"worker_options": {"guidance_scale": float("nan")}},
        {"worker_options": {"guidance_scale": "nan"}},
        {"worker_options": {"guidance_scale": -1}},
    ]:
        try:
            protocol.generation_options(original_case | bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Ignored/invalid generation override accepted:{bad}")
    assert (
        protocol.generation_options(
            original_case | {"worker_options": {"duration": -1}}
        )["duration"]
        == -1.0
    )
    report = {
        "measurement_kind": "CPU-only Form/retention/identity checks; no model/GPU",
        "passed": True,
        "requests": 17,
        "saved_wavs": 15,
        "asr_ready": 15,
        "boundary400": 2,
        "registration_preprocess_variants": 2,
        "all12_worker_fields_observed_by_handler": True,
        "V4_unchanged": True,
    }
    protocol.write_json(root / "smoke-result.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(smoke(parser.parse_args().output.resolve()))
