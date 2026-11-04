#!/usr/bin/env python3
"""CPU-only synthetic HTTP protocol check. No TTS, ASR or GPU model is loaded."""

import argparse
import asyncio
import hashlib
import io
import json
import resource
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import run as protocol
import soundfile as sf
from verify import verify


async def smoke(root):
    root.mkdir(parents=True, exist_ok=False)
    # A separate CPU process can change its own limits without affecting our host.
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,run; print(json.dumps(run.configure_file_limits(6000)))",
        ],
        cwd=Path(protocol.__file__).parent,
        text=True,
    )
    checked_limits = json.loads(result)
    assert checked_limits["after_soft_hard"][0] == resource.RLIM_INFINITY or (
        checked_limits["after_soft_hard"][0] >= 13024
    )
    with patch.object(resource, "getrlimit", return_value=(1024, 2048)):
        try:
            protocol.configure_file_limits(1200)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Insufficient hard file limit did not fail preflight")
    protocol.write_json(root / "cpu-file-limit-check.json", checked_limits)
    samples = np.sin(np.arange(2400, dtype=np.float32) * 0.08) * 0.2
    wav = io.BytesIO()
    sf.write(wav, samples, 24000, format="WAV", subtype="PCM_16")
    body = wav.getvalue()
    reference = root / "reference.wav"
    reference.write_bytes(body)
    corpus = [
        {
            "id": "en",
            "text": "Hello there.",
            "language": "en",
            "api_language": "en",
            "omnilingual_language": "eng_Latn",
            "length": "short",
            "mode": "clone",
            "reference_id": "en",
            "reference_language": "en",
            "voice_profile": "local_target_language_reference",
        },
        {
            "id": "es",
            "text": "Hola, ¿cómo estás?",
            "language": "es",
            "api_language": "es",
            "omnilingual_language": "spa_Latn",
            "length": "short",
            "mode": "clone",
            "reference_id": "en",
            "reference_language": "en",
            "voice_profile": "explicit_cross_language_reference",
        },
        {
            "id": "unmapped",
            "text": "Synthetic unmapped fixture.",
            "language": "zz",
            "api_language": "zz",
            "omnilingual_language": None,
            "length": "short",
            "mode": "design",
        },
        {
            "id": "reject",
            "text": "reject this fixture",
            "language": "en",
            "api_language": "en",
            "omnilingual_language": "eng_Latn",
            "length": "short",
            "mode": "design",
        },
    ]
    refs = [
        {
            "id": "en",
            "path": reference.name,
            "text": "Reference fixture.",
            "sha256": hashlib.sha256(body).hexdigest(),
        }
    ]
    phases = [
        {"id": "sequential", "kind": "sequential", "content_profile": "unique"},
        {"id": "burst4", "kind": "burst", "content_profile": "unique"},
        {
            "id": "staggered",
            "kind": "staggered",
            "arrival_rate_rps": 100,
            "arrival_distribution": "poisson",
            "arrival_seed": 4,
        },
    ]
    for name, data in [
        ("corpus.json", corpus),
        ("references.json", refs),
        ("input-plan.json", {"phases": phases}),
    ]:
        protocol.write_json(root / name, data)
    args = SimpleNamespace(
        corpus=root / "corpus.json",
        references=root / "references.json",
        plan=root / "input-plan.json",
        output=root / "run",
        label="cpu-mock",
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
        connections=16,
        keepalive_expiry=600,
        request_timeout=5,
        phase_timeout=10,
        startup_timeout=5,
    )
    args.output.mkdir()
    (args.output / "audio").mkdir()
    protocol.write_json(
        args.output / "metadata.json",
        {
            "hardware": {
                "measurement_kind": "CPU-only fake HTTP; no GPU/model inference"
            },
            "command": [],
            "requested_args": vars(args)
            | {
                "corpus": str(args.corpus),
                "references": str(args.references),
                "plan": str(args.plan),
                "output": str(args.output),
            },
        },
    )
    protocol.write_json(args.output / "corpus.json", corpus)
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
        "note": "Synthetic health response for protocol validation only",
    }
    registrations = []

    async def handler(request):
        if request.url.path in ["/health", "/healthz"]:
            return httpx.Response(200, json=health)
        if request.url.path == "/languages":
            return httpx.Response(
                200, json={"languages": [{"id": "en"}, {"id": "es"}, {"id": "zz"}]}
            )
        if request.url.path == "/clone-prompts":
            registrations.append(request)
            return httpx.Response(200, json={"prompt_id": "mock-reference"})
        await asyncio.sleep(0.001)
        if b"reject+this" in request.content:
            return httpx.Response(429, json={"detail": "Synthetic saturation"})
        return httpx.Response(200, content=body, headers={"content-type": "audio/wav"})

    async def ready(*args):
        return health

    original_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return original_client(*args, **kwargs, transport=httpx.MockTransport(handler))

    with (
        patch.object(protocol, "wait_ready", ready),
        patch.object(httpx, "AsyncClient", client),
    ):
        await protocol.execute(args, None)
    protocol.write_json(args.output / "outcome.json", {"status": "completed"})
    assert len(registrations) == 1, (
        "Native and cross-language cases must reuse explicitref"
    )
    result = verify(args.output)
    assert result["attempts"] == 12 and result["http200_hashes_verified"] == 9
    assert result["asr_eligible"] == 6 and result["unmapped_requires_review"] == 3
    summaries = json.loads((args.output / "summary.json").read_text())
    assert not summaries["burst4"]["all_offers_completed_200_within_2s"]
    bad_args = SimpleNamespace(**(vars(args) | {"audio_decode_batch_size": 8}))
    try:
        protocol.validate_runtime(bad_args, health)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing forwarded codec flag did not fail runtime check")
    command_args = SimpleNamespace(
        **(
            vars(args)
            | {
                "model": "mock-model",
                "server_extra_flags_json": "[]",
                "audio_decode_batch_size": 8,
                "response_cache_max_bytes": 67108864,
                "response_cache_ttl": 0.0,
                "startup_seed": 20261103,
            }
        )
    )
    command = protocol.build_command(command_args)
    assert Path(command[1]).name == "seeded_server.py"
    for flag, value in [
        ("--audio-decode-batch-size", "8"),
        ("--response-cache-max-bytes", "67108864"),
        ("--response-cache-ttl", "0.0"),
        ("--seed", "20261103"),
    ]:
        assert command[command.index(flag) + 1] == value
    cache_args = SimpleNamespace(
        **(
            vars(args)
            | {
                "response_cache_max_bytes": 67108864,
                "response_cache_ttl": 0.0,
            }
        )
    )
    cache_health = {**health, "response_cache": {"max_bytes": 67108864, "ttl_s": 0.0}}
    protocol.validate_runtime(cache_args, cache_health)
    try:
        protocol.validate_runtime(
            cache_args, {**health, "response_cache": {"max_bytes": 0, "ttl_s": 0.0}}
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Cache flag recorded but not applied was accepted")
    bad_plan = root / "bad-plan.json"
    protocol.write_json(
        bad_plan,
        [
            {
                "id": "misleading-unique",
                "kind": "burst",
                "case_ids": ["en"],
                "requests": 2,
                "content_profile": "unique",
            }
        ],
    )
    try:
        protocol.load_inputs(args.corpus, bad_plan, args.references)
    except AssertionError:
        pass
    else:
        raise AssertionError("Repeated input falsely qualified as unique")
    protocol.write_json(
        root / "smoke-result.json",
        {
            "measurement_kind": (
                "CPU-only synthetic protocol validation; no GPU/model inference"
            ),
            "passed": True,
            "attempts": 12,
            "retained_http200_wavs": 9,
            "asr_ready": 6,
            "unmapped_review": 3,
            "checks": [
                "Explicit native/cross-language reference reuse",
                "Sequential/burst/Poisson arrival accounting",
                "Every200 WAV/hash retained;429 does not meet fullcohort2starget",
                "Unmapped ASR remains visible",
                "Runtime flag mismatch rejected",
                "Repeatedcontent cannot claim unique inference",
            ],
        },
    )
    print(json.dumps({"passed": True, "kind": "CPU-only fake HTTP", **result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(smoke(parser.parse_args().output.resolve()))
