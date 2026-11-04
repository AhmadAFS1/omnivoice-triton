"""Unchanged bounded-gate contracts extracted from the prior132 helper; CPU/file checks only."""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def idle():
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    assert not value, value
    return value


def check(config):
    idle()
    physical = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
        text=True,
    )
    assert config["GPU_UUID"] in physical and config["GPU_name"] in physical
    for filename, expected in config["frozen_files"].items():
        assert sha(ROOT / filename) == expected, filename
    source = ROOT / config["source_dir"]
    tree = hashlib.sha256()
    for path in sorted((source / "src").rglob("*.py")):
        tree.update(
            str(path.relative_to(source)).encode() + b"\0" + sha(path).encode() + b"\n"
        )
    assert tree.hexdigest() == config["source_tree_sha256"]
    refs = ROOT / config["references_path"]
    for ref in read(refs):
        assert sha(refs.parent / ref["path"]) == ref["sha256"]


def verify(output, command):
    import numpy as np
    import soundfile as sf

    proc = subprocess.run(
        [command[0], str(ROOT / "load/verify_v5.py"), str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (output / "independent-protocol-verification.json").write_text(proc.stdout)
    (output / "independent-protocol-verification-stderr.txt").write_text(proc.stderr)
    assert proc.returncode == 0, "Retain protocol integrity failure and stop"
    requests = [
        json.loads(x) for x in (output / "requests.jsonl").read_text().splitlines()
    ]
    expected = [c["id"] for c in read(HERE / "corpus132.json")]
    assert len(requests) == 132 and [r["case"]["id"] for r in requests] == expected
    audited = []
    for q in read(output / "quality-manifest.json"):
        path = output / q["path"]
        assert sha(path) == q["sha256"]
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 24000 and audio.shape[1] == 1
        finite = bool(np.isfinite(audio).all())
        rms = (
            float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            if len(audio)
            else 0.0
        )
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        audited.append(
            {
                "id": q["id"],
                "sha256": q["sha256"],
                "frames": len(audio),
                "sample_rate": rate,
                "finite": finite,
                "RMS": rms,
                "peak": peak,
                "basic_valid": bool(
                    len(audio) and finite and rms > 1e-5 and peak > 1e-4
                ),
                "harness_signal_valid": q["audio"]["signal_valid"],
            }
        )
    write(output / "independent-waveform-audit.json", audited)
    health = read(output / "health-after.json")
    assert health.get("cuda_fault") is None
    log = (output / "server.log").read_text().casefold()
    assert not any(
        s in log
        for s in (
            "cudagraphcaptureerror",
            "device-side assert triggered",
            "cuda error:",
            "cuda out of memory",
        )
    ), "Fatal CUDA context fault"
    return requests, {
        "requests": 132,
        "status_counts": dict(Counter(str(r.get("status")) for r in requests)),
        "WAV_hashes_and_decodes_verified": len(audited),
        "basic_valid": sum(r["basic_valid"] for r in audited),
        "harness_signal_valid": sum(r["harness_signal_valid"] for r in audited),
        "protocol_verifier_exit": 0,
        "health_after": health,
    }
