#!/usr/bin/env python3
"""Check completed downloader evidence and isolated packages; no model inference."""

import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = {
    "omniASR_tokenizer_written_v2": (
        91481,
        "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
    ),
    "omniASR_LLM_Unlimited_1B_v2": (
        9118766620,
        "f3cda9ddd8d7991c396cce58714269824457ae3c8807ca6cd0297e8d275bea66",
    ),
    "omniASR_CTC_1B_v2": (
        3902956068,
        "354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c",
    ),
}
assets = json.loads((root / "asr-assets.json").read_text())
rows = {row["name"]: row for row in assets["assets"]}
assert rows.keys() == expected.keys()
for name, (size, sha) in expected.items():
    row = rows[name]
    assert row["bytes"] == size and row["sha256"] == sha, name
    assert Path(row["path"]).stat().st_size == size, name
before = root / "baseline-packages-before-asr-uv.txt"
after = root / "baseline-packages-after-asr-uv.txt"
assert before.read_bytes() == after.read_bytes()
pins = {
    "torch": "2.8.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "fairseq2": "0.6",
    "fairseq2n": "0.6+cu128",
    "pyarrow": "20.0.0",
}
for package, version in pins.items():
    assert importlib.metadata.version(package) == version, package
marker = {
    "measurement_kind": "CPU-only completed download/hash/package evidence check",
    "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "assets": list(rows.values()),
    "baseline_package_freeze_unchanged": True,
    "baseline_freeze_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
    "packages": pins,
    "method": "Separate ASR virtualenv; UV reused baseline wheel cache with online "
    "index metadata. No shared .pth, Torch symlinks or baseline package mutations.",
    "gpu_model_load_or_inference_performed": False,
    "note": "Hashes were computed over full files by prefetch.py after download. "
    "GPU execution remains reserved for root release after TTS timing.",
}
path = root / "ASR_ASSETS_VERIFIED.json"
path.write_text(json.dumps(marker, indent=2) + "\n")
print(json.dumps({"marker": str(path), "assets_verified": len(rows)}))
