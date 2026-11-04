"""Offline third-family ASR for preselected word/speaker diagnostics only."""

import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import mlx.core as mx
import mlx_whisper
import numpy as np
import soundfile as sf
from mlx_whisper.tokenizer import LANGUAGES
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "quality/whisper-fixed-diagnostics-v1"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    assert not OUT.exists(), "Preserve every prior attempt"
    assert MODEL.is_dir(), "Only the existing local model; no download"
    assert (
        digest(MODEL / "weights.safetensors")
        == "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6"
    )
    source = ROOT / "quality/whisper-fixed-selected-joint-v1.json"
    selected = json.loads(source.read_text())
    unique, logical, skipped = {}, [], []
    for pair in selected:
        route = pair["language"]
        language = "ar" if route.startswith("ar-") else route
        if language not in LANGUAGES:
            skipped.append(
                {
                    "case": pair["comparison_original_case_id"],
                    "variant": pair["variant"],
                    "language": route,
                    "reason": "Whisper language not supported; no language/proxy substitution",
                }
            )
            continue
        for side in ("before", "after"):
            row = pair[side]
            path = Path(row["path"])
            assert digest(path) == row["sha256"]
            key = (row["sha256"], language)
            if key not in unique:
                unique[key] = {
                    "id": "audio-"
                    + hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                    "path": str(path),
                    "sha256": row["sha256"],
                    "language_hint": language,
                }
            logical.append(
                {
                    "case": pair["comparison_original_case_id"],
                    "variant": pair["variant"],
                    "side": side,
                    "input_id": unique[key]["id"],
                    "language": route,
                    "voice": pair["voice"],
                    "expected": row["expected"],
                    "sha256": row["sha256"],
                    "Meta_evidence": pair["ASR_evidence"][side],
                    "note": "Arabic routes use family-language hint ar, not a dialect-specific recognition claim"
                    if route.startswith("ar-")
                    else None,
                }
            )
    OUT.mkdir()
    write(
        OUT / "selection.json",
        {
            "kind": "Purposive preselected diagnostic cases, not a random failure-rate sample",
            "source_sha256": digest(source),
            "logical": logical,
            "unique_inputs": list(unique.values()),
            "skipped": skipped,
        },
    )
    write(
        OUT / "metadata.json",
        {
            "at": dt.datetime.now(dt.UTC).isoformat(),
            "measurement_kind": "Offline Apple integrated Metal GPU ASR; no inference API or NVIDIA access",
            "platform": platform.platform(),
            "device_info": mx.device_info(),
            "model_revision": MODEL.name,
            "weights_sha256": digest(MODEL / "weights.safetensors"),
            "packages": {
                x: importlib.metadata.version(x)
                for x in (
                    "mlx",
                    "mlx-metal",
                    "mlx-whisper",
                    "numpy",
                    "scipy",
                    "soundfile",
                )
            },
            "script_sha256": digest(Path(__file__)),
            "settings": {
                "temperature": 0,
                "condition_on_previous_text": False,
                "initial_prompt": None,
                "prefix": None,
                "word_timestamps": True,
            },
            "coload": "Other Mac CPU/Metal activity unmeasured; rented TTS/Meta ASR run on separate hosts",
            "quality_certified": False,
        },
    )
    started = time.monotonic()
    completed = []
    raw_path = OUT / "transcripts.jsonl"
    with raw_path.open("x") as raw_file:
        for item in unique.values():
            assert time.monotonic() - started < 580, (
                "Finite diagnostic budget exhausted"
            )
            assert digest(Path(item["path"])) == item["sha256"]
            samples, rate = sf.read(item["path"], dtype="float32", always_2d=True)
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(
                samples.mean(axis=1), 16000 // divisor, rate // divisor
            ).astype(np.float32)
            begin = time.monotonic()
            # Expected text, Meta hypotheses and voice labels are never passed
            # into transcription. Only the waveform and supported language.
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=str(MODEL),
                language=item["language_hint"],
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=None,
                prefix=None,
                word_timestamps=True,
                verbose=None,
            )
            row = {
                **item,
                "transcript": result["text"],
                "raw_result": result,
                "asr_wall_s": time.monotonic() - begin,
            }
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_file.flush()
            completed.append(row)
            write(
                OUT / "progress.json",
                {
                    "completed": len(completed),
                    "required": len(unique),
                    "elapsed_s": time.monotonic() - started,
                },
            )
    spec = importlib.util.spec_from_file_location(
        "quality_report", ROOT / "quality/report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    by_id = {r["id"]: r for r in completed}
    scored = []
    for row in logical:
        transcript = by_id[row["input_id"]]["transcript"]
        scored.append(
            {
                **row,
                "Whisper_transcript": transcript,
                "Whisper_metrics": report.text_metrics(row["expected"], transcript),
            }
        )
    write(OUT / "scored-logical.json", scored)
    write(
        OUT / "complete.json",
        {
            "completed": len(completed),
            "required": len(unique),
            "logical": len(scored),
            "skipped_pairs": len(skipped),
            "elapsed_s": time.monotonic() - started,
            "peak_allocated_bytes": mx.get_peak_memory(),
            "quality_certified": False,
            "interpretation": "Third-family diagnostic; ASR agreement/disagreement is not native listening, MOS, speaker identity, or a population failure rate.",
        },
    )
    print((OUT / "complete.json").read_text())


if __name__ == "__main__":
    main()
