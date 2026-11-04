"""CPU-only relative speaker screen for synthetic voices; not MOS certification."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "quality/speaker-model"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def describe(values):
    return (
        {
            "n": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }
        if values
        else {"n": 0}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", action="append", default=[], type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve earlier screens"
    args.output.mkdir(parents=True)
    assets = json.loads((MODEL / "ASSETS_VERIFIED.json").read_text())
    for row in assets["files"]:
        assert digest(MODEL / row["name"]) == row["sha256"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    classifier = EncoderClassifier.from_hparams(
        source=str(MODEL),
        savedir=str(args.output / "loader-state"),
        run_opts={"device": "cpu"},
        overrides={"pretrained_path": str(MODEL)},
    )
    classifier.eval()
    assert next(classifier.mods.embedding_model.parameters()).device.type == "cpu"
    assert not classifier.mods.embedding_model.training
    metadata = {
        "at": dt.datetime.now(dt.UTC).isoformat(),
        "measurement_kind": "CPU-only "
        "SpeechBrain ECAPA-TDNN embedding extraction and paired cosine comparisons",
        "GPU": "None; CUDA disabled and device explicitly CPU; Metal not used",
        "CPU": "Apple M5 Pro",
        "platform": platform.platform(),
        "physical_shared_RAM_bytes": int(
            __import__("subprocess").check_output(["sysctl", "-n", "hw.memsize"])
        ),
        "runtime": {
            p: importlib.metadata.version(p)
            for p in [
                "speechbrain",
                "torch",
                "torchaudio",
                "numpy",
                "scipy",
                "soundfile",
                "huggingface_hub",
            ]
        },
        "model_assets": assets,
        "runner_sha256": digest(Path(__file__)),
        "threads": 4,
        "embedding_normalize_argument": False,
        "sample_rate": 16000,
        "amplitude_normalization": "None",
        "inference_API_calls": 0,
        "co_load": "Other local CPU activity unmeasured; "
        "NVIDIA TTS/ASR runs on separate rented hosts",
        "interpretation": "Model trained on VoxCeleb; synthetic/multilingual "
        "domain is not certified. Compare the same route/text/voice against "
        "full-reference baseline and seven other stock voices. No universal "
        "threshold, MOS, accent or age/gender certification is inferred.",
    }
    write(args.output / "metadata.json", metadata)
    embeddings = {}

    def encode(path, expected_sha):
        assert digest(path) == expected_sha
        if expected_sha in embeddings:
            return np.array(embeddings[expected_sha]["embedding"], dtype=np.float64)
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        assert mono.size and np.isfinite(mono).all() and np.any(mono)
        divisor = math.gcd(sr, 16000)
        samples = resample_poly(mono, 16000 // divisor, sr // divisor).astype(
            np.float32
        )
        started = time.perf_counter()
        with torch.inference_mode():
            result = classifier.encode_batch(
                torch.from_numpy(samples).unsqueeze(0), normalize=False
            )
        vector = result.reshape(-1).cpu().double().numpy()
        assert vector.size == 192 and np.isfinite(vector).all()
        norm = np.linalg.norm(vector)
        assert norm > 0
        vector /= norm
        embeddings[expected_sha] = {
            "path": str(path),
            "sha256": expected_sha,
            "embedding": vector.tolist(),
            "audio_duration_s": len(mono) / sr,
            "CPU_encode_wall_s": time.perf_counter() - started,
        }
        with (args.output / "embeddings.jsonl").open("a") as stream:
            stream.write(json.dumps(embeddings[expected_sha]) + "\n")
        return vector

    references = json.loads(
        (ROOT / "quality/voice-controls/references/manifest.json").read_text()
    )
    anchors = {}
    for row in references:
        if row["language"] == "en":
            anchors[row["voice_id"]] = encode(
                ROOT / "quality/voice-controls/references" / row["path"], row["sha256"]
            )
    assert len(anchors) == 8

    def similarities(vector, voice):
        scores = {
            name: float(np.clip(np.dot(vector, anchor), -1, 1))
            for name, anchor in anchors.items()
        }
        own = scores[voice]
        others = {name: score for name, score in scores.items() if name != voice}
        nearest = max(others, key=others.get)
        return {
            "own_cosine": own,
            "nearest_other_voice": nearest,
            "nearest_other_cosine": others[nearest],
            "own_margin": own - others[nearest],
            "own_is_top1": own >= max(scores.values()),
            "all_stock_scores": scores,
        }

    controls = []
    for row in json.loads(
        (ROOT / "quality/reference-prefix-v1/manifest.json").read_text()
    ):
        vector = encode(
            ROOT / "quality/reference-prefix-v1" / row["path"], row["sha256"]
        )
        controls.append(
            {
                "id": row["id"],
                "voice": row["voice_id"],
                "word_count": row["word_count"],
                **similarities(vector, row["voice_id"]),
            }
        )
    write(args.output / "reference-controls.json", controls)
    generated = []
    for directory in args.run_dir:
        records = json.loads((directory / "quality-manifest.json").read_text())
        for row in records:
            if not row.get("audio", {}).get("signal_valid"):
                generated.append(
                    {
                        "run": directory.name,
                        "id": row["id"],
                        "unscored_invalid_signal": True,
                    }
                )
                continue
            case = row["case"]
            vector = encode(directory / row["path"], row["sha256"])
            generated.append(
                {
                    "run": directory.name,
                    "id": row["id"],
                    "language": case["language"],
                    "voice": case["voice_id"],
                    "original_case": case["comparison_original_case_id"],
                    "unchanged_reference_control": case["unchanged_reference_control"],
                    "audio_duration_s": row["audio"]["duration_s"],
                    **similarities(vector, case["voice_id"]),
                }
            )
        print(
            json.dumps(
                {"processed": directory.name, "unique_embeddings": len(embeddings)}
            ),
            flush=True,
        )
    write(args.output / "generated-scores.json", generated)
    paired = []
    full = {
        r["original_case"]: r
        for r in generated
        if r["run"].endswith("-full") and "original_case" in r
    }
    for row in generated:
        before = full.get(row.get("original_case"))
        if before is None or row["run"].endswith("-full"):
            continue
        paired.append(
            {
                "run": row["run"],
                "original_case": row["original_case"],
                "language": row["language"],
                "voice": row["voice"],
                "unchanged_reference_control": row["unchanged_reference_control"],
                "before_own_cosine": before["own_cosine"],
                "after_own_cosine": row["own_cosine"],
                "delta_own_cosine": row["own_cosine"] - before["own_cosine"],
                "before_margin": before["own_margin"],
                "after_margin": row["own_margin"],
                "before_top1": before["own_is_top1"],
                "after_top1": row["own_is_top1"],
            }
        )
    write(args.output / "paired-deltas.json", paired)
    summary = {
        "metadata": metadata,
        "unique_embeddings": len(embeddings),
        "reference_controls": len(controls),
        "generated": len(generated),
        "paired": len(paired),
        "quality_certified": False,
        "variants": {},
    }
    for variant in sorted({r["run"] for r in paired}):
        rows = [
            r
            for r in paired
            if r["run"] == variant and not r["unchanged_reference_control"]
        ]
        summary["variants"][variant] = {
            "changed_cases": len(rows),
            "paired_own_cosine_delta": describe([r["delta_own_cosine"] for r in rows]),
            "before_top1": sum(r["before_top1"] for r in rows),
            "after_top1": sum(r["after_top1"] for r in rows),
            "lost_top1": sum(r["before_top1"] and not r["after_top1"] for r in rows),
        }
    write(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"}), flush=True)


if __name__ == "__main__":
    main()
