#!/usr/bin/env python3
"""Prepare coverage, then optionally run both local GPU ASRs after TTS release.

Actual ASR GPU/VRAM/runtime is collected by evaluate.py. Original reference
controls were prerecorded; their generation hardware is unverified. No API
transcription service or expected-word prompt is used.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("asr_evaluation", HERE / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def prepare(root, output, references_path, include):
    combined, coverage = [], {}
    for manifest in sorted((root / "results").rglob("quality-manifest.json")):
        arm = manifest.parent.relative_to(root / "results").as_posix()
        if include and arm not in include:
            continue
        all_audio = ev.read_rows(manifest)
        ready = ev.read_rows(manifest.parent / "quality-asr-ready.json")
        ready_ids = {row["id"] for row in ready}
        requests = ev.read_rows(manifest.parent / "requests.jsonl")
        request_by_id = {row["id"]: row for row in requests}
        success_ids = {row["id"] for row in requests if row.get("status") == 200}
        retained_ids = {row["id"] for row in all_audio}
        assert success_ids == retained_ids, f"HTTP200 audio gap in {arm}"
        assert ready_ids <= retained_ids
        coverage[arm] = {
            "attempts": len(requests),
            "http_status_counts": dict(
                Counter(str(row.get("status")) for row in requests)
            ),
            "retained_http200_audio": len(all_audio),
            "asr_ready": len(ready),
            "unscorable_review": [
                row for row in all_audio if row["id"] not in ready_ids
            ],
            "target_languages": sorted({row["language"] for row in all_audio}),
            "note": (
                "Unmapped/unreadable cases are retained and are not quality passes."
            ),
        }
        for case in ready:
            path = Path(case["path"])
            if not path.is_absolute():
                path = manifest.parent / path
            assert ev.sha256(path) == case["sha256"]
            request = request_by_id[case["id"]]
            combined.append(
                {
                    **case,
                    "id": f"{arm}::{case['id']}",
                    "original_id": case["id"],
                    "path": str(path.resolve()),
                    "run": arm,
                    "quality_arm": arm,
                    "response_source": request.get("headers", {}).get(
                        "x-omnivoice-response-source", "unreported_legacy"
                    ),
                    "inference_executed": request.get("headers", {}).get(
                        "x-omnivoice-inference-executed", "unreported_legacy"
                    ),
                }
            )
    refs = ev.read_rows(references_path)
    for reference in refs:
        path = Path(reference.get("path") or reference["file"])
        if not path.is_absolute():
            path = references_path.parent / path
        assert ev.sha256(path) == reference["sha256"]
        combined.append(
            {
                "id": f"reference-controls::{reference['id']}",
                "original_id": reference["id"],
                "path": str(path.resolve()),
                "sha256": reference["sha256"],
                "language": reference["language"],
                "omnilingual_language": reference["omnilingual_language"],
                "expected": reference["text"],
                "text": reference["text"],
                "run": "reference-controls",
                "quality_arm": "reference-controls",
                "source_reference": reference,
                "generation_hardware": "Prerecorded Gemini control; GPU unverified",
            }
        )
    combined.sort(
        key=lambda row: (row["quality_arm"] != "reference-controls", row["id"])
    )
    assert combined and len({row["id"] for row in combined}) == len(combined)
    ev.write_json(output / "combined-manifest.json", combined)
    ev.write_json(
        output / "http-and-audio-coverage.json",
        {
            "measurement_kind": "CPU-only file/HTTP coverage bookkeeping",
            "arms": coverage,
            "logical_asr_cases": len(combined),
            "distinct_wav_hashes": len({row["sha256"] for row in combined}),
            "distinct_wav_language_pairs": len(
                {(row["sha256"], row["omnilingual_language"]) for row in combined}
            ),
            "note": (
                "Duplicate cache/coalesced WAVs are not independent quality evidence."
            ),
        },
    )
    return combined


def split(output, models):
    combined = output / "combined"
    cases = ev.read_rows(combined / "prepared.json")
    transcripts = {
        model: {row["id"]: row for row in ev.read_rows(combined / f"{model}.jsonl")}
        for model in models
    }
    index = {}
    for arm in sorted({case["source"]["quality_arm"] for case in cases}):
        directory = output / "by-arm" / arm
        directory.mkdir(parents=True)
        selected = [case for case in cases if case["source"]["quality_arm"] == arm]
        prepared = [
            {**case, "id": case["source"]["original_id"], "combined_id": case["id"]}
            for case in selected
        ]
        ev.write_json(directory / "prepared.json", prepared)
        for model in models:
            with (directory / f"{model}.jsonl").open("w") as file:
                for case in selected:
                    row = transcripts[model][case["id"]]
                    file.write(
                        json.dumps(
                            {
                                **row,
                                "combined_id": row["id"],
                                "id": case["source"]["original_id"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            shutil.copy2(
                combined / f"{model}-metadata.json",
                directory / f"{model}-metadata.json",
            )
        ev.score(directory, models)
        index[arm] = json.loads((directory / "summary.json").read_text())
    ev.write_json(
        output / "index.json",
        {
            "arms": index,
            "limitations": [
                "ASR variants share a family, not independent ground truth",
                "Support/mapping membership is not native quality qualification",
                "Case proxy/script/source caveats remain in source metadata",
                "No human MOS or speaker similarity study",
                "Full-WAV hashes retain cache/coalescing identity",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--include-arms", nargs="*")
    parser.add_argument("--allow-gpu-inference", action="store_true")
    parser.add_argument("--models", nargs="+", default=ev.DEFAULT_MODELS)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = prepare(
        args.root.resolve(), output, args.references.resolve(), args.include_arms
    )
    print(
        json.dumps({"prepared_cases": len(cases), "gpu_inference": False}), flush=True
    )
    if not args.allow_gpu_inference:
        return
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    assert not processes, "GPU still has compute processes; TTS must release it first"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "evaluate.py"),
            "--manifest",
            str(output / "combined-manifest.json"),
            "--output",
            str(output / "combined"),
            "--models",
            *args.models,
        ],
        check=True,
    )
    split(output, args.models)


if __name__ == "__main__":
    main()
