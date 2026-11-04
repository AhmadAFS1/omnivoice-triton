#!/usr/bin/env python3
"""Root-released finite CTC/LLM batch1/2/4 probe, preserving old709 evidence."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from batch_prototype import MODELS, idle


def snapshot(root):
    files = list((root / "asr-results/stage1").rglob("*.jsonl"))
    files += [root / "asr/evaluate.py", root / "asr/run_bounded.py"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--allow-gpu-inference", action="store_true")
    args = parser.parse_args()
    assert args.allow_gpu_inference, "Explicit root release required"
    root = args.root.resolve()
    idle()
    before = snapshot(root)
    output = root / "asr-results/batch-probe-v1"
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen709-before.json").write_text(json.dumps(before, indent=2) + "\n")
    started = time.monotonic()
    deadline = started + args.max_seconds
    results = {}
    for model in MODELS:
        if time.monotonic() + 30 >= deadline:
            break
        idle()
        command = [
            sys.executable,
            str(root / "asr/batch_prototype.py"),
            "--manifest",
            str(root / "quality/asr-batch-selection/manifest.json"),
            "--output",
            str(output / model),
            "--model",
            model,
            "--allow-gpu-inference",
            "--max-seconds",
            str(max(1, deadline - time.monotonic())),
        ]
        with (output / f"{model}.log").open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        comparison = output / model / "comparison.json"
        results[model] = {
            "exit_code": code,
            "comparison": json.loads(comparison.read_text())
            if comparison.exists()
            else None,
        }
        if code:
            break
    after = snapshot(root)
    (output / "frozen709-after.json").write_text(json.dumps(after, indent=2) + "\n")
    assert before == after, "Old709/evaluator/controller evidence changed"
    idle()
    complete = len(results) == len(MODELS) and all(
        value.get("comparison")
        and all(
            outcome.get("status") == "completed"
            for outcome in value["comparison"]["outcomes"].values()
        )
        for value in results.values()
    )
    marker = {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "measurement_kind": "Actual GPU ASR batching probe; no TTS on this device",
        "output": str(output),
        "elapsed_s": time.monotonic() - started,
        "models": results,
        "fixed_inputs": 41,
        "frozen709_unchanged": True,
        "gpu_released_idle": True,
        "production_defaults_changed": False,
        "probe_execution_status": "completed" if complete else "partial_or_failed",
        "strict_parity_all_models": complete
        and all(
            all(value["comparison"]["strict_parity_gate"].values())
            for value in results.values()
        ),
    }
    (root / "ASR_BATCH_PROBE_COMPLETE.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "marker": str(root / "ASR_BATCH_PROBE_COMPLETE.json"),
                "gpu_idle": True,
                "elapsed_s": marker["elapsed_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
