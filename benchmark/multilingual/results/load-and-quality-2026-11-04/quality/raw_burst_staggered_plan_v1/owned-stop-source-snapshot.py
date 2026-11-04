#!/usr/bin/env python3
"""Root-authorized16-control micro only; standard-library supervisor, no model import."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

CONTROLLER_SHA = "6f2f258dfd8e4452fa1e6736cb56bc00b451a84bdc786a185f2e9026bd28e888"
EVALUATOR_SHA = "db2e933104fd0cdd939f6cdf6661331e9e72cc3403e59f049a771b0a83567cdd"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def idle():
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=15,
    ).strip()
    assert not value, f"GPU not idle:{value}"
    return {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "compute_processes": value,
    }


def stop_tree(process):
    """Workers start their own sessions; terminate only this controller's tree."""
    rows = subprocess.check_output(["ps", "-eo", "pid,ppid,pgid"], text=True)
    table = [tuple(map(int, line.split())) for line in rows.splitlines()[1:]]
    owned = {process.pid}
    changed = True
    while changed:
        before = len(owned)
        owned.update(pid for pid, parent, _ in table if parent in owned)
        changed = len(owned) != before
    groups = {group for pid, _, group in table if pid in owned}
    for group in groups:
        if group == os.getpgrp():
            raise RuntimeError("Refusing to terminate supervisor's process group")
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=["modern", "primary"], required=True)
    parser.add_argument("--execute-root-released-micro", action="store_true")
    args = parser.parse_args()
    assert args.execute_root_released_micro
    root = args.root.resolve()
    model = (
        "omniASR_LLM_Unlimited_1B_v2" if args.role == "modern" else "omniASR_CTC_1B_v2"
    )
    namespace = "modern59071593-batch1-v1" if args.role == "modern" else "legacy-batch1"
    label = f"{args.role}-prefix16-{'LLM' if args.role == 'modern' else 'CTC'}-micro-v1"
    output = root / "asr-results" / label
    evidence = root / "asr-results" / (label + "-preflight")
    evidence.mkdir(parents=True, exist_ok=False)
    before = idle()
    assert sha(root / "asr/run_selected_models.py") == CONTROLLER_SHA
    assert sha(root / "asr/evaluate.py") == EVALUATOR_SHA
    prepared = root / "quality/reference-prefix-asr16-v1"
    cases = json.loads((prepared / "asr-input-manifest.json").read_text())
    assert len(cases) == len({row["id"] for row in cases}) == 16
    assert all(row["omnilingual_language"] == "eng_Latn" for row in cases)
    for row in cases:
        assert sha(Path(row["path"])) == row["sha256"]
    original_marker = root / (
        "MODERN_ASR_ENV_ASSETS_STAGED.json"
        if args.role == "modern"
        else "ASR_ASSETS_VERIFIED.json"
    )
    marker = json.loads(original_marker.read_text())
    assert (
        marker.get(
            "baseline_package_freeze_unchanged", marker.get("baseline_freeze_unchanged")
        )
        is True
    )
    assert (
        len(marker["assets"]) == 3
        and sum(row["bytes"] for row in marker["assets"]) == 13021814169
    )
    packages = {
        re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(): dist.version
        for dist in importlib.metadata.distributions()
    }
    source = importlib.util.find_spec("omnilingual_asr")
    base = Path(next(iter(source.submodule_search_locations)))
    source_hashes = {
        str(path.relative_to(base)): sha(path) for path in sorted(base.rglob("*.py"))
    }
    direct = importlib.metadata.distribution("omnilingual-asr").read_text(
        "direct_url.json"
    )
    assert "81f51e224ce9e74b02cc2a3eaf21b2d91d743455" in direct
    write(
        evidence / "environment-and-source.json",
        {
            "measurement_kind": "CPU-only package/source snapshot before the limited GPU micro",
            "packages": packages,
            "package_count": len(packages),
            "source_direct_url": json.loads(direct),
            "ASR_source_files": source_hashes,
            "controller_sha256": CONTROLLER_SHA,
            "evaluator_sha256": EVALUATOR_SHA,
            "original_asset_marker": str(original_marker),
            "original_asset_marker_sha256": sha(original_marker),
            "namespace": namespace,
            "model": model,
        },
    )
    marker.update(
        packages=packages,
        baseline_package_freeze_unchanged=True,
        original_marker_sha256=sha(original_marker),
        original_marker_path=str(original_marker),
        authorization="Root limited16 controls;300s budget/reserve45; no broader model work",
        ASR_source_files=source_hashes,
        protocol_namespace=namespace,
        micro_only=True,
        full_primary_modern_freeze_comparison_still_required=True,
    )
    micro_marker = evidence / "micro-assets.json"
    write(micro_marker, marker)
    command = [
        sys.executable,
        str(root / "asr/run_selected_models.py"),
        "--prepared",
        str(prepared),
        "--output",
        str(output),
        "--models",
        model,
        "--protocol-namespace",
        namespace,
        "--asset-marker",
        str(micro_marker),
        "--budget-seconds",
        "300",
        "--reserve-seconds",
        "45",
        "--allow-gpu-inference",
    ]
    write(
        evidence / "launch.json",
        {"command": command, "idle_before": before, "worker_snapshot_in_output": True},
    )
    process = None
    result = {
        "status": "failed",
        "role": args.role,
        "model": model,
        "namespace": namespace,
        "output": str(output),
        "preflight": str(evidence),
        "broader_inference_authorized": False,
    }
    try:
        idle()
        with (evidence / "controller.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                stop_tree(process)
                raise TimeoutError("Finite prefix micro controller deadline")
        outcome = json.loads((output / "outcome.json").read_text())
        result.update(
            status="completed"
            if code == 0 and outcome["requested_models_complete"]
            else "partial_or_failed",
            exit_code=code,
            outcome=outcome,
        )
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_tree(process)
        for attempt in range(10):
            try:
                after = idle()
                break
            except AssertionError:
                if attempt == 9:
                    raise
                time.sleep(1)
        result.update(idle_after=after, gpu_released_idle=True)
        write(evidence / "micro-outcome.json", result)
        write(root / f"{args.role.upper()}_PREFIX16_MICRO_COMPLETE.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
