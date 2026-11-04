#!/usr/bin/env python3
"""Finite primary RTX4070 diagnostics; dry run unless --execute is explicit."""

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/omnivoice-max-concurency")
REF = "47bd743e1796995d5481ff5889b76fa5c89ace66"
RUNNER_SHA = "df51c081d8108bf7dae5778adc2f21398e59d0ca8de394601f4251a55c521652"


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def gpu_processes():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def stop_owned(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.plan.read_text())
    assert document["source_ref"] == REF
    if not args.execute:
        print(json.dumps(document, indent=2))
        return
    assert args.budget_seconds > 0
    assert hashlib.sha256((ROOT / "load/run.py").read_bytes()).hexdigest() == RUNNER_SHA
    assert not (ROOT / "CANDIDATE47_CORE_OUTCOME.json").exists()
    assert not gpu_processes(), "Require explicit previous-owner release and idle GPU"
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    assert "GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33" in hardware
    signal.signal(signal.SIGTERM, interrupt)
    env = os.environ | {"HF_HOME": str(ROOT / "hf-cache"), "HF_HUB_OFFLINE": "1"}
    outcomes = []
    started = time.monotonic()
    completed = False
    child = None
    try:
        for job in document["jobs"]:
            remaining = args.budget_seconds - (time.monotonic() - started)
            assert remaining > 120, "Finite controller budget exhausted before next job"
            assert not gpu_processes(), "Another GPU owner is active"
            original = job.get("original_command", job["command"])
            assert original[original.index("--source-ref") + 1] == REF
            output = Path(original[original.index("--output") + 1])
            assert not output.exists(), "No silent retry or output overwrite"
            row = {"id": job["id"], "started_at": now()}
            outcomes.append(row)
            with (ROOT / f"{job['id']}.log").open("w") as log:
                child = subprocess.Popen(
                    job["command"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write(
                    ROOT / "CANDIDATE47_CORE_CURRENT_JOB.json",
                    row | {"pid": child.pid, "instance_id": 59058485},
                )
                try:
                    code = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    # Telemetry handles SIGTERM by interrupting its owned harness;
                    # the harness then closes its separately owned model process.
                    stop_owned(child)
                    row["budget_timeout"] = True
                    raise
            child = None
            row.update(exit_code=code, finished_at=now())
            write(ROOT / "candidate47-core-outcomes.json", outcomes)
            print(json.dumps(row), flush=True)
            assert code == 0, "Stop on workload failure; retain every outcome"
            for script, destination, extra in [
                ("verify.py", "independent-protocol-verification.json", [str(output)]),
                (
                    "arrival_report.py",
                    "arrival-report.json",
                    [
                        "--run-dir",
                        str(output),
                        "--output",
                        str(output / "arrival-report.json"),
                    ],
                ),
            ]:
                check = subprocess.run(
                    [original[0], str(ROOT / "load" / script), *extra],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(
                        1, min(60, args.budget_seconds - (time.monotonic() - started))
                    ),
                )
                stdout_path = (
                    destination
                    if script == "verify.py"
                    else destination + ".stdout.txt"
                )
                (output / stdout_path).write_text(check.stdout)
                (output / (destination + ".stderr.txt")).write_text(check.stderr)
                row[script + "_exit"] = check.returncode
                assert check.returncode == 0, "Protocol validation failed"
            health = json.loads((output / "health-after.json").read_text())
            assert health.get("cuda_fault") is None, (
                "Stop after fatal CUDA context fault"
            )
            write(ROOT / "candidate47-core-outcomes.json", outcomes)
        completed = True
    finally:
        stop_owned(child)
        active = gpu_processes()
        write(
            ROOT / "CANDIDATE47_CORE_OUTCOME.json",
            {
                "at": now(),
                "instance_id": 59058485,
                "source_ref": REF,
                "completed": completed,
                "outcomes": outcomes,
                "gpu_released": not bool(active),
                "compute_processes": active,
                "interpretation": "Real single-GPU complete-body diagnostics; "
                "all failed offers remain failures. ASR quality is a separate gate.",
            },
        )


if __name__ == "__main__":
    main()
