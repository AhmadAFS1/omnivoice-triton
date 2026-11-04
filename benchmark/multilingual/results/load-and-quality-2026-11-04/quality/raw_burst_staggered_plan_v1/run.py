"""CPU dry-only unless a new root lease releases modern after raw5k+LLM ASR."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def phase_readout(rows, offered):
    successful = [row for row in rows if row.get("status") == 200]
    valid = [row for row in successful if row.get("audio", {}).get("signal_valid")]

    def arrival_latency(row):
        body, late = row.get("complete_ms"), row.get("dispatch_lateness_ms")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in (body, late)
        ):
            return None
        return body + late

    arrivals = [
        latency for row in successful if (latency := arrival_latency(row)) is not None
    ]
    valid_arrivals = [
        latency for row in valid if (latency := arrival_latency(row)) is not None
    ]
    wall = (
        max((row.get("completion_from_barrier_ms", 0) for row in rows), default=0)
        / 1000
    )
    return {
        "offers": offered,
        "recorded_outcomes": len(rows),
        "missing_or_not_dispatched": offered - len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "HTTP200": len(successful),
        "HTTP200_signal_valid": len(valid),
        "rejected429": sum(row.get("status") == 429 for row in rows),
        "transport_errors": sum(bool(row.get("error")) for row in rows),
        "valid_audio_per_offer": len(valid) / offered,
        "valid_audio_within500ms_alloffers": sum(
            row["complete_ms"] <= 500 for row in valid
        ),
        "valid_audio_dispatch_within2s_alloffers": sum(
            row["complete_ms"] <= 2000 for row in valid
        ),
        "valid_audio_scheduledarrival_within500ms_alloffers": sum(
            value <= 500 for value in valid_arrivals
        ),
        "valid_audio_scheduledarrival_within2s_alloffers": sum(
            value <= 2000 for value in valid_arrivals
        ),
        "scheduledarrival_formula": "complete_ms(actualPOSTdispatch_to_fullbody) + dispatch_lateness_ms",
        "scheduledarrival_HTTP200_timing_observations": len(arrivals),
        "scheduledarrival_missing_timing_alloffers": offered
        - sum(arrival_latency(row) is not None for row in rows),
        "valid_audio_per_network_wall_s": len(valid) / wall if wall else None,
        "HTTP200_only_p95_ms": percentile(
            [row["complete_ms"] for row in successful], 95
        ),
        **{
            f"HTTP200_actualdispatch_only_p{p}_ms": percentile(
                [row["complete_ms"] for row in successful], p
            )
            for p in (50, 95, 99)
        },
        **{
            f"HTTP200_scheduledarrival_only_p{p}_ms": percentile(arrivals, p)
            for p in (50, 95, 99)
        },
        "dispatch_lateness_p95_ms": percentile(
            [
                row["dispatch_lateness_ms"]
                for row in rows
                if "dispatch_lateness_ms" in row
            ],
            95,
        ),
        "new_TCP_connect_events": sum(
            "connection.connect_tcp.started" in row.get("trace", {}) for row in rows
        ),
        "network_wall_s": wall,
        "quality": "Basic signal validity only; words/voice not certified.429/errors are never successful audio throughput.",
    }


def owned_groups(pid):
    table = [
        tuple(map(int, line.split()))
        for line in subprocess.check_output(
            ["ps", "-eo", "pid,ppid,pgid"], text=True
        ).splitlines()[1:]
    ]
    owned = {pid}
    for _ in range(len(table)):
        updated = owned | {child for child, parent, _ in table if parent in owned}
        if updated == owned:
            break
        owned = updated
    return {group for child, _, group in table if child in owned}


def stop_groups(groups):
    for group in groups:
        assert group != os.getpgrp(), "Never stop supervisor group"
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--lease", type=Path)
    args = parser.parse_args()
    config = read(HERE / "config.json")
    assert sha(HERE / "plan.json") == config["plan_sha256"]
    for relative, expected in config["frozen_files"].items():
        assert sha(ROOT / relative) == expected, relative
    if not args.execute:
        print(
            json.dumps(
                {
                    "GPU_executed": False,
                    "measurement_kind": "CPU-only future plan; no model import",
                    "config": config,
                },
                indent=2,
            )
        )
        return
    assert args.lease is not None
    lease = read(args.lease)
    assert all(
        lease[key]
        for key in (
            "root_authorized_staggered_calls",
            "raw5k_retention_idle_complete",
            "post_raw_modern_LLM_ASR_idle_complete",
            "CPU_IO_quiet",
        )
    )
    assert lease["controller_sha256"] == sha(Path(__file__)) and lease[
        "config_sha256"
    ] == sha(HERE / "config.json")
    assert (
        lease["source_ref"] == config["source_ref"]
        and lease["GPU_UUID"] == config["GPU_UUID"]
    )
    assert 0 <= time.time() - dt.datetime.fromisoformat(lease["at"]).timestamp() < 600
    assert sha(ROOT / config["stop_helper_path"]) == config["stop_helper_sha256"]
    started = time.monotonic()
    query = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    assert not subprocess.check_output(query, text=True, timeout=5).strip()
    physical = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=5,
    )
    assert config["GPU_UUID"] in physical and "RTX 4070 Ti" in physical
    source = ROOT / "candidate-v8-audio-guard"
    assert {
        str(path.relative_to(source)) for path in (source / "src").rglob("*.py")
    } == {row["path"] for row in config["source_files"]}
    for row in config["source_files"]:
        assert sha(source / row["path"]) == row["sha256"]
    for row in config["model_files"]:
        path = ROOT / "model" / row["path"]
        assert path.stat().st_size == row["bytes"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                assert time.monotonic() - started < 30
                digest.update(block)
        assert digest.hexdigest() == row["sha256"]
    output = Path(config["output"])
    assert not output.exists() and not (HERE / "execution.json").exists()
    write(
        HERE / "execution.json",
        {
            "at": dt.datetime.now(dt.UTC).isoformat(),
            "lease": lease,
            "controller_sha256": sha(Path(__file__)),
            "config_sha256": sha(HERE / "config.json"),
        },
    )
    child, status, groups = None, "failed", set()
    previous = {}

    def interrupted(_number, _frame):
        raise KeyboardInterrupt("Stop only owned frozenV4 worker tree")

    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, interrupted)
    try:
        with (HERE / "worker.log").open("w") as log:
            child = subprocess.Popen(
                config["command"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(
                    os.environ, HF_HOME=str(ROOT / "hf-cache"), HF_HUB_OFFLINE="1"
                ),
            )
            groups.add(child.pid)
            write(HERE / "worker-pid.json", {"pid": child.pid})
            deadline = started + config["worker_bound_s"]
            while child.poll() is None and time.monotonic() < deadline:
                groups |= owned_groups(child.pid)
                time.sleep(0.5)
            if child.poll() is None:
                status = "deadline_partial"
            else:
                status = (
                    "completed" if child.returncode == 0 else "worker_failed_partial"
                )
    finally:
        if child is not None:
            groups |= owned_groups(child.pid)
            stop_groups(groups)
            if child.poll() is None:
                child.wait(timeout=10)
        phases = read(HERE / "plan.json")["phases"]
        file = output / "requests.jsonl"
        rows = (
            [json.loads(line) for line in file.read_text().splitlines()]
            if file.exists()
            else []
        )
        report = {
            phase["id"]: phase_readout(
                [row for row in rows if row.get("phase") == phase["id"]],
                phase.get("requests", len(phase["case_ids"])),
            )
            for phase in phases
        }
        write(
            HERE / "readout.json",
            {
                "status": status,
                "phases": report,
                "no_429_or_error_counted_as_completed_audio": True,
            },
        )
        active = subprocess.check_output(query, text=True, timeout=5).strip()
        write(
            ROOT / "MODERN_STAGGERED_CALLS_COMPLETE_IDLE.json",
            {
                "at": dt.datetime.now(dt.UTC).isoformat(),
                "status": status,
                "elapsed_s": time.monotonic() - started,
                "gpu_released_idle": not bool(active),
                "compute_processes": active,
                "word_quality_certified": False,
            },
        )
        for number, handler in previous.items():
            signal.signal(number, handler)
    assert status == "completed" and not active


if __name__ == "__main__":
    main()
