#!/usr/bin/env python3
"""CPU-only overlay: wrap every planned job with equivalent passive telemetry."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sampler", type=Path, required=True, help="Sampler path on execution machine"
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--thread-limit", type=int, default=8)
    args = parser.parse_args()
    assert args.interval >= 1 and args.thread_limit >= 0
    assert not args.output.exists(), "Use a fresh output plan"
    document = json.loads(args.plan.read_text())
    for job in document["jobs"]:
        command = job["command"]
        assert "original_command" not in job, "Plan is already wrapped"
        original = list(command)
        run_output = Path(command[command.index("--output") + 1])
        telemetry = run_output.with_name(run_output.name + "--telemetry")
        job["original_command"] = original
        job["telemetry"] = {
            "output": str(telemetry),
            "interval_s": args.interval,
            "thread_detail_limit": args.thread_limit,
            "source": "Passive OS/NVML; no model imports",
        }
        job["command"] = [
            command[0],
            str(args.sampler),
            "--output",
            str(telemetry),
            "--interval",
            str(args.interval),
            "--thread-limit",
            str(args.thread_limit),
            "--command-json",
            json.dumps(original),
        ]
    document["telemetry_overlay"] = {
        "measurement_kind": "CPU-only plan transformation",
        "source_plan": str(args.plan.resolve()),
        "source_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "sampler_on_execution_machine": str(args.sampler),
        "same_settings_every_job": True,
        "note": "This only writes commands; it launches no job or sampler.",
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"output": str(args.output), "jobs_wrapped": len(document["jobs"])})
    )


if __name__ == "__main__":
    main()
