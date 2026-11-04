"""Finite modern eager API132 OFF/ON experiment; explicit root GPU lease required."""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import contract_helpers as helper
from owned_process import RecordedProcess, record_servers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-root-authorized", action="store_true")
    args = parser.parse_args()
    config = helper.read(HERE / "config.json")
    if not args.execute_root_authorized:
        print(json.dumps(config, indent=2))
        return
    lease = helper.read(HERE / "authorization.json")
    assert lease["root_authorized_modern264"] and lease["CPU_IO_quiet"]
    assert lease["source_ref"] == config["source_ref"]
    assert not (HERE / "execution.json").exists()
    helper.check(config)
    stop_path = ROOT / "run_prefix_micro.py"
    assert helper.sha(stop_path) == config["stop_helper_sha256"]
    spec = importlib.util.spec_from_file_location("owned_stop", stop_path)
    stopper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stopper)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    helper.write(
        HERE / "execution.json",
        {
            "at": helper.now(),
            "lease": lease,
            "config_sha256": helper.sha(HERE / "config.json"),
            "controller_sha256": helper.sha(Path(__file__)),
        },
    )
    started = time.monotonic()
    status, outcomes, baseline = "failed", [], None
    try:
        for job in config["jobs"]:
            helper.check(config)
            remaining = config["budget_s"] - (time.monotonic() - started)
            assert remaining > 60, "Insufficient remaining finite budget for next arm"
            output = ROOT / "results" / job["id"]
            assert not output.exists()
            command = [
                job["command"][0],
                str(ROOT / "load/telemetry.py"),
                "--output",
                str(output.with_name(output.name + "--telemetry")),
                "--interval",
                "1",
                "--thread-limit",
                "8",
                "--command-json",
                json.dumps(job["command"]),
            ]
            child, servers = None, {}
            row = {"id": job["id"], "fused_rope": job["fused_rope"], "at": helper.now()}
            outcomes.append(row)
            try:
                with (HERE / (job["id"] + ".log")).open("w") as log:
                    child = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=dict(
                            os.environ,
                            HF_HOME=str(ROOT / "hf-cache"),
                            HF_HUB_OFFLINE="1",
                        ),
                    )
                    helper.write(HERE / "current.json", row | {"pid": child.pid})
                    deadline = started + config["budget_s"] - 30
                    while child.poll() is None:
                        record_servers(child.pid, servers)
                        helper.write(HERE / "owned-servers.json", servers)
                        if time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(command, remaining - 30)
                        time.sleep(0.5)
                    code = child.wait(timeout=1)
                row["exit_code"] = code
                assert code == 0, "Preserve failed arm; no hidden retry"
                requests, checks = helper.verify(output, job["command"])
                row.update(checks)
                assert checks["status_counts"] == {"200": 132}
                assert checks["basic_valid"] == checks["harness_signal_valid"] == 132
                actual = checks["health_after"]["inference_optimizations"]
                assert actual["fused_rope_enabled"] == job["fused_rope"]
                assert actual["fused_rope_ready"] == job["fused_rope"]
                if job["fused_rope"]:
                    rotary = actual["fused_rope_metrics"]
                    assert rotary["fused_calls"] > 0 and not rotary["fallbacks"]
                    assert baseline is not None and len(baseline) == 132
                    paired = [
                        {
                            "id": r["case"]["id"],
                            "off_sha256": baseline[r["case"]["id"]],
                            "on_sha256": r["sha256"],
                            "exact": r["sha256"] == baseline[r["case"]["id"]],
                        }
                        for r in requests
                    ]
                    helper.write(output / "paired-off-on.json", paired)
                    row["exact_WAVs"] = sum(x["exact"] for x in paired)
                    assert row["exact_WAVs"] == 132
                else:
                    baseline = {r["case"]["id"]: r["sha256"] for r in requests}
            finally:
                if child is not None:
                    stopper.stop_tree(child)
                for pid, stamp in servers.items():
                    process = RecordedProcess(pid, stamp)
                    if process.current():
                        stopper.stop_tree(process)
                row["finished_at"] = helper.now()
                helper.write(HERE / "outcomes.json", outcomes)
            helper.idle()
        status = "completed_exact_132_paired264"
    finally:
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        helper.write(
            ROOT / "MODERN_FUSED_ROPE_EAGER264_COMPLETE.json",
            {
                "at": helper.now(),
                "status": status,
                "outcomes": outcomes,
                "elapsed_s": time.monotonic() - started,
                "GPU_idle": not bool(processes),
                "compute_processes": processes,
                "source_ref": config["source_ref"],
                "interpretation": "Paired eager B1 API latency and exact bodies; not novel burst capacity",
            },
        )


if __name__ == "__main__":
    main()
