"""One finite API startup/132-wave parity gate; root authorization required."""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


class RecordedProcess:
    """Small stop_tree adapter, qualified by PID plus kernel start time."""

    def __init__(self, pid, started):
        self.pid, self.started = pid, started

    def current(self):
        return process_start(self.pid) == self.started

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.current() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.current():
            raise subprocess.TimeoutExpired(str(self.pid), timeout)


def record_servers(leader, records):
    table = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    rows = [line.split(None, 2) for line in table.splitlines()[1:]]
    owned = {leader}
    while True:
        expanded = owned | {int(r[0]) for r in rows if int(r[1]) in owned}
        if expanded == owned:
            break
        owned = expanded
    for row in rows:
        if int(row[0]) in owned and len(row) == 3 and "seeded_server.py" in row[2]:
            pid = int(row[0])
            started = process_start(pid)
            if started is not None:
                records[pid] = started


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-root-authorized", action="store_true")
    args = parser.parse_args()
    helper = load_module(
        "previous_gate", ROOT / "quality/primary-deferred-capture132-v1/run.py"
    )
    config = helper.read(HERE / "config.json")
    if not args.execute_root_authorized:
        print(json.dumps(config, indent=2))
        return
    lease = helper.read(HERE / "authorization.json")
    assert lease["root_authorized_API132"] and lease["CPU_IO_quiet"]
    assert lease["source_ref"] == config["source_ref"]
    assert not (HERE / "execution.json").exists()
    model = helper.read(ROOT / "FUSED_ROPE_PRODUCT_MODEL_JOINED_PARITY.json")
    assert model["source_ref"] == config["source_ref"]
    assert model["six_pairs_exact"] and model["distinct_complete_pairs"] == 6
    helper.check(config)
    stopper = load_module("owned_stop", ROOT / "run_prefix_micro.py")
    assert helper.sha(ROOT / "run_prefix_micro.py") == config["stop_helper_sha256"]

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    helper.write(
        HERE / "execution.json",
        {
            "at": helper.now(),
            "lease": lease,
            "controller_sha256": helper.sha(Path(__file__)),
            "config_sha256": helper.sha(HERE / "config.json"),
        },
    )
    child, servers = None, {}
    started = time.monotonic()
    status, result = "failed", {}
    try:
        job = config["jobs"][0]
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
        with (HERE / "API132.log").open("w") as log:
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
            helper.write(HERE / "current.json", {"pid": child.pid})
            deadline = time.monotonic() + 740
            while child.poll() is None:
                record_servers(child.pid, servers)
                helper.write(HERE / "owned-servers.json", servers)
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, 740)
                time.sleep(0.5)
            code = child.wait(timeout=1)
        assert code == 0, "Preserve failed gate; no hidden retry"
        requests, result = helper.verify(output, job["command"])
        assert result["status_counts"] == {"200": 132}
        assert result["basic_valid"] == result["harness_signal_valid"] == 132
        baseline = helper.read(
            ROOT / "quality/primary-deferred-capture132-v1/N0_BASELINE_VALID.json"
        )["sha256_by_case_id"]
        paired = [
            {
                "case_id": row["case"]["id"],
                "baseline_sha256": baseline[row["case"]["id"]],
                "fused_sha256": row["sha256"],
                "exact": row["sha256"] == baseline[row["case"]["id"]],
            }
            for row in requests
        ]
        helper.write(output / "paired-to-source656-N0.json", paired)
        result["exact_baseline_WAVs"] = sum(row["exact"] for row in paired)
        metrics = result["health_after"]["inference_optimizations"]
        assert metrics["fused_rope_enabled"] and metrics["fused_rope_ready"]
        rotary = metrics["fused_rope_metrics"]
        assert rotary["forward_calls"] > 0 and rotary["fused_calls"] > 0
        assert not rotary["fallbacks"]
        assert result["exact_baseline_WAVs"] == 132
        status = "completed_exact_132"
    finally:
        if child is not None:
            stopper.stop_tree(child)
        for pid, started_at in servers.items():
            recorded = RecordedProcess(pid, started_at)
            if recorded.current():
                stopper.stop_tree(recorded)
        active = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        helper.write(
            ROOT / "FUSED_ROPE_PRODUCT_API132_COMPLETE.json",
            {
                "at": helper.now(),
                "status": status,
                "results": result,
                "elapsed_s": time.monotonic() - started,
                "compute_processes": active,
                "GPU_idle": not bool(active),
                "source_archive_sha256": config["source_ref"],
                "no_quality_improvement_claim": True,
            },
        )


if __name__ == "__main__":
    main()
