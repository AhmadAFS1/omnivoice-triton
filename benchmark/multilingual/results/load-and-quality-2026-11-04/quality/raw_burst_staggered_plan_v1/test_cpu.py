"""CPU-only denominator and fixed-input plan checks; no model imports."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("staggered_control", HERE / "run.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class MetricTests(unittest.TestCase):
    def test_late_fast_body_fails_scheduledarrival_500ms_SLO(self):
        row = {
            "status": 200,
            "audio": {"signal_valid": True},
            "complete_ms": 200,
            "dispatch_lateness_ms": 400,
            "completion_from_barrier_ms": 600,
        }
        result = control.phase_readout([row], 2)
        self.assertEqual(result["valid_audio_within500ms_alloffers"], 1)
        self.assertEqual(
            result["valid_audio_scheduledarrival_within500ms_alloffers"], 0
        )
        self.assertEqual(result["valid_audio_scheduledarrival_within2s_alloffers"], 1)
        self.assertEqual(result["HTTP200_scheduledarrival_only_p95_ms"], 600)
        self.assertEqual(result["missing_or_not_dispatched"], 1)

    def test_recorded_separate_server_group_stops_after_leader_exits(self):
        code = """
import subprocess,sys
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
print(child.pid,flush=True)
sys.stdin.readline()
"""
        leader = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        server = int(leader.stdout.readline().strip())
        groups = set()
        try:
            groups = control.owned_groups(leader.pid)
            self.assertIn(server, groups)
            self.assertIn(leader.pid, groups)
            leader.stdin.write("exit\n")
            leader.stdin.flush()
            leader.wait(timeout=5)
            control.stop_groups(groups)
            deadline = time.monotonic() + 3
            live = True
            while live and time.monotonic() < deadline:
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(server)],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                live = bool(state and not state.startswith("Z"))
                if live:
                    time.sleep(0.01)
            self.assertFalse(live)
        finally:
            for pid in (server, leader.pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            leader.wait(timeout=5)
            leader.stdin.close()
            leader.stdout.close()

    def test_rejections_and_invalid_fast_audio_are_not_successful_throughput(self):
        rows = [
            {
                "status": 200,
                "audio": {"signal_valid": True},
                "complete_ms": 600,
                "completion_from_barrier_ms": 1000,
            },
            {
                "status": 200,
                "audio": {"signal_valid": False},
                "complete_ms": 1,
                "completion_from_barrier_ms": 1000,
            },
            {"status": 429, "complete_ms": 0.1, "completion_from_barrier_ms": 1000},
            {
                "status": None,
                "error": "timeout",
                "complete_ms": 1000,
                "completion_from_barrier_ms": 1000,
            },
        ]
        report = control.phase_readout(rows, 5)
        self.assertEqual(report["valid_audio_per_offer"], 0.2)
        self.assertEqual(report["valid_audio_per_network_wall_s"], 1)
        self.assertEqual(report["valid_audio_within500ms_alloffers"], 0)
        self.assertEqual(report["missing_or_not_dispatched"], 1)
        self.assertEqual(report["rejected429"], 1)
        self.assertEqual(report["transport_errors"], 1)

    def test_unexecuted_phase_has_zero_success_and_full_missing_denominator(self):
        report = control.phase_readout([], 15000)
        self.assertEqual(report["HTTP200_signal_valid"], 0)
        self.assertEqual(report["missing_or_not_dispatched"], 15000)
        self.assertIsNone(report["valid_audio_per_network_wall_s"])

    def test_fixed_rates_counts_pool_and_pending_limits(self):
        config = json.loads((HERE / "config.json").read_text())
        self.assertEqual(
            [row["rate_rps"] for row in config["schedule"]], [0.5, 5, 50, 250]
        )
        self.assertEqual(
            [row["offers"] for row in config["schedule"]], [30, 300, 3000, 15000]
        )
        command = config["command"]
        self.assertEqual(command[command.index("--max-pending-requests") + 1], "4")
        self.assertEqual(command[command.index("--connections") + 1], "256")
        self.assertEqual(command[command.index("--collect-ms") + 1], "0")
        self.assertEqual(command[command.index("--response-cache-max-bytes") + 1], "0")
        self.assertEqual(config["total_bound_s"], 360)


if __name__ == "__main__":
    unittest.main()
