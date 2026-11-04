"""Synthetic proc/sys fixtures only; no GPU query, model or real benchmark."""

import tempfile
import unittest
from pathlib import Path

import telemetry as meter


class TelemetryProtocol(unittest.TestCase):
    def test_proc_stat_thread_identity_and_parent(self):
        fields = ["S"] + ["0"] * 49
        for index, value in {
            1: 40,
            11: 123,
            12: 17,
            17: 8,
            19: 555,
            20: 4096,
            21: 20,
            36: 7,
        }.items():
            fields[index] = str(value)
        parsed = meter.parse_stat("88 (worker (with) spaces) " + " ".join(fields))
        self.assertEqual(parsed["name"], "worker (with) spaces")
        self.assertEqual(
            (parsed["pid"], parsed["ppid"], parsed["start_ticks"]), (88, 40, 555)
        )
        self.assertEqual((parsed["user_ticks"], parsed["last_cpu"]), (123, 7))
        table = {40: {"ppid": 1}, 88: {"ppid": 40}, 90: {"ppid": 88}, 99: {"ppid": 1}}
        self.assertEqual(meter.descendants([40], table), {40, 88, 90})

    def test_cgroup_v2_pressure_and_throttle_units(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc = root / "proc"
            sysroot = root / "sys"
            (proc / "self").mkdir(parents=True)
            (proc / "self/cgroup").write_text("0::/bench\n")
            cgroup = sysroot / "fs/cgroup/bench"
            cgroup.mkdir(parents=True)
            (cgroup / "cpu.stat").write_text(
                "usage_usec 1000\nnr_throttled 3\nthrottled_usec 200\n"
            )
            (cgroup / "cpu.pressure").write_text(
                "some avg10=1.20 avg60=0.40 avg300=0.1 total=1234\n"
            )
            identity = meter.detect_cgroup(proc, sysroot)
            self.assertEqual(identity["version"], 2)
            values = meter.cgroup_sample(identity)
            self.assertEqual(values["cpu.stat"]["throttled_usec"], 200)
            self.assertEqual(values["cpu.pressure"]["some"]["total"], 1234)
            self.assertEqual(
                meter.delta(values["cpu.stat"], {"throttled_usec": 150}),
                {"throttled_usec": 50},
            )

    def test_per_cpu_frequency_fallback_and_guest_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc = root / "proc"
            sysroot = root / "sys"
            proc.mkdir()
            sysroot.mkdir()
            (proc / "cpuinfo").write_text(
                "processor : 0\ncpu MHz : 2300.5\nprocessor : 1\ncpu MHz : 2800\n"
            )
            self.assertEqual(
                meter.cpu_frequencies(proc, sysroot)["mhz"],
                {"cpu0": 2300.5, "cpu1": 2800},
            )
        old = {"cpu0": [0] * 10}
        new = {"cpu0": [20, 0, 10, 50, 20, 0, 0, 0, 15, 0]}
        self.assertEqual(meter.cpu_usage(new, old)["cpu0"]["busy_pct"], 30)
        self.assertEqual(meter.cpu_usage(new, old)["cpu0"]["iowait_pct"], 20)


if __name__ == "__main__":
    unittest.main()
