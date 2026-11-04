#!/usr/bin/env python3
"""Observe future benchmark jobs without importing CUDA or a model framework.

One persistent nvidia-smi process records GPU CSV at 1Hz. This process samples
Linux CPU/cgroup/process files and records its own measured overhead. Use the
same settings around both comparison arms. Existing workers are never modified.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROC = Path("/proc")
SYS = Path("/sys")
GPU_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "driver_version",
    "pci.bus_id",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.graphics",
    "clocks.current.memory",
    "pstate",
    "clocks_event_reasons.active",
    "clocks_throttle_reasons.active",
]


def read(path):
    try:
        return path.read_text()
    except (OSError, ValueError):
        return None


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_stat(value):
    """Linux /proc/<pid>/stat; comm may contain spaces and parentheses."""
    left, right = value.find("("), value.rfind(")")
    fields = value[right + 1 :].split()
    return {
        "pid": int(value[:left]),
        "name": value[left + 1 : right],
        "state": fields[0],
        "ppid": int(fields[1]),
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "threads": int(fields[17]),
        "start_ticks": int(fields[19]),
        "virtual_bytes": int(fields[20]),
        "resident_pages": int(fields[21]),
        "last_cpu": int(fields[36]),
    }


def key_values(value):
    result = {}
    for line in (value or "").splitlines():
        fields = line.replace(":", " ", 1).split()
        if len(fields) >= 2:
            result[fields[0]] = " ".join(fields[1:])
    return result


def pressure(value):
    result = {}
    for line in (value or "").splitlines():
        fields = line.split()
        if fields:
            result[fields[0]] = {
                key: int(number) if key == "total" else float(number)
                for key, number in (field.split("=") for field in fields[1:])
            }
    return result or None


def stat_numbers(value):
    return {
        key: int(number)
        for key, number in key_values(value).items()
        if number.isdigit()
    }


def delta(current, previous):
    return {
        key: value - previous[key]
        for key, value in current.items()
        if key in previous
        and isinstance(value, (int, float))
        and isinstance(previous[key], (int, float))
        and value >= previous[key]
    }


def detect_cgroup(proc=PROC, sysroot=SYS):
    entries = []
    for line in (read(proc / "self/cgroup") or "").splitlines():
        group, controllers, path = line.split(":", 2)
        entries.append({"hierarchy": group, "controllers": controllers, "path": path})
        if group == "0" and not controllers:
            candidates = [
                sysroot / "fs/cgroup" / path.lstrip("/"),
                sysroot / "fs/cgroup",
            ]
            for candidate in candidates:
                if (candidate / "cpu.stat").exists():
                    return {
                        "version": 2,
                        "directory": str(candidate),
                        "membership": entries,
                    }
    directories = {}
    for item in entries:
        for controller in item["controllers"].split(","):
            for name in [item["controllers"], controller, "cpu,cpuacct", "cpuacct,cpu"]:
                candidate = sysroot / "fs/cgroup" / name / item["path"].lstrip("/")
                if candidate.exists():
                    directories[controller] = str(candidate)
                    break
    return {
        "version": 1 if directories else None,
        "directories": directories,
        "membership": entries,
        "unavailable": not bool(directories),
    }


def cgroup_sample(identity):
    if identity.get("version") == 2:
        directory = Path(identity["directory"])
        result = {}
        for name in [
            "cpu.stat",
            "cpu.max",
            "cpu.weight",
            "cpu.pressure",
            "memory.pressure",
            "io.pressure",
            "memory.current",
            "memory.max",
            "memory.high",
            "memory.events",
            "memory.numa_stat",
            "cpuset.cpus.effective",
            "cpuset.mems.effective",
            "pids.current",
            "pids.max",
        ]:
            value = read(directory / name)
            if value is not None:
                result[name] = (
                    pressure(value)
                    if name.endswith("pressure")
                    else (
                        stat_numbers(value)
                        if name in {"cpu.stat", "memory.events"}
                        else value.strip()
                    )
                )
        return result
    result = {}
    for controller, directory in identity.get("directories", {}).items():
        names = {
            "cpu": ["cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us"],
            "cpuacct": ["cpuacct.usage"],
            "cpuset": ["cpuset.cpus", "cpuset.mems"],
            "memory": [
                "memory.usage_in_bytes",
                "memory.limit_in_bytes",
                "memory.numa_stat",
            ],
        }
        for name in names.get(controller, []):
            value = read(Path(directory) / name)
            if value is not None:
                result[name] = (
                    stat_numbers(value) if name == "cpu.stat" else value.strip()
                )
    return result


def cpu_frequencies(proc=PROC, sysroot=SYS):
    rows = {}
    for path in (sysroot / "devices/system/cpu").glob(
        "cpu[0-9]*/cpufreq/scaling_cur_freq"
    ):
        value = read(path)
        if value and value.strip().isdigit():
            rows[path.parts[-3]] = int(value.strip()) / 1000
    if rows:
        return {"source": "sysfs scaling_cur_freq (kHz converted toMHz)", "mhz": rows}
    current = None
    for line in (read(proc / "cpuinfo") or "").splitlines():
        if line.startswith("processor"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("cpu MHz") and current is not None:
            rows["cpu" + current] = float(line.split(":", 1)[1])
    return {
        "source": "/proc/cpuinfo reported MHz" if rows else "unavailable",
        "mhz": rows,
    }


def cpu_counters(proc=PROC):
    rows = {}
    for line in (read(proc / "stat") or "").splitlines():
        fields = line.split()
        if fields and (fields[0] == "cpu" or fields[0][3:].isdigit()):
            if fields[0].startswith("cpu"):
                rows[fields[0]] = [int(value) for value in fields[1:]]
    return rows


def cpu_usage(current, previous):
    result = {}
    for name, values in current.items():
        if name not in previous:
            continue
        # Linux guest counters are already included in user/nice; exclude them.
        changes = [
            max(0, value - old) for value, old in zip(values[:8], previous[name][:8])
        ]
        total = sum(changes)
        if total:
            result[name] = {
                "busy_pct": 100 * (total - changes[3] - changes[4]) / total,
                "iowait_pct": 100 * changes[4] / total,
                "delta_ticks": changes,
            }
    return result


def static_topology(proc=PROC, sysroot=SYS):
    cpumodel = next(
        (
            line.split(":", 1)[1].strip()
            for line in (read(proc / "cpuinfo") or "").splitlines()
            if line.startswith("model name") and ":" in line
        ),
        None,
    )
    nodes = {
        path.parent.name: (read(path) or "").strip()
        for path in (sysroot / "devices/system/node").glob("node[0-9]*/cpulist")
    }
    gpu_nodes = {}
    for path in (sysroot / "bus/pci/devices").glob("*/vendor"):
        if (read(path) or "").strip() == "0x10de":
            gpu_nodes[path.parent.name] = {
                "numa_node": read(path.parent / "numa_node"),
                "local_cpu_list": read(path.parent / "local_cpulist"),
            }
    return {
        "cpu_model": cpumodel,
        "logical_host_cpu_count": os.cpu_count(),
        "sampler_allowed_cpus": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "numa_node_cpu_lists": nodes,
        "nvidia_pci_numa_affinity": gpu_nodes,
        "physical_memory": key_values(read(proc / "meminfo")).get("MemTotal"),
    }


def process_table(proc=PROC):
    rows = {}
    for path in proc.glob("[0-9]*/stat"):
        value = read(path)
        if value:
            try:
                row = parse_stat(value)
                rows[row["pid"]] = row
            except (ValueError, IndexError):
                continue
    return rows


def descendants(roots, table):
    found = set(roots) & table.keys()
    while True:
        added = {pid for pid, row in table.items() if row["ppid"] in found} - found
        if not added:
            return found
        found.update(added)


def process_sample(pid, row, previous, elapsed, ticks, thread_limit, proc=PROC):
    directory = proc / str(pid)
    status = key_values(read(directory / "status"))
    result = dict(row)
    result["rss_bytes"] = row["resident_pages"] * os.sysconf("SC_PAGE_SIZE")
    result["affinity_cpu_list"] = status.get("Cpus_allowed_list")
    result["allowed_memory_nodes"] = status.get("Mems_allowed_list")
    result["cgroup_membership"] = read(directory / "cgroup")
    result["io_counters"] = stat_numbers(read(directory / "io"))
    result["context_switches"] = {
        key: status.get(key)
        for key in ["voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"]
    }
    key = (pid, row["start_ticks"])
    old = previous.get(key, {})
    if old and elapsed > 0:
        result["cpu_pct_one_core_100"] = (
            (
                row["user_ticks"]
                + row["system_ticks"]
                - old["user_ticks"]
                - old["system_ticks"]
            )
            / ticks
            / elapsed
            * 100
        )
    threads = []
    for path in (directory / "task").glob("[0-9]*/stat"):
        value = read(path)
        if not value:
            continue
        try:
            thread = parse_stat(value)
        except (ValueError, IndexError):
            continue
        prior = previous.get(("thread", pid, thread["pid"], thread["start_ticks"]), {})
        work = thread["user_ticks"] + thread["system_ticks"]
        thread["cpu_delta_ticks"] = max(0, work - prior.get("ticks", work))
        previous[("thread", pid, thread["pid"], thread["start_ticks"])] = {
            "ticks": work
        }
        threads.append(thread)
    result["thread_ids"] = sorted(thread["pid"] for thread in threads)
    result["top_threads"] = sorted(
        threads, key=lambda item: item["cpu_delta_ticks"], reverse=True
    )[:thread_limit]
    for thread in result["top_threads"]:
        thread_status = key_values(
            read(directory / "task" / str(thread["pid"]) / "status")
        )
        thread["affinity_cpu_list"] = thread_status.get("Cpus_allowed_list")
        thread["allowed_memory_nodes"] = thread_status.get("Mems_allowed_list")
    previous[key] = row
    return result


def start_gpu(output, interval, disabled):
    if disabled:
        return None, {"available": False, "reason": "Explicit CPU-only sampler check"}
    try:
        help_text = subprocess.check_output(
            ["nvidia-smi", "--help-query-gpu"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        fields = [name for name in GPU_FIELDS if f'"{name}"' in help_text]
        if "clocks_event_reasons.active" in fields:
            fields = [
                name for name in fields if name != "clocks_throttle_reasons.active"
            ]
        if not fields:
            raise RuntimeError("No requested fields found in nvidia-smi query help")
        command = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
            f"--loop-ms={round(interval * 1000)}",
        ]
        if shutil.which("stdbuf"):
            command = ["stdbuf", "-oL"] + command
        with (output / "gpu.csv").open("w") as csvfile:
            csv.writer(csvfile).writerow(fields)
        with (
            (output / "gpu.csv").open("a") as csvfile,
            (output / "gpu-errors.log").open("w") as err,
        ):
            process = subprocess.Popen(
                command,
                stdout=csvfile,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=os.environ | {"TZ": "UTC"},
            )
        return process, {
            "available": True,
            "fields": fields,
            "command": command,
            "pid": process.pid,
            "units": "nvidia-smi nounits; field schema retained",
        }
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return None, {"available": False, "reason": str(error)}


def stop_child(process, interrupt=False):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT if interrupt else signal.SIGTERM)
        try:
            process.wait(timeout=40 if interrupt else 5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--thread-limit", type=int, default=8)
    parser.add_argument("--track-pid", type=int, action="append", default=[])
    parser.add_argument("--duration", type=float)
    parser.add_argument("--command-json", help="Optional benchmark argv JSON array")
    parser.add_argument(
        "--command-file", type=Path, help="File containing benchmark argv JSON"
    )
    parser.add_argument("--disable-gpu-query", action="store_true")
    args = parser.parse_args()
    assert args.interval >= 1 and args.thread_limit >= 0
    assert not (args.command_json and args.command_file)
    raw_command = args.command_json or (
        args.command_file.read_text() if args.command_file else None
    )
    assert raw_command or (args.duration and args.duration > 0)
    assert not (raw_command and args.duration), "Wrapped jobs run until completion"
    command = json.loads(raw_command) if raw_command else None
    assert command is None or (
        isinstance(command, list)
        and command
        and all(isinstance(value, str) for value in command)
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    start_io = stat_numbers(read(PROC / "self/io"))
    group = detect_cgroup()
    ticks = os.sysconf("SC_CLK_TCK")
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    gpu, gpu_meta = start_gpu(output, args.interval, args.disable_gpu_query)
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    setup_child_cpu = (children_after.ru_utime + children_after.ru_stime) - (
        children_before.ru_utime + children_before.ru_stime
    )
    metadata = {
        "measurement_kind": "Passive OS/NVML telemetry; no CUDA/model imports",
        "started_utc": utc(),
        "monotonic_ns": time.monotonic_ns(),
        "sampler_pid": os.getpid(),
        "platform": platform.platform(),
        "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interval_s": args.interval,
        "thread_detail_limit": args.thread_limit,
        "clock_ticks_per_second": ticks,
        "cgroup": group,
        "topology": static_topology(),
        "gpu": gpu_meta,
        "wrapped_command": command,
        "tracked_root_pids": args.track_pid,
        "scope_notes": [
            "100% processCPU means oneCPU core",
            "Host pressure/load may include work outside this container",
            "CPU MHz is the kernel-reported value, not a guaranteed clock",
            "GPU timestamps use TZ=UTC; physicalVRAM uses nvidia-smi memory.total. "
            "Torch-visibleVRAM is in paired TTS metadata.",
            "NUMA evidence uses allowed nodes and cgroup memory.numa_stat; "
            "no expensive per-process numa_maps walk is performed",
        ],
    }
    write(output / "metadata.json", metadata)
    target = None
    interrupted = None
    rows = 0
    prior = {}
    previous_cpu = {}
    previous_cgroup = {}
    previous_time = started
    next_sample = started
    gpu_last_cpu = 0.0
    overruns = 0

    def receive(signum, _frame):
        raise KeyboardInterrupt(f"Signal{signum}")

    signal.signal(signal.SIGTERM, receive)
    try:
        if command:
            with (output / "wrapped-job.log").open("w") as log:
                target = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            metadata["wrapped_pid"] = target.pid
            write(output / "metadata.json", metadata)
        while True:
            now = time.monotonic()
            if now >= next_sample:
                cpu_before = resource.getrusage(resource.RUSAGE_SELF)
                sample_start = time.perf_counter()
                table = process_table()
                roots = args.track_pid + ([target.pid] if target else [])
                tracked = descendants(roots, table)
                elapsed = now - previous_time
                processes = [
                    process_sample(
                        pid, table[pid], prior, elapsed, ticks, args.thread_limit
                    )
                    for pid in sorted(tracked)
                ]
                own = {
                    pid: table[pid]
                    for pid in [os.getpid(), gpu.pid if gpu else -1]
                    if pid in table
                }
                if gpu and gpu.pid in own:
                    item = own[gpu.pid]
                    gpu_last_cpu = (item["user_ticks"] + item["system_ticks"]) / ticks
                counters = cpu_counters()
                cg = cgroup_sample(group)
                record = {
                    "utc": utc(),
                    "monotonic_ns": time.monotonic_ns(),
                    "from_start_s": now - started,
                    "elapsed_since_previous_s": elapsed,
                    "host_loadavg": list(os.getloadavg()),
                    "host_loadavg_raw": read(PROC / "loadavg"),
                    "host_pressure": {
                        name: pressure(read(PROC / "pressure" / name))
                        for name in ["cpu", "memory", "io"]
                    },
                    "cpu_frequencies": cpu_frequencies(),
                    "cpu_usage": cpu_usage(counters, previous_cpu),
                    "cgroup": cg,
                    "cgroup_cpu_stat_delta": delta(
                        cg.get("cpu.stat", {}), previous_cgroup.get("cpu.stat", {})
                    ),
                    "processes": processes,
                    "sampler_processes": list(own.values()),
                    "sample_collection_wall_ms": (time.perf_counter() - sample_start)
                    * 1000,
                    "gpu_sampler_exit_code": gpu.poll() if gpu else None,
                }
                cpu_after = resource.getrusage(resource.RUSAGE_SELF)
                record["collection_cpu_ms"] = (
                    (cpu_after.ru_utime + cpu_after.ru_stime)
                    - (cpu_before.ru_utime + cpu_before.ru_stime)
                ) * 1000
                append(output / "samples.jsonl", record)
                rows += 1
                previous_cpu, previous_cgroup = counters, cg
                previous_time = now
                next_sample += args.interval
                if time.monotonic() > next_sample:
                    overruns += 1
                    next_sample = time.monotonic() + args.interval
            if target and target.poll() is not None:
                break
            if args.duration and time.monotonic() - started >= args.duration:
                break
            time.sleep(min(0.1, max(0.001, next_sample - time.monotonic())))
    except KeyboardInterrupt as error:
        interrupted = str(error)
    finally:
        stop_child(target, interrupt=True)
        if gpu:
            value = read(PROC / str(gpu.pid) / "stat")
            if value:
                final_gpu = parse_stat(value)
                gpu_last_cpu = (
                    final_gpu["user_ticks"] + final_gpu["system_ticks"]
                ) / ticks
        stop_child(gpu)
        elapsed = time.monotonic() - started
        end_usage = resource.getrusage(resource.RUSAGE_SELF)
        own_cpu = (end_usage.ru_utime + end_usage.ru_stime) - (
            start_usage.ru_utime + start_usage.ru_stime
        )
        write(
            output / "outcome.json",
            {
                "finished_utc": utc(),
                "wall_s": elapsed,
                "samples": rows,
                "sampling_overruns": overruns,
                "interrupted": interrupted,
                "wrapped_exit_code": target.returncode if target else None,
                "sampler_self_cpu_s": own_cpu,
                "gpu_sampler_observed_cpu_s": gpu_last_cpu,
                "sampler_setup_child_cpu_s": setup_child_cpu,
                "observed_total_sampler_cpu_pct_one_core_100": (
                    (own_cpu + gpu_last_cpu + setup_child_cpu) / elapsed * 100
                    if elapsed
                    else None
                ),
                "sampler_max_rss_raw": end_usage.ru_maxrss,
                "sampler_io_counter_delta": delta(
                    stat_numbers(read(PROC / "self/io")), start_io
                ),
                "sampler_max_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
                "overhead_note": "SelfCPU includes sampling/serialization/setup/waits. "
                "Nvidia childCPU uses lifetimeCPU; targetCPU stays separate.",
            },
        )
    if interrupted:
        raise SystemExit(130)
    if target and target.returncode:
        raise SystemExit(target.returncode)


if __name__ == "__main__":
    main()
