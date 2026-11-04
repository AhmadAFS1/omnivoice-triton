"""Exact reviewed API132 PID/starttime cleanup support; no GPU calls."""

import subprocess
import time
from pathlib import Path


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
