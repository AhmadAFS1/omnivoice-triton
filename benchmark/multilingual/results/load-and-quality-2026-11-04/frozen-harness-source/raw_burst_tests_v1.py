"""CPU-only raw-HTTP parser/socket/counter protocol tests, never a TTS test."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.parse
import wave
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RAW = module("raw_burst", "raw_burst_v1.py")
SERVER = module("raw_server", "raw_burst_server_v1.py")


def wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(struct.pack("<240h", *([1000, -1000] * 120)))
    return buffer.getvalue()


def response(body, status=200, extra=b""):
    return (
        f"HTTP/1.1 {status} Test\r\nContent-Length: {len(body)}\r\n".encode()
        + extra
        + b"\r\n"
        + body
    )


class ParserTests(unittest.IsolatedAsyncioTestCase):
    async def parse(self, body, **kwargs):
        reader = asyncio.StreamReader(limit=RAW.HEADER_LIMIT)
        reader.feed_data(body)
        reader.feed_eof()
        return await RAW.read_response(reader, **kwargs)

    async def test_complete_binary_wav_and_non200_bodies(self):
        for status, body in (
            (200, wav()),
            (429, b"overload"),
            (500, b"ordinary failure"),
        ):
            with self.subTest(status=status):
                actual, headers, received = await self.parse(response(body, status))
                self.assertEqual(actual, status)
                self.assertEqual(
                    hashlib.sha256(received).digest(), hashlib.sha256(body).digest()
                )
                self.assertEqual(int(headers["content-length"]), len(body))

    async def test_rejects_chunked_duplicate_missing_signed_or_invalid_length(self):
        inputs = [
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: +0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: x\r\n\r\n",
            b"HTTP/1.1 100 Continue\r\nContent-Length: 0\r\n\r\n",
        ]
        for value in inputs:
            with self.subTest(value=value), self.assertRaises(RAW.ResponseError):
                await self.parse(value)

    async def test_truncated_body_preserves_status_partial_bytes_and_budget(self):
        budget = RAW.RAMBudget(100)
        with self.assertRaises(RAW.ResponseError) as caught:
            await self.parse(
                b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort", budget=budget
            )
        error = caught.exception
        self.assertEqual(
            (error.category, error.status, error.body, error.declared),
            ("truncated", 200, b"short", 10),
        )
        self.assertEqual((budget.peak, budget.reserved), (10, 5))

    async def test_body_and_aggregate_caps_reject_before_allocating(self):
        with self.assertRaises(RAW.ResponseError) as caught:
            await self.parse(response(b"1234"), max_body=3)
        self.assertEqual(caught.exception.category, "body_limit")
        budget = RAW.RAMBudget(3)
        with self.assertRaises(RAW.ResponseError) as caught:
            await self.parse(response(b"1234"), budget=budget)
        self.assertEqual(caught.exception.category, "RAM_limit")
        self.assertEqual(budget.reserved, 0)

    async def test_cancelled_read_retains_received_bytes(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\npart")
        task = asyncio.create_task(RAW.read_response(reader))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(RAW.ResponseError) as caught:
            await task
        self.assertEqual(caught.exception.category, "cancelled")
        self.assertEqual(caught.exception.body, b"part")


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.post_connections = []
        self.connections = 0
        self.writers = []
        self.body = wav()

        async def handler(reader, writer):
            self.connections += 1
            connection = self.connections
            self.writers.append(writer)
            try:
                while True:
                    head = await reader.readuntil(b"\r\n\r\n")
                    lines = head.split(b"\r\n")
                    _method, target, _ = lines[0].split()
                    headers = dict(
                        line.split(b":", 1) for line in lines[1:] if b":" in line
                    )
                    body = await reader.readexactly(
                        int(headers.get(b"Content-Length", b"0"))
                    )
                    if target == b"/healthz":
                        writer.write(response(b'{"ok":true}'))
                        await writer.drain()
                        continue
                    self.post_connections.append(connection)
                    mode = urllib.parse.parse_qs(body.decode()).get("fixture", ["ok"])[
                        0
                    ]
                    if mode == "truncated":
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial"
                        )
                    elif mode == "chunked":
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
                        )
                    elif mode == "429":
                        writer.write(response(b"overload", 429))
                    elif mode == "500":
                        writer.write(response(b"ordinary failure", 500))
                    else:
                        writer.write(
                            response(self.body, extra=b"Connection: close\r\n")
                        )
                    await writer.drain()
                    break
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                writer.close()

        self.server = await asyncio.start_server(handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        for writer in self.writers:
            writer.close()
        await asyncio.gather(
            *(writer.wait_closed() for writer in self.writers), return_exceptions=True
        )

    async def test_preconnected_distinct_socket_cohort_preserves_all_outcomes(self):
        modes = ["ok", "429", "500", "truncated", "chunked", "close"]
        streams, pre, ok = await RAW.preconnect(
            "127.0.0.1", self.port, len(modes), width=2
        )
        self.assertTrue(ok)
        self.assertEqual(len({tuple(row["local"]) for row in pre}), len(modes))
        payloads = [
            RAW.raw_request(
                "POST",
                "/generate",
                f"127.0.0.1:{self.port}",
                urllib.parse.urlencode({"fixture": mode}).encode(),
                close=True,
            )
            for mode in modes
        ]
        rows, bodies, info = await RAW.cohort(
            streams, payloads, [{"id": mode} for mode in modes], timeout=5
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(set(self.post_connections)), 6)
        self.assertEqual(self.connections, 6)  # No replacement/retry sockets.
        self.assertEqual([row["status"] for row in rows[:4]], [200, 429, 500, 200])
        self.assertFalse(rows[3]["full_body_received"])
        self.assertFalse(rows[4]["full_body_received"])
        self.assertEqual(bodies[3], b"partial")
        self.assertEqual(bodies[0], self.body)
        self.assertEqual(bodies[5], self.body)
        self.assertTrue(
            all(row["inference_executed_header"] == "unreported" for row in rows)
        )
        self.assertFalse(info["timed_out"])


class CounterTests(unittest.TestCase):
    def test_benchmark_counters_preserve_return_and_exception(self):
        class Batcher:
            def _execute_batch(self, batch):
                for request in batch:
                    request.future.set_result("original output")
                return "original return"

            def queue_snapshot(self):
                return {"pending_requests": 0}

        SERVER.install_counters(Batcher)
        batcher = Batcher()
        futures = [SimpleNamespace(future=Future()) for _ in range(2)]
        self.assertEqual(batcher._execute_batch(futures), "original return")
        self.assertEqual(futures[0].future.result(), "original output")
        counters = batcher.queue_snapshot()["benchmark_raw_burst_v1"]
        self.assertEqual(
            counters,
            {
                "batches_started": 1,
                "members_started": 2,
                "members_succeeded": 2,
                "members_failed": 0,
            },
        )

        class Failing(Batcher):
            def _execute_batch(self, batch):
                raise RuntimeError("unchanged error")

        SERVER.install_counters(Failing)
        failing = Failing()
        with self.assertRaisesRegex(RuntimeError, "unchanged error"):
            failing._execute_batch([SimpleNamespace(future=Future())])
        self.assertEqual(
            failing.queue_snapshot()["benchmark_raw_burst_v1"]["members_failed"], 1
        )

    def test_default_dry_checks_fixtures_without_neural_imports(self):
        guard = """
import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','numpy','soundfile','omnivoice','omnivoice_triton'}:
  raise RuntimeError('Dry plan attempted neural/dependency import:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", guard, str(HERE / "raw_burst_v1.py")],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        value = json.loads(result.stdout)
        self.assertFalse(value["GPU_executed"])
        self.assertEqual(value["config"]["requests"], 5000)
        self.assertEqual(value["config"]["cohort_inputs"], 1000)


class CleanupTests(unittest.TestCase):
    def test_surviving_descendant_is_stopped_after_group_leader_exits(self):
        # A real subprocess starts a same-group descendant, then exits first.
        # The descendant ignores TERM so the KILL path must still target the
        # group even though Popen.poll() reports the leader already completed.
        program = """
import subprocess,sys
child=subprocess.Popen([sys.executable,'-c',
 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'],
 stdout=subprocess.PIPE,text=True)
assert child.stdout.readline().strip()=='ready'
print(child.pid,flush=True)
"""
        leader = subprocess.Popen(
            [sys.executable, "-c", program],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        descendant = int(leader.stdout.readline().strip())
        leader.wait(timeout=5)
        self.assertIsNotNone(leader.poll())
        try:
            os.kill(descendant, 0)  # Descendant survived its completed leader.
            RAW.stop_owned_group(leader, term_timeout=0.01, kill_timeout=0.01)
            deadline = time.monotonic() + 3
            live = True
            while live and time.monotonic() < deadline:
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(descendant)],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                live = bool(state and not state.startswith("Z"))
                if live:
                    time.sleep(0.01)
            self.assertFalse(live, "Same-group descendant remained live")
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            leader.stdout.close()

    def test_early_server_pid_metadata_error_still_stops_server(self):
        class FakeServer:
            pid = 12345
            returncode = None

            def __init__(self):
                self.signals = []
                self.waited = False

            def poll(self):
                return self.returncode

            def send_signal(self, number):
                self.signals.append(number)

            def wait(self, timeout):
                self.waited = True
                self.returncode = 0
                return 0

        server = FakeServer()
        fake_legacy = SimpleNamespace(
            build_command=lambda args: [sys.executable, "seeded-placeholder.py"]
        )
        original_write = RAW.write

        def fail_pid(path, value):
            if path.name == "server-pid.json":
                raise OSError("injected early metadata failure")
            return original_write(path, value)

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                output=Path(folder), source=Path(folder), model=Path(folder), port=1
            )
            with (
                patch.object(RAW, "import_file", return_value=fake_legacy),
                patch.object(RAW.subprocess, "Popen", return_value=server),
                patch.object(RAW, "write", side_effect=fail_pid),
                self.assertRaisesRegex(OSError, "injected early metadata failure"),
            ):
                RAW.child(args, {}, [], [])
            self.assertEqual(server.signals, [signal.SIGKILL])
            self.assertTrue(server.waited)
            self.assertIn(
                "injected early metadata failure",
                (Path(folder) / "child-failure.json").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
