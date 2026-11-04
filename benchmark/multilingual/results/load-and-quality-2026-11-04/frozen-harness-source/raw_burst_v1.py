"""Strict preconnected raw-HTTP5k load assay; CPU dry plan by default.

5000 inference requests cycle1000 published inputs, never called5000 novel inputs.
Only an exact root lease enables the owned2400s modern-GPU controller.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COUNT = 5000
SOURCE = "ce48b253e66b951d87ea42919f945828c62e96a4"
GPU_UUID = "GPU-5aada52f-9af2-4506-47f7-75e99aaf51ee"
HEADER_LIMIT = 65536
BODY_LIMIT = 2 * 1024**2
RAM_LIMIT = 1024**3
PHASE = "raw5000-cyclic-original1000"


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def streaming_sha(path, deadline):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            assert time.monotonic() < deadline, "Model-asset preflight deadline"
            result.update(chunk)
    return result.hexdigest()


class ResponseError(Exception):
    def __init__(
        self,
        message,
        *,
        category="protocol",
        status=None,
        headers=None,
        body=b"",
        head=b"",
        declared=None,
    ):
        super().__init__(message)
        self.category, self.status, self.headers = category, status, headers or {}
        self.body, self.head, self.declared = body, head, declared


class RAMBudget:
    def __init__(self, limit=RAM_LIMIT):
        self.limit, self.reserved, self.peak = limit, 0, 0

    def reserve(self, size):
        if self.reserved + size > self.limit:
            raise ResponseError(
                "Aggregate body RAM budget exceeded", category="RAM_limit"
            )
        self.reserved += size
        self.peak = max(self.peak, self.reserved)

    def shrink(self, reserved, retained):
        self.reserved -= reserved - retained


async def read_response(reader, budget=None, *, max_body=BODY_LIMIT):
    """Read exactly the declared response body; never parse chunks/EOF as success."""
    status, headers, head, declared, reserved = None, {}, b"", None, 0
    chunks = []
    received = 0
    try:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.LimitOverrunError, asyncio.IncompleteReadError) as error:
            raise ResponseError(
                "Truncated/oversize HTTP header", head=getattr(error, "partial", b"")
            ) from error
        if len(head) > HEADER_LIMIT:
            raise ResponseError("HTTP header limit exceeded", head=head[:HEADER_LIMIT])
        lines = head[:-4].split(b"\r\n")
        match = re.fullmatch(rb"HTTP/1\.[01] ([2-5][0-9]{2})(?: [^\r\n]*)?", lines[0])
        if not match:
            raise ResponseError("Expected final HTTP/1.x status")
        status = int(match[1])
        for line in lines[1:]:
            if b":" not in line or line.startswith((b" ", b"\t")):
                raise ResponseError("Malformed/folded HTTP header")
            key, value = line.split(b":", 1)
            if not re.fullmatch(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key):
                raise ResponseError("Invalid HTTP header name")
            name = key.decode("ascii").lower()
            if name in headers:
                raise ResponseError("Duplicate HTTP header: " + name)
            if any(byte < 32 and byte != 9 or byte == 127 for byte in value):
                raise ResponseError("Control byte in HTTP header")
            headers[name] = value.decode("latin1").strip()
        if "transfer-encoding" in headers:
            raise ResponseError(
                "Transfer-Encoding is unsupported; strict Content-Length required"
            )
        if headers.get("content-encoding", "identity").lower() != "identity":
            raise ResponseError("Compressed response is unsupported")
        length = headers.get("content-length", "")
        if not length.isascii() or not re.fullmatch(r"[0-9]+", length):
            raise ResponseError("Missing/invalid Content-Length")
        declared = int(length)
        if declared > max_body:
            raise ResponseError(
                "Declared body exceeds per-response cap", category="body_limit"
            )
        if budget:
            budget.reserve(declared)
            reserved = declared
        while received < declared:
            part = await reader.read(min(65536, declared - received))
            if not part:
                raise ResponseError(
                    "EOF before complete Content-Length body", category="truncated"
                )
            chunks.append(part)
            received += len(part)
        return status, headers, b"".join(chunks)
    except BaseException as error:
        body = b"".join(chunks)
        if budget and reserved:
            budget.shrink(reserved, len(body))
        if isinstance(error, ResponseError):
            error.status, error.headers = (
                status or error.status,
                headers or error.headers,
            )
            error.body, error.head, error.declared = (
                body or error.body,
                head or error.head,
                declared,
            )
            raise
        if isinstance(error, asyncio.CancelledError):
            raise ResponseError(
                "Request cancelled/deadline before full body",
                category="cancelled",
                status=status,
                headers=headers,
                body=body,
                head=head,
                declared=declared,
            ) from error
        raise ResponseError(
            repr(error),
            category="transport",
            status=status,
            headers=headers,
            body=body,
            head=head,
            declared=declared,
        ) from error


def raw_request(method, target, authority, body=b"", *, close=False):
    assert method in {"GET", "POST"} and target in {"/healthz", "/generate"}
    headers = {
        "Host": authority,
        "Connection": "close" if close else "keep-alive",
        "Accept-Encoding": "identity",
        "Content-Length": str(len(body)),
    }
    if method == "POST":
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return (
        f"{method} {target} HTTP/1.1\r\n".encode()
        + b"".join(f"{key}: {value}\r\n".encode() for key, value in headers.items())
        + b"\r\n"
        + body
    )


def parameters(case, prompt_ids):
    return {
        "mode": case["mode"],
        "text": case["text"],
        "language": case["api_language"],
        "num_step": "16",
        "speed": "1.0",
        "denoise": "true",
        "postprocess_mode": "off",
        "insert_pauses": "false",
        "prompt_id": prompt_ids[case["reference_id"]],
    }


async def preconnect(host, port, count, *, timeout=120, width=256):
    streams, rows = [None] * count, [None] * count
    semaphore = asyncio.Semaphore(width)
    authority = f"{host}:{port}"

    async def one(index):
        writer = None
        started = time.perf_counter()
        try:
            async with semaphore:
                reader, writer = await asyncio.open_connection(
                    host, port, limit=HEADER_LIMIT
                )
                writer.get_extra_info("socket").setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
                )
                writer.write(raw_request("GET", "/healthz", authority))
                await writer.drain()
                status, headers, _ = await read_response(reader, max_body=65536)
                if (
                    status != 200
                    or headers.get("connection", "").lower() == "close"
                    or reader.at_eof()
                ):
                    raise RuntimeError("Preconnected socket cannot be reused")
                streams[index] = (reader, writer)
                rows[index] = {
                    "index": index,
                    "status": status,
                    "fd": writer.get_extra_info("socket").fileno(),
                    "local": list(writer.get_extra_info("sockname")),
                    "peer": list(writer.get_extra_info("peername")),
                    "setup_ms": (time.perf_counter() - started) * 1000,
                }
        except Exception as error:  # noqa: BLE001 -- preserve every preconnect failure; never retry
            rows[index] = {"index": index, "error": repr(error)}
            if writer is not None:
                writer.close()

    try:
        await asyncio.wait_for(
            asyncio.gather(*(one(i) for i in range(count))), timeout=timeout
        )
    except TimeoutError:
        for index, row in enumerate(rows):
            if row is None:
                rows[index] = {
                    "index": index,
                    "error": "Preconnect deadline before reusable socket proof",
                }
        await close_streams(streams)
        return streams, rows, False
    except BaseException:
        await close_streams(streams)
        raise
    valid = [row for row in rows if row and "error" not in row]
    ok = (
        len(valid) == count
        and len({row["fd"] for row in valid}) == count
        and len({tuple(row["local"]) for row in valid}) == count
    )
    if not ok:
        await close_streams(streams)
    return streams, rows, ok


async def close_streams(streams):
    writers = [pair[1] for pair in streams if pair]
    for writer in writers:
        writer.close()
    await asyncio.gather(
        *(writer.wait_closed() for writer in writers), return_exceptions=True
    )


async def cohort(
    streams, payloads, cases, *, timeout=2200, fatal_event=None, ram_limit=RAM_LIMIT
):
    count = len(streams)
    assert count == len(payloads) == len(cases)
    budget, event = RAMBudget(ram_limit), asyncio.Event()
    bodies, rows = (
        {},
        [
            {
                "id": f"{PHASE}-{i:05d}",
                "phase": PHASE,
                "index": i,
                "status": None,
                "dispatched_from_barrier_ms": None,
                "case": case,
                "cycle_index": i // 1000,
                "original_input_index": i % 1000,
                "no_retries": True,
            }
            for i, case in enumerate(cases)
        ],
    )
    barrier, barrier_unix = 0.0, 0.0

    async def one(index):
        row, pair = rows[index], streams[index]
        await event.wait()
        started = time.perf_counter()
        row.update(
            started_at=now(),
            dispatched_from_barrier_ms=(started - barrier) * 1000,
            phase_barrier_unix=barrier_unix,
            new_tcp_connections=0,
        )
        try:
            reader, writer = pair
            row["POST_write_called_from_barrier_ms"] = (
                time.perf_counter() - barrier
            ) * 1000
            writer.write(payloads[index])
            await writer.drain()
            row["POST_drain_finished_from_barrier_ms"] = (
                time.perf_counter() - barrier
            ) * 1000
            status, headers, body = await read_response(reader, budget)
            row["network_finished_perf"] = time.perf_counter()
            row.update(
                status=status,
                headers=headers,
                full_body_received=True,
                response_bytes=len(body),
            )
            bodies[index] = body
            if (
                fatal_event is not None
                and status >= 500
                and any(
                    word in body.lower()
                    for word in (
                        b"cuda error",
                        b"cudagraphcaptureerror",
                        b"device-side assert",
                        b"cuda out of memory",
                    )
                )
            ):
                fatal_event.set()
        except ResponseError as error:
            row.update(
                status=error.status,
                headers=error.headers,
                error=str(error),
                error_category=error.category,
                full_body_received=False,
                declared_content_length=error.declared,
                response_bytes=len(error.body),
                raw_response_head=error.head.decode("latin1"),
            )
            if error.body:
                bodies[index] = error.body
        except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 -- every socket outcome is retained
            row.update(
                error=repr(error),
                error_category="cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "transport",
                full_body_received=False,
            )
        finally:
            finished = row.get("network_finished_perf", time.perf_counter())
            row.update(
                complete_ms=(finished - started) * 1000,
                completion_from_barrier_ms=(finished - barrier) * 1000,
                network_finished_perf=finished,
                completed_at=now(),
            )
            pair[1].close()

    tasks = [asyncio.create_task(one(i)) for i in range(count)]
    await asyncio.sleep(0)
    barrier, barrier_unix = time.perf_counter(), time.time()
    event.set()
    gather = asyncio.gather(*tasks, return_exceptions=True)
    fatal = asyncio.create_task(fatal_event.wait()) if fatal_event is not None else None
    timed_out, fatal_stop = False, False
    try:
        wait_for = {gather} | ({fatal} if fatal is not None else set())
        done, _ = await asyncio.wait(
            wait_for, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        timed_out = not done
        fatal_stop = fatal is not None and fatal in done and fatal.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if fatal is not None:
            fatal.cancel()
            await asyncio.gather(fatal, return_exceptions=True)
        await close_streams(streams)
    finished = time.perf_counter()
    for row in rows:
        if row["dispatched_from_barrier_ms"] is None:
            row.update(
                error="Cancelled before POST dispatch",
                error_category="not_dispatched",
                full_body_received=False,
            )
        row["inference_executed_header"] = row.get("headers", {}).get(
            "x-omnivoice-inference-executed", "unreported"
        )
    return (
        rows,
        bodies,
        {
            "cohort_wall_s": finished - barrier,
            "barrier_unix": barrier_unix,
            "timed_out": timed_out,
            "fatal_CUDA_stop": fatal_stop,
            "body_RAM_reserved_peak_bytes": budget.peak,
            "body_RAM_limit_bytes": ram_limit,
        },
    )


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_checks(args, config):
    assert config["source_ref"] == SOURCE and config["requests"] == COUNT
    assert config["cohort_inputs"] == 1000 and config["warmup_inputs"] == 10
    for name, expected in config["frozen_files"].items():
        assert sha(ROOT / name) == expected, name
    if args.source.exists():
        assert {
            str(path.relative_to(args.source))
            for path in (args.source / "src").rglob("*.py")
        } == {row["path"] for row in config["source_files"]}
    for row in config["source_files"]:
        if args.source.exists():
            actual = sha(args.source / row["path"])
        else:
            raw = subprocess.check_output(
                ["git", "-C", str(args.source_git), "show", SOURCE + ":" + row["path"]],
                timeout=10,
            )
            actual = hashlib.sha256(raw).hexdigest()
        assert actual == row["sha256"], row["path"]
    corpus = read(ROOT / config["corpus"])
    corpus = corpus["cases"] if isinstance(corpus, dict) else corpus
    by_id = {case["id"]: case for case in corpus}
    phases = read(ROOT / config["plan"])["phases"]
    warm = [by_id[key] for key in phases[0]["case_ids"]]
    original = [by_id[key] for key in phases[1]["case_ids"]]
    assert len(warm) == 10 and len(original) == 1000
    signature = lambda case: json.dumps(
        [
            case["mode"],
            case["text"],
            case["api_language"],
            case["reference_id"],
            case.get("instruct"),
        ],
        ensure_ascii=False,
    )
    assert len({signature(case) for case in original}) == 1000
    assert not {signature(case) for case in warm} & {
        signature(case) for case in original
    }
    refs = read(ROOT / config["references"])
    assert len(refs) == 9
    by_ref = {}
    for ref in refs:
        path = Path(ref.get("path", ref.get("file")))
        path = (
            path if path.is_absolute() else (ROOT / config["references"]).parent / path
        )
        assert sha(path) == ref["sha256"]
        by_ref[ref["id"]] = ref
    assert all(
        case["mode"] == "clone" and case["reference_id"] in by_ref
        for case in warm + original
    )
    return warm, original


def smi():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=5,
    ).strip()


def stop_owned_group(process, *, term_timeout=15, kill_timeout=5):
    """Stop this controller's group even after its original leader has exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=term_timeout)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=kill_timeout)


def validate_lease(args, config):
    lease = read(args.lease)
    assert (
        lease["root_authorized_raw_burst5000"]
        and lease["modern_GPU_explicitly_released"]
        and lease["CPU_IO_quiet"]
    )
    assert lease["controller_sha256"] == sha(Path(__file__))
    assert lease["server_wrapper_sha256"] == sha(HERE / "raw_burst_server_v1.py")
    assert lease["config_sha256"] == sha(args.config)
    assert (
        lease["source_ref"] == SOURCE
        and lease["GPU_UUID"] == GPU_UUID
        and lease["requests"] == COUNT
    )
    age = time.time() - dt.datetime.fromisoformat(lease["at"]).timestamp()
    assert 0 <= age < 600
    return lease


def build_args(args, config):
    return SimpleNamespace(
        model=str(args.model),
        runner="triton",
        port=args.port,
        collect_ms=0.0,
        max_batch_requests=1,
        max_pending_requests=6000,
        max_pending_gpu_tasks=4,
        max_batch_target_tokens=4096,
        max_batch_conditioning_tokens=4096,
        batch_bucket_sizes="1,2,4,8,16,32",
        max_cuda_graphs=0,
        audio_decode_batch_size=1,
        response_cache_max_bytes=0,
        response_cache_ttl=600.0,
        server_extra_flags_json='["--disable-unused-kv-cache"]',
        startup_seed=20261103,
        output=args.output,
        expected_health_json=json.dumps(
            {
                "max_cuda_graph_row_tokens": 0,
                "disable_unused_kv_cache": True,
                "llm_use_cache": False,
                "split_cfg": False,
                "sage_attention": False,
                "full_triton_patch": False,
                "response_cache.enabled": False,
            }
        ),
    )


async def execute(args, config, legacy, runtime, warm, original, server):
    import httpx

    url = f"http://127.0.0.1:{args.port}"
    health = await runtime.wait_ready(url, server, 120)
    checks = legacy.validate_runtime(build_args(args, config), health)
    write(args.output / "health-startup.json", health)
    write(args.output / "runtime-validation.json", checks)
    assert (
        read(args.output / "server-benchmark-wrapper.json")["timeout_keep_alive"] == 600
    )
    by_id, references, phases = legacy.load_inputs(
        ROOT / config["corpus"], ROOT / config["plan"], ROOT / config["references"]
    )
    assert [legacy.effective_signature(x) for x in original] == [
        legacy.effective_signature(x) for x in legacy.expand_cases(phases[1], by_id)
    ]
    prompt_ids, all_rows, quality = {}, [], []
    (args.output / "audio").mkdir()
    (args.output / "errors").mkdir()
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        language_body = (await client.get(url + "/languages")).json()
        write(args.output / "language-support.json", language_body)
        supported = {row["id"] for row in language_body["languages"]}
        assert all(case["api_language"] in supported for case in warm + original)
        for ref_id, ref in references.items():
            path = Path(ref["path"])
            response = await client.post(
                url + "/clone-prompts",
                data={"ref_text": ref["text"], "preprocess_prompt": "true"},
                files={"ref_audio": (path.name, path.read_bytes(), "audio/wav")},
            )
            runtime.append_json(
                args.output / "registrations.jsonl",
                {
                    "reference_id": ref_id,
                    "status": response.status_code,
                    "reference": ref,
                    "body": response.json(),
                },
            )
            response.raise_for_status()
            prompt_ids[ref_id] = response.json()["prompt_id"]
        write(args.output / "registered-prompt-ids.json", prompt_ids)
        for index, case in enumerate(warm):
            identity = f"warmup-disjoint-inputs-{index:05d}"
            params = parameters(case, prompt_ids)
            started = time.perf_counter()
            response = await client.post(url + "/generate", data=params)
            finished = time.perf_counter()
            row = {
                "id": identity,
                "phase": "warmup-disjoint-inputs",
                "case": case,
                "parameters": params,
                "status": response.status_code,
                "headers": dict(response.headers),
                "complete_ms": (finished - started) * 1000,
                "network_finished_perf": finished,
                "full_body_received": True,
            }
            all_rows.append(row)
            response.raise_for_status()
            quality += legacy.retain_wave_batch(
                args.output, args.output.name, {identity: (response.content, row, case)}
            )
            write(args.output / "warmup-quality-manifest.json", quality)
            write(args.output / "warmup-requests.json", all_rows)
        streams, pre_rows, ok = await preconnect("127.0.0.1", args.port, COUNT)
        write(args.output / "preconnected-sockets.json", pre_rows)
        assert ok, (
            "All5000 distinct proven reusable TCP sockets required; no reduced cohort"
        )
        selected = [original[i % 1000] for i in range(COUNT)]
        params = [parameters(case, prompt_ids) for case in selected]
        payloads = [
            raw_request(
                "POST",
                "/generate",
                f"127.0.0.1:{args.port}",
                urllib.parse.urlencode(value).encode(),
                close=True,
            )
            for value in params
        ]
        before = (await client.get(url + "/health")).json()
        legacy.validate_runtime(build_args(args, config), before)
        write(args.output / "health-before.json", before)
        assert all(
            not writer.is_closing() and not reader.at_eof()
            for reader, writer in streams
        )
        log_offset = (args.output / "server.log").stat().st_size
        write(args.output / "cohort-server-log-offset.json", {"bytes": log_offset})
        fatal, stop = asyncio.Event(), asyncio.Event()

        async def monitor():
            while not stop.is_set():
                try:
                    response = await client.get(url + "/health", timeout=5)
                    body = response.json()
                    runtime.append_json(
                        args.output / "health-probes.jsonl",
                        {"at": now(), "status": response.status_code, "body": body},
                    )
                    if body.get("cuda_fault") is not None or server.poll() is not None:
                        fatal.set()
                        return
                except (httpx.HTTPError, ValueError) as error:
                    runtime.append_json(
                        args.output / "health-probes.jsonl",
                        {"at": now(), "error": repr(error)},
                    )
                    if server.poll() is not None:
                        fatal.set()
                        return
                try:
                    await asyncio.wait_for(stop.wait(), timeout=5)
                except TimeoutError:
                    pass

        monitor_task = asyncio.create_task(monitor())
        try:
            rows, bodies, info = await cohort(
                streams, payloads, selected, fatal_event=fatal
            )
            import resource

            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            info["client_maxRSS_at_network_end_bytes"] = (
                peak_rss if sys.platform == "darwin" else peak_rss * 1024
            )
            info["retained_body_bytes"] = sum(map(len, bodies.values()))
        finally:
            stop.set()
            await monitor_task
        for index, row in enumerate(rows):
            row["parameters"] = params[index]
        all_rows += rows
        captured = {
            row["id"]: (bodies[i], row, selected[i])
            for i, row in enumerate(rows)
            if row.get("status") == 200
            and row.get("full_body_received")
            and i in bodies
        }
        quality += await asyncio.to_thread(
            legacy.retain_wave_batch, args.output, args.output.name, captured
        )
        for i, row in enumerate(rows):
            if i in bodies and row["id"] not in captured:
                path = args.output / "errors" / (row["id"] + ".body")
                path.write_bytes(bodies[i])
                row.update(
                    error_body_path=str(path.relative_to(args.output)),
                    error_body_sha256=sha(path),
                    error_body=bodies[i][:1500].decode("utf8", errors="replace"),
                )
        try:
            after = (
                (await client.get(url + "/health", timeout=10)).json()
                if not fatal.is_set()
                else {"cuda_fault": "fatal event; no control retry"}
            )
        except (httpx.HTTPError, ValueError) as error:
            after = {"control_error": repr(error)}
        write(args.output / "health-after.json", after)
        write(args.output / "quality-manifest.json", quality)
        write(
            args.output / "quality-asr-ready.json",
            [
                row
                for row in quality
                if row.get("omnilingual_language") and row["audio"].get("asr_readable")
            ],
        )
        write(
            args.output / "quality-unmapped.json",
            [row for row in quality if not row.get("omnilingual_language")],
        )
        write(
            args.output / "quality-invalid-audio.json",
            [row for row in quality if not row["audio"].get("asr_readable")],
        )
        with (args.output / "requests.jsonl").open("w") as handle:
            handle.writelines(
                json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows
            )
        after_counter = after.get("batcher_queue", {}).get("benchmark_raw_burst_v1", {})
        counts = {
            key: after_counter[key]
            - before["batcher_queue"]["benchmark_raw_burst_v1"][key]
            if key in after_counter
            else None
            for key in (
                "batches_started",
                "members_started",
                "members_succeeded",
                "members_failed",
            )
        }
        dispatch = [
            row["POST_write_called_from_barrier_ms"]
            for row in rows
            if "POST_write_called_from_barrier_ms" in row
        ]
        full = [
            row
            for row in rows
            if row.get("status") == 200 and row.get("full_body_received")
        ]
        valid = {q["id"] for q in quality if q["audio"].get("signal_valid")}
        summary = {
            **info,
            "offers": COUNT,
            "original_effective_inputs": 1000,
            "cycles": 5,
            "content_profile": "5000 fresh inference requests cycling1000 original inputs; not5000 novel inputs",
            "HTTP_status_counts": dict(Counter(str(row.get("status")) for row in rows)),
            "error_category_counts": dict(
                Counter(
                    row["error_category"] for row in rows if row.get("error_category")
                )
            ),
            "HTTP200_complete_bodies": len(full),
            "HTTP200_complete_signal_valid": sum(row["id"] in valid for row in full),
            "transport_or_protocol_failures": sum(
                bool(row.get("error")) for row in rows
            ),
            "POST_dispatch_spread_ms": max(dispatch) - min(dispatch)
            if dispatch
            else None,
            "POST_dispatch_p95_ms": runtime.percentile(dispatch, 95),
            "complete_ms_p50": runtime.percentile(
                [row["complete_ms"] for row in full], 50
            ),
            "complete_ms_p95": runtime.percentile(
                [row["complete_ms"] for row in full], 95
            ),
            "whole5000_full_body_ms": max(
                row["completion_from_barrier_ms"] for row in full
            )
            if len(full) == COUNT
            else None,
            "batcher_benchmark_counter_delta": counts,
            "neural_work_proof_pass": counts
            == {
                "batches_started": COUNT,
                "members_started": COUNT,
                "members_succeeded": COUNT,
                "members_failed": 0,
            },
            "legacy_Inference_Executed_headers": dict(
                Counter(row["inference_executed_header"] for row in rows)
            ),
            "cache_disabled_before": before["response_cache"]["enabled"] is False,
            "cache_disabled_after": after.get("response_cache", {}).get("enabled")
            is False,
            "quality": "Signal validity only. Word/voice fidelity pending blind ASR/native review. No quality certification.",
        }
        with (args.output / "server.log").open("rb") as log:
            log.seek(log_offset)
            cohort_log = log.read().decode("utf8", errors="replace")
        summary["independent_server_log_members_executing"] = sum(
            map(int, re.findall(r"Executing batch lane=\S+ size=(\d+)", cohort_log))
        )
        summary["independent_server_log_members_completed"] = sum(
            map(int, re.findall(r"Completed batch lane=\S+ size=(\d+)", cohort_log))
        )
        summary["all5000_server_log_neural_work_proof"] = (
            summary["independent_server_log_members_executing"]
            == summary["independent_server_log_members_completed"]
            == COUNT
        )
        summary["all5000_inference_full_body_signal_gate"] = (
            len(full) == COUNT
            and all(row["id"] in valid for row in full)
            and summary["neural_work_proof_pass"]
            and summary["all5000_server_log_neural_work_proof"]
            and summary["cache_disabled_after"]
            and not info["fatal_CUDA_stop"]
        )
        write(args.output / "summary.json", summary)
        return summary


def child(args, config, warm, original):
    sys.path.insert(0, str(HERE))
    legacy = import_file("frozen_v4", HERE / "run.py")
    runtime = import_file("frozen_runtime", HERE / "runtime.py")
    server_args = build_args(args, config)
    command = legacy.build_command(server_args)
    command[1] = str(HERE / "raw_burst_server_v1.py")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LINGUA_") and key != "WORKER_ID"
    }
    environment.update(
        PYTHONPATH=str(args.source / "src"),
        HF_HOME=str(ROOT / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMNIVOICE_LOG_LEVEL="INFO",
    )
    write(args.output / "server-command.json", command)
    with (args.output / "server.log").open("w") as log:
        fatal = False
        server = subprocess.Popen(
            command,
            cwd=args.source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            write(
                args.output / "server-pid.json",
                {"pid": server.pid, "inherits_owned_process_group": True},
            )
            process_limits = Path(f"/proc/{server.pid}/limits")
            if process_limits.exists():
                contents = process_limits.read_text()
                (args.output / "server-process-limits.txt").write_text(contents)
                open_limit = next(
                    line.split()[3]
                    for line in contents.splitlines()
                    if line.startswith("Max open files")
                )
                assert open_limit == "unlimited" or int(open_limit) >= 16384
            summary = asyncio.run(
                execute(args, config, legacy, runtime, warm, original, server)
            )
            fatal = summary["fatal_CUDA_stop"]
            write(
                args.output / "child-outcome.json",
                {
                    "status": "completed_diagnostic",
                    "summary_gate": summary["all5000_inference_full_body_signal_gate"],
                },
            )
        except BaseException as error:
            fatal = True  # No model/CUDA cleanup in a possibly broken context.
            write(
                args.output / "child-failure.json", {"at": now(), "error": repr(error)}
            )
            raise
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGKILL if fatal else signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def supervise(args, config, warm, original):
    started = time.monotonic()
    lease = validate_lease(args, config)
    assert args.source.exists() and not smi()
    physical = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=5,
    ).strip()
    assert GPU_UUID in physical and "RTX 4070 Ti" in physical
    for row in config["model_files"]:
        relative = Path(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = args.model / relative
        assert path.stat().st_size == row["bytes"]
        assert streaming_sha(path, started + 45) == row["sha256"]
    sys.path.insert(0, str(HERE))
    legacy = import_file("frozen_v4", HERE / "run.py")
    limits = legacy.configure_file_limits(COUNT)
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < 16384:
        resource.setrlimit(resource.RLIMIT_NOFILE, (16384, hard))
    limits["raw_burst_required_limit"] = 16384
    limits["final_soft_hard"] = list(resource.getrlimit(resource.RLIMIT_NOFILE))
    assert (
        limits["final_soft_hard"][0] == resource.RLIM_INFINITY
        or limits["final_soft_hard"][0] >= 16384
    )
    args.output.mkdir(parents=True, exist_ok=False)
    hardware = json.loads(
        subprocess.check_output(
            [sys.executable, str(Path(__file__)), "--collect-hardware"],
            text=True,
            timeout=45,
        )
    )
    for package, version in {
        "torch": "2.8.0+cu128",
        "triton": "3.4.0",
        "transformers": "5.5.4",
        "omnivoice": "0.1.4",
    }.items():
        assert hardware["packages"][package] == version
    write(
        args.output / "metadata.json",
        {
            "started_at": now(),
            "label": args.output.name,
            "hardware": hardware,
            "source_ref": SOURCE,
            "model_revision": config["model_revision"],
            "config": config,
            "lease": lease,
            "file_descriptor_limits": limits,
            "protocol_sha256": sha(Path(__file__)),
            "wrapper_sha256": sha(HERE / "raw_burst_server_v1.py"),
            "profile": "triton-eager FP16 B1/KVoff/16steps/speed1/postoff/codec1/cache0; raw5000 sockets, cyclic1000text",
            "ASR_co_load": False,
        },
    )
    process = None
    status = "failed"
    previous = {}

    def interrupted(_number, _frame):
        raise KeyboardInterrupt("Stop owned controller/server group")

    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, interrupted)
    try:
        with (args.output / "controller.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--child",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(args.output / "controller-pid.json", {"pid": process.pid})
            code = process.wait(timeout=max(1, 2350 - (time.monotonic() - started)))
            status = "completed_diagnostic" if code == 0 else "child_failed"
    except subprocess.TimeoutExpired:
        status = "overall_deadline"
    finally:
        if process is not None:
            # Server inherits this exact owned group. Also stop an orphan server
            # if its controller has already exited unexpectedly.
            stop_owned_group(process)
        try:
            active = smi()
        except (OSError, subprocess.SubprocessError) as error:
            active = "unverified: " + repr(error)
        write(
            args.output / "RAW_BURST5000_COMPLETE_IDLE.json",
            {
                "at": now(),
                "status": status,
                "elapsed_s": time.monotonic() - started,
                "compute_processes": active,
                "gpu_released_idle": not bool(active),
                "quality_certified": False,
            },
        )
        for number, handler in previous.items():
            signal.signal(number, handler)
    assert status == "completed_diagnostic" and not active


def main():
    if sys.argv[1:] == ["--collect-hardware"]:
        sys.path.insert(0, str(HERE))
        runtime = import_file("frozen_runtime", HERE / "runtime.py")
        print(json.dumps(runtime.collect_hardware()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "raw_burst_v1-plan/config.json"
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "candidate-v8-audio-guard"
    )
    parser.add_argument(
        "--source-git",
        type=Path,
        default=Path("/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton"),
    )
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--lease", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/triton-eager-raw5000-cyclic1000-v1",
    )
    parser.add_argument("--port", type=int, default=18780)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = read(args.config)
    warm, original = fixture_checks(args, config)
    if not args.execute:
        assert not args.child
        print(
            json.dumps(
                {
                    "measurement_kind": "CPU-only source/ref/fixture dry plan; no GPU calls or neural imports",
                    "GPU_executed": False,
                    "config": config,
                    "controller_sha256": sha(Path(__file__)),
                    "wrapper_sha256": sha(HERE / "raw_burst_server_v1.py"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    assert args.lease is not None
    validate_lease(args, config)
    if args.child:
        child(args, config, warm, original)
    else:
        supervise(args, config, warm, original)


if __name__ == "__main__":
    main()
