"""Optional cached/coalesced transport assay, CPU dry by default; never freshRPS."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE_SHA = "b204405036ae236be8ceefc4f760271fcd13fc18428a86fded9110471ea94665"
WRAPPER_SHA = "7a9087c32a7bca6f69afcab96818e83eb9d939250b4a2b80248021f10475bac6"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core():
    path = ROOT / "load/raw_burst_v1.py"
    assert (
        sha(path) == CORE_SHA
        and sha(ROOT / "load/raw_burst_server_v1.py") == WRAPPER_SHA
    )
    spec = importlib.util.spec_from_file_location("frozen_raw_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response_proof(rows, bodies, expected, delta, wanted_delta, expected_sha=None):
    hashes = {hashlib.sha256(body).hexdigest() for body in bodies.values()}
    sources = Counter(
        row.get("headers", {}).get("x-omnivoice-response-source", "unreported")
        for row in rows
    )
    executed = Counter(
        row.get("headers", {}).get("x-omnivoice-inference-executed", "unreported")
        for row in rows
    )
    mapping_pass = all(
        (source := row.get("headers", {}).get("x-omnivoice-response-source"))
        in {"generated", "coalesced", "cache"}
        and row.get("headers", {}).get("x-omnivoice-inference-executed")
        == ("true" if source == "generated" else "false")
        for row in rows
    )
    good = len(rows) == len(bodies) == expected and all(
        row.get("status") == 200 and row.get("full_body_received") for row in rows
    )
    identical = len(hashes) == 1 and (expected_sha is None or hashes == {expected_sha})
    counter_pass = delta == {
        "batches_started": wanted_delta,
        "members_started": wanted_delta,
        "members_succeeded": wanted_delta,
        "members_failed": 0,
    }
    source_pass = (
        (
            sources.get("generated", 0) == 1
            and executed.get("true", 0) == 1
            and sum(sources.values()) == expected
            and set(sources) <= {"generated", "coalesced", "cache"}
            and executed.get("false", 0) == expected - 1
        )
        if wanted_delta == 1
        else sources == {"cache": expected} and executed == {"false": expected}
    )
    return {
        "offers": expected,
        "HTTP200_complete": sum(
            row.get("status") == 200 and row.get("full_body_received") for row in rows
        ),
        "source_counts": dict(sources),
        "actual_inference_header_counts": dict(executed),
        "unique_body_hashes": sorted(hashes),
        "all_complete_WAV_bytes_identical": identical,
        "benchmark_neural_member_delta": delta,
        "counter_pass": counter_pass,
        "source_header_pass": source_pass and mapping_pass,
        "per_response_source_inference_mapping_pass": mapping_pass,
        "protocol_complete_pass": good,
        "gate_pass": good
        and identical
        and counter_pass
        and source_pass
        and mapping_pass,
        "interpretation": "Cached/coalesced exact-input transport, never novel inference throughput or independent word-quality observations",
    }


def counter_delta(before, after):
    a = before["batcher_queue"]["benchmark_raw_burst_v1"]
    b = after["batcher_queue"]["benchmark_raw_burst_v1"]
    return {key: b[key] - a[key] for key in a}


def prepare(args, config, core):
    assert (
        sha(ROOT / "load/raw_burst_v1-plan/config.json")
        == "0a5ea9e8d0f30a58193b195932b07e221b598a5b3953d5e280946ae37cdf1bf4"
    )
    rawconfig = read(ROOT / "load/raw_burst_v1-plan/config.json")
    core.fixture_checks(args, rawconfig)
    cases = read(ROOT / rawconfig["corpus"])
    cases = cases["cases"] if isinstance(cases, dict) else cases
    target = next(case for case in cases if case["id"] == "en-short")
    warm = next(
        case
        for case in cases
        if case["language"] == "en" and case["text"] != target["text"]
    )
    assert warm["reference_id"] == target["reference_id"] == "en-aoede"
    assert (
        config["target_case_id"] == target["id"]
        and config["warmup_case_id"] == warm["id"]
    )
    return rawconfig, warm, target


def check_lease(args, config):
    lease = read(args.lease)
    assert all(
        lease[key]
        for key in (
            "root_authorized_cached_assay",
            "core_ASR_retained_idle_complete",
            "staggered_retained_idle_complete",
            "CPU_IO_quiet",
        )
    )
    assert lease["controller_sha256"] == sha(Path(__file__)) and lease[
        "config_sha256"
    ] == sha(HERE / "config.json")
    assert (
        lease["GPU_UUID"] == config["GPU_UUID"]
        and lease["source_ref"] == config["source_ref"]
    )
    assert 0 <= time.time() - dt.datetime.fromisoformat(lease["at"]).timestamp() < 600
    return lease


async def execute(args, config, core, rawconfig, warm, target, legacy, runtime, server):
    import httpx

    url = f"http://127.0.0.1:{args.port}"
    server_args = core.build_args(args, rawconfig)
    server_args.response_cache_max_bytes = 128 * 1024**2
    expected = json.loads(server_args.expected_health_json)
    expected["response_cache.enabled"] = True
    server_args.expected_health_json = json.dumps(expected)
    initial = await runtime.wait_ready(url, server, 60)
    legacy.validate_runtime(server_args, initial)
    assert initial["response_cache"]["entries"] == 0
    write(args.output / "health-startup.json", initial)
    raw_refs = read(ROOT / rawconfig["references"])
    ref = next(row for row in raw_refs if row["id"] == target["reference_id"])
    refpath = Path(ref.get("path", ref.get("file")))
    refpath = (
        refpath
        if refpath.is_absolute()
        else (ROOT / rawconfig["references"]).parent / refpath
    )
    all_rows, quality, summaries = [], [], []
    (args.output / "audio").mkdir()
    (args.output / "errors").mkdir()
    fatal = asyncio.Event()
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        registered = await client.post(
            url + "/clone-prompts",
            data={"ref_text": ref["text"], "preprocess_prompt": "true"},
            files={"ref_audio": (refpath.name, refpath.read_bytes(), "audio/wav")},
        )
        write(
            args.output / "registration.json",
            {
                "status": registered.status_code,
                "body": registered.json(),
                "reference": ref,
            },
        )
        registered.raise_for_status()
        prompt_ids = {target["reference_id"]: registered.json()["prompt_id"]}
        # Exactly one different text key; target has not been generated in this
        # fresh process before the cold cohort. Registration is not generation.
        started = time.perf_counter()
        r = await client.post(url + "/generate", data=core.parameters(warm, prompt_ids))
        finished = time.perf_counter()
        warmrow = {
            "id": "disjoint-warmup-00000",
            "phase": "disjoint-warmup",
            "case": warm,
            "parameters": core.parameters(warm, prompt_ids),
            "status": r.status_code,
            "headers": dict(r.headers),
            "complete_ms": (finished - started) * 1000,
            "network_finished_perf": finished,
            "full_body_received": True,
        }
        write(args.output / "warmup-request.json", warmrow)
        if r.status_code != 200:
            (args.output / "errors/warmup-response.body").write_bytes(r.content)
        r.raise_for_status()
        quality += legacy.retain_wave_batch(
            args.output, args.output.name, {warmrow["id"]: (r.content, warmrow, warm)}
        )
        all_rows.append(warmrow)
        write(args.output / "quality-manifest.json", quality)
        assert quality[-1]["audio"]["signal_valid"]
        cold_hash = None
        for label, count, wanted in (
            ("cold-identical1000", 1000, 1),
            ("hot-identical5000", 5000, 0),
        ):
            before = (await client.get(url + "/health")).json()
            legacy.validate_runtime(server_args, before)
            write(args.output / (label + "-health-before.json"), before)
            if wanted == 1:
                assert before["response_cache"]["entries"] == 1
            streams, pre, ok = await core.preconnect(
                "127.0.0.1", args.port, count, timeout=30, width=256
            )
            write(args.output / (label + "-sockets.json"), pre)
            assert ok
            params = core.parameters(target, prompt_ids)
            payload = core.raw_request(
                "POST",
                "/generate",
                f"127.0.0.1:{args.port}",
                urllib.parse.urlencode(params).encode(),
                close=True,
            )
            core.PHASE = label  # Cohorts are serial; frozen parser/core code unchanged.
            rows, bodies, info = await core.cohort(
                streams,
                [payload] * count,
                [target] * count,
                timeout=35,
                fatal_event=fatal,
            )
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            info["client_peakRSS_bytes"] = (
                rss if sys.platform == "darwin" else rss * 1024
            )
            for row in rows:
                row.update(
                    parameters=params,
                    cycle_index=None,
                    original_input_index=0,
                    content_profile="one identical cached target key; not fresh-input capacity",
                )
            all_rows += rows
            captured = {
                row["id"]: (bodies[i], row, target)
                for i, row in enumerate(rows)
                if i in bodies
                and row.get("status") == 200
                and row.get("full_body_received")
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
                    )
            write(args.output / "quality-manifest.json", quality)
            write(
                args.output / "quality-asr-ready.json",
                [
                    q
                    for q in quality
                    if q.get("omnilingual_language") and q["audio"].get("asr_readable")
                ],
            )
            with (args.output / "requests.jsonl").open("w") as handle:
                handle.writelines(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows
                )
            assert not fatal.is_set(), (
                "FatalCUDA response; retain and terminate owned process"
            )
            after = (await client.get(url + "/health")).json()
            write(args.output / (label + "-health-after.json"), after)
            proof = response_proof(
                rows, bodies, count, counter_delta(before, after), wanted, cold_hash
            )
            proof.update(
                phase=label,
                transport=info,
                whole_complete_body_ms=max(
                    row.get("completion_from_barrier_ms", 0) for row in rows
                ),
                POST_dispatch_spread_ms=max(
                    row["POST_write_called_from_barrier_ms"] for row in rows
                )
                - min(row["POST_write_called_from_barrier_ms"] for row in rows),
            )
            full = [
                row
                for row in rows
                if row.get("status") == 200 and row.get("full_body_received")
            ]
            proof["HTTP200_fullbody_dispatch_latency_ms"] = {
                f"p{p}": runtime.percentile([row["complete_ms"] for row in full], p)
                for p in (50, 95, 99)
            }
            proof["alloffers_HTTP200_complete_within2s"] = sum(
                row["completion_from_barrier_ms"] <= 2000 for row in full
            )
            proof["independent_word_quality_observations_max"] = 1
            proof["signal_valid"] = sum(
                q["audio"].get("signal_valid", False)
                for q in quality
                if q["phase"] == label
            )
            proof["gate_pass"] = proof["gate_pass"] and proof["signal_valid"] == count
            summaries.append(proof)
            write(args.output / "summary.json", summaries)
            assert proof["gate_pass"], "Preserve failed cached assay; no hidden retry"
            cold_hash = proof["unique_body_hashes"][0]
            del captured, bodies
    return summaries


def child(args, config, core, rawconfig, warm, target):
    sys.path.insert(0, str(ROOT / "load"))
    legacy = core.import_file("frozen_v4", ROOT / "load/run.py")
    runtime = core.import_file("frozen_runtime", ROOT / "load/runtime.py")
    server_args = core.build_args(args, rawconfig)
    server_args.response_cache_max_bytes = 128 * 1024**2
    command = legacy.build_command(server_args)
    command[1] = str(ROOT / "load/raw_burst_server_v1.py")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LINGUA_") and key != "WORKER_ID"
    }
    env.update(
        PYTHONPATH=str(args.source / "src"),
        HF_HOME=str(ROOT / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMNIVOICE_LOG_LEVEL="INFO",
    )
    write(args.output / "server-command.json", command)
    with (args.output / "server.log").open("w") as log:
        failed = True
        server = subprocess.Popen(
            command, cwd=args.source, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            write(
                args.output / "server-pid.json",
                {"pid": server.pid, "inherits_owned_group": True},
            )
            summaries = asyncio.run(
                execute(
                    args, config, core, rawconfig, warm, target, legacy, runtime, server
                )
            )
            write(
                args.output / "child-outcome.json",
                {"status": "completed_cached_not_novel", "phases": summaries},
            )
            failed = False
        except BaseException as error:
            write(
                args.output / "failure.json",
                {"at": dt.datetime.now(dt.UTC).isoformat(), "error": repr(error)},
            )
            raise
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGKILL if failed else signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def supervise(args, config, core, rawconfig, warm, target):
    started = time.monotonic()
    lease = check_lease(args, config)
    assert not core.smi()
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
    for row in rawconfig["model_files"]:
        file = args.model / row["path"]
        assert (
            file.stat().st_size == row["bytes"]
            and core.streaming_sha(file, started + 30) == row["sha256"]
        )
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < 16384:
        resource.setrlimit(resource.RLIMIT_NOFILE, (16384, hard))
    args.output.mkdir(parents=True, exist_ok=False)
    hardware = json.loads(
        subprocess.check_output(
            [sys.executable, str(ROOT / "load/raw_burst_v1.py"), "--collect-hardware"],
            text=True,
            timeout=30,
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
            "at": dt.datetime.now(dt.UTC).isoformat(),
            "hardware": hardware,
            "source_ref": config["source_ref"],
            "profile": "cached/coalesced exactinput,never fresh inference capacity",
            "config": config,
            "lease": lease,
            "controller_sha256": sha(Path(__file__)),
        },
    )
    process, status = None, "failed"
    previous = {}

    def interrupt(_number, _frame):
        raise KeyboardInterrupt("Stop owned cached-assay group")

    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, interrupt)
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
            code = process.wait(timeout=max(1, 95 - (time.monotonic() - started)))
            status = "completed_cached_not_novel" if code == 0 else "child_failed"
    except subprocess.TimeoutExpired:
        status = "bounded_timeout"
    finally:
        if process is not None:
            core.stop_owned_group(process, term_timeout=10, kill_timeout=5)
        active = core.smi()
        write(
            args.output / "MODERN_CACHED_ASSAY_COMPLETE_IDLE.json",
            {
                "at": dt.datetime.now(dt.UTC).isoformat(),
                "status": status,
                "elapsed_s": time.monotonic() - started,
                "gpu_released_idle": not bool(active),
                "compute_processes": active,
                "novel_inference_throughput": False,
                "word_quality_certified": False,
            },
        )
        for number, handler in previous.items():
            signal.signal(number, handler)
    assert status == "completed_cached_not_novel" and not active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lease", type=Path)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "candidate-v8-audio-guard"
    )
    parser.add_argument(
        "--source-git",
        type=Path,
        default=Path("/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton"),
    )
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--port", type=int, default=18782)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/triton-eager-cached-cold1k-hot5k-v1",
    )
    args = parser.parse_args()
    config = read(HERE / "config.json")
    core = load_core()
    rawconfig, warm, target = prepare(args, config, core)
    if not args.execute:
        assert not args.child
        print(
            json.dumps(
                {
                    "GPU_executed": False,
                    "config": config,
                    "warmup_case": warm,
                    "target_case": target,
                    "controller_sha256": sha(Path(__file__)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    assert args.lease is not None
    check_lease(args, config)
    if args.child:
        child(args, config, core, rawconfig, warm, target)
    else:
        supervise(args, config, core, rawconfig, warm, target)


if __name__ == "__main__":
    main()
