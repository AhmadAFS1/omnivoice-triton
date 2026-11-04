#!/usr/bin/env python3
"""Separate blind ASR batching prototype; existing evaluate/controller stay frozen.

Default prints a CPU-only plan. --prepare-audio hashes/copies ASR inputs using
the unchanged evaluator. GPU execution requires --allow-gpu-inference and idle
GPU verification. No transcript cache or expected-text prompt is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["omniASR_CTC_1B_v2", "omniASR_LLM_Unlimited_1B_v2"]
MODEL = MODELS[1]
BINS = [3, 6, 12, 30.001]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator():
    spec = importlib.util.spec_from_file_location(
        "immutable_evaluator", HERE / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol_id(batch, dtype, pipeline_sha, model_sha):
    specification = {
        "name": "omniASR-blind-duration-batching-v1",
        "model": MODEL,
        "dtype": dtype,
        "max_batch_size": batch,
        "duration_bins_s": BINS,
        "call_window_chunks": 32,
        "scheduler": "controls-first/duration/round-robin-language",
        "model_defaults": "official pipeline beam/decode defaults unchanged",
        "expected_text_prompt": False,
        "language_conditioning": "Known exact script per chunk; CTC ignores it"
        if "CTC" in MODEL
        else "Known exact script per chunk",
        "prepare_source_sha256": digest(HERE / "evaluate.py"),
        "prototype_source_sha256": digest(Path(__file__).resolve()),
        "pipeline_source_sha256": pipeline_sha,
        "model_asset_sha256": model_sha,
    }
    serialized = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return "omniasr-batch-v1-" + hashlib.sha256(serialized.encode()).hexdigest()[
        :20
    ], specification


def schedule(cases, window=32):
    """No transcript/gold content affects selection or batch ordering."""
    groups = defaultdict(list)
    for case in cases:
        source = case.get("source", {})
        control = "reference-control" in str(source.get("run", case.get("run", "")))
        for index, chunk in enumerate(case["chunks"]):
            duration = chunk["end_s"] - chunk["start_s"]
            assert 0 < duration <= 30.001
            bucket = next(i for i, maximum in enumerate(BINS) if duration <= maximum)
            groups[(0 if control else 1, bucket)].append(
                {
                    "id": case["id"],
                    "chunk_index": index,
                    "path": chunk["path"],
                    "sha256": chunk["sha256"],
                    "lang_id": case["lang_id"],
                    "duration_s": duration,
                }
            )
    result = []
    for key, rows in sorted(groups.items()):
        languages = defaultdict(deque)
        for row in sorted(
            rows, key=lambda r: (r["duration_s"], r["id"], r["chunk_index"])
        ):
            languages[row["lang_id"]].append(row)
        ordered = []
        while any(languages.values()):
            for lang in sorted(languages):
                if languages[lang]:
                    ordered.append(languages[lang].popleft())
        result += [
            {"group": list(key), "chunks": ordered[start : start + window]}
            for start in range(0, len(ordered), window)
        ]
    assert sum(len(r["chunks"]) for r in result) == sum(len(r["chunks"]) for r in cases)
    return result


def transcribe(cases, pipeline, batch, on_call=None):
    """Official API only: audio paths, per-row language, configured batch size."""
    observations = {case["id"]: {} for case in cases}
    results = []
    calls = []
    by_id = {case["id"]: case for case in cases}
    for number, item in enumerate(schedule(cases)):
        chunks = item["chunks"]
        before = on_call("before", number, chunks) if on_call else {}
        started = time.perf_counter()
        texts = pipeline.transcribe(
            [r["path"] for r in chunks],
            lang=[r["lang_id"] for r in chunks],
            batch_size=batch,
        )
        after = on_call("after", number, chunks) if on_call else {}
        elapsed = time.perf_counter() - started
        assert len(texts) == len(chunks) and all(
            isinstance(text, str) for text in texts
        )
        call = {
            "index": number,
            "group": item["group"],
            "chunks": chunks,
            "max_batch_size": batch,
            "internal_batch_count": (len(chunks) + batch - 1) // batch,
            "mixed_language_count": len({r["lang_id"] for r in chunks}),
            "transcribe_call_wall_s": elapsed,
            "before": before,
            "after": after,
        }
        calls.append(call)
        for chunk, text in zip(chunks, texts):
            observations[chunk["id"]][chunk["chunk_index"]] = {
                "text": text,
                "call": number,
                "attributed_wall_s": elapsed / len(chunks),
            }
        for identity, observed in observations.items():
            case = by_id[identity]
            if len(observed) == len(case["chunks"]) and not observed.get("published"):
                ordered = [observed[index] for index in range(len(case["chunks"]))]
                row = {
                    "id": identity,
                    "sha256": case["sha256"],
                    "lang_id": case["lang_id"],
                    "model_card": MODEL,
                    "transcript": " ".join(r["text"].strip() for r in ordered),
                    "chunk_transcripts": [r["text"] for r in ordered],
                    "asr_wall_s": sum(r["attributed_wall_s"] for r in ordered),
                    "asr_wall_attribution": "Equal call-wall share per chunk",
                    "call_indices": [r["call"] for r in ordered],
                    "confidence": None,
                }
                results.append(row)
                observed["published"] = True
                if on_call:
                    on_call("row", number, row)
        if on_call:
            on_call("complete", number, call)
    return results, calls


def idle():
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=15,
    ).strip()
    assert not value, f"GPU has existing compute processes:{value}"


def worker(args):
    assert args.allow_gpu_inference, "Root GPU release must be explicit"
    idle()
    import inspect

    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

    cases = json.loads((args.output / "input/prepared.json").read_text())
    assert all(case["lang_id"] in supported_langs for case in cases)
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    assert free >= total * 0.9, "Need at least90% free visible VRAM before model load"
    assets = json.loads(args.asset_marker.read_text())
    asset = next(row for row in assets["assets"] if row["name"] == MODEL)
    pipeline_sha = digest(Path(inspect.getfile(ASRInferencePipeline)))
    config, specification = protocol_id(
        args.worker_batch, args.dtype, pipeline_sha, asset["sha256"]
    )
    output = args.output / f"batch{args.worker_batch}"
    output.mkdir(exist_ok=False)
    ev = evaluator()
    meta = ev.hardware()
    meta.update(
        {
            "asr_protocol_config_id": config,
            "specification": specification,
            "model_card": MODEL,
            "dtype": args.dtype,
            "batch_size": args.worker_batch,
            "blind": True,
            "expected_text_prompt": False,
            "confidence": None,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "free_before_model_bytes": free,
            "visible_total_bytes": total,
            "immutable_full_WAVs": True,
            "cache_reuse": False,
            "worker_pid": os.getpid(),
            "GPU_idle_checked_before_model": True,
        }
    )
    started = time.perf_counter()
    write(output / f"{MODEL}-metadata.json", meta)
    try:
        pipeline = ASRInferencePipeline(
            model_card=MODEL, device=args.device, dtype=getattr(torch, args.dtype)
        )
    except BaseException as error:
        write(
            output / "outcome.json",
            {
                "status": "model_load_failed",
                "error": repr(error),
                "asr_protocol_config_id": config,
                "load_wall_s": time.perf_counter() - started,
            },
        )
        raise
    torch.cuda.synchronize(device)
    meta["load_wall_s"] = time.perf_counter() - started
    write(output / f"{MODEL}-metadata.json", meta)
    write(output / "prepared.json", cases)
    case_path = output / f"{MODEL}.jsonl"
    call_path = output / "calls.jsonl"
    peak_allocated = 0
    peak_reserved = 0
    with case_path.open("w") as rows_file, call_path.open("w") as calls_file:

        def hook(event, number, payload):
            nonlocal peak_allocated, peak_reserved
            if event == "before":
                torch.cuda.synchronize(device)
                available, _ = torch.cuda.mem_get_info(device)
                assert available >= 512 * 1024**2, (
                    "Less than512MiB memory headroom before call"
                )
                torch.cuda.reset_peak_memory_stats(device)
                return {"free_bytes": available}
            if event == "after":
                torch.cuda.synchronize(device)
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                peak_allocated = max(peak_allocated, allocated)
                peak_reserved = max(peak_reserved, reserved)
                return {
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "free_bytes": torch.cuda.mem_get_info(device)[0],
                }
            if event == "row":
                payload["asr_protocol_config_id"] = config
                rows_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rows_file.flush()
            if event == "complete":
                calls_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                calls_file.flush()
                print(
                    json.dumps(
                        {
                            "batch": args.worker_batch,
                            "call": number,
                            "chunks": len(payload["chunks"]),
                            "wall_s": payload["transcribe_call_wall_s"],
                        }
                    ),
                    flush=True,
                )
                assert peak_allocated <= total * 0.9, (
                    "Peak allocation exceeded90% VRAM guard"
                )
            return {}

        started = time.perf_counter()
        try:
            rows, calls = transcribe(cases, pipeline, args.worker_batch, hook)
            outcome = {
                "status": "completed",
                "complete_cases": len(rows),
                "expected_cases": len(cases),
                "worker_wall_s": time.perf_counter() - started,
                "transcribe_calls_wall_s": sum(
                    r["transcribe_call_wall_s"] for r in calls
                ),
                "chunks": sum(len(r["chunks"]) for r in calls),
            }
        except BaseException as error:
            outcome = {
                "status": "failed",
                "error": repr(error),
                "worker_wall_s": time.perf_counter() - started,
            }
            raise
        finally:
            outcome.update(
                asr_protocol_config_id=config,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                visible_total_bytes=total,
            )
            write(output / "outcome.json", outcome)


def compare(root):
    ev = evaluator()
    by_batch = {}
    outcomes = {}
    for batch in [1, 2, 4]:
        directory = root / f"batch{batch}"
        path = directory / f"{MODEL}.jsonl"
        by_batch[batch] = (
            {r["id"]: r for r in ev.read_rows(path)} if path.exists() else {}
        )
        outcomes[str(batch)] = (
            json.loads((directory / "outcome.json").read_text())
            if (directory / "outcome.json").exists()
            else {"status": "not_executed"}
        )
    base = by_batch[1]
    differences = {}
    for batch in [2, 4]:
        differences[str(batch)] = [
            {
                "id": identity,
                "batch1": row["transcript"],
                "candidate": by_batch[batch].get(identity, {}).get("transcript"),
                "raw_equal": row["transcript"]
                == by_batch[batch].get(identity, {}).get("transcript"),
                "normalized_equal": ev.normalize(row["transcript"])
                == ev.normalize(
                    by_batch[batch].get(identity, {}).get("transcript", "")
                ),
            }
            for identity, row in base.items()
        ]
    result = {
        "measurement_kind": "Blind ASR prototype comparison; TTS metrics separate",
        "outcomes": outcomes,
        "comparisons": differences,
        "strict_parity_gate": {
            str(batch): bool(base)
            and outcomes["1"]["status"] == "completed"
            and set(base) == set(by_batch[batch])
            and all(row["raw_equal"] for row in differences[str(batch)])
            and outcomes[str(batch)]["status"] == "completed"
            for batch in [2, 4]
        },
        "cache_reuse": False,
        "expected_words_fed_to_model": False,
        "note": "Keep protocol IDs separate in quality reports. "
        "Parity on this finite fixture does not certify all future recordings.",
    }
    baseline_time = outcomes["1"].get("transcribe_calls_wall_s")
    result["inference_speedup_vs_batch1"] = {
        str(batch): baseline_time / outcomes[str(batch)]["transcribe_calls_wall_s"]
        if baseline_time and outcomes[str(batch)].get("transcribe_calls_wall_s")
        else None
        for batch in [2, 4]
    }
    write(root / "comparison.json", result)
    return result


def main():
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-audio", action="store_true")
    parser.add_argument("--allow-gpu-inference", action="store_true")
    parser.add_argument("--worker-batch", type=int, choices=[1, 2, 4])
    parser.add_argument("--model", choices=MODELS, default=MODEL)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument(
        "--asset-marker", type=Path, default=HERE.parent / "ASR_ASSETS_VERIFIED.json"
    )
    args = parser.parse_args()
    MODEL = args.model
    args.output = args.output.resolve()
    if args.worker_batch:
        worker(args)
        return
    preview = {
        "measurement_kind": "CPU-only plan until explicit GPU release",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest(args.manifest),
        "prototype_sha256": digest(Path(__file__).resolve()),
        "model": MODEL,
        "batch_sizes": [1, 2, 4],
        "cache_reuse": False,
        "max_seconds": args.max_seconds,
        "original_evaluate_sha256": digest(HERE / "evaluate.py"),
        "duration_bins_s": BINS,
    }
    print(json.dumps(preview), flush=True)
    if not (args.prepare_audio or args.allow_gpu_inference):
        return
    if args.output.exists():
        original = json.loads((args.output / "plan.json").read_text())
        for field in [
            "manifest_sha256",
            "prototype_sha256",
            "original_evaluate_sha256",
        ]:
            assert original[field] == preview[field], (
                "Prepared source changed; use a fresh output"
            )
    else:
        args.output.mkdir(parents=True)
        write(args.output / "plan.json", preview)
    if not (args.output / "input/prepared.json").exists():
        directory = args.output / "input"
        directory.mkdir()
        evaluator().prepare(args.manifest.resolve(), directory)
        write(
            args.output / "schedule.json",
            schedule(json.loads((directory / "prepared.json").read_text())),
        )
    if not args.allow_gpu_inference:
        return
    assert args.asset_marker.exists()
    deadline = time.monotonic() + args.max_seconds
    attempts = []
    for batch in [1, 2, 4]:
        assert not (args.output / f"batch{batch}").exists(), (
            "Fresh config output required; no overwrite"
        )
        if time.monotonic() + 30 >= deadline:
            break
        idle()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(args.manifest),
            "--output",
            str(args.output),
            "--worker-batch",
            str(batch),
            "--model",
            MODEL,
            "--allow-gpu-inference",
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--asset-marker",
            str(args.asset_marker),
        ]
        with (args.output / f"batch{batch}.log").open("w") as log:
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
            attempts.append(
                {
                    "batch_size": batch,
                    "command": command,
                    "exit_code": process.returncode,
                    "wall_s": time.monotonic() - started,
                }
            )
            write(args.output / "attempts.json", attempts)
            if process.returncode:
                break
    result = compare(args.output)
    print(
        json.dumps(
            {
                "comparison": str(args.output / "comparison.json"),
                "strict_parity_gate": result["strict_parity_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
