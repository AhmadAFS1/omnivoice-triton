#!/usr/bin/env python3
"""Bounded/resumable execution of the unchanged blind Omnilingual workers.

Reads canonical quality/report.py's globally SHA+script deduplicated manifest.
Default is a CPU-only plan. Root must release the GPU and pass
--allow-gpu-inference. CTC runs first so early LLM records have both judges.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["omniASR_CTC_1B_v2", "omniASR_LLM_Unlimited_1B_v2"]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(row):
    ids = row.get("logical_ids", []) + [row.get("run", "")]
    if any("reference-control" in identity for identity in ids):
        group = 0
    elif any("all299" in identity for identity in ids):
        group = 1
    elif any("probe" in identity for identity in ids):
        group = 2
    elif any("controlled-" in identity for identity in ids):
        group = 3
    else:
        group = 4
    phase = row.get("phase", "")
    length = next(
        (n for n, word in enumerate(["short", "medium", "critical"]) if word in phase),
        3,
    )
    return group, length, row.get("language", ""), row["id"]


def validate_manifest(rows):
    assert rows, "No verified ASR inputs"
    assert len({r["id"] for r in rows}) == len(rows), "Duplicate input IDs"
    keys = [(r["sha256"], r["omnilingual_language"]) for r in rows]
    assert len(set(keys)) == len(rows), "Input is not globally SHA+script deduplicated"
    assert all(
        r.get("lang_id", r["omnilingual_language"]) == r["omnilingual_language"]
        for r in rows
    )


def gpu_idle():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=15,
    ).strip()
    if result:
        raise RuntimeError(
            f"GPU is not released/idle; existing compute processes:{result}"
        )
    return {
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "query": "nvidia-smi compute processes",
        "result": result,
    }


def safe_rows(path, prepared, model, allow_reuse=False):
    """Keep complete valid records; retain malformed/truncated raw bytes separately."""
    valid, issues = {}, []
    if not path.exists():
        return [], [{"issue": "worker_output_missing", "path": str(path)}]
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
            source = prepared.get(row["id"])
            assert source is not None
            assert row["sha256"] == source["sha256"]
            assert row.get("model_card") == model
            assert isinstance(row.get("transcript"), str)
            if row.get("asr_reused_from_id"):
                assert allow_reuse, "Fresh global inputs are already unique"
                assert row.get("asr_reuse_verified_key") == [
                    source["sha256"],
                    source["lang_id"],
                ]
                assert row["asr_wall_s"] == 0
            assert row["id"] not in valid
            valid[row["id"]] = row
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            issues.append(
                {
                    "issue": "invalid_or_incomplete_worker_row",
                    "line": line_no,
                    "reason": str(error),
                    "raw_source": str(path),
                }
            )
    return list(valid.values()), issues


def publish(raw, output, model, allow_reuse=False):
    rows = json.loads((raw / "prepared.json").read_text())
    by_id = {row["id"]: row for row in rows}
    valid, issues = safe_rows(raw / f"{model}.jsonl", by_id, model, allow_reuse)
    meta = raw / f"{model}-metadata.json"
    if valid:
        assert meta.exists(), "No blind-worker metadata for completed transcript rows"
        evidence = json.loads(meta.read_text())
        assert evidence["model_card"] == model
        assert evidence["blind"] is True and evidence["expected_text_prompt"] is False
    assert not output.exists(), "Never overwrite prior published transcripts"
    temporary = output.with_name("." + output.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    shutil.copy2(raw / "prepared.json", temporary / "prepared.json")
    with (temporary / f"{model}.jsonl").open("w") as stream:
        for row in valid:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if meta.exists():
        shutil.copy2(meta, temporary / meta.name)
    write(
        temporary / "publication.json",
        {
            "raw_attempt": str(raw),
            "raw_output_sha256": digest(raw / f"{model}.jsonl")
            if (raw / f"{model}.jsonl").exists()
            else None,
            "validated_complete_transcript_rows": len(valid),
            "issues": issues,
            "note": "Only complete records are published; raw evidence is unchanged.",
        },
    )
    temporary.rename(output)
    return valid, issues


def collect_existing(output, expected):
    covered = {model: {} for model in MODELS}
    directories = sorted(
        path
        for path in (output / "published").glob("*")
        if path.is_dir() and not path.name.startswith(".")
    )
    for directory in directories:
        prepared = json.loads((directory / "prepared.json").read_text())
        by_id = {row["id"]: row for row in prepared}
        for identity, source in by_id.items():
            assert identity in expected, "Resume input set changed"
            target = expected[identity]
            assert (source["sha256"], source["lang_id"]) == (
                target["sha256"],
                target["lang_id"],
            ), "Resume input hash/script changed"
        for model in MODELS:
            path = directory / f"{model}.jsonl"
            if not path.exists():
                continue
            rows, issues = safe_rows(path, by_id, model, allow_reuse=True)
            assert not issues, "Published transcript evidence failed revalidation"
            if rows:
                evidence = json.loads(
                    (directory / f"{model}-metadata.json").read_text()
                )
                assert evidence["model_card"] == model and evidence["blind"] is True
                assert evidence["expected_text_prompt"] is False
            for row in rows:
                prior = covered[model].get(row["id"])
                assert prior is None or prior["transcript"] == row["transcript"], (
                    "Conflicting prior transcripts; retain and inspect both"
                )
                covered[model][row["id"]] = row
    return covered, directories


def import_prior(prior_roots, output, expected, evaluator_sha256, dtype):
    """Import matching verified audio/script evidence from immutable older stages."""
    by_key = {(row["sha256"], row["lang_id"]): row for row in expected.values()}
    imported = []
    for prior_root in prior_roots:
        prior_root = prior_root.resolve()
        plan = json.loads((prior_root / "plan.json").read_text())
        assert plan["evaluator_sha256"] == evaluator_sha256
        assert plan["dtype"] == dtype
        for directory in sorted((prior_root / "published").glob("*")):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            old_prepared = {
                r["id"]: r
                for r in json.loads((directory / "prepared.json").read_text())
            }
            for model in MODELS:
                path = directory / f"{model}.jsonl"
                if not path.exists():
                    continue
                prior, issues = safe_rows(path, old_prepared, model, allow_reuse=True)
                assert not issues, "Prior published transcript cache failed validation"
                selected, sources = [], []
                for row in prior:
                    source = old_prepared[row["id"]]
                    key = (source["sha256"], source["lang_id"])
                    current = by_key.get(key)
                    if current:
                        selected.append(
                            {
                                **row,
                                "id": current["id"],
                                "asr_wall_s": 0.0,
                                "peak_allocated_bytes": None,
                                "asr_reused_from_id": row["id"],
                                "asr_reused_from_source": str(path),
                                "asr_reuse_verified_key": list(key),
                                "original_asr_wall_s": row.get(
                                    "original_asr_wall_s", row["asr_wall_s"]
                                ),
                            }
                        )
                        sources.append(current)
                if not selected:
                    continue
                key = hashlib.sha256((str(path) + digest(path)).encode()).hexdigest()[
                    :20
                ]
                label = f"import-{key}-{model}"
                destination = output / "published" / label
                if destination.exists():
                    continue
                raw = output / "cache-imports" / label
                raw.mkdir(parents=True, exist_ok=False)
                write(raw / "prepared.json", sources)
                with (raw / f"{model}.jsonl").open("w") as stream:
                    for row in selected:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                shutil.copy2(
                    directory / f"{model}-metadata.json", raw / f"{model}-metadata.json"
                )
                write(
                    raw / "prior-provenance.json",
                    {
                        "prior_root": str(prior_root),
                        "source_jsonl": str(path),
                        "source_jsonl_sha256": digest(path),
                        "source_prepared_sha256": digest(directory / "prepared.json"),
                        "note": "Transcript reuse only; no new GPU inference",
                    },
                )
                publish(raw, destination, model, allow_reuse=True)
                imported.append(
                    {
                        "source": str(path),
                        "imported_rows": len(selected),
                        "published": str(destination),
                    }
                )
    return imported


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Canonical quality helper preparation directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=3300)
    parser.add_argument("--reserve-seconds", type=float, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--asset-marker", type=Path, default=HERE.parent / "ASR_ASSETS_VERIFIED.json"
    )
    parser.add_argument("--allow-gpu-inference", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prior-asr-root",
        type=Path,
        action="append",
        default=[],
        help="Reuse validated transcripts from an earlier preparation stage",
    )
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--quality-helper", type=Path, default=HERE.parent / "quality/report.py"
    )
    parser.add_argument("--pair", action="append", default=[])
    args = parser.parse_args()
    args.prepared, args.output = args.prepared.resolve(), args.output.resolve()
    manifest = args.prepared / "asr-input-manifest.json"
    rows = json.loads(manifest.read_text())
    validate_manifest(rows)
    rows.sort(key=priority)
    preview = {
        "measurement_kind": "CPU-only execution plan until explicit GPU release",
        "unique_SHA_script_inputs": len(rows),
        "model_order": MODELS,
        "budget_seconds": args.budget_seconds,
        "reserve_seconds": args.reserve_seconds,
        "prepared_manifest": str(manifest),
        "prepared_manifest_sha256": digest(manifest),
        "evaluator_sha256": digest(HERE / "evaluate.py"),
        "dtype": args.dtype,
        "device": args.device,
        "prior_asr_roots": [str(p.resolve()) for p in args.prior_asr_root],
        "priority": "Reference controls, all299 arms, probes, control micro, then load",
        "models_receive": "Only audio paths and known language/script",
        "resume": "Only validated transcripts are reused; missing records are retried",
    }
    print(json.dumps(preview), flush=True)
    if not args.allow_gpu_inference:
        return
    assert args.budget_seconds > args.reserve_seconds > 0
    assert args.asset_marker.exists(), "Full model/package verification is required"
    marker = json.loads(args.asset_marker.read_text())
    assert marker["baseline_package_freeze_unchanged"]
    assert len(marker["assets"]) == 3
    deadline = time.monotonic() + args.budget_seconds
    if args.output.exists():
        assert args.resume, "Use a fresh output or explicit --resume"
        original = json.loads((args.output / "plan.json").read_text())
        for field in [
            "prepared_manifest_sha256",
            "evaluator_sha256",
            "dtype",
            "device",
        ]:
            assert original[field] == preview[field], "Resume source changed"
    else:
        args.output.mkdir(parents=True)
        write(args.output / "plan.json", preview)
    ordered = args.output / "ordered-input-manifest.json"
    if not ordered.exists():
        write(ordered, rows)
    prepared_dir = args.output / "input"
    if not (prepared_dir / "prepared.json").exists():
        prepared_dir.mkdir(exist_ok=True)
        spec = importlib.util.spec_from_file_location(
            "blind_evaluator", HERE / "evaluate.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.prepare(ordered, prepared_dir)  # CPU-only original WAV hashes/chunks.
    prepared = json.loads((prepared_dir / "prepared.json").read_text())
    expected = {row["id"]: row for row in prepared}
    assert expected.keys() == {row["id"] for row in rows}
    imported = import_prior(
        args.prior_asr_root,
        args.output,
        expected,
        preview["evaluator_sha256"],
        args.dtype,
    )
    write(args.output / "prior-imports.json", imported)
    covered, directories = collect_existing(args.output, expected)
    attempts = []
    outcome_path = args.output / "attempts.json"
    if outcome_path.exists():
        attempts = json.loads(outcome_path.read_text())
    worker_error = False
    for model in MODELS:
        remaining = [row for row in prepared if row["id"] not in covered[model]]
        if not remaining:
            continue
        if time.monotonic() >= deadline - args.reserve_seconds:
            break
        idle = gpu_idle()
        index = len(list((args.output / "raw").glob("*")))
        label = f"{index:03d}-{model}"
        raw = args.output / "raw" / label
        raw.mkdir(parents=True, exist_ok=False)
        write(raw / "prepared.json", remaining)
        remaining_by_id = {row["id"]: row for row in remaining}
        command = [
            sys.executable,
            str(HERE / "evaluate.py"),
            "--output",
            str(raw),
            "--worker-model",
            model,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
        ]
        started = time.monotonic()
        timed_out = False
        with (raw / "worker.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline - args.reserve_seconds:
                        timed_out = True
                        break
                    observed, _ = safe_rows(
                        raw / f"{model}.jsonl", remaining_by_id, model
                    )
                    observed_seconds = sum(row.get("asr_wall_s", 0) for row in observed)
                    mean_seconds = (
                        observed_seconds / len(observed) if observed else None
                    )
                    write(
                        args.output / "progress.json",
                        {
                            "model": model,
                            "pid": process.pid,
                            "attempt": label,
                            "model_elapsed_s": time.monotonic() - started,
                            "budget_remaining_s": max(0, deadline - time.monotonic()),
                            "prior_covered": {
                                key: len(value) for key, value in covered.items()
                            },
                            "current_completed_transcripts": len(observed),
                            "current_ASR_compute_s": observed_seconds,
                            "current_mean_ASR_compute_s": mean_seconds,
                            "estimated_remaining_model_compute_s": mean_seconds
                            * (len(remaining) - len(observed))
                            if observed
                            else None,
                        },
                    )
                    time.sleep(2)
            finally:
                stop_owned(process)
        published = args.output / "published" / label
        valid, issues = publish(raw, published, model)
        attempts.append(
            {
                "attempt": label,
                "model": model,
                "command": command,
                "gpu_idle_check": idle,
                "exit_code": process.returncode,
                "budget_timeout": timed_out,
                "valid_rows": len(valid),
                "publication_issues": issues,
                "elapsed_s": time.monotonic() - started,
            }
        )
        write(outcome_path, attempts)
        covered, directories = collect_existing(args.output, expected)
        if not timed_out and (
            process.returncode or len(valid) != len(remaining) or issues
        ):
            worker_error = True
            break
    summary = {
        "measurement_kind": "Local GPU ASR only; original TTS metrics unchanged",
        "unique_inputs": len(prepared),
        "covered_by_model": {key: len(value) for key, value in covered.items()},
        "both_models_covered": len(
            set.intersection(*(set(v) for v in covered.values()))
        ),
        "missing_by_model": {
            key: sorted(expected.keys() - value.keys())
            for key, value in covered.items()
        },
        "status": "completed"
        if all(len(v) == len(prepared) for v in covered.values())
        else "worker_error"
        if worker_error
        else "budget_exhausted",
        "published_directories": [str(path) for path in directories],
        "caveat": "ASR coverage/agreement is not human quality certification.",
    }
    write(args.output / "outcome.json", summary)
    score_command = [
        sys.executable,
        str(args.quality_helper),
        "score",
        "--prepared",
        str(args.prepared),
    ]
    for directory in directories:
        score_command += ["--asr-dir", str(directory)]
    for pair in args.pair:
        score_command += ["--pair", pair]
    if args.report_output:
        score_command += ["--output", str(args.report_output)]
    write(args.output / "score-command.json", score_command)
    if args.report_output:
        subprocess.run(score_command, check=True)
    print(json.dumps(summary), flush=True)
    if summary["status"] != "completed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
