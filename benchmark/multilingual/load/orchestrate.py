#!/usr/bin/env python3
"""Prepare finite fresh-worker experiments; GPU work needs --execute explicitly.

Default is a CPU-only dry run. Root owns the rental watchdog and GPU release.
Every command uses a new output directory; before/candidate model settings match.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


def make_jobs(args):
    root = args.root.resolve()
    plans = root / "load/generated-plans-v3"
    common = [
        str(root / "baseline/.venv/bin/python"),
        str(root / "load/run.py"),
        "--model",
        str(root / "model"),
        "--model-revision",
        "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        "--references",
        str(root / "languages/references/manifest.json"),
        "--steps",
        "16",
        "--max-cuda-graphs",
        "8",
        "--audio-decode-batch-size",
        "1",
    ]
    sources = {
        "before": (args.baseline_source or root / "baseline", args.baseline_ref),
        "candidate": (args.candidate_source or root / "candidate", args.candidate_ref),
    }
    jobs = []

    def job(name, source_kind, corpus, plan, extra=None, experimental=False):
        source, ref = sources[source_kind]
        server_extra = []
        expected_health = {}
        if source_kind == "candidate":
            server_extra = json.loads(args.candidate_flags_json)
            expected_health = json.loads(args.candidate_health_json)
        if experimental:
            assert args.experimental_flags_json and args.experimental_health_json
            server_extra = json.loads(args.experimental_flags_json)
            expected_health = json.loads(args.experimental_health_json)
        command = common + [
            "--source",
            str(source.resolve()),
            "--source-ref",
            ref,
            "--label",
            name,
            "--output",
            str(root / "results" / name),
            "--corpus",
            str(corpus),
            "--plan",
            str(plan),
        ]
        if source_kind == "candidate":
            command += [
                "--response-cache-max-bytes",
                "0",
                "--response-cache-ttl",
                "600",
            ]
        if extra:
            # Dedicated values are changed before launch, never appended twice.
            for flag, value in extra:
                if flag == "--server-extra-flags-json":
                    server_extra.extend(json.loads(value))
                    continue
                if flag == "--expected-health-json":
                    expected_health.update(json.loads(value))
                    continue
                if flag in command:
                    command[command.index(flag) + 1] = str(value)
                else:
                    command.extend([flag, str(value)])
        if server_extra or expected_health:
            command += [
                "--server-extra-flags-json",
                json.dumps(server_extra),
                "--expected-health-json",
                json.dumps(expected_health),
            ]
        jobs.append(
            {
                "id": name,
                "command": command,
                "source_kind": source_kind,
                "experimental": experimental,
            }
        )

    quality = plans / "quality-all-languages.json"
    corpus = root / "languages/corpus.json"
    job("before-all299", "before", corpus, quality, [("--startup-seed", 20261103)])
    job(
        "candidate-all299", "candidate", corpus, quality, [("--startup-seed", 20261103)]
    )
    if args.experimental_flags_json:
        job(
            "experimental-all299",
            "candidate",
            corpus,
            quality,
            [("--startup-seed", 20261103)],
            experimental=True,
        )
    for name, plan in [
        ("cache-cold-identical1000", "cold-identical1000.json"),
        ("cache-hot-identical1000", "hot-identical1000.json"),
        ("cache-hot-multilingual1000", "hot-multilingual1000.json"),
    ]:
        job(
            name,
            "candidate",
            corpus,
            plans / plan,
            [
                ("--response-cache-max-bytes", args.cache_bytes),
                ("--response-cache-ttl", 600),
                ("--connections", 1200),
            ],
        )
    for kind in ["before", "candidate"]:
        job(
            f"{kind}-short-curve",
            kind,
            corpus,
            plans / "basic-bursts.json",
            [("--max-pending-requests", 1200)],
        )
        job(
            f"{kind}-staggered",
            kind,
            corpus,
            plans / "staggered.json",
            [("--max-pending-requests", 16)],
        )
    # Long GPU queues are diagnostics; their full completions must not be called SLO.
    for kind in ["before", "candidate"]:
        job(
            f"{kind}-unique1000",
            kind,
            plans / "unique-load-corpus.json",
            plans / "burst1000-unique-short.json",
            [
                ("--max-pending-requests", 1200),
                ("--connections", 1200),
                ("--request-timeout", 1500),
                ("--phase-timeout", 1200),
            ],
        )
    if args.include_experimental_load:
        assert args.experimental_flags_json and args.experimental_health_json
        job(
            "experimental-short-curve",
            "candidate",
            corpus,
            plans / "basic-bursts.json",
            [("--max-pending-requests", 1200)],
            experimental=True,
        )
        job(
            "experimental-staggered",
            "candidate",
            corpus,
            plans / "staggered.json",
            [("--max-pending-requests", 16)],
            experimental=True,
        )
        job(
            "experimental-unique1000",
            "candidate",
            plans / "unique-load-corpus.json",
            plans / "burst1000-unique-short.json",
            [
                ("--max-pending-requests", 1200),
                ("--connections", 1200),
                ("--request-timeout", 1500),
                ("--phase-timeout", 1200),
            ],
            experimental=True,
        )
    if args.include_optional5000:
        job(
            "candidate-staggered5000",
            "candidate",
            corpus,
            plans / "staggered5000-optional.json",
            [("--max-pending-requests", 16)],
        )
        job(
            "cache-hot-identical5000",
            "candidate",
            corpus,
            plans / "hot-identical5000-optional.json",
            [
                ("--response-cache-max-bytes", args.cache_bytes),
                ("--response-cache-ttl", 600),
                ("--connections", 6000),
                (
                    "--server-extra-flags-json",
                    '["--response-cache-max-waiters","6000"]',
                ),
                ("--expected-health-json", '{"response_cache.max_waiters":6000}'),
            ],
        )
    known = {job["id"] for job in jobs}
    if args.jobs:
        selected = args.jobs.split(",")
        assert len(selected) == len(set(selected)), "Duplicate selected job"
        assert set(selected) <= known, "Unknown selected job"
        by_id = {job["id"]: job for job in jobs}
        jobs = [by_id[name] for name in selected]
    if args.skip_labels:
        skipped = set(args.skip_labels.split(","))
        assert skipped <= known, "Unknown skip label"
        jobs = [job for job in jobs if job["id"] not in skipped]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--baseline-ref", default="c79ce35")
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument(
        "--candidate-flags-json", default='["--disable-unused-kv-cache"]'
    )
    parser.add_argument(
        "--candidate-health-json",
        default=json.dumps(
            {
                "disable_unused_kv_cache": True,
                "llm_use_cache": False,
                "split_cfg": False,
                "inference_optimizations.split_cfg_enabled": False,
            }
        ),
    )
    parser.add_argument("--experimental-flags-json")
    parser.add_argument("--experimental-health-json")
    parser.add_argument("--cache-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--jobs", default="")
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--include-optional5000", action="store_true")
    parser.add_argument("--include-experimental-load", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=9000)
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()
    jobs = make_jobs(args)
    document = {
        "measurement_kind": "CPU-only command planning until explicit execution",
        "target_gpu": "RTX4070; VRAM and runtime verified per real run",
        "jobs": jobs,
        "explicitly_skipped_labels": args.skip_labels.split(",")
        if args.skip_labels
        else [],
        "budget_seconds": args.budget_seconds,
        "note": "500responses/s requires all1000 complete bodies in2seconds. "
        "Cached/coalesced and novel generated responses are separate results.",
    }
    if args.plan_output:
        args.plan_output.write_text(json.dumps(document, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(document, indent=2))
        return
    root = args.root.resolve()
    deadline = time.monotonic() + args.budget_seconds
    environment = os.environ.copy()
    environment.update(HF_HOME=str(root / "hf-cache"), HF_HUB_OFFLINE="1")
    log_path = root / "suite-outcomes.json"
    outcomes = json.loads(log_path.read_text()) if log_path.exists() else []
    for job in jobs:
        if job["id"] == "cache-hot-identical5000":
            # This optional arm repeats the same target as measured hot1000.
            # Reserve50% extra beyond raw response bodies, with an8GiB ceiling.
            prior = root / "results/cache-hot-identical1000/requests.jsonl"
            rows = [json.loads(line) for line in prior.read_text().splitlines()]
            sizes = [row["response_bytes"] for row in rows if row.get("status") == 200]
            assert sizes, "Measure hot1000 retained-body sizes before optional5000"
            assert max(sizes) * 5000 * 1.5 < 8 * 1024**3, (
                "Optional5000 estimated retained bodies plus reserve exceeds8GiB"
            )
        output = Path(job["command"][job["command"].index("--output") + 1])
        if output.exists():
            if args.resume_completed and (output / "outcome.json").exists():
                outcome = json.loads((output / "outcome.json").read_text())
                assert outcome["status"] == "completed", (
                    "Existing job failed; do not hide retry"
                )
                outcomes.append({"job": job["id"], "skipped_existing_complete": True})
                continue
            raise FileExistsError(
                f"Output exists; choose explicit remaining jobs:{output}"
            )
        if time.monotonic() + 120 > deadline:
            outcomes.append(
                {"job": job["id"], "not_started": "soft suite budget exhausted"}
            )
            break
        with (root / f"{job['id']}.log").open("w") as log:
            code = subprocess.call(
                job["command"], env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        outcomes.append(
            {
                "job": job["id"],
                "exit_code": code,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        log_path.write_text(json.dumps(outcomes, indent=2) + "\n")
        print(json.dumps(outcomes[-1]), flush=True)
        if code:
            raise SystemExit(
                f"Stop after {job['id']} failure; inspect retained evidence"
            )
    log_path.write_text(json.dumps(outcomes, indent=2) + "\n")


if __name__ == "__main__":
    main()
