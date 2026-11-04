#!/usr/bin/env python3
"""Prepare500-offer independent staggered profiles; CPU-only, never launches jobs."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Execution task root")
    parser.add_argument(
        "--language-dir", type=Path, required=True, help="Local fixture directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh local plan directory"
    )
    parser.add_argument("--baseline-ref", default="c79ce35")
    parser.add_argument("--candidate-source", default="candidate-v5")
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--remote-plan-dir", default="load/stagger500-plans-v2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.language_dir
    pool = json.loads((source / "load-pool.json").read_text())
    corpus = json.loads((source / "corpus.json").read_text())
    if isinstance(pool, dict):
        pool = pool["cases"]
    if isinstance(corpus, dict):
        corpus = corpus["cases"]
    signatures = [
        (
            r["mode"],
            r["text"],
            r["api_language"],
            r.get("reference_id"),
            r.get("instruct"),
        )
        for r in pool
    ]
    assert len(set(signatures)) == len(pool) and len(pool) >= 1010
    assert len({r["language"] for r in pool[:500]}) == 104
    assert not set(signatures[:500]) & set(signatures[1000:1010])
    short = [r for r in corpus if r["length"] == "short"]
    assert len(short) == 104
    target = next(r for r in short if r["language"] == "en")
    write(args.output / "unique-load-corpus.json", pool)
    jobs = []
    remote = args.root / args.remote_plan_dir
    base = [
        str(args.root / "baseline/.venv/bin/python"),
        str(args.root / "load/run.py"),
        "--model",
        str(args.root / "model"),
        "--model-revision",
        "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        "--references",
        str(args.root / "languages/references/manifest.json"),
        "--steps",
        "16",
        "--audio-decode-batch-size",
        "1",
        "--max-cuda-graphs",
        "8",
        "--startup-seed",
        "20261103",
        "--connections",
        "1200",
    ]
    for profile in ["unique", "hot-identical", "hot-multilingual"]:
        for window in [1, 2]:
            name = f"stagger500-{profile}-{window}s"
            warm_ids = (
                [r["id"] for r in pool[1000:1010]]
                if profile == "unique"
                else (
                    [target["id"]]
                    if profile == "hot-identical"
                    else [r["id"] for r in short]
                )
            )
            warm = {
                "id": "disjoint-warmup" if profile == "unique" else "fill-cache-keys",
                "kind": "sequential",
                "case_ids": warm_ids,
                "content_profile": "unique",
                "warmup": True,
                "deadline_s": 1200,
            }
            main = {
                "id": name,
                "kind": "staggered",
                "requests": 500,
                "case_ids": [r["id"] for r in pool[:500]]
                if profile == "unique"
                else (
                    [target["id"]]
                    if profile == "hot-identical"
                    else [r["id"] for r in short]
                ),
                "content_profile": "unique" if profile == "unique" else profile,
                "arrival_distribution": "uniform",
                "arrival_rate_rps": 500 / window,
                "arrival_window_s": window,
                "first_scheduled_arrival_s": 0,
                "last_scheduled_arrival_s": 499 / (500 / window),
                "preconnect": 500,
                "deadline_s": 1200 if profile == "unique" else 180,
                "interpretation": "500 independent offers; body and cohort latency "
                "are both reported. A429 never counts as fast success. "
                + (
                    "Fresh unique inputs, outputcacheoff; longqueue is diagnostic."
                    if profile == "unique"
                    else "Warm reuse; inspect source headers, not novel inference."
                ),
            }
            write(args.output / f"{name}.json", {"phases": [warm, main]})
            for kind in (
                ["before", "candidate"] if profile == "unique" else ["candidate"]
            ):
                label = f"{kind}-{name}"
                command = base + [
                    "--source",
                    str(
                        args.root
                        / ("baseline" if kind == "before" else args.candidate_source)
                    ),
                    "--source-ref",
                    args.baseline_ref if kind == "before" else args.candidate_ref,
                    "--label",
                    label,
                    "--output",
                    str(args.root / "results" / label),
                    "--corpus",
                    str(remote / "unique-load-corpus.json")
                    if profile == "unique"
                    else str(args.root / "languages/corpus.json"),
                    "--plan",
                    str(remote / f"{name}.json"),
                    "--max-pending-requests",
                    "1200" if profile == "unique" else "16",
                    "--request-timeout",
                    "1500",
                    "--phase-timeout",
                    "1200",
                ]
                if kind == "candidate":
                    command += [
                        "--response-cache-max-bytes",
                        "0" if profile == "unique" else str(128 * 1024**2),
                        "--response-cache-ttl",
                        "600",
                        "--server-extra-flags-json",
                        '["--disable-unused-kv-cache"]',
                        "--expected-health-json",
                        json.dumps(
                            {
                                "disable_unused_kv_cache": True,
                                "llm_use_cache": False,
                                "split_cfg": False,
                                "inference_optimizations.split_cfg_enabled": False,
                            }
                        ),
                    ]
                jobs.append(
                    {
                        "id": label,
                        "command": command,
                        "profile": profile,
                        "full_success_required": 500,
                        "arrival_window_s": window,
                    }
                )
    rng = random.Random(20261103)
    offsets = [0.0]
    for _ in range(1499):
        offsets.append(offsets[-1] + rng.expovariate(25))
    realistic = {
        "id": "calls500-poisson-25rps",
        "kind": "staggered",
        "case_ids": [r["id"] for r in corpus],
        "requests": 1500,
        "arrival_distribution": "poisson",
        "arrival_rate_rps": 25,
        "arrival_seed": 20261103,
        "expected_arrival_horizon_s": 60,
        "seeded_last_arrival_s": offsets[-1],
        "preconnect": 128,
        "content_profile": "mixed-content-cyclic",
        "deadline_s": 180,
        "interpretation": "500 activecalls assuming one response/20s; cacheoff; "
        "pending16 overload profile. Allrejections remain in failuredenominator.",
    }
    warm = {
        "id": "warmup-short10",
        "kind": "sequential",
        "warmup": True,
        "case_ids": [r["id"] for r in short[:10]],
        "deadline_s": 120,
    }
    write(args.output / "realistic-calls500.json", {"phases": [warm, realistic]})
    for kind in ["before", "candidate"]:
        template = next(
            job
            for job in jobs
            if job["id"].startswith(kind + "-") and job["profile"] == "unique"
        )
        command = list(template["command"])
        label = f"{kind}-realistic-calls500"
        changes = {
            "--connections": 128,
            "--label": label,
            "--output": str(args.root / "results" / label),
            "--corpus": str(args.root / "languages/corpus.json"),
            "--plan": str(remote / "realistic-calls500.json"),
            "--max-pending-requests": 16,
            "--request-timeout": 300,
            "--phase-timeout": 180,
        }
        for flag, value in changes.items():
            command[command.index(flag) + 1] = str(value)
        jobs.append(
            {
                "id": label,
                "command": command,
                "profile": "realistic-calls500",
                "offered_assistant_turns": 1500,
                "assumed_turn_interval_s": 20,
            }
        )
    write(
        args.output / "jobs.json",
        {
            "measurement_kind": "CPU-only planned commands; none launched",
            "goal": "1000 concurrent plus500 staggered; this plans the500 profiles",
            "jobs": jobs,
            "fixture_integrity": {
                "first500_languages": 104,
                "first500_unique_inputs": 500,
                "warmups_disjoint": True,
                "source_pool_sha256": hashlib.sha256(
                    (source / "load-pool.json").read_bytes()
                ).hexdigest(),
            },
            "realistic500_pending_required": 16,
            "cache_and_novel_results_separate": True,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "planned_jobs": len(jobs),
                "uniform_offers_each": 500,
                "covered_routes": 104,
            }
        )
    )


if __name__ == "__main__":
    main()
