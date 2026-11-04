#!/usr/bin/env python3
"""Create auditable finite plans; this program does not launch model inference."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def arrival_metadata(count, rate, seed):
    """Match run.py's finite first-at-zero exponential-gap schedule exactly."""
    rng = random.Random(seed)
    offsets = [0.0]
    gaps = [rng.expovariate(rate) for _ in range(count - 1)]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    ordered_gaps = sorted(gaps)
    return {
        "schedule_kind": "fixed count, independent exponential inter-arrival gaps",
        "first_arrival_s": 0,
        "count_over_rate_expected_horizon_s": count / rate,
        "last_arrival_expected_s": (count - 1) / rate,
        "seeded_last_arrival_s": offsets[-1],
        "seeded_arrivals_before_60s": sum(value < 60 for value in offsets),
        "seeded_gap_min_s": min(gaps),
        "seeded_gap_p50_s": ordered_gaps[len(gaps) // 2],
        "seeded_gap_p95_s": ordered_gaps[int((len(gaps) - 1) * 0.95)],
        "seeded_gap_max_s": max(gaps),
        "offsets_sha256": hashlib.sha256(
            json.dumps(offsets, separators=(",", ":")).encode()
        ).hexdigest(),
        "note": "Finite count is fixed; final time is random rather than exactly60s. "
        "Actual dispatch times and lateness remain per request in requests.jsonl.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-pool", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.corpus.read_text())
    cases = raw["cases"] if isinstance(raw, dict) else raw
    args.output.mkdir(parents=True, exist_ok=False)
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    short = [case for case in cases if case["length"] == "short"]
    assert short
    # Mix scripts/languages deliberately before taking shorter load subsets.
    preferred = ["en", "es", "ar", "hi", "zh", "ja", "ru", "ko", "fr", "sw"]
    ordered = sorted(
        short,
        key=lambda case: (
            preferred.index(case["language"]) if case["language"] in preferred else 100,
            case["language"],
        ),
    )
    unique_pool = []
    signatures = set()
    for case in ordered + cases:
        signature = (case["text"], case["api_language"], case.get("reference_id"))
        if signature not in signatures:
            signatures.add(signature)
            unique_pool.append(case)
    qa = [
        {
            "id": f"qa-{length}-all-languages",
            "kind": "sequential",
            "case_ids": [case["id"] for case in cases if case["length"] == length],
            "content_profile": "unique",
            "deadline_s": 1200,
            "reference_mode": "registered",
            "warmup": False,
        }
        for length in ["short", "medium", "critical"]
        if any(case["length"] == length for case in cases)
    ]
    write(args.output / "quality-all-languages.json", {"phases": qa})
    warm = {
        "id": "warmup-language-short",
        "kind": "sequential",
        "case_ids": [case["id"] for case in ordered[:10]],
        "warmup": True,
        "content_profile": "unique",
        "deadline_s": 120,
    }
    bursts = [warm]
    for count in [1, 10, 100]:
        pool = unique_pool[:count]
        if len(pool) == count:
            bursts.append(
                {
                    "id": f"burst{count}-unique",
                    "kind": "burst",
                    "case_ids": [case["id"] for case in pool],
                    "content_profile": "unique",
                    "deadline_s": 600,
                    "preconnect": min(count, 128),
                }
            )
    bursts.append(
        {
            "id": "burst1000-repeat-short-compute",
            "kind": "burst",
            "case_ids": [case["id"] for case in ordered[:10]],
            "requests": 1000,
            "content_profile": "repeated-short-compute",
            "deadline_s": 900,
            "preconnect": 128,
            "interpretation": "Repeated content only. If output caching is added, "
            "this must be reported as cache-hit service, not unique inference.",
        }
    )
    write(args.output / "bursts.json", {"phases": bursts})
    write(args.output / "basic-bursts.json", {"phases": bursts[:-1]})
    staggered = [warm]
    for calls, rps, count in [
        (10, 0.5, 30),
        (100, 5, 300),
        (1000, 50, 3000),
        (5000, 250, 15000),
    ]:
        staggered.append(
            {
                "id": f"calls{calls}-poisson-{rps}rps",
                "kind": "staggered",
                "case_ids": [case["id"] for case in unique_pool],
                "requests": count,
                "arrival_rate_rps": rps,
                "arrival_distribution": "poisson",
                "arrival_seed": 20261103,
                "deadline_s": 180,
                "expected_arrival_horizon_s": 60,
                "arrival_schedule_statistics": arrival_metadata(count, rps, 20261103),
                "content_profile": "mixed-content-cyclic",
                "preconnect": 128,
                "interpretation": f"{calls} active-call scenario assumes one assistant "
                "response per20seconds. Fresh generations with output cache disabled; "
                "offered rate is independent of server completions. "
                "Use pending16 overload admission, never the pending1200 deep queue.",
            }
        )
    write(args.output / "staggered.json", {"phases": staggered[:-1]})
    write(
        args.output / "staggered5000-optional.json", {"phases": [warm, staggered[-1]]}
    )
    pool_path = args.load_pool or args.corpus.parent / "load-pool.json"
    unique = json.loads(pool_path.read_text()) if pool_path.exists() else list(cases)
    if isinstance(unique, dict):
        unique = unique["cases"]
    signatures = {
        (case["text"], case["api_language"], case.get("reference_id"))
        for case in unique
    }
    assert len(signatures) == len(unique), "Unique pool repeats effective inputs"
    write(args.output / "unique-load-corpus.json", unique)
    if len(unique) >= 1010:
        write(
            args.output / "burst1000-unique-short.json",
            {
                "phases": [
                    {
                        "id": "warmup-disjoint-inputs",
                        "kind": "sequential",
                        "case_ids": [case["id"] for case in unique[1000:1010]],
                        "content_profile": "unique",
                        "warmup": True,
                        "deadline_s": 120,
                    },
                    {
                        "id": "burst1000-unique-published",
                        "kind": "burst",
                        "case_ids": [case["id"] for case in unique[:1000]],
                        "content_profile": "unique",
                        "deadline_s": 1200,
                        "preconnect": 1000,
                        "cache_intent": "cold-output-cache",
                        "interpretation": (
                            "1000 unique published text/language/voice inputs; "
                            "warmups are disjoint. Spoken durations vary. "
                            "Coverage must be checked from the actual selected IDs."
                        ),
                    },
                ]
            },
        )
    cache_case = next(
        (case for case in ordered if case["language"] == "en"), ordered[0]
    )
    other = next((case for case in cases if case["id"] != cache_case["id"]), None)
    cold = []
    if other:
        cold.append(
            {
                "id": "warmup-disjoint-from-cold-key",
                "kind": "sequential",
                "case_ids": [other["id"]],
                "warmup": True,
                "content_profile": "unique",
                "deadline_s": 120,
            }
        )
    cold.append(
        {
            "id": "cold-identical1000",
            "kind": "burst",
            "case_ids": [cache_case["id"]],
            "requests": 1000,
            "content_profile": "identical-inflight-coalescing",
            "cache_intent": "target-key-cold",
            "preconnect": 1000,
            "deadline_s": 180,
        }
    )
    write(args.output / "cold-identical1000.json", {"phases": cold})
    write(
        args.output / "hot-identical1000.json",
        {
            "phases": [
                {
                    "id": "fill-one-cache-key",
                    "kind": "sequential",
                    "case_ids": [cache_case["id"]],
                    "warmup": True,
                    "content_profile": "unique",
                    "deadline_s": 120,
                },
                {
                    "id": "hot-identical1000",
                    "kind": "burst",
                    "case_ids": [cache_case["id"]],
                    "requests": 1000,
                    "content_profile": "identical-warm-cache",
                    "cache_intent": "warm",
                    "preconnect": 1000,
                    "deadline_s": 180,
                },
            ]
        },
    )
    write(
        args.output / "hot-multilingual1000.json",
        {
            "phases": [
                {
                    "id": "fill-all-language-short-keys",
                    "kind": "sequential",
                    "case_ids": [case["id"] for case in ordered],
                    "warmup": True,
                    "content_profile": "unique",
                    "deadline_s": 1200,
                },
                {
                    "id": "hot-multilingual1000",
                    "kind": "burst",
                    "case_ids": [case["id"] for case in ordered],
                    "requests": 1000,
                    "content_profile": "multilingual-warm-cache",
                    "cache_intent": "warm",
                    "preconnect": 1000,
                    "deadline_s": 180,
                },
            ]
        },
    )
    hot_one = json.loads((args.output / "hot-identical1000.json").read_text())
    hot_one["phases"][-1].update(id="hot-identical5000", requests=5000, preconnect=5000)
    hot_one["phases"][-1]["interpretation"] = (
        "Optional retained-WAV warm-cache service stress. "
        "Use cache max-waiters6000; no novel inference capacity claim."
    )
    write(args.output / "hot-identical5000-optional.json", hot_one)
    languages = sorted({case["language"] for case in cases})
    counts = {
        "corpus_cases": len(cases),
        "target_languages": languages,
        "language_count": len(languages),
        "qa_requests": sum(len(p["case_ids"]) for p in qa),
        "burst_requests": sum(p.get("requests", len(p["case_ids"])) for p in bursts),
        "staggered_requests": sum(
            p.get("requests", len(p["case_ids"])) for p in staggered[:-1]
        ),
        "optional_staggered5000_requests": 15010,
        "optional_hot5000_requests": 5001,
        "expanded_unique_inputs": len(unique),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "hardware": (
            "Unverified until execution; no GPU inference performed by plan creation"
        ),
    }
    write(args.output / "counts.json", counts)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
