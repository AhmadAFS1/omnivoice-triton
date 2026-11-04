#!/usr/bin/env python3
"""CPU-only full-body/arrival/drain accounting from retained V4/V5 requests."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    left = int(index)
    right = min(len(values) - 1, left + 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def phase_report(phase, rows, offered):
    assert len({row["id"] for row in rows}) == len(rows), "Duplicate request IDs"
    assert len(rows) <= offered, "More observed rows than offered requests"
    normal = [r for r in rows if not r.get("validation_only", False)]
    successful = [r for r in normal if r.get("status") == 200]
    all200 = len(normal) == offered and len(successful) == offered and offered > 0
    decoded = sum(bool(r.get("audio", {}).get("asr_readable")) for r in successful)
    signal_valid = sum(bool(r.get("audio", {}).get("signal_valid")) for r in successful)
    body = [r["complete_ms"] for r in successful]
    cohort = [r["completion_from_barrier_ms"] for r in successful]
    scheduled = [r["scheduled_offset_ms"] for r in normal]
    dispatch = [r["dispatched_from_barrier_ms"] for r in normal]
    wire = [
        r["dispatched_from_barrier_ms"]
        + r["trace"]["http11.send_request_body.complete"]
        for r in normal
        if "http11.send_request_body.complete" in r.get("trace", {})
    ]

    def span(values):
        return max(values) - min(values) if values else None

    last_scheduled = max(scheduled) if scheduled else None
    last_dispatch = max(dispatch) if dispatch else None
    last_success = max(cohort) if cohort else None
    return {
        "phase": phase["id"],
        "kind": phase["kind"],
        "warmup": phase.get("warmup", False),
        "offered": offered,
        "attempted": len(rows),
        "successful_full_bodies": len(successful),
        "decoded_WAV_bodies": decoded,
        "signal_valid_bodies": signal_valid,
        "all_offers_http200": all200,
        "not_dispatched": offered - len(rows),
        "status_counts": dict(Counter(str(r.get("status")) for r in rows)),
        "per_request_full_body_ms": {
            "p50": percentile(body, 0.5),
            "p95": percentile(body, 0.95),
            "p99": percentile(body, 0.99),
            "max": max(body) if body else None,
        },
        "per_request_http200_within_500ms": sum(x <= 500 for x in body),
        "per_request_http200_within_2s": sum(x <= 2000 for x in body),
        "all_offers_http200_and_p95_under_500ms": all200
        and percentile(body, 0.95) < 500,
        "all_offers_http200_and_each_under_2s": all200 and max(body) <= 2000,
        "all_offers_http200_and_cohort_under_2s": all200 and max(cohort) <= 2000,
        "all_offers_signal_valid_and_cohort_under_2s": all200
        and signal_valid == offered
        and max(cohort) <= 2000,
        "cohort_completion_from_barrier_ms": {
            "p95": percentile(cohort, 0.95),
            "last": last_success,
        },
        "scheduled_arrival_span_ms": span(scheduled),
        "last_scheduled_arrival_ms": last_scheduled,
        "client_dispatch_span_ms": span(dispatch),
        "last_client_dispatch_ms": last_dispatch,
        "client_wire_body_sent_span_ms": span(wire),
        "client_wire_events": len(wire),
        "all200_last_scheduled_arrival_to_last_completion_ms": (
            last_success - last_scheduled if all200 else None
        ),
        "all200_last_dispatch_to_last_completion_ms": (
            last_success - last_dispatch if all200 else None
        ),
        "max_dispatch_lateness_ms": max(
            (r.get("dispatch_lateness_ms", 0) for r in normal), default=None
        ),
        "response_source_counts": dict(
            Counter(
                r.get("headers", {}).get(
                    "x-omnivoice-response-source", "unreported_legacy"
                )
                for r in successful
            )
        ),
        "inference_executed_counts": dict(
            Counter(
                r.get("headers", {}).get(
                    "x-omnivoice-inference-executed", "unreported_legacy"
                )
                for r in successful
            )
        ),
        "distinct_WAV_SHA256": len(
            {r.get("sha256") for r in successful if r.get("sha256")}
        ),
        "distinct_input_SHA256": len({r["effective_input_sha256"] for r in normal}),
        "interpretation": "HTTP200 latency is conditional; every offer must succeed. "
        "Dispatch/wire times are client observations, not exact server arrival. "
        "Cached/coalesced responses are separate from novel inference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh report path"
    root = args.run_dir
    plan = json.loads((root / "plan.json").read_text())
    summaries = json.loads((root / "summary.json").read_text())
    rows = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    metadata = json.loads((root / "metadata.json").read_text())
    results = []
    for phase in plan:
        if phase["id"] not in summaries:
            continue
        count = summaries[phase["id"]]["offered_count"]
        results.append(
            phase_report(phase, [r for r in rows if r["phase"] == phase["id"]], count)
        )
    report = {
        "measurement_kind": "CPU-only retained full-body timing analysis",
        "original_TTS_hardware": metadata["hardware"],
        "run": str(root.resolve()),
        "request_log_sha256": hashlib.sha256(
            (root / "requests.jsonl").read_bytes()
        ).hexdigest(),
        "phases": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "phases": len(results)}))


if __name__ == "__main__":
    main()
