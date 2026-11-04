#!/usr/bin/env python3
"""Independent retained-file/protocol accounting; no model inference."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def verify(root):
    metadata = read(root / "metadata.json")
    args = metadata["requested_args"]
    file_limits = metadata.get("file_descriptor_limits")
    if file_limits:
        soft, hard = file_limits["after_soft_hard"]
        assert soft == -1 or soft >= file_limits["minimum_required"]
        assert hard == file_limits["before_soft_hard"][1]
        assert file_limits["configured_http_connection_cap"] == args["connections"]
        child_limits = root / "server-process-limits.txt"
        if child_limits.exists():
            row = next(
                line.split()
                for line in child_limits.read_text().splitlines()
                if line.startswith("Max open files")
            )
            assert (
                row[3] == "unlimited"
                or int(row[3]) >= (file_limits["minimum_required"])
            )
    health = read(root / "health-before.json")
    assert metadata.get("runtime_validation"), "Actual runtime was not validated"
    requested = metadata["runtime_validation"]["actual_health_checked"]
    for key, value in requested.items():
        actual = health
        for part in key.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        assert actual == value, f"Runtime setting mismatch:{key}"
    for name in ["max_pending_requests", "max_pending_gpu_tasks"]:
        assert health["batcher_queue"][name] == args[name]
    if args["startup_seed"] is not None:
        seed = read(root / "server-seed.json")
        assert any(Path(x).name == "seeded_server.py" for x in metadata["command"])
        assert all(
            seed[key] == args["startup_seed"]
            for key in ["requested_seed", "torch_initial_seed", "cuda_initial_seed"]
        )
        assert seed["applied_before_module_import"] is True
    rows = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    assert len({row["id"] for row in rows}) == len(rows), "Duplicate request ID"
    quality = read(root / "quality-manifest.json")
    assert len({row["id"] for row in quality}) == len(quality)
    successes = {row["id"]: row for row in rows if row.get("status") == 200}
    assert set(successes) == {row["id"] for row in quality}, (
        "HTTP200 audio coverage gap"
    )
    paths = set()
    for row in quality:
        path = (root / row["path"]).resolve()
        assert path.is_relative_to(root.resolve()) and path not in paths
        paths.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"] == successes[row["id"]]["sha256"]
    ready = read(root / "quality-asr-ready.json")
    expected_ready = {
        row["id"]
        for row in quality
        if row.get("omnilingual_language") and row["audio"].get("asr_readable")
    }
    assert expected_ready == {row["id"] for row in ready}, "ASR eligibility mismatch"
    unmapped = read(root / "quality-unmapped.json")
    assert {row["id"] for row in unmapped} == {
        row["id"] for row in quality if not row.get("omnilingual_language")
    }
    summaries = read(root / "summary.json")
    plan = read(root / "plan.json")
    corpus = {row["id"]: row for row in read(root / "corpus.json")}
    # Validate independent phase counts from declared selection, not observed logs.
    planned = {}
    for phase in plan:
        selected = (
            list(corpus.values())
            if phase.get("case_ids", "all") == "all"
            else [corpus[key] for key in phase["case_ids"]]
        )
        if phase.get("lengths"):
            selected = [
                row for row in selected if row.get("length") in phase["lengths"]
            ]
        if phase.get("languages"):
            selected = [
                row for row in selected if row["language"] in phase["languages"]
            ]
        planned[phase["id"]] = phase.get("requests", len(selected))
    outcome = read(root / "outcome.json")
    for phase_id, summary in summaries.items():
        phase_rows = [row for row in rows if row["phase"] == phase_id]
        assert summary["offered_count"] == planned[phase_id]
        assert summary["attempts"] == len(phase_rows)
        assert summary["not_dispatched_count"] == planned[phase_id] - len(phase_rows)
        assert summary["status_counts"] == dict(
            Counter(str(row.get("status")) for row in phase_rows)
        )
        assert summary["http200"] == sum(row.get("status") == 200 for row in phase_rows)
        expected_target = len(phase_rows) == planned[phase_id] and all(
            row.get("status") == 200 and row["completion_from_barrier_ms"] <= 2000
            for row in phase_rows
        )
        assert summary["all_offers_completed_200_within_2s"] == expected_target
        for row in phase_rows:
            assert (
                abs(
                    row["completion_from_barrier_ms"]
                    - row["dispatched_from_barrier_ms"]
                    - row["complete_ms"]
                )
                < 0.001
            )
    if outcome["status"] == "completed":
        assert set(summaries) == set(planned), "Completed run skipped requested phase"
        assert Counter(row["phase"] for row in rows) == planned
    return {
        "measurement_kind": "CPU-only retained protocol/hash validation",
        "source_gpu": metadata["hardware"],
        "outcome": outcome["status"],
        "attempts": len(rows),
        "http200_hashes_verified": len(paths),
        "asr_eligible": len(ready),
        "unmapped_requires_review": len(unmapped),
        "planned_phase_counts": planned,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    for directory in parser.parse_args().runs:
        print(json.dumps(verify(directory.resolve())))
