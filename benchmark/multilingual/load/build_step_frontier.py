#!/usr/bin/env python3
"""Prepare a quality-gated step-count experiment; CPU-only, no GPU launch."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", default="candidate-v6")
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.fixtures / "all104-all8voices-and-config-smoke.json"
    grid = json.loads(original.read_text())
    selected = [
        case
        for case in grid
        if case.get("voice_id") == "Aoede"
        and case["config_type"] == "lingua-video"
        and case["mode"] == "clone"
    ]
    assert len(selected) == len({case["language"] for case in selected}) == 104
    assert all(case["worker_options"]["num_step"] == 16 for case in selected)
    corpus = []
    jobs = []
    remote_dir = args.root / "load/step-frontier-v1"
    for steps in (16, 8, 4):
        cases = []
        for source_case in selected:
            case = copy.deepcopy(source_case)
            case["comparison_original_case_id"] = source_case["id"]
            case["comparison_family"] = "step-count-frontier-v1"
            case["id"] = f"{source_case['language']}-short--Aoede--steps{steps}"
            case["config_type"] = f"experimental-num-step-{steps}"
            case["worker_options"]["num_step"] = steps
            cases.append(case)
        corpus.extend(cases)
        label = f"candidate-step-frontier-{steps}"
        write(
            args.output / f"steps{steps}.json",
            {
                "phases": [
                    {
                        "id": f"step-frontier-{steps}-all104",
                        "kind": "sequential",
                        "case_ids": [case["id"] for case in cases],
                        "deadline_s": 1200,
                        "content_profile": "quality-gated-step-count-frontier",
                        "interpretation": "Experimental step count, never a default "
                        "promotion. Compare identical published text/reference/speed "
                        "on all104routes with blind ASR and retained full WAVs.",
                    }
                ]
            },
        )
        jobs.append(
            {
                "id": label,
                "command": [
                    str(args.root / "baseline/.venv/bin/python"),
                    str(args.root / "load/run_v5.py"),
                    "--source",
                    str(args.root / args.source),
                    "--source-ref",
                    args.source_ref,
                    "--model",
                    str(args.root / "model"),
                    "--model-revision",
                    "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
                    "--corpus",
                    str(remote_dir / "corpus312.json"),
                    "--references",
                    str(args.root / "quality/voice-controls/references/manifest.json"),
                    "--plan",
                    str(remote_dir / f"steps{steps}.json"),
                    "--label",
                    label,
                    "--output",
                    str(args.root / "results" / label),
                    "--steps",
                    str(steps),
                    "--max-cuda-graphs",
                    "2",
                    "--audio-decode-batch-size",
                    "1",
                    "--startup-seed",
                    "20261103",
                    "--response-cache-max-bytes",
                    "0",
                    "--server-extra-flags-json",
                    '["--disable-unused-kv-cache"]',
                    "--expected-health-json",
                    json.dumps(
                        {
                            "disable_unused_kv_cache": True,
                            "llm_use_cache": False,
                            "split_cfg": False,
                            "response_cache.enabled": False,
                        }
                    ),
                ],
            }
        )
    write(args.output / "corpus312.json", corpus)
    write(
        args.output / "jobs.json",
        {
            "measurement_kind": "CPU-only plan preparation; GPU unverified until run",
            "source_fixture_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "jobs": jobs,
            "planned_cases": 312,
            "interpretation": "Source/voice/text/order/startup seed fixed per arm; "
            "onlynum_step changes. Concurrent RNG is not involved. Step count is "
            "a quality-changing experiment requiring independent word/signal review. "
            "No inference, throughput result or quality approval is implied.",
        },
    )
    print(json.dumps({"output": str(args.output), "cases": len(corpus)}))


if __name__ == "__main__":
    main()
