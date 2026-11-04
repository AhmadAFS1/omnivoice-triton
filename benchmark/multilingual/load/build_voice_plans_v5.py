#!/usr/bin/env python3
"""Finite voice/configuration QA plans; CPU-only, never starts model inference."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Execution taskroot")
    parser.add_argument("--candidate-source", default="candidate-v5")
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--remote-plan-dir", default="load/voice-plans-v5")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fullfile = args.fixtures / "all104-all8voices-and-config-smoke.json"
    matrixfile = args.fixtures / "pairwise-and-boundary-matrix.json"
    invalidfile = args.fixtures / "validation-only-matrix.json"
    full = json.loads(fullfile.read_text())
    matrix = json.loads(matrixfile.read_text())
    invalid = json.loads(invalidfile.read_text())
    assert len(full) == 2704 and len(matrix) == 130 and len(invalid) == 28
    assert len({r["id"] for r in full}) == 2704
    voices = list(dict.fromkeys(r["voice_id"] for r in full if r["mode"] == "clone"))
    configs = ["lingua-default", "lingua-video", "lingua-tutor"]
    assert len(voices) == 8
    phases = []
    for voice in voices:
        for config in configs:
            cases = [
                r
                for r in full
                if r["mode"] == "clone"
                and r["voice_id"] == voice
                and r["config_type"] == config
            ]
            assert len(cases) == 104 and len({r["language"] for r in cases}) == 104
            phases.append(
                {
                    "id": f"all104-{voice}-{config}",
                    "kind": "sequential",
                    "case_ids": [r["id"] for r in cases],
                    "content_profile": "voice-configuration-grid",
                    "deadline_s": 1200,
                    "voice_id": voice,
                    "config_type": config,
                    "interpretation": "104 routes; voice/reference origin is explicit. "
                    "One full-WAV sample per route/preset; native review remains open.",
                }
            )
    for mode in ["auto", "design"]:
        cases = [r for r in full if r["mode"] == mode]
        assert len(cases) == 104
        phases.append(
            {
                "id": f"all104-worker-{mode}",
                "kind": "sequential",
                "case_ids": [r["id"] for r in cases],
                "deadline_s": 1200,
                "content_profile": "worker-mode-grid",
                "mode": mode,
            }
        )
    write(args.output / "all2704-voice-configs.json", {"phases": phases})
    gate = [r for r in full if r["language"] in {"ar", "en", "es", "hi"}]
    assert len(gate) == 104
    write(
        args.output / "gate104cases-4languages.json",
        {
            "phases": [
                {
                    "id": "gate104cases-4languages",
                    "kind": "sequential",
                    "case_ids": [r["id"] for r in gate],
                    "deadline_s": 1200,
                    "content_profile": "voice-configuration-grid",
                    "interpretation": "104 cases in four languages and eight voices, "
                    "three presets plus auto/design. Hindi uses English references.",
                }
            ]
        },
    )
    groups = list(dict.fromkeys(r.get("config_type", "matrix") for r in matrix))
    write(
        args.output / "matrix130.json",
        {
            "phases": [
                {
                    "id": f"matrix130-group{index}",
                    "kind": "sequential",
                    "deadline_s": 1200,
                    "case_ids": [
                        r["id"]
                        for r in matrix
                        if r.get("config_type", "matrix") == group
                    ],
                    "content_profile": "configuration-pairwise-and-boundary",
                    "configuration_group": group,
                }
                for index, group in enumerate(groups)
            ]
        },
    )
    assert all(
        r.get("validation_only") and r.get("expected_http_status") == 400
        for r in invalid
    )
    write(
        args.output / "validation28.json",
        {
            "phases": [
                {
                    "id": "API-boundary-validation28",
                    "kind": "sequential",
                    "validation_only": True,
                    "reference_mode": "upload",
                    "case_ids": [r["id"] for r in invalid],
                    "deadline_s": 180,
                    "interpretation": "Expected400 boundary checks only. "
                    "CPU fake-API first; numeric and semantic scopes stay separate. "
                    "These checks are excluded from throughput and word quality.",
                }
            ]
        },
    )
    remote = args.root / args.remote_plan_dir
    fixture_remote = args.root / "quality/voice-controls"
    jobs = []
    common = [
        str(args.root / "baseline/.venv/bin/python"),
        str(args.root / "load/run_v5.py"),
        "--model",
        str(args.root / "model"),
        "--model-revision",
        "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        "--steps",
        "16",
        "--max-cuda-graphs",
        "8",
        "--audio-decode-batch-size",
        "1",
        "--startup-seed",
        "20261103",
        "--references",
        str(fixture_remote / "references/manifest.json"),
    ]
    for plan, corpus, kinds in [
        ("gate104cases-4languages", fullfile.name, ["before", "candidate"]),
        ("all2704-voice-configs", fullfile.name, ["before", "candidate"]),
        ("matrix130", matrixfile.name, ["before", "candidate"]),
        ("validation28", invalidfile.name, ["candidate"]),
    ]:
        for kind in kinds:
            label = f"{kind}-voice-{plan}"
            command = common + [
                "--source",
                str(
                    args.root
                    / ("baseline" if kind == "before" else args.candidate_source)
                ),
                "--source-ref",
                "c79ce35" if kind == "before" else args.candidate_ref,
                "--corpus",
                str(fixture_remote / corpus),
                "--plan",
                str(remote / f"{plan}.json"),
                "--label",
                label,
                "--output",
                str(args.root / "results" / label),
            ]
            if kind == "candidate":
                command += [
                    "--response-cache-max-bytes",
                    "0",
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
                    "quality_gate_before_full": True,
                    "validation_only": plan == "validation28",
                }
            )
    write(
        args.output / "jobs.json",
        {
            "measurement_kind": "CPU-only command preparation; no GPU launch",
            "jobs": jobs,
            "counts": {
                "full": 2704,
                "full_phases": len(phases),
                "gate": 104,
                "gate_languages": 4,
                "matrix": 130,
                "validation_only": 28,
            },
            "mode_counts": dict(Counter(r["mode"] for r in full)),
            "fixture_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [fullfile, matrixfile, invalidfile]
            },
            "note": "Finish gate and both judges before full2704. Retain all bodies. "
            "Boundary checks have separate denominators. V4 remains unchanged.",
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "planned_jobs": len(jobs),
                "full_requests_per_arm": 2704,
                "full_phases": len(phases),
            }
        )
    )


if __name__ == "__main__":
    main()
