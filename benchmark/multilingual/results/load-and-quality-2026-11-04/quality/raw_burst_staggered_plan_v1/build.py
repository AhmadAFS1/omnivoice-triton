"""CPU-only four-load Poisson assay plans, using frozen V4 helper bodies."""

from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = Path("/workspace/omnivoice-max-concurency")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build():
    raw = json.loads((ROOT / "load/raw_burst_v1-plan/config.json").read_text())
    original_plan = json.loads((ROOT / raw["plan"]).read_text())
    original_ids = original_plan["phases"][1]["case_ids"]
    warm = dict(original_plan["phases"][0])
    warm["deadline_s"] = 40
    phases = [warm]
    schedule = []
    for calls, rate, count in (
        (10, 0.5, 30),
        (100, 5, 300),
        (1000, 50, 3000),
        (5000, 250, 15000),
    ):
        phase = {
            "id": f"calls{calls}-poisson-{str(rate).replace('.', '_')}rps",
            "kind": "staggered",
            "case_ids": original_ids,
            "requests": count,
            "arrival_rate_rps": rate,
            "arrival_distribution": "poisson",
            "arrival_seed": 20261104,
            "deadline_s": 75,
            "content_profile": "cyclic_original1000_cacheoff_offers_not_all_neural_execution",
            "assumed_active_calls": calls,
            "turn_interval_s": 20,
            "reference_mode": "registered",
        }
        phases.append(phase)
        rng, offset = random.Random(phase["arrival_seed"]), 0.0
        for _ in range(count - 1):
            offset += rng.expovariate(rate)
        schedule.append(
            {
                "phase": phase["id"],
                "calls": calls,
                "rate_rps": rate,
                "offers": count,
                "last_planned_offset_s": offset,
                "declared_phase_deadline_s": 75,
            }
        )
    write(HERE / "plan.json", {"phases": phases})
    # Compile only exact hash-pinned synchronous input helpers; no dependency,
    # Torch, CUDA, model or network import is involved in this CPU validation.
    protocol = ROOT / "load/run.py"
    assert digest(protocol) == raw["frozen_files"]["load/run.py"]
    wanted = {"digest", "load_inputs", "effective_signature", "expand_cases"}
    tree = ast.parse(protocol.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Path": Path, "json": json, "hashlib": hashlib, "random": random}
    exec(  # noqa: S102 -- exact locally hash-pinned helper bodies
        compile(ast.Module(body=definitions, type_ignores=[]), str(protocol), "exec"),
        namespace,
    )
    by_id, refs, actual_phases = namespace["load_inputs"](
        ROOT / raw["corpus"], HERE / "plan.json", ROOT / raw["references"]
    )
    for phase, row in zip(actual_phases[1:], schedule, strict=True):
        cases = namespace["expand_cases"](phase, by_id)
        row["routes_offered"] = len({case["language"] for case in cases})
        row["unique_effective_inputs"] = len(
            {namespace["effective_signature"](case) for case in cases}
        )
        assert len(cases) == row["offers"] and row["rate_rps"] == row["calls"] / 20
        assert row["last_planned_offset_s"] < row["declared_phase_deadline_s"]
    assert (
        len(refs) == 9 and len(namespace["expand_cases"](actual_phases[0], by_id)) == 10
    )
    output = REMOTE / "results/triton-eager-staggered-overload-calls-v1"
    command = [
        str(REMOTE / "baseline/.venv/bin/python"),
        str(REMOTE / "load/run.py"),
        "--source",
        str(REMOTE / "candidate-v8-audio-guard"),
        "--source-ref",
        raw["source_ref"],
        "--model",
        str(REMOTE / "model"),
        "--model-revision",
        raw["model_revision"],
        "--corpus",
        str(REMOTE / raw["corpus"]),
        "--plan",
        str(REMOTE / "load/raw_burst_staggered_plan_v1/plan.json"),
        "--references",
        str(REMOTE / raw["references"]),
        "--output",
        str(output),
        "--label",
        output.name,
        "--runner",
        "triton",
        "--port",
        "18781",
        "--collect-ms",
        "0",
        "--max-batch-requests",
        "1",
        "--max-pending-requests",
        "4",
        "--max-pending-gpu-tasks",
        "4",
        "--max-cuda-graphs",
        "0",
        "--audio-decode-batch-size",
        "1",
        "--steps",
        "16",
        "--response-cache-max-bytes",
        "0",
        "--startup-seed",
        "20261103",
        "--connections",
        "256",
        "--keepalive-expiry",
        "600",
        "--startup-timeout",
        "60",
        "--request-timeout",
        "20",
        "--phase-timeout",
        "75",
        "--server-extra-flags-json",
        '["--disable-unused-kv-cache"]',
        "--expected-health-json",
        json.dumps(
            {
                "runner": "triton",
                "disable_unused_kv_cache": True,
                "llm_use_cache": False,
                "split_cfg": False,
                "sage_attention": False,
                "full_triton_patch": False,
                "max_cuda_graph_row_tokens": 0,
                "response_cache.enabled": False,
            }
        ),
    ]
    write(
        HERE / "config.json",
        {
            "measurement_kind": "CPU-only future plan; no neural launch",
            "source_ref": raw["source_ref"],
            "source_tree_sha256": raw["source_tree_sha256"],
            "model_revision": raw["model_revision"],
            "GPU_UUID": raw["GPU_UUID"],
            "planned_GPU": "NVIDIA GeForce RTX4070Ti; fresh hardware/idle evidence required",
            "source_files": raw["source_files"],
            "model_files": raw["model_files"],
            "frozen_files": raw["frozen_files"],
            "plan_sha256": digest(HERE / "plan.json"),
            "stop_helper_path": "run_prefix_micro.py",
            "stop_helper_sha256": "8d9350f5cbbd13da22d12f6e1a7a97789a053e885fa57a3c0cd8602fb4e7f7bd",
            "command": command,
            "output": str(output),
            "total_bound_s": 360,
            "worker_bound_s": 325,
            "warmup_offers": 10,
            "cohort_offers": 18330,
            "schedule": schedule,
            "timing_scope": "Four Poisson schedules (~248.4s total offered span), one startup and ten disjoint warmups outside cohorts; finite360s outer bound",
            "transport": "Frozen V4 standardHTTPX max256 pool, clientkeepalive600; product/Uvicorn defaults unchanged. Report actual connect events and dispatchlateness, no5000socketburst claim.",
            "future_gate": "Only after raw5k retained+idle and subsequent modernLLMASR released+idle, plus new exact-scope root lease. No execution authorized by this CPU plan.",
            "failure_policy": "Frozen V4 stops after phase on non200/429 or transporterror; partial/later-unexecuted phases remain explicit, no hidden retry.",
            "quality": "HTTP200/basic signal validity do not certify words/voice. Count accepted validaudio separately from429/admission/errors; never call rejections throughput.",
            "SLO_clocks": "Retain actualPOSTdispatch-to-fullbody metrics and separately compute scheduledarrival-to-fullbody=complete_ms+dispatch_lateness_ms. Report p50/p95/p99 with explicit HTTP200 timing denominator; valid-audio <=500ms/2s counts always divide by all configured offers, including missing/rejected/invalid failures.",
        },
    )
    write(
        HERE / "CPU-validation.json",
        {
            "measurement_kind": "CPU-only exactV4 helper input/ref/expansion validation",
            "frozen_V4_sha256": digest(protocol),
            "references_verified": len(refs),
            "warmups_disjoint": True,
            "offers_total": 18340,
            "cohort_offers": 18330,
            "schedule": schedule,
            "GPU_executed": False,
        },
    )
    print(
        json.dumps(
            {"GPU_executed": False, "cohort_offers": 18330, "schedule": schedule},
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
