#!/usr/bin/env python3
"""Select fixed provenance cohorts from a canonical quality preparation, CPU only.

No WAV hashing, models, transcript scores or worker calls. Canonical prep and
all original logical outcomes remain unchanged. Outputs are evaluator manifests,
not replacement quality preparations; score against the canonical preparation.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

FRONTIERS = {f"candidate-step-frontier-{steps}" for steps in (16, 8, 4)}
DIAGNOSTICS = {
    f"modern-v6-diagnostic-{language}-speed{speed}-post{post}"
    for language in ("tk", "so")
    for speed in ("1.0", "1.3")
    for post in ("off", "full")
}
INVALID_ORIGINALS = {
    ("modern-v6-voice-primary300", "primary100-Aoede-lingua-default-00049"),
    ("modern-v6-voice-primary300", "primary100-Aoede-lingua-default-00071"),
    ("modern-v6-voice-matrix130", "matrix130-group0-00018"),
    (
        "modern-v6-diagnostic-grid-10-remaining100-Charon-lingua-default",
        "remaining100-Charon-lingua-default-00071",
    ),
    (
        "modern-v6-diagnostic-grid-13-remaining100-Iapetus-lingua-default",
        "remaining100-Iapetus-lingua-default-00003",
    ),
}
LOW_SIGNAL_DIAGNOSTICS = {
    ("modern-v6-voice-gate104", "gate4lang-all8-lingua-default-00002"),
    (
        "modern-v6-diagnostic-grid-03-remaining100-Zephyr-lingua-video",
        "remaining100-Zephyr-lingua-video-00041",
    ),
    (
        "modern-v6-diagnostic-grid-05-remaining100-Zephyr-lingua-tutor",
        "remaining100-Zephyr-lingua-tutor-00041",
    ),
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def arm_name(row):
    return row["arm"].rsplit("/", 1)[-1]


def choose(logical, unique, inventory, cohort):
    by_id = {row["id"]: row for row in unique}
    assert len(by_id) == len(unique), "Duplicate canonical ASR IDs"
    routes = {row["language"] for row in inventory}
    if cohort == "priority104-step16":
        chosen = [
            row
            for row in logical
            if row["kind"] == "reference_control"
            or arm_name(row) == "candidate-step-frontier-16"
        ]
        cases = [row for row in chosen if row["kind"] == "request"]
        counts = Counter(row["language"] for row in cases)
        assert set(counts) == routes and all(n == 1 for n in counts.values()), (
            "Priority104 needs exactly one step16 request for every inventory route; "
            "do not silently replace missing/failed routes with another configuration"
        )
    elif cohort == "frontier-diagnostics":
        chosen = [
            row
            for row in logical
            if row["kind"] == "reference_control"
            or arm_name(row) in FRONTIERS | DIAGNOSTICS
            or (arm_name(row), row["original_id"])
            in INVALID_ORIGINALS | LOW_SIGNAL_DIAGNOSTICS
        ]
    elif cohort == "all":
        chosen = list(logical)
    else:
        raise ValueError(cohort)
    ids = {row["asr_input_id"] for row in chosen if row.get("asr_input_id")}
    missing = ids - by_id.keys()
    assert not missing, (
        f"Canonical logical mapping references missing ASR IDs: {missing}"
    )
    selected = [row for row in unique if row["id"] in ids]
    summary = {
        "cohort": cohort,
        "selection_basis": (
            "Fixed source arm/route/control provenance; "
            "diagnostic cohort additionally includes predefined zero-frame and "
            "weak/short signal cases. No ASR transcript or score selection"
        ),
        "selected_logical_rows": len(chosen),
        "selected_unique_ASR_inputs": len(selected),
        "logical_kinds": dict(Counter(row["kind"] for row in chosen)),
        "request_routes": sorted(
            {row["language"] for row in chosen if row["kind"] == "request"}
        ),
        "arms": dict(Counter(row["arm"] for row in chosen)),
        "unscorable_logical_outcomes": [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "language",
                    "http_status",
                    "issues",
                    "declared_asr_ready",
                )
            }
            for row in chosen
            if not row.get("asr_input_id")
        ],
        "reference_controls": sum(row["kind"] == "reference_control" for row in chosen),
        "interpretation": (
            "Unavailable/zero-frame/validation outcomes remain explicit "
            "and never become ASR inputs or passes"
        ),
    }
    return selected, chosen, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = {
        name: hashlib.sha256((args.prepared / name).read_bytes()).hexdigest()
        for name in (
            "asr-input-manifest.json",
            "logical-cases.jsonl",
            "inventory.json",
            "preparation.json",
        )
    }
    logical = [
        json.loads(line)
        for line in (args.prepared / "logical-cases.jsonl").read_text().splitlines()
    ]
    unique = json.loads((args.prepared / "asr-input-manifest.json").read_text())
    inventory = json.loads((args.prepared / "inventory.json").read_text())
    summaries = []
    for cohort in ("priority104-step16", "frontier-diagnostics", "all"):
        selected, chosen, summary = choose(logical, unique, inventory, cohort)
        output = args.output / cohort
        output.mkdir()
        write(output / "asr-input-manifest.json", selected)
        write(output / "selected-logical-outcomes.json", chosen)
        summary.update(
            source_prepared=str(args.prepared.resolve()), source_file_sha256=source
        )
        write(output / "selection.json", summary)
        summaries.append(summary)
    write(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
