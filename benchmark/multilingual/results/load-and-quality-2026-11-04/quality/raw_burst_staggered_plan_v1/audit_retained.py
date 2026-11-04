"""CPU-only SHA/decoded-signal and all-offer timing audit of retained V4 phases."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("frozen_staggered", here / "run.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    rows = [json.loads(line) for line in (args.run / "requests.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows), "Duplicate outcome IDs"
    quality = json.loads((args.run / "quality-manifest.json").read_text())
    assert {row["id"] for row in quality} == {row["id"] for row in rows if row.get("status") == 200}
    audited = []
    for q in quality:
        row = by_id[q["id"]]
        assert q["expected"] == row["case"]["text"] and q["language"] == row["case"]["language"]
        path = args.run / q["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"] == q["sha256"]
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 24000 and samples.shape[1] == 1
        finite = bool(np.isfinite(samples).all())
        peak = float(np.abs(samples).max()) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
        valid = bool(samples.size and finite and peak > 0.001 and rms > 1e-5)
        assert valid == bool(q["audio"]["signal_valid"])
        audited.append({"id": q["id"], "phase": q["phase"], "sha256": q["sha256"],
                        "path": q["path"], "language": q["language"], "frames": len(samples),
                        "sample_rate": rate, "channels": 1, "finite": finite, "peak": peak,
                        "RMS": rms, "signal_valid": valid, "bytes": len(raw)})
    plan = json.loads((here / "plan.json").read_text())
    phases = {}
    for phase in plan["phases"]:
        offered = phase.get("requests", len(phase["case_ids"]))
        group = [row for row in rows if row["phase"] == phase["id"]]
        assert len(group) <= offered
        phases[phase["id"]] = controller.phase_readout(group, offered)
        phases[phase["id"]]["warmup"] = bool(phase.get("warmup"))
    summary = {
        "measurement_kind": "Apple M5 Pro CPU-only retained SHA/SoundFile/timing audit; no model inference",
        "TTS_hardware": json.loads((args.run / "metadata.json").read_text())["hardware"],
        "controller_sha256": hashlib.sha256((here / "run.py").read_bytes()).hexdigest(),
        "planned_cohort_offers": sum(p.get("requests", len(p["case_ids"])) for p in plan["phases"] if not p.get("warmup")),
        "recorded_outcomes": len(rows),
        "status_counts_including_warmup": dict(Counter(str(row.get("status")) for row in rows)),
        "verified_WAVs": len(audited), "verified_WAV_bytes": sum(row["bytes"] for row in audited),
        "invalid_signal_ids": [row["id"] for row in audited if not row["signal_valid"]],
        "all_offers_retain_missing_rejected_invalid_as_failure": True,
        "phases": phases,
        "word_quality": "Unscored here. Signal-valid HTTP200 does not establish words, speaker identity or native quality.",
        "legacy_inference_headers": "Unreported headers remain unreported; cache-off/neural evidence is separate.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "verified-audio.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audited))
    print(json.dumps({key: summary[key] for key in ("planned_cohort_offers", "recorded_outcomes", "verified_WAVs", "verified_WAV_bytes", "invalid_signal_ids")}, indent=2))


if __name__ == "__main__":
    main()
