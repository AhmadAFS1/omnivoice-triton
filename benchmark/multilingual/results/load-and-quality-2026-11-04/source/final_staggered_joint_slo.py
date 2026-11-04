"""CPU-only exact SHA/script join of words, signal and arrival-to-body latency."""

import datetime as dt
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ROOT / "retained-59071593/results/triton-eager-staggered-overload-calls-v1/requests.jsonl"
WORDS = ROOT / "quality/final-staggered-word-analysis-v1/scored533.jsonl"
OUT = ROOT / "quality/final-staggered-joint-slo-v1"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    requests, words = rows(REQUESTS), rows(WORDS)
    by_id = {r["original_id"]: r for r in words}
    assert len(requests) == 18340 and len(by_id) == len(words) == 533
    result, protocols = [], {}
    for request in requests:
        valid = request["status"] == 200 and request.get("audio", {}).get("signal_valid", False)
        word = by_id.get(request["id"])
        assert valid == bool(word)
        both_exact = both_zero = False
        latency = None
        if valid:
            assert word["hash_verified"] and word["actual_sha256"] == request["sha256"]
            assert word["expected"] == request["case"]["text"]
            assert word["omnilingual_language"] == request["case"]["omnilingual_language"]
            assert len(word["judges"]) == 2
            both_exact = all(r["exact_normalized"] for r in word["judges"].values())
            both_zero = all(r["cer"] == 0 for r in word["judges"].values())
            for model, judge in word["judges"].items():
                assert judge["sha256"] == request["sha256"]
                protocols.setdefault(model, set()).add(judge["asr_protocol_config_id"])
            if request["phase"] != "warmup-disjoint-inputs":
                latency = request["complete_ms"] + request["dispatch_lateness_ms"]
                assert math.isfinite(latency) and latency >= 0
        result.append({"id": request["id"], "phase": request["phase"], "language": request["case"]["language"], "status": request["status"], "signal_valid": valid, "ASR_proxy": request["case"].get("asr_mapping_is_proxy", False), "scheduled_arrival_to_body_ms": latency, "both_normalized_exact": both_exact, "both_zero_CER": both_zero, **{f"valid_and_both_{kind}_within_{bound}ms": bool(valid and latency is not None and latency <= bound and passed) for kind, passed in (("exact", both_exact), ("zeroCER", both_zero)) for bound in (500, 2000)}})
    assert all(len(v) == 1 for v in protocols.values())
    phases = {}
    cohort = [r for r in result if r["phase"] != "warmup-disjoint-inputs"]
    assert len(cohort) == 18330 and sum(r["signal_valid"] for r in cohort) == 523
    assert Counter(r["status"] for r in cohort) == {200: 523, 429: 17807}
    for phase in sorted({r["phase"] for r in cohort}):
        selected = [r for r in cohort if r["phase"] == phase]
        phases[phase] = {"offers": len(selected), "valid_WAVs": sum(r["signal_valid"] for r in selected), "rejected429": sum(r["status"] == 429 for r in selected), "both_exact": sum(r["both_normalized_exact"] for r in selected), "both_zero_CER": sum(r["both_zero_CER"] for r in selected), **{key: sum(r[key] for r in selected) for key in selected[0] if key.startswith("valid_and_both_")}}
    OUT.mkdir()
    summary = {"at": dt.datetime.now(dt.UTC).isoformat(), "kind": "Independent CPU-only strict SHA/expected Unicode/script join; no inference", "offers": 18330, "warmups_excluded": 10, "phases": phases, "protocols": {k: next(iter(v)) for k, v in protocols.items()}, "quality_certified": False, "interpretation": "Joint all-offer diagnostic: rejected, missing/invalid and slow audio cannot pass. ASR exactness/CER is not native/MOS/speaker certification. Actual scheduled arrival includes dispatch lateness."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "joined18340.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result))
    (OUT / "provenance.json").write_text(json.dumps({"source_files": [{"path": str(p), "sha256": sha(p)} for p in (REQUESTS, WORDS, Path(__file__))]}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
