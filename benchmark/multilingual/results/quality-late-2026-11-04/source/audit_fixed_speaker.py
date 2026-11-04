"""Independent scalar/hash audit and strict word join; no inference or network."""

import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "quality"
SOURCE = QUALITY / "speaker-fixed396-v1"
OUT = QUALITY / "speaker-fixed396-independent-v1"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(
        math.fsum(x * x for x in a) * math.fsum(y * y for y in b)
    )


def main():
    assert not OUT.exists(), "Preserve previous audits"
    meta = read(SOURCE / "metadata.json")
    assert meta["inference_API_calls"] == 0 and meta["CPU"] == "Apple M5 Pro"
    assert meta["runner_sha256"] == sha(QUALITY / "speaker_similarity.py")
    for asset in meta["model_assets"]["files"]:
        assert sha(QUALITY / "speaker-model" / asset["name"]) == asset["sha256"]
    vectors = {}
    for row in lines(SOURCE / "embeddings.jsonl"):
        assert sha(Path(row["path"])) == row["sha256"]
        vector = row["embedding"]
        assert len(vector) == 192 and all(math.isfinite(x) for x in vector)
        assert abs(math.fsum(x * x for x in vector) - 1) < 1e-12
        assert row["sha256"] not in vectors
        vectors[row["sha256"]] = vector
    assert len(vectors) == 376
    references = read(QUALITY / "voice-controls/references/manifest.json")
    anchors = {r["voice_id"]: vectors[r["sha256"]] for r in references if r["language"] == "en"}
    assert len(anchors) == 8
    original = {(r["run"], r["id"]): r for r in read(SOURCE / "generated-scores.json")}
    scored_file = QUALITY / "report-global7405-v1/scored-logical-cases.jsonl"
    word_rows = {(r.get("arm"), r.get("original_id")): r for r in lines(scored_file) if r.get("kind") == "request"}
    pairs, protocols, count = [], {}, 0
    for case in lines(QUALITY / "reference-fixed-duration-analysis-v1/paired132.jsonl"):
        arms = {}
        for variant, clip in case["arms"].items():
            run = "reference-fixed-duration-v1-" + variant
            baseline = original[(run, clip["id"])]
            vector = vectors[clip["sha256"]]
            scores = {voice: max(-1.0, min(1.0, cosine(vector, anchor))) for voice, anchor in anchors.items()}
            assert all(abs(score - baseline["all_stock_scores"][voice]) < 1e-12 for voice, score in scores.items())
            own = scores[clip["voice_id"]]
            margin = own - max(s for voice, s in scores.items() if voice != clip["voice_id"])
            top1 = own >= max(scores.values())
            assert top1 == baseline["own_is_top1"]
            word = word_rows[(run, clip["id"])]
            assert word["hash_verified"] and word["actual_sha256"] == clip["sha256"]
            assert word["expected"] == clip["expected"] and len(word["judges"]) == 2
            for model, judge in word["judges"].items():
                protocols.setdefault(model, set()).add(judge["asr_protocol_config_id"])
            arms[variant] = {**clip, "run": run, "own_cosine": own, "own_margin": margin, "top1": top1, "judges": word["judges"]}
            count += 1
        before = arms["full"]
        for variant in ("prefix10", "prefix17"):
            after = arms[variant]
            assert before["frames"] == after["frames"]
            both_before = all(j["exact_normalized"] for j in before["judges"].values())
            both_after = all(j["exact_normalized"] for j in after["judges"].values())
            delta = {model: after["judges"][model]["cer"] - judge["cer"] for model, judge in before["judges"].items()}
            row = {"case": case["comparison_original_case_id"], "variant": variant, "language": before["language"], "voice": before["voice_id"], "control": before["unchanged_reference_control"], "before": before, "after": after, "delta_own_cosine": after["own_cosine"] - before["own_cosine"], "lost_top1": before["top1"] and not after["top1"], "gained_top1": not before["top1"] and after["top1"], "lost_both_exact": both_before and not both_after, "gained_both_exact": both_after and not both_before, "both_CER_worse": all(x > 0 for x in delta.values()), "CER_delta": delta}
            if row["control"]:
                assert before["sha256"] == after["sha256"] and row["delta_own_cosine"] == 0
            pairs.append(row)
    assert count == 396 and len(pairs) == 264 and all(len(v) == 1 for v in protocols.values())
    summary = {"at": dt.datetime.now(dt.UTC).isoformat(), "measurement_kind": "CPU-only independent scalar cosine, audio hash and strict SHA+script ASR join; no new inference", "generated": count, "pairs": len(pairs), "unique_embeddings": len(vectors), "unchanged_controls": sum(r["control"] for r in pairs), "quality_certified": False, "ASR_protocols": {k: next(iter(v)) for k, v in protocols.items()}, "variants": {}}
    for variant in ("prefix10", "prefix17"):
        rows = [r for r in pairs if r["variant"] == variant and not r["control"]]
        summary["variants"][variant] = {"n": len(rows), "median_own_cosine_delta": statistics.median(r["delta_own_cosine"] for r in rows), **{key: sum(r[key] for r in rows) for key in ("lost_top1", "gained_top1", "lost_both_exact", "gained_both_exact", "both_CER_worse")}}
    OUT.mkdir()
    (OUT / "pairs264.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    review = [r for r in pairs if r["lost_top1"] or r["lost_both_exact"] or r["both_CER_worse"]]
    (OUT / "flagged-review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    inputs = [Path(__file__), scored_file, QUALITY / "reference-fixed-duration-analysis-v1/paired132.jsonl", *[p for p in SOURCE.iterdir() if p.is_file()]]
    (OUT / "provenance.json").write_text(json.dumps({"inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs], "interpretation": "ECAPA is a relative synthetic/multilingual speaker screen; Meta ASR can disagree or mistranscribe. No native listening or MOS certification, no default promotion."}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
