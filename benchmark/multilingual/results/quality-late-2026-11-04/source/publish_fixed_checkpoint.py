"""Package retained fixed-duration samples and diagnostic evidence; no inference."""

import datetime as dt
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton")
PACKAGE = REPO / "benchmark/multilingual"
DEST = PACKAGE / "results/quality-late-2026-11-04"
SAMPLES = PACKAGE / "samples/rtx4070ti-fixed-reference-2026-11-04"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() == "max-concurency"
    assert not DEST.exists() and not SAMPLES.exists()
    DEST.mkdir(parents=True)
    SAMPLES.mkdir(parents=True)
    provenance = []
    folders = ["reference-fixed-duration-analysis-v1", "reference-fixed-duration-word-analysis-v1", "runner-word-analysis132-v1", "global2704-word-readout-v1", "speaker-fixed396-v1", "speaker-fixed396-independent-v1", "whisper-prefix-diagnostics-v1", "whisper-fixed-diagnostics-v1", "report-global7405-v1"]
    for name in folders:
        source = ROOT / "quality" / name
        output = DEST / name
        output.mkdir()
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".csv", ".py"}:
                continue
            target = output / path.name
            if path.stat().st_size >= 20_000_000:
                target = target.with_suffix(target.suffix + ".gz")
                with path.open("rb") as original, target.open("wb") as saved:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=saved, mtime=0) as compressed:
                        shutil.copyfileobj(original, compressed)
                with gzip.open(target, "rb") as saved:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: saved.read(1024 * 1024), b""):
                        h.update(chunk)
                assert h.hexdigest() == sha(path)
                adaptation = "lossless gzip, decompressed SHA verified"
            else:
                shutil.copyfile(path, target)
                assert sha(path) == sha(target)
                adaptation = "none; historical absolute paths remain provenance"
            provenance.append({"source": str(path.relative_to(ROOT)), "target": str(target.relative_to(PACKAGE)), "source_sha256": sha(path), "saved_sha256": sha(target), "adaptation": adaptation})
    (DEST / "source").mkdir()
    for name in ["audit_fixed_speaker.py", "speaker_similarity.py", "crosscheck_prefix_whisper.py", "crosscheck_fixed_whisper.py", "publish_fixed_checkpoint.py"]:
        path = ROOT / "quality" / name
        if not path.exists():
            continue
        shutil.copyfile(path, DEST / "source" / name)
        provenance.append({"source": str(path.relative_to(ROOT)), "target": str((DEST / "source" / name).relative_to(PACKAGE)), "source_sha256": sha(path), "saved_sha256": sha(DEST / "source" / name), "adaptation": "none"})
    manifest = []
    for variant in ["full", "prefix10", "prefix17"]:
        source = ROOT / "retained-59071593/results" / ("reference-fixed-duration-v1-" + variant)
        (SAMPLES / variant).mkdir()
        for line in (source / "requests.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["status"] == 200 and row["audio"]["signal_valid"]
            original = source / row["path"]
            target = SAMPLES / variant / original.name
            assert sha(original) == row["sha256"]
            shutil.copyfile(original, target)
            assert sha(target) == row["sha256"]
            case = row["case"]
            manifest.append({"arm": variant, "file": str(target.relative_to(SAMPLES)), "sha256": row["sha256"], "request_id": row["id"], "comparison_original_case_id": case["comparison_original_case_id"], "language": case["language"], "omnilingual_language": case["omnilingual_language"], "voice_id": case["voice_id"], "expected": case["expected"], "conditioning_reference_sha256": case["conditioning_reference_sha256"], "unchanged_reference_control": case["unchanged_reference_control"], "generation_options": row["normalized_worker_options"], "audio": row["audio"], "target_tokens": int(row["headers"]["x-omnivoice-batch-target-tokens"]), "source": case["source"]})
    assert len(manifest) == 396 and len({r["language"] for r in manifest}) == 104 and len({r["voice_id"] for r in manifest}) == 8
    by_case = {}
    for row in manifest:
        by_case.setdefault(row["comparison_original_case_id"], []).append(row)
    assert len(by_case) == 132 and all(len(rows) == 3 and len({r["target_tokens"] for r in rows}) == 1 and len({r["audio"]["duration_s"] for r in rows}) == 1 for rows in by_case.values())
    write(SAMPLES / "manifest.json", manifest)
    write(SAMPLES / "verification.json", {"at": dt.datetime.now(dt.UTC).isoformat(), "kind": "CPU-only copy/hash and matched target/duration verification of previously independently decoded WAVs", "WAVs": 396, "bytes": sum((SAMPLES / r["file"]).stat().st_size for r in manifest), "unique_sha256": len({r["sha256"] for r in manifest}), "routes": 104, "voices": 8, "default_promotion": False, "conditioning_WAVs_bundled": False})
    (SAMPLES / "README.md").write_text("""# Fixed-duration reference samples, RTX4070Ti, November4

All396 generated full WAVs: full references, ten-word English prefixes and
seventeen-word English prefixes,132cases/arm across104routes and eight voices.
Every arm holds the original target frame count and output duration fixed.
Use `manifest.json` to join the same original case across arms, verify SHA and
inspect target text, full generation settings and source/license attribution.
Conditioning reference WAVs and model weights are excluded.

These are experimental crops. Word and relative speaker regressions remain;
neither crop is a default. Read the [joint word/speaker review](../../results/quality-late-2026-11-04/speaker-fixed396-independent-v1/report.md)
and [word results](../../results/quality-late-2026-11-04/reference-fixed-duration-word-analysis-v1/summary.json).
Raw scored Meta records are preserved in the losslessly compressed full7405
report; selected third-family Whisper results and all skip lists are adjacent.
ASR and ECAPA do not certify native quality, MOS or speaker identity.
""")
    write(DEST / "publication-provenance.json", {"at": dt.datetime.now(dt.UTC).isoformat(), "files": provenance, "samples": str(SAMPLES.relative_to(PACKAGE)), "sample_manifest_sha256": sha(SAMPLES / "manifest.json"), "publisher_sha256": sha(Path(__file__)), "inference_calls": 0})
    print(json.dumps({"report_files": len(provenance), "WAVs": len(manifest), "sample_manifest_sha256": sha(SAMPLES / "manifest.json")}))


if __name__ == "__main__":
    main()
