"""Package completed benchmark evidence, excluding model/conditioning audio/keys."""

import argparse
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
SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".log", ".md", ".py"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--full-report", type=Path)
    parser.add_argument("--extra-folder", action="append", default=[], type=Path)
    parser.add_argument("--extra-run", action="append", default=[], type=Path)
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() == "max-concurency"
    destination = PACKAGE / "results" / args.name
    assert not destination.exists(), "Preserve earlier publications"
    destination.mkdir(parents=True)
    copied = []

    def copy_file(source, target):
        assert source.is_file() and source.suffix in SUFFIXES
        before = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), "No duplicate target or overwrite"
        compressed = source.stat().st_size >= 2_000_000
        if compressed:
            target = target.with_suffix(target.suffix + ".gz")
            with source.open("rb") as original, target.open("wb") as output:
                with gzip.GzipFile(fileobj=output, filename="", mode="wb", mtime=0) as compressed_output:
                    shutil.copyfileobj(original, compressed_output)
            digest = hashlib.sha256()
            with gzip.open(target, "rb") as output:
                for chunk in iter(lambda: output.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == before
        else:
            shutil.copyfile(source, target)
            assert sha(target) == before
        assert sha(source) == before, "Source changed during publication"
        copied.append({"source": str(source.relative_to(ROOT)), "target": str(target.relative_to(PACKAGE)), "source_sha256": before, "saved_sha256": sha(target), "source_bytes": source.stat().st_size, "saved_bytes": target.stat().st_size, "adaptation": "lossless gzip; decompressed SHA verified" if compressed else "none; historical absolute paths retained as provenance"})

    def copy_folder(source, label, recursive=False):
        assert source.is_dir()
        paths = source.rglob("*") if recursive else source.iterdir()
        for path in sorted(paths):
            if path.is_file() and path.suffix in SUFFIXES:
                relative = path.relative_to(source)
                assert not any(part in {"keys", "model", "hf-cache", "loader-state", "history-before-cleanup-review", "history-before-scheduled-arrival-SLO", "__pycache__"} for part in relative.parts)
                copy_file(path, destination / label / relative)

    modern = ROOT / "retained-59071593/results"
    runs = [modern / name for name in ("before-unique1000-graph2", "triton-eager-novel1000-B1-v1", "triton-eager-novel1000-B8-v1", "triton-eager-raw5000-cyclic1000-v1")]
    runs += [source.resolve() for source in args.extra_run]
    for source in runs:
        assert (source / "summary.json").exists()
        copy_folder(source, "runs/" + source.name)
    for name in ("novel1000-B1-B8-word-analysis-v1", "novel1000-c79-vs-eager-word-analysis-v1", "split-reduction-gpu-root-v1-analysis"):
        copy_folder(ROOT / "quality" / name, "quality/" + name)
    copy_folder(ROOT / "load/primary-deferral132-analysis-v1", "primary-deferral132-analysis-v1")
    copy_folder(ROOT / "load/raw_burst_v1-plan/retained-audit-v1", "raw5000-independent-audit")
    for name in ("raw_burst_v1.py", "raw_burst_server_v1.py", "raw_burst_tests_v1.py", "run.py", "verify.py", "run_v5.py", "verify_v5.py", "runtime.py", "seeded_server.py", "telemetry.py"):
        copy_file(ROOT / "load" / name, destination / "frozen-harness-source" / name)
    for name in ("config.json", "README.md", "CPU_READY.json", "root-arrival-recomputation-v1.json", "dry-plan.json"):
        path = ROOT / "load/raw_burst_v1-plan" / name
        if path.exists():
            copy_file(path, destination / "raw5000-plan" / name)
    for name in ("audit_split_reduction.py", "crosscheck_novel1000_whisper.py", "crosscheck_raw_outliers_whisper.py", "publish_final_evidence.py"):
        copy_file(ROOT / "quality" / name, destination / "source" / name)
    for source in args.extra_folder:
        source = source.resolve()
        copy_folder(source, "quality/" + source.name)
    if args.full_report is not None:
        args.full_report = args.full_report.resolve()
        assert (args.full_report / "summary.json").exists()
        copy_folder(args.full_report, "full-two-judge-report")
    (destination / "publication-provenance.json").write_text(json.dumps({"at": dt.datetime.now(dt.UTC).isoformat(), "kind": "CPU-only copy/hash/lossless compression of retained evidence; no inference", "publisher_sha256": sha(Path(__file__)), "files": copied, "WAV_policy": "All measured WAVs retained locally with exact hashes; generated before/after samples are published separately. No model weights, conditioning WAVs or transfer keys bundled."}, ensure_ascii=False, indent=2) + "\n")
    (destination / "README.md").write_text("""# Complete-audio load evidence

Read the [results and scaling report](../../../../docs/max-concurrency-results-2026-11-04.md)
for workload, hardware, quality and comparison limits. Each run preserves all
offers, full-body latency, exact source/config/package/seed metadata and failures.
Cached/coalesced traffic and fresh synthesis must remain separate. Rejections,
missing/invalid audio and protocol errors never count as completed speech.

Large JSON/JSONL/CSV/log files use lossless gzip. `publication-provenance.json`
records both saved and decompressed-original SHA256 values; decompression is
independently verified during packaging. Original absolute paths remain source
provenance; generated-audio hashes link the separate sample manifests or the
complete locally retained WAVs. Weights, conditioning audio and keys are excluded.

The harness copies are archival source snapshots with pinned helper/input hashes,
not self-contained deployments. Reproduction needs the recorded model assets,
matching reference WAVs, runtime and full fixture tree. ASR and signal checks do
not certify native speech quality, MOS or speaker identity.
""")
    print(json.dumps({"destination": str(destination), "files": len(copied), "saved_bytes": sum(row["saved_bytes"] for row in copied), "source_bytes": sum(row["source_bytes"] for row in copied)}))


if __name__ == "__main__":
    main()
