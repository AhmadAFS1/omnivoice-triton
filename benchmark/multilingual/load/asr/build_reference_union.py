#!/usr/bin/env python3
"""CPU-only hash-verified portable union; source manifests/WAVs are unchanged."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources, ids = [], [], set()
    for manifest in args.manifest:
        manifest = manifest.resolve()
        sources.append(
            {"path": os.path.relpath(manifest, args.output), "sha256": sha(manifest)}
        )
        records = json.loads(manifest.read_text())
        for record in records:
            identity = record.get("reference_id", record["id"])
            assert identity not in ids, f"Duplicate reference ID:{identity}"
            ids.add(identity)
            path = manifest.parent / record.get("path", record["file"])
            path = path.resolve(strict=True)
            assert path.is_file() and sha(path) == record["sha256"], identity
            relative = os.path.relpath(path, args.output)
            rows.append(
                {
                    **record,
                    "path": relative,
                    "file": relative,
                    "union_source_manifest": os.path.relpath(manifest, args.output),
                    "union_source_manifest_sha256": sha(manifest),
                }
            )
    output = args.output / "manifest.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "measurement_kind": "CPU-only verified reference-manifest union; no inference",
        "source_manifests": sources,
        "references": len(rows),
        "unique_wav_hashes": len({row["sha256"] for row in rows}),
        "output_sha256": sha(output),
        "source_files_modified": False,
        "portability": (
            "Keep source quality/ directory layout on target; "
            "file/path both resolve relative to this union manifest"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
