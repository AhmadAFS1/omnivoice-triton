"""Post-run CPU retention: exact payload fetch plan and original-path hardlinks.

First copy all raw metadata and errors, excluding audio/. `plan` emits one audio
path per safe SHA plus every unsafe original path. Fetch those actual remote
files, then `restore --decode` verifies bytes/unique WAVs and all logical paths.
No model or network call is made here. Invalid/partial/error bytes are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(root, path):
    path = Path(path)
    assert not path.is_absolute() and ".." not in path.parts
    assert (root / path).resolve().is_relative_to(root.resolve())
    return path


def plan(root):
    quality = json.loads((root / "quality-manifest.json").read_text())
    requests = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    by_id = {row["id"]: row for row in requests}
    assert len(by_id) == len(requests)
    groups, unsafe = defaultdict(list), []
    for q in quality:
        row = by_id[q["id"]]
        path = str(relative(root, q["path"]))
        assert q["sha256"] == row["sha256"] and q["expected"] == row["case"]["text"]
        safe = (
            row.get("status") == 200
            and row.get("full_body_received") is True
            and row.get("headers", {}).get("content-type", "").split(";")[0].lower()
            == "audio/wav"
            and q.get("audio", {}).get("signal_valid") is True
            and q.get("audio", {}).get("asr_readable") is True
        )
        item = {"id": q["id"], "path": path, "sha256": q["sha256"]}
        if safe:
            groups[q["sha256"]].append(item)
        else:
            unsafe.append(item)
    # Conflicting validation flags for identical bytes stay uncollapsed.
    unsafe_hashes = {item["sha256"] for item in unsafe}
    for digest in list(groups):
        if digest in unsafe_hashes:
            unsafe.extend(groups.pop(digest))
    needed = [items[0]["path"] for items in groups.values()] + [
        item["path"] for item in unsafe
    ]
    error_files = []
    for row in requests:
        if row.get("error_body_path"):
            path = str(relative(root, row["error_body_path"]))
            error_files.append(
                {"id": row["id"], "path": path, "sha256": row["error_body_sha256"]}
            )
    return {
        "raw_requests": len(requests),
        "quality_rows": len(quality),
        "metadata_SHA256": {
            name: sha(root / name)
            for name in ("quality-manifest.json", "requests.jsonl")
        },
        "safe_groups": dict(groups),
        "unsafe_originals": unsafe,
        "separate_error_files": error_files,
        "remote_audio_paths_needed": needed,
        "policy": "Safe full200 signal-valid audio only; invalid/partial/error bytes are not deduplicated as valid. Raw metadata/original remote files unchanged.",
    }


def restore(root, manifest, decode=False):
    for name, expected in manifest["metadata_SHA256"].items():
        assert sha(root / name) == expected
    payloads, hardlinks, logical = 0, 0, 0
    decoded = []
    for digest, items in manifest["safe_groups"].items():
        representative = root / relative(root, items[0]["path"])
        assert sha(representative) == digest
        payloads += representative.stat().st_size
        if decode:
            import numpy as np
            import soundfile as sf

            values, rate = sf.read(representative, dtype="float32", always_2d=True)
            assert (
                rate == 24000
                and values.shape[1] == 1
                and values.size
                and np.isfinite(values).all()
            )
            rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
            peak = float(np.max(np.abs(values)))
            assert rms > 1e-5 and peak > 0.001
            decoded.append(
                {
                    "sha256": digest,
                    "rate": rate,
                    "channels": values.shape[1],
                    "frames": len(values),
                    "RMS": rms,
                    "peak": peak,
                    "finite": True,
                }
            )
        for item in items:
            destination = root / relative(root, item["path"])
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(representative, destination)
                hardlinks += 1
            assert sha(destination) == item["sha256"]
            logical += 1
    for item in manifest["unsafe_originals"] + manifest["separate_error_files"]:
        assert sha(root / relative(root, item["path"])) == item["sha256"]
    return {
        "measurement_kind": "CPU-only SHA/hardlink/optional SoundFile decode; no model/GPU run",
        "raw_recorded_requests": manifest["raw_requests"],
        "quality_rows": manifest["quality_rows"],
        "safe_logical_paths_SHA_verified": logical,
        "safe_unique_actual_remote_payloads": len(manifest["safe_groups"]),
        "actual_remote_payload_bytes_fetched": payloads,
        "hardlinks_created": hardlinks,
        "unsafe_originals_separately_retained": len(manifest["unsafe_originals"]),
        "error_body_files_separately_retained": len(manifest["separate_error_files"]),
        "unique_payload_decodes": decoded,
        "metadata_SHA256": manifest["metadata_SHA256"],
        "retention_kind": "One actual remote WAV per verified safe SHA, byte-identical original pathnames restored as local hardlinks. All original raw remote WAVs remain retained. No synthetic audio substitution.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "restore"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    if args.mode == "plan":
        value = plan(root)
        (root / "local-retention-plan.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )
        (root / "remote-audio-files-needed.txt").write_text(
            "".join(path + "\n" for path in value["remote_audio_paths_needed"])
        )
        print(
            json.dumps(
                {
                    "safe_unique_payloads": len(value["safe_groups"]),
                    "unsafe_originals": len(value["unsafe_originals"]),
                    "logical_quality_rows": value["quality_rows"],
                },
                indent=2,
            )
        )
    else:
        value = restore(
            root,
            json.loads((root / "local-retention-plan.json").read_text()),
            args.decode,
        )
        (root / "LOCAL_DEDUP_RETENTION_PROVENANCE.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
