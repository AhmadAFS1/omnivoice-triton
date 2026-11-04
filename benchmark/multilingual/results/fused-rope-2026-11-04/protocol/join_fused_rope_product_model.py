"""CPU-only verification joining a bounded five-pair gate and its remaining pair."""

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/omnivoice-max-concurency")
SOURCE = "966a9afcf5913c8a4d9a3f29888e95ac2cfff18042356522e5416a805c2e2ccb"
RUNS = [
    "fused-rope-product-model-gpu-v2",
    "fused-rope-product-model-gpu-v3-continuation",
]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    expected = {(c, b) for c in ["en-short", "es-short", "ar-short"] for b in [1, 8]}
    joined, bodies, metadata_sources = {}, [], []
    for label in RUNS:
        root = ROOT / "prototypes" / label
        meta = read(root / "metadata.json")
        assert meta["source_ref"] == SOURCE
        metadata_sources.append(meta["source_files"])
        outcome = read(root / "outcome.json")
        rows = read(root / "measurements.json")
        assert outcome["pairs_checked"] == len(rows)
        if label.endswith("continuation"):
            assert outcome["status"] == "completed" and len(rows) == 1
        for row in rows:
            key = (row["case_id"], row["batch"])
            assert key in expected and key not in joined and row["exact"]
            stem = f"{key[0]}-B{key[1]}"
            parity = read(root / (stem + "-parity.json"))
            assert parity["every_full_logit_bit_and_input_token_wave_equal"]
            assert parity["patched_python_calls"] > 0
            assert not parity["product_metrics"]["fallbacks"]
            assert parity["product_metrics"]["patch_range"] == [0, 24]
            logits = [
                read(root / f"{stem}-{a}-full-logit-hashes.json")
                for a in ["reference", "fused-rope-first24"]
            ]
            assert len(logits[0]) == 16 and logits[0] == logits[1]
            for kind in ["tokens", "float-audio"]:
                paths = [
                    root / f"{stem}-{a}-parity-r0-{kind}.npz"
                    for a in ["reference", "fused-rope-first24"]
                ]
                with np.load(paths[0]) as first, np.load(paths[1]) as second:
                    assert first.files == second.files
                    for name in first.files:
                        x, y = first[name], second[name]
                        assert (
                            x.shape == y.shape
                            and x.dtype == y.dtype
                            and x.tobytes() == y.tobytes()
                        )
            for index in range(key[1]):
                pair = [
                    root / f"{stem}-{a}-parity-r0-{index:03d}.wav"
                    for a in ["reference", "fused-rope-first24"]
                ]
                assert sha(pair[0]) == sha(pair[1])
                bodies.extend({"path": str(p), "sha256": sha(p)} for p in pair)
            joined[key] = {
                "case": key[0],
                "batch": key[1],
                "run": label,
                "parity_sha256": sha(root / (stem + "-parity.json")),
            }
    assert metadata_sources[0] == metadata_sources[1]
    assert set(joined) == expected and len(bodies) == 54
    proof = {
        "at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_ref": SOURCE,
        "six_pairs_exact": True,
        "distinct_complete_pairs": 6,
        "verified_WAVs_for_complete_pairs": 54,
        "unique_WAV_sha256": len({r["sha256"] for r in bodies}),
        "pairs": list(joined.values()),
        "bodies": bodies,
        "scope": "CPU-verified exact16step full-logit/input hashes, raw token/float array bytes and PCM WAV bytes. Five pairs in bounded v2 plus explicitly authorized remaining AR B8 v3. Failed/partial prior outputs retained; no timing claim.",
    }
    p = ROOT / "FUSED_ROPE_PRODUCT_MODEL_JOINED_PARITY.json"
    p.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k not in ["bodies", "pairs"]}))


if __name__ == "__main__":
    main()
