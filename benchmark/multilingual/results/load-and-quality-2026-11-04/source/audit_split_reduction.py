"""Independent retained-array/WAV comparisons; CPU only, no model or network."""

import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "quality/split-reduction-gpu-root-v1-retained/prototypes/split-reduction-gpu-root-v1"
OUT = ROOT / "quality/split-reduction-gpu-root-v1-analysis"
ARMS = ("concat-true", "concat-false", "split-false", "split-true")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_npz(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        return all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes() for k in a.files)


def main():
    assert not OUT.exists()
    rows = read(SOURCE / "quality-manifest.json")
    assert len(rows) == 108
    verified = []
    for row in rows:
        path = SOURCE / row["path"]
        assert sha(path) == row["sha256"]
        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 24000 and len(wave) and wave.shape[1] == 1 and np.isfinite(wave).all()
        valid = bool(np.abs(wave).max() > .001 and np.sqrt(np.mean(wave.astype(np.float64) ** 2)) > 1e-5)
        assert valid == row["signal_valid"]
        verified.append({"id": row["id"], "sha256": row["sha256"], "frames": len(wave), "signal_valid": valid})
    results = []
    for job in read(SOURCE / "summary.json"):
        batch = job["batch"]
        comparisons = []
        for left, right in itertools.combinations(ARMS, 2):
            stems = [f"en-short-B{batch}-{arm}-parity" for arm in (left, right)]
            tokens = equal_npz(*[SOURCE / (s + "-tokens.npz") for s in stems])
            floats = equal_npz(*[SOURCE / (s + "-float-audio.npz") for s in stems])
            pcm = all(sha(SOURCE / (stems[0] + f"-{i:03d}.wav")) == sha(SOURCE / (stems[1] + f"-{i:03d}.wav")) for i in range(batch))
            original = next(x for x in job["comparisons"] if (x["left"], x["right"]) == (left, right))
            assert (tokens, floats, pcm) == tuple(original[k] for k in ("tokens_equal", "float_WAV_equal", "PCM_WAV_equal"))
            comparisons.append({"left": left, "right": right, "tokens_exact": tokens, "float_WAV_exact": floats, "PCM_WAV_exact": pcm, "target_trace": original["target_trace"]})
        repeated = {}
        for arm in ARMS:
            stem = f"en-short-B{batch}-{arm}"
            repeated[arm] = all(equal_npz(SOURCE / (stem + "-parity-" + kind + ".npz"), SOURCE / (stem + "-timed-" + kind + ".npz")) for kind in ("tokens", "float-audio"))
            assert repeated[arm] == job["timed"][arm]["parity_and_timed_outputs_identical"]
        results.append({"case": "en-short", "batch": batch, "seed": job["seed"], "comparisons": comparisons, "parity_vs_timing_exact": repeated, "one_warm_generation_ms": {arm: job["timed"][arm]["full_model_ms"] for arm in ARMS}})
    assert len(results) == 2
    OUT.mkdir()
    summary = {"at": dt.datetime.now(dt.UTC).isoformat(), "kind": "Independent CPU SHA/SoundFile/raw-array-byte comparison; no model inference", "verified_WAVs": len(rows), "unique_WAVs": len({r["sha256"] for r in rows}), "signal_valid": sum(r["signal_valid"] for r in verified), "native_or_word_quality_certified": False, "default_promotion": False, "results": results}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "verified-WAVs.json").write_text(json.dumps(verified, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({"audit_sha256": sha(Path(__file__)), "source_files": {name: sha(SOURCE / name) for name in ("summary.json", "quality-manifest.json", "metadata.json", "outcome.json", "supervisor-outcome.json")}}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
