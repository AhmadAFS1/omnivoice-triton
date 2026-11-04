"""CPU file-retention contracts; synthetic bytes, no model/GPU/ASR."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cache_retention", Path(__file__).with_name("retain_deduplicated.py")
)
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


class Tests(unittest.TestCase):
    def make(self, root, valids):
        rows, quality = [], []
        for i, valid in enumerate(valids):
            digest = hashlib.sha256(b"actual payload").hexdigest()
            name = f"audio/{i}.wav"
            rows.append(
                {
                    "id": str(i),
                    "sha256": digest,
                    "status": 200,
                    "full_body_received": True,
                    "headers": {"content-type": "audio/wav"},
                    "case": {"text": "gold"},
                }
            )
            quality.append(
                {
                    "id": str(i),
                    "sha256": digest,
                    "path": name,
                    "expected": "gold",
                    "audio": {"signal_valid": valid, "asr_readable": True},
                }
            )
        (root / "quality-manifest.json").write_text(json.dumps(quality))
        (root / "requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return rows, quality

    def test_one_actual_payload_restores_every_original_hardlink_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make(root, [True, True, True])
            plan = retention.plan(root)
            self.assertEqual(plan["remote_audio_paths_needed"], ["audio/0.wav"])
            (root / "audio").mkdir()
            (root / "audio/0.wav").write_bytes(b"actual payload")
            proof = retention.restore(root, plan)
            self.assertEqual(proof["safe_logical_paths_SHA_verified"], 3)
            self.assertEqual(proof["hardlinks_created"], 2)
            self.assertTrue((root / "audio/0.wav").samefile(root / "audio/2.wav"))

    def test_invalid_flag_conflict_does_not_collapse_as_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make(root, [True, False])
            plan = retention.plan(root)
            self.assertFalse(plan["safe_groups"])
            self.assertEqual(len(plan["unsafe_originals"]), 2)
            self.assertEqual(
                set(plan["remote_audio_paths_needed"]), {"audio/0.wav", "audio/1.wav"}
            )

    def test_partial_error_bytes_remain_separate_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, _ = self.make(root, [])
            (root / "errors").mkdir()
            body = root / "errors/partial.body"
            body.write_bytes(b"partial bytes")
            rows.append(
                {
                    "id": "failed",
                    "status": 200,
                    "full_body_received": False,
                    "error_body_path": "errors/partial.body",
                    "error_body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
                }
            )
            (root / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            proof = retention.restore(root, retention.plan(root))
            self.assertEqual(proof["safe_logical_paths_SHA_verified"], 0)
            self.assertEqual(proof["error_body_files_separately_retained"], 1)
            self.assertEqual(body.read_bytes(), b"partial bytes")


if __name__ == "__main__":
    unittest.main()
