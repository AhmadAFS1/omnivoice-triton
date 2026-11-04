"""CPU-only transcript retention/resume checks; no audio model or GPU is loaded."""

import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_bounded as bounded


class BoundedProtocol(unittest.TestCase):
    def test_cpu_subprocesses_complete_and_resume_without_new_workers(self):
        # The worker is a synthetic JSON writer. No model library is imported.
        fake = """import argparse,json
from pathlib import Path
def prepare(manifest,output):
 rows=json.loads(Path(manifest).read_text())
 (output/'prepared.json').write_text(json.dumps(rows))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for x in ['output','worker-model','device','dtype']:p.add_argument('--'+x)
 a=p.parse_args();r=Path(a.output);m=a.worker_model
 rows=json.loads((r/'prepared.json').read_text())
 (r/(m+'-metadata.json')).write_text(json.dumps({'model_card':m,'blind':True,
 'expected_text_prompt':False,'measurement_kind':'CPU-only synthetic writer'}))
 with (r/(m+'.jsonl')).open('w') as f:
  for row in rows:f.write(json.dumps({'id':row['id'],'sha256':row['sha256'],
   'model_card':m,'transcript':'Synthetic fixture','asr_wall_s':0.001})+'\\n')
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root / "helpers"
            helper.mkdir()
            (helper / "evaluate.py").write_text(fake)
            prepared = root / "prepared"
            prepared.mkdir()
            rows = [
                {
                    "id": "cpu-only",
                    "sha256": "a",
                    "lang_id": "eng_Latn",
                    "omnilingual_language": "eng_Latn",
                    "language": "en",
                }
            ]
            bounded.write(prepared / "asr-input-manifest.json", rows)
            marker = root / "assets.json"
            bounded.write(
                marker,
                {"baseline_package_freeze_unchanged": True, "assets": [{}, {}, {}]},
            )
            output = root / "output"
            argv = [
                "test",
                "--prepared",
                str(prepared),
                "--output",
                str(output),
                "--budget-seconds",
                "10",
                "--reserve-seconds",
                "1",
                "--asset-marker",
                str(marker),
                "--allow-gpu-inference",
            ]
            original_sleep = time.sleep
            with (
                patch.object(bounded, "HERE", helper),
                patch.object(bounded, "gpu_idle", return_value={"CPU_only_mock": True}),
                patch.object(
                    bounded.time, "sleep", side_effect=lambda _: original_sleep(0.01)
                ),
                patch("sys.argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                bounded.main()
                outcome = json.loads((output / "outcome.json").read_text())
                self.assertEqual(outcome["status"], "completed")
                self.assertEqual(outcome["both_models_covered"], 1)
                self.assertEqual(len(list((output / "raw").iterdir())), 2)
                with patch("sys.argv", argv + ["--resume"]):
                    bounded.main()
                self.assertEqual(len(list((output / "raw").iterdir())), 2)
                original_hashes = {
                    str(p): bounded.digest(p)
                    for p in (output / "published").rglob("*.jsonl")
                }
                final_prepared = root / "final-prepared"
                final_prepared.mkdir()
                more = rows + [rows[0] | {"id": "new-waveform", "sha256": "b"}]
                bounded.write(final_prepared / "asr-input-manifest.json", more)
                final_output = root / "final-output"
                final_argv = list(argv)
                final_argv[final_argv.index("--prepared") + 1] = str(final_prepared)
                final_argv[final_argv.index("--output") + 1] = str(final_output)
                final_argv += ["--prior-asr-root", str(output)]
                with patch("sys.argv", final_argv):
                    bounded.main()
                result = json.loads((final_output / "outcome.json").read_text())
                self.assertEqual(result["both_models_covered"], 2)
                for raw in (final_output / "raw").iterdir():
                    self.assertEqual(
                        [
                            r["id"]
                            for r in json.loads((raw / "prepared.json").read_text())
                        ],
                        ["new-waveform"],
                    )
                for p, sha in original_hashes.items():
                    self.assertEqual(bounded.digest(Path(p)), sha)
                imports = json.loads((final_output / "prior-imports.json").read_text())
                self.assertEqual(sum(r["imported_rows"] for r in imports), 2)
                # Same WAV bytes with a different exact script are not a cache hit.
                mismatch = {
                    "other-script": rows[0]
                    | {"id": "other-script", "lang_id": "spa_Latn"}
                }
                self.assertEqual(
                    bounded.import_prior(
                        [output],
                        root / "mismatch",
                        mismatch,
                        bounded.digest(helper / "evaluate.py"),
                        "bfloat16",
                    ),
                    [],
                )

    def test_priority_and_unique_language_namespace(self):
        row = {
            "id": "same-bytes-en",
            "sha256": "one",
            "language": "en",
            "omnilingual_language": "eng_Latn",
        }
        other = row | {"id": "same-bytes-es", "omnilingual_language": "spa_Latn"}
        bounded.validate_manifest([row, other])
        with self.assertRaises(AssertionError):
            bounded.validate_manifest([row, row | {"id": "duplicate"}])
        self.assertLess(
            bounded.priority(row | {"run": "reference-controls"}),
            bounded.priority(row | {"run": "before-all299"}),
        )
        self.assertLess(
            bounded.priority(row | {"run": "candidate-all299"}),
            bounded.priority(row | {"run": "cache-hot-identical1000"}),
        )

    def test_partial_publication_retains_raw_and_resumes_only_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw/000-test"
            raw.mkdir(parents=True)
            model = bounded.MODELS[0]
            prepared = [
                {"id": "good", "sha256": "a", "lang_id": "eng_Latn"},
                {"id": "wrong", "sha256": "b", "lang_id": "spa_Latn"},
            ]
            bounded.write(raw / "prepared.json", prepared)
            valid = {
                "id": "good",
                "sha256": "a",
                "model_card": model,
                "transcript": "Recognized from synthetic fixture.",
            }
            wrong = valid | {"id": "wrong", "sha256": "invalid"}
            content = json.dumps(valid) + "\n" + json.dumps(wrong) + '\n{"truncated":'
            path = raw / f"{model}.jsonl"
            path.write_text(content)
            bounded.write(
                raw / f"{model}-metadata.json",
                {
                    "model_card": model,
                    "blind": True,
                    "expected_text_prompt": False,
                    "measurement_kind": "CPU-only synthetic fixture, not GPU inference",
                },
            )
            published = root / "published/000-test"
            rows, issues = bounded.publish(raw, published, model)
            self.assertEqual(path.read_text(), content)
            self.assertEqual(rows, [valid])
            self.assertEqual(len(issues), 2)
            expected = {r["id"]: r for r in prepared}
            covered, dirs = bounded.collect_existing(root, expected)
            self.assertEqual(set(covered[model]), {"good"})
            self.assertEqual(dirs, [published])
            expected["good"] = expected["good"] | {"lang_id": "spa_Latn"}
            with self.assertRaises(AssertionError):
                bounded.collect_existing(root, expected)

    def test_missing_blind_metadata_never_becomes_published_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            raw.mkdir()
            model = bounded.MODELS[0]
            bounded.write(raw / "prepared.json", [{"id": "x", "sha256": "a"}])
            (raw / f"{model}.jsonl").write_text(
                json.dumps(
                    {
                        "id": "x",
                        "sha256": "a",
                        "model_card": model,
                        "transcript": "No metadata evidence",
                    }
                )
                + "\n"
            )
            with self.assertRaises(AssertionError):
                bounded.publish(raw, Path(temp) / "published/000", model)
            self.assertFalse((Path(temp) / "published/000").exists())


if __name__ == "__main__":
    unittest.main()
