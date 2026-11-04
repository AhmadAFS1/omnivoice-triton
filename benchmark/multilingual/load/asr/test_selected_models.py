"""CPU-only selected-model/configuration contracts; no GPU or model inference."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_selected_models as controller
from run_selected_models import (
    CONTROLLER,
    MODELS,
    collect_existing,
    coverage_summary,
    digest,
    import_prior,
    protocol_id,
    publish,
    select_models,
    validate_resume,
    write,
)


class SelectedModelsTests(unittest.TestCase):
    def test_same_namespace_imports_all_published_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior, target = root / "prior", root / "target"
            prior.mkdir()
            target.mkdir()
            model = MODELS[1]
            protocols = {MODELS[0]: "legacy-batch1", model: "modern-same"}
            write(
                prior / "plan.json", {"evaluator_sha256": "worker", "dtype": "bfloat16"}
            )
            expected = {}
            for index in range(2):
                identity = f"item{index}"
                source = {
                    "id": identity,
                    "sha256": f"wav{index}",
                    "lang_id": "eng_Latn",
                }
                expected[identity] = source
                raw = root / f"raw{index}"
                raw.mkdir()
                write(raw / "prepared.json", [source])
                row = {
                    **source,
                    "model_card": model,
                    "transcript": f"words{index}",
                    "asr_wall_s": 1.0,
                }
                (raw / f"{model}.jsonl").write_text(json.dumps(row) + "\n")
                write(
                    raw / f"{model}-metadata.json",
                    {"model_card": model, "blind": True, "expected_text_prompt": False},
                )
                publish(
                    raw,
                    prior / "published" / str(index),
                    model,
                    protocol=protocols[model],
                )
            imported = import_prior(
                [prior], target, expected, "worker", "bfloat16", [model], protocols
            )
            self.assertEqual([row["imported_rows"] for row in imported], [1, 1])
            covered, _ = collect_existing(target, expected, protocols)
            self.assertEqual(set(covered[model]), {"item0", "item1"})
            self.assertTrue(
                all(row["asr_wall_s"] == 0 for row in covered[model].values())
            )

    def test_full_controller_executes_only_selected_fake_cpu_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake-evaluator"
            fake.mkdir()
            # Real child process, but this evaluator only copies fixture metadata.
            (fake / "evaluate.py").write_text("""import argparse,json
from pathlib import Path
def prepare(manifest,output):
    rows=json.loads(Path(manifest).read_text())
    for row in rows: row['lang_id']=row['omnilingual_language']
    (Path(output)/'prepared.json').write_text(json.dumps(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    p.add_argument('--worker-model');p.add_argument('--device');p.add_argument('--dtype')
    a=p.parse_args();rows=json.loads((a.output/'prepared.json').read_text())
    with (a.output/(a.worker_model+'.jsonl')).open('w') as f:
        for row in rows:
            value=dict(id=row['id'],sha256=row['sha256'],model_card=a.worker_model,
                       transcript='CPU fixture',asr_wall_s=0.01)
            f.write(json.dumps(value)+'\\n')
    meta=dict(model_card=a.worker_model,blind=True,expected_text_prompt=False,
              physical_gpu_nvidia_smi='CPU-only fake worker; no CUDA')
    (a.output/(a.worker_model+'-metadata.json')).write_text(json.dumps(meta))
""")
            prepared = root / "prepared"
            prepared.mkdir()
            write(
                prepared / "asr-input-manifest.json",
                [
                    dict(
                        id="a",
                        sha256="wav-a",
                        omnilingual_language="eng_Latn",
                        language="en",
                    ),
                    dict(
                        id="b",
                        sha256="wav-b",
                        omnilingual_language="arb_Arab",
                        language="ar",
                    ),
                ],
            )
            marker = root / "assets.json"
            write(
                marker,
                {
                    "baseline_package_freeze_unchanged": True,
                    "assets": [
                        {"name": name, "sha256": "CPU-fixture"}
                        for name in [*MODELS, "omniASR_tokenizer_written_v2"]
                    ],
                },
            )
            output = root / "output"
            argv = [
                "run_selected_models.py",
                "--prepared",
                str(prepared),
                "--output",
                str(output),
                "--models",
                MODELS[1],
                "--protocol-namespace",
                "CPU-fixture-host",
                "--asset-marker",
                str(marker),
                "--budget-seconds",
                "15",
                "--reserve-seconds",
                "1",
                "--allow-gpu-inference",
            ]
            with (
                patch.object(controller, "HERE", fake),
                patch.object(
                    controller, "gpu_idle", return_value={"CPU_only_fixture": True}
                ),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                controller.main()
            result = json.loads((output / "outcome.json").read_text())
            self.assertEqual(result["status"], "selected_models_completed")
            self.assertEqual(result["covered_by_model"], {MODELS[0]: 0, MODELS[1]: 2})
            attempts = json.loads((output / "attempts.json").read_text())
            self.assertEqual([row["model"] for row in attempts], [MODELS[1]])
            score_command = json.loads((output / "score-command.json").read_text())
            self.assertNotIn("--models", score_command)  # required default stays both
            meta = json.loads(
                next((output / "published").rglob("*metadata.json")).read_text()
            )
            self.assertEqual(
                meta["physical_gpu_nvidia_smi"], "CPU-only fake worker; no CUDA"
            )
            self.assertTrue(
                meta["asr_protocol_config_id"].startswith("CPU-fixture-host-")
            )

    def test_default_both_and_duplicate_unknown_rejection(self):
        self.assertEqual(select_models(), MODELS)
        self.assertEqual(select_models([MODELS[1]]), [MODELS[1]])
        for models in ([], [MODELS[0], MODELS[0]], ["unknown"]):
            with self.assertRaises(ValueError):
                select_models(models)

    def test_one_completed_judge_cannot_be_both_completed(self):
        expected = {"a": {}, "b": {}}
        covered = {MODELS[0]: dict(expected), MODELS[1]: {}}
        result = coverage_summary(covered, expected, [MODELS[0]])
        self.assertTrue(result["requested_models_complete"])
        self.assertFalse(result["required_models_complete"])
        self.assertEqual(result["both_models_covered"], 0)
        self.assertEqual(result["status"], "selected_models_completed")
        self.assertEqual(result["missing_by_model"][MODELS[1]], ["a", "b"])
        covered[MODELS[1]] = dict(expected)
        self.assertEqual(
            coverage_summary(covered, expected, MODELS)["status"], "completed"
        )

    def test_resume_locks_selected_scope_namespace_and_immutable_legacy_output(self):
        plan = dict(
            controller_kind=CONTROLLER,
            prepared_manifest_sha256="audio",
            evaluator_sha256="worker",
            dtype="bfloat16",
            device="cuda:0",
            model_order=[MODELS[1]],
            protocol_namespace="modern",
            protocol_ids={MODELS[1]: "modern-abc"},
        )
        validate_resume(plan, dict(plan))
        for field, change in [
            ("model_order", MODELS),
            ("protocol_namespace", "other"),
            ("protocol_ids", {MODELS[1]: "legacy-batch1"}),
        ]:
            with self.assertRaises(AssertionError):
                validate_resume(plan, {**plan, field: change})
        with self.assertRaisesRegex(AssertionError, "Legacy output is immutable"):
            validate_resume({}, plan)

    def test_protocol_identity_changes_for_dtype_worker_assets_packages_and_namespace(
        self,
    ):
        model = MODELS[1]
        base = [
            model,
            "modern",
            "bfloat16",
            "worker",
            [{"name": model, "sha256": "weights"}],
        ]
        baseline = protocol_id(*base, packages={"torch": "2.8"})
        self.assertEqual(protocol_id(*base, packages={"torch": "2.8"}), baseline)
        changes = [
            (1, "other-host"),
            (2, "float16"),
            (3, "changed-worker"),
            (4, [{"name": model, "sha256": "different"}]),
        ]
        for index, value in changes:
            args = list(base)
            args[index] = value
            self.assertNotEqual(protocol_id(*args, packages={"torch": "2.8"}), baseline)
        self.assertNotEqual(protocol_id(*base, packages={"torch": "2.9"}), baseline)
        self.assertEqual(
            protocol_id(model, "legacy-batch1", "bfloat16", "worker", []),
            "legacy-batch1",
        )

    def test_publication_preserves_raw_and_cross_namespace_import_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = {"id": "a", "sha256": "wav", "lang_id": "eng_Latn"}
            raw = root / "raw"
            raw.mkdir()
            write(raw / "prepared.json", [source])
            model = MODELS[1]
            row = {
                "id": "a",
                "sha256": "wav",
                "model_card": model,
                "transcript": "Blind words",
                "asr_wall_s": 3.0,
            }
            (raw / f"{model}.jsonl").write_text(json.dumps(row) + "\n")
            write(
                raw / f"{model}-metadata.json",
                {
                    "model_card": model,
                    "blind": True,
                    "expected_text_prompt": False,
                    "physical_gpu_nvidia_smi": "fixture CPU only",
                },
            )
            before = {p.name: digest(p) for p in raw.iterdir()}
            prior = root / "prior"
            published = prior / "published/000"
            published.parent.mkdir(parents=True)
            valid, issues = publish(raw, published, model, protocol="modern-A")
            self.assertFalse(issues)
            self.assertEqual(valid[0]["asr_protocol_config_id"], "modern-A")
            self.assertEqual(before, {p.name: digest(p) for p in raw.iterdir()})
            protocols = {MODELS[0]: "legacy-batch1", model: "modern-A"}
            covered, _ = collect_existing(prior, {"a": source}, protocols)
            self.assertEqual(len(covered[model]), 1)
            with self.assertRaisesRegex(AssertionError, "configuration changed"):
                collect_existing(prior, {"a": source}, {**protocols, model: "modern-B"})
            write(
                prior / "plan.json", {"evaluator_sha256": "worker", "dtype": "bfloat16"}
            )
            target = root / "target"
            target.mkdir()
            skipped = import_prior(
                [prior],
                target,
                {"a": source},
                "worker",
                "bfloat16",
                [model],
                {**protocols, model: "modern-B"},
            )
            self.assertEqual(skipped[0]["reason"], "different_judge_protocol")
            self.assertFalse((target / "published").exists())
            imported = import_prior(
                [prior], target, {"a": source}, "worker", "bfloat16", [model], protocols
            )
            self.assertEqual(imported[0]["imported_rows"], 1)
            result, _ = collect_existing(target, {"a": source}, protocols)
            self.assertEqual(result[model]["a"]["asr_wall_s"], 0.0)
            self.assertEqual(result[model]["a"]["asr_reused_from_id"], "a")
            self.assertEqual(result[MODELS[0]], {})


if __name__ == "__main__":
    unittest.main()
