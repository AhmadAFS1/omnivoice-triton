#!/usr/bin/env python3
"""CPU-only synthetic regressions. No model/GPU/network operation is performed."""

import importlib.util
import io
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "quality_report", Path(__file__).with_name("report.py")
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportTest(unittest.TestCase):
    def test_ASR_batch_protocol_identity_cannot_mix_or_borrow_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            model = report.MODELS[0]
            unique = [
                {"id": "audio", "sha256": "a" * 64, "omnilingual_language": "eng_Latn"}
            ]
            row = {
                "id": "audio",
                "sha256": "a" * 64,
                "lang_id": "eng_Latn",
                "model_card": model,
                "transcript": "hello",
            }
            report.write_jsonl(first / f"{model}.jsonl", [row])
            old, _, _, metadata = report.load_asr([first], [model], unique, [])
            self.assertEqual(
                old[model][("a" * 64, "eng_Latn")]["asr_protocol_config_id"],
                "legacy-batch1",
            )
            self.assertEqual(
                metadata[f"{model}:coverage"]["asr_protocol_config_id"], "legacy-batch1"
            )
            report.write_jsonl(
                second / f"{model}.jsonl",
                [{**row, "asr_protocol_config_id": "prototype-batch2-pinned-hash"}],
            )
            with self.assertRaisesRegex(ValueError, "Mixed ASR protocol configs"):
                report.load_asr([first, second], [model], unique, [])
            report.write_json(
                first / f"{model}-metadata.json",
                {"asr_protocol_config_id": "prototype-batch2-pinned-hash"},
            )
            with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
                report.load_asr([first], [model], unique, [])
            report.write_jsonl(
                first / f"{model}.jsonl",
                [{**row, "asr_protocol_config_id": "prototype-batch2-pinned-hash"}],
            )
            merged, _, _, metadata = report.load_asr(
                [first, second], [model], unique, []
            )
            self.assertEqual(len(merged[model]), 1)
            self.assertEqual(
                metadata[f"{model}:coverage"]["asr_protocol_config_id"],
                "prototype-batch2-pinned-hash",
            )

    def test_all64_reference_controls_remain_separate(self):
        rows = []
        for language in ["ar", "da", "de", "el", "en", "es", "fi", "fr"]:
            for voice in range(8):
                reference = f"{language}-voice{voice}"
                rows.append(
                    {
                        "id": f"reference-controls::{reference}",
                        "kind": "reference_control",
                        "reference_id": reference,
                        "language": language,
                        "voice_id": f"voice{voice}",
                        "actual_sha256": f"synthetic-{reference}",
                        "omnilingual_language": "synthetic-script",
                        "judges": {
                            model: {
                                "exact_normalized": not (
                                    language == "en" and voice == 0
                                )
                            }
                            for model in report.MODELS
                        },
                    }
                )
        calibration = report.calibration_status(rows, report.MODELS)
        self.assertEqual(len(calibration["by_reference_id"]), 64)
        self.assertEqual(len(calibration["by_language"]), 8)
        self.assertEqual(calibration["by_language"]["en"]["controls"], 8)
        self.assertEqual(
            calibration["by_reference_id"]["en-voice0"]["status"],
            "control_ASR_mismatch_review",
        )
        self.assertEqual(
            calibration["by_reference_id"]["en-voice1"]["status"],
            "control_ASR_screen_matches",
        )
        self.assertEqual(
            calibration["by_language"]["en"]["status"],
            "control_ASR_partial_or_mismatch_review",
        )

    def test_expected_rejections_and_unexpected200_stay_outside_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            audio = run / "fake.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x20" * 240)
            sha = report.digest(audio)
            case = {
                "id": "en-short",
                "text": "Hello.",
                "language": "en",
                "api_language": "en",
                "omnilingual_language": "eng_Latn",
                "mode": "clone",
                "reference_id": "en-aoede",
                "reference_language": "en",
                "voice_id": "Aoede",
                "config_type": "lingua-video",
                "voice_profile": "local_target_language_reference",
            }
            identity = {
                "sha256": "synthetic-reference-hash",
                "text": "Reference.",
                "preprocess_prompt": True,
            }
            rows, quality = [], []
            for name, status, validation in [
                ("normal", 200, False),
                ("rejected", 400, True),
                ("guardfailed", 200, True),
            ]:
                current_case = {
                    **case,
                    "id": name,
                    "validation_only": validation,
                    "expected_http_status": 400 if validation else 200,
                }
                row = {
                    "id": name,
                    "case": current_case,
                    "phase": "checks",
                    "status": status,
                    "validation_only": validation,
                    "expected_http_status": current_case["expected_http_status"],
                    "generation_options": {
                        "guidance_scale": "nan" if validation else "0"
                    },
                    "reference_identity": identity,
                    "reference_transport": "registered",
                    "preprocessing_applied_at": "registration",
                }
                rows.append(row)
                if status == 200:
                    quality.append(
                        {
                            "id": name,
                            "case": current_case,
                            "path": "fake.wav",
                            "sha256": sha,
                            "language": "en",
                            "omnilingual_language": "eng_Latn",
                            "expected": "Hello.",
                            "phase": "checks",
                        }
                    )
            report.write_jsonl(run / "requests.jsonl", rows)
            report.write_json(run / "quality-manifest.json", quality)
            report.write_json(run / "quality-asr-ready.json", quality)
            report.write_json(
                run / "metadata.json",
                {
                    "hardware": {
                        "measurement_kind": "CPU-only synthetic validation/HTTP fixture; no GPU/model inference"
                    }
                },
            )
            with redirect_stdout(io.StringIO()):
                prep = report.prepare(
                    SimpleNamespace(
                        output=root / "p",
                        inventory=report.LANGUAGES / "active-languages.json",
                        corpus=report.LANGUAGES / "corpus.json",
                        run_dir=[run],
                        runs_root=None,
                        references=None,
                    )
                )
                scored = report.score(
                    SimpleNamespace(
                        output=root / "s",
                        prepared=root / "p",
                        asr_dir=[],
                        models=report.MODELS,
                        pair=[],
                    )
                )
            self.assertEqual(prep["request_rows"], 1)
            self.assertEqual(prep["validation_rows"], 2)
            self.assertEqual(prep["unique_ASR_inputs"], 1)
            logical = report.read_rows(root / "p/logical-cases.jsonl")
            failed = next(r for r in logical if r["original_id"] == "guardfailed")
            self.assertTrue(failed["hash_verified"])
            self.assertIsNone(failed["asr_input_id"])
            self.assertIn(
                "validation_guard_failure_unexpected_status", failed["issues"]
            )
            normal = next(r for r in logical if r["original_id"] == "normal")
            self.assertEqual(normal["generation_options"], {"guidance_scale": "0"})
            self.assertEqual(normal["reference_identity"], identity)
            self.assertEqual(normal["voice_id"], "Aoede")
            self.assertEqual(
                scored["logical_validation_statuses"],
                {
                    "validation_expected_status_matches": 1,
                    "validation_guard_failure": 1,
                },
            )
            route = next(
                r
                for r in report.read_rows(root / "s/per-route.json")
                if r["language"] == "en"
            )
            self.assertEqual(route["requests"], 1)
            self.assertEqual(route["http_status_counts"], {"200": 1})
            self.assertEqual(route["validation_guard_failures"], 1)
            self.assertEqual(route["validation_unexpected200_retained"], 1)
            config = report.read_rows(root / "s/per-configuration.json")
            self.assertEqual(len(config), 1)
            self.assertEqual(config[0]["requests"], 1)
            self.assertEqual(config[0]["voice_id"], "Aoede")
            changed = {
                **rows[0],
                "reference_identity": {**identity, "sha256": "different"},
            }
            self.assertNotEqual(
                report.semantic_identity(case, rows[0]),
                report.semantic_identity(case, changed),
            )
            self.assertFalse(
                report.validation_status_matches(
                    {"expected_http_status": None, "http_status": None}
                )
            )

    def test_model_probes_never_become_HTTP_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = next(
                c
                for c in report.read_rows(report.LANGUAGES / "corpus.json")
                if c["id"] == "en-short"
            )
            probe = root / "results" / "split-cfg-probe-v1"
            probe.mkdir(parents=True)
            report.write_json(
                probe / "metadata.json",
                {
                    "prototype_sha256": "synthetic-script-hash",
                    "hardware": {
                        "measurement_kind": "CPU-only synthetic model-probe fixture; no GPU/model inference"
                    },
                },
            )
            audio = probe / "audio.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x20" * 240)
            for arm in ["baseline", "split-cfg"]:
                directory = probe / arm
                directory.mkdir()
                report.write_json(
                    directory / "quality-manifest.json",
                    [
                        {
                            "id": f"en-short-b1-{i:03d}",
                            "path": str(audio),
                            "sha256": report.digest(audio),
                            "expected": source["text"],
                            "language": "en",
                            "omnilingual_language": "eng_Latn",
                            "phase": "fixed-seed-split-cfg",
                            "signal_valid": i == 0,
                            "reference_language": "en",
                        }
                        for i in range(2)
                    ],
                )
            with redirect_stdout(io.StringIO()):
                prepared = report.prepare(
                    SimpleNamespace(
                        output=root / "p",
                        inventory=report.LANGUAGES / "active-languages.json",
                        corpus=report.LANGUAGES / "corpus.json",
                        run_dir=[],
                        runs_root=root / "results",
                        references=None,
                    )
                )
                summary = report.score(
                    SimpleNamespace(
                        output=root / "s",
                        prepared=root / "p",
                        asr_dir=[],
                        models=report.MODELS,
                        pair=[
                            "split-cfg-probe-v1/baseline=split-cfg-probe-v1/split-cfg"
                        ],
                    )
                )
            self.assertEqual(prepared["request_rows"], 0)
            self.assertEqual(prepared["model_probe_rows"], 4)
            self.assertEqual(prepared["unique_ASR_inputs"], 1)
            self.assertEqual(summary["logical_request_screen_statuses"], {})
            rows = report.read_rows(root / "p/logical-cases.jsonl")
            self.assertTrue(
                all(
                    r["kind"] == "model_probe" and r["http_status"] is None
                    for r in rows
                )
            )
            self.assertTrue(all(r["source"] == source["source"] for r in rows))
            route = next(
                r
                for r in report.read_rows(root / "s/per-route.json")
                if r["language"] == "en"
            )
            self.assertEqual(route["requests"], 0)
            self.assertEqual(route["http_status_counts"], {})
            self.assertEqual(route["model_probe_outputs"], 4)
            self.assertEqual(route["model_probe_ASR_ready"], 2)

    def test_ambiguous_language_wrong_hash_and_conflicts_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = "a" * 64
            unique = [
                {"id": "en", "sha256": sha, "omnilingual_language": "eng_Latn"},
                {"id": "es", "sha256": sha, "omnilingual_language": "spa_Latn"},
            ]
            model = report.MODELS[0]
            report.write_jsonl(
                root / f"{model}.jsonl",
                [
                    {
                        "id": "unknown",
                        "sha256": sha,
                        "transcript": "Cannot infer language from this hash",
                    },
                    {
                        "id": "en",
                        "sha256": "b" * 64,
                        "transcript": "Incorrect claimed hash",
                    },
                    {"id": "es", "sha256": sha, "transcript": "uno"},
                    {"id": "es", "sha256": sha, "transcript": "dos"},
                ],
            )
            transcripts, issues, _, _ = report.load_asr([root], [model], unique, [])
            self.assertEqual(transcripts[model], {})
            self.assertEqual(len(issues), 3)

    def test_unicode_and_spacing(self):
        self.assertEqual(report.normalize("إِنَّهُ الـيَوْمَ!"), "انه اليوم")
        result = report.text_metrics("today", "to day")
        self.assertGreater(result["wer"], 0)
        self.assertEqual(result["cer"], 0)
        self.assertTrue(result["spacing_only_difference"])
        self.assertEqual(report.text_metrics("你好世界", "你好世")["cer"], 0.25)
        self.assertTrue(
            report.text_metrics("50 dollars", "fifty dollars")[
                "numeric_representation_differs"
            ]
        )
        self.assertEqual(report.edit_counts(["a", "b"], ["a", "c"])["substitutions"], 1)

    def test_dedup_propagation_failure_coverage_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "fixture.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x20" * 240)
            sha = report.digest(audio)
            inventory = report.read_rows(report.LANGUAGES / "active-languages.json")
            model_outputs = root / "asr"
            model_outputs.mkdir()
            for arm in ["before", "after"]:
                run = root / "runs" / arm
                run.mkdir(parents=True)
                requests, quality = [], []
                for i in range(1000):
                    case = {
                        "id": "en-short",
                        "language": "en",
                        "api_language": "en",
                        "text": "Hello there.",
                        "omnilingual_language": "eng_Latn",
                        "length": "short",
                        "mode": "clone",
                        "reference_id": "en-aoede",
                        "reference_language": "en",
                        "voice_profile": "local_target_language_reference",
                    }
                    if i == 999:
                        case = {**case, "id": "en-wrong-expected", "text": "Goodbye."}
                    identity = f"qa-{i:05d}"
                    request = {
                        "id": identity,
                        "phase": "qa-short-all-languages",
                        "status": 200,
                        "case": case,
                        "parameters": {
                            "text": case["text"],
                            "language": "en",
                            "prompt_id": arm + "-uuid",
                        },
                        "headers": {
                            "x-omnivoice-response-source": "cache"
                            if i
                            else "generated",
                            "x-omnivoice-origin-request-id": "origin-en",
                        },
                    }
                    q = {
                        "id": identity,
                        "path": str(audio),
                        "sha256": sha,
                        "case": case,
                        **case,
                        "expected": case["text"],
                        "phase": request["phase"],
                        "audio": {"asr_readable": True, "signal_valid": True},
                    }
                    q["id"] = identity
                    requests.append(request)
                    quality.append(q)
                if arm == "before":
                    for language in ["es", "tw"]:
                        route = next(r for r in inventory if r["language"] == language)
                        case = {
                            "id": language + "-short",
                            "text": "Hello there.",
                            "language": language,
                            "api_language": route["api_language"],
                            "omnilingual_language": route["omnilingual_language"],
                            "reference_id": "en-aoede",
                            "reference_language": "en",
                            "length": "short",
                            "voice_profile": "cross_language_english_reference",
                            "asr_mapping_is_proxy": language == "tw",
                        }
                        identity = language + "-request"
                        requests.append(
                            {
                                "id": identity,
                                "status": 200,
                                "case": case,
                                "phase": "extra",
                            }
                        )
                        quality.append(
                            {
                                **case,
                                "id": identity,
                                "path": str(audio),
                                "sha256": sha,
                                "case": case,
                                "expected": case["text"],
                                "phase": "extra",
                            }
                        )
                    requests.append(
                        {
                            "id": "failed",
                            "status": 503,
                            "phase": "extra",
                            "case": {"id": "en-failed", "language": "en"},
                        }
                    )
                    requests.append(
                        {
                            "id": "corrupt",
                            "status": 200,
                            "phase": "extra",
                            "case": {"id": "en-corrupt", "language": "en"},
                        }
                    )
                    quality.append(
                        {
                            "id": "corrupt",
                            "language": "en",
                            "omnilingual_language": "eng_Latn",
                            "path": str(audio),
                            "sha256": "0" * 64,
                            "expected": "Hello there.",
                            "phase": "extra",
                        }
                    )
                report.write_jsonl(run / "requests.jsonl", requests)
                report.write_json(run / "quality-manifest.json", quality)
                report.write_json(run / "quality-asr-ready.json", quality)
                report.write_json(
                    run / "metadata.json",
                    {
                        "hardware": {
                            "measurement_kind": "CPU-only synthetic fake HTTP; no GPU/model inference"
                        }
                    },
                )
            controls = root / "references.json"
            report.write_json(
                controls,
                [
                    {
                        "id": "en-aoede",
                        "language": "en",
                        "omnilingual_language": "eng_Latn",
                        "path": str(audio),
                        "text": "Hello there.",
                        "sha256": sha,
                    }
                ],
            )
            prep_args = SimpleNamespace(
                output=root / "prepared",
                inventory=report.LANGUAGES / "active-languages.json",
                corpus=report.LANGUAGES / "corpus.json",
                run_dir=[],
                runs_root=root / "runs",
                references=controls,
            )
            with redirect_stdout(io.StringIO()):
                prepared = report.prepare(prep_args)
            self.assertEqual(prepared["inventory_routes"], 104)
            self.assertEqual(prepared["corpus_cases"], 299)
            self.assertEqual(prepared["logical_rows"], 2005)
            self.assertEqual(prepared["unique_ASR_inputs"], 3)
            self.assertEqual(prepared["hash_reads"], 1)
            inputs = report.read_rows(root / "prepared/asr-input-manifest.json")
            self.assertEqual(len({r["sha256"] for r in inputs}), 1)
            self.assertEqual(len({r["omnilingual_language"] for r in inputs}), 3)
            for model in report.MODELS:
                report.write_jsonl(
                    model_outputs / f"{model}.jsonl",
                    [
                        {
                            "id": r["id"],
                            "sha256": r["sha256"],
                            "model_card": model,
                            "transcript": "Hello there.",
                            "asr_wall_s": 0.0,
                        }
                        for r in inputs
                    ],
                )
                report.write_json(
                    model_outputs / f"{model}-metadata.json",
                    {
                        "measurement_kind": "CPU-only synthetic transcript fixture; no GPU/model inference"
                    },
                )
            score_args = SimpleNamespace(
                output=root / "scored",
                prepared=root / "prepared",
                asr_dir=[model_outputs],
                models=report.MODELS,
                pair=["before=after"],
            )
            with redirect_stdout(io.StringIO()):
                scored = report.score(score_args)
            self.assertFalse(scored["quality_certified"])
            self.assertEqual(scored["paired"]["before=after"]["hash_identical"], 1000)
            rows = report.read_rows(root / "scored/scored-logical-cases.jsonl")
            wrong = next(r for r in rows if r["id"] == "before::qa-00999")
            self.assertIn("word_mismatch_review", wrong["flags"])
            self.assertGreater(wrong["judges"][report.MODELS[0]]["wer"], 0)
            correct = next(r for r in rows if r["id"] == "before::qa-00000")
            self.assertEqual(correct["screen_status"], "ASR_screen_matches")
            proxy = next(r for r in rows if r["language"] == "tw")
            self.assertIn("proxy_language_mapping_unqualified", proxy["flags"])
            self.assertIn(
                "target_language_ASR_uncalibrated_or_control_review", proxy["flags"]
            )
            corrupt = next(r for r in rows if r["id"] == "before::corrupt")
            self.assertEqual(corrupt["screen_status"], "not_scored")
            routes = report.read_rows(root / "scored/per-route.json")
            self.assertEqual(len(routes), 104)
            self.assertEqual(
                next(r for r in routes if r["language"] == "en")["http_status_counts"][
                    "503"
                ],
                1,
            )
            self.assertEqual(
                next(r for r in routes if r["language"] == "ja")["status"], "not_run"
            )
            origins = report.read_rows(root / "scored/origin-propagation.json")
            self.assertIn("same_origin_different_semantic_inputs", origins[0]["issues"])
            self.assertIn("CPU-only", (root / "scored/report.md").read_text())
            (model_outputs / f"{report.MODELS[1]}.jsonl").unlink()
            score_args.output = root / "partial"
            with redirect_stdout(io.StringIO()):
                partial = report.score(score_args)
            self.assertEqual(partial["routes_with_both_ASR_outputs"], 0)
            self.assertEqual(
                partial["logical_request_screen_statuses"].get("ASR_screen_matches", 0),
                0,
            )
            with self.assertRaisesRegex(ValueError, "fake/CPU-only"):
                report.export_samples(
                    SimpleNamespace(
                        report=root / "scored",
                        inventory=prep_args.inventory,
                        pair="before=after",
                        phase="qa-short-all-languages",
                        expected_gpu="RTX 4070",
                    )
                )

    def test_no_outputs_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(io.StringIO()):
                prep = report.prepare(
                    SimpleNamespace(
                        output=root / "p",
                        inventory=report.LANGUAGES / "active-languages.json",
                        corpus=report.LANGUAGES / "corpus.json",
                        run_dir=[],
                        runs_root=None,
                        references=None,
                    )
                )
                summary = report.score(
                    SimpleNamespace(
                        output=root / "s",
                        prepared=root / "p",
                        asr_dir=[],
                        models=report.MODELS,
                        pair=[],
                    )
                )
            self.assertEqual(prep["unique_ASR_inputs"], 0)
            self.assertEqual(summary["status"], "awaiting_ASR_outputs")
            self.assertEqual(summary["routes_with_both_ASR_outputs"], 0)
            self.assertTrue(
                all(
                    r["status"] == "not_run"
                    for r in report.read_rows(root / "s/per-route.json")
                )
            )


if __name__ == "__main__":
    unittest.main()
