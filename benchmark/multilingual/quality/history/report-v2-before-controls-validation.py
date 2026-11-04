#!/usr/bin/env python3
"""CPU-only preparation and postprocessing. This program never loads an ASR/TTS model.

prepare verifies retained WAV hashes, keeps every HTTP outcome, and exports one ASR
input per (audio hash, exact language/script). score reads existing blind-ASR JSONL
outputs and propagates evidence to every logical request without treating cached
copies as independent quality observations. All outputs go to a fresh directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGUAGES = HERE.parent / "languages"
MODELS = ["omniASR_LLM_Unlimited_1B_v2", "omniASR_CTC_1B_v2"]
CAVEATS = [
    "ASR agreement/mismatch is a screen, not perceptual quality or native-language certification.",
    "The two Omnilingual variants share a model family and can share recognition errors.",
    "WER is not comparable across scripts/tokenization; CER ignores whitespace only after documented normalization.",
    "Published texts and existing reference transcripts have not been independently reviewed by native speakers.",
    "An English clone reference is explicitly cross-language; it is not a native target-language voice reference.",
    "Twi uses the broader aka_Latn Akan ASR proxy; exact Twi recognition quality is unqualified.",
    "Repeated cached/coalesced WAVs are not independent quality evidence or fresh-inference throughput.",
    "No calibrated ASR confidence, MOS, speaker similarity, or human listening judgment is inferred.",
    "Hash differences can result from unseeded generation; a difference alone is not a quality regression.",
]


def now():
    return datetime.now(UTC).isoformat()


def read_rows(path):
    raw = Path(path).read_text()
    return (
        json.loads(raw)
        if raw.lstrip().startswith("[")
        else [json.loads(x) for x in raw.splitlines() if x.strip()]
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        stream.writelines(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def cpu_metadata():
    def sysctl(key):
        try:
            return subprocess.check_output(
                ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unverified"

    return {
        "measurement_kind": "CPU-only file hashing, text scoring and report generation; no GPU/model inference",
        "date_utc": now(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": sysctl("machdep.cpu.brand_string")
        if sys.platform == "darwin"
        else platform.processor(),
        "physical_memory_bytes": sysctl("hw.memsize")
        if sys.platform == "darwin"
        else "unverified",
        "python": sys.version,
        "script_sha256": digest(__file__),
    }


def hardware_label(hardware):
    """Do not invent GPU identity/VRAM from an arm's name or allocator limits."""
    if not hardware:
        return "TTS GPU model, physical VRAM, visible VRAM and co-load are unverified."
    if "CPU-only" in hardware.get("measurement_kind", ""):
        return hardware["measurement_kind"]
    physical = hardware.get(
        "nvidia_smi_gpu", hardware.get("physical_gpu_nvidia_smi", "unverified")
    )
    visible = hardware.get(
        "torch_cuda_devices", hardware.get("torch_visible_gpu", "unverified")
    )
    coload = hardware.get(
        "nvidia_smi_processes",
        hardware.get(
            "co_resident_gpu_processes",
            hardware.get("co_resident_processes_at_start", "unverified"),
        ),
    )
    runtime = hardware.get(
        "torch_cuda_runtime", hardware.get("cuda_runtime", "unverified")
    )
    date = hardware.get("collected_at", hardware.get("date_utc", "unverified"))
    return f"Physical GPU/VRAM evidence: {physical}; visible GPU/VRAM: {json.dumps(visible)}; CUDA: {runtime}; co-load: {coload}; measured UTC: {date}"


def provenance(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def semantic_identity(case, request):
    parameters = dict(request.get("parameters", {}))
    # Prompt UUIDs are server-local. Use the actual reference identity instead.
    parameters.pop("prompt_id", None)
    parameters.pop("ref_audio", None)
    return object_digest(
        {
            "parameters": parameters,
            "text": case.get("text", case.get("expected")),
            "language": case.get("language"),
            "api_language": case.get("api_language"),
            "reference_id": case.get("reference_id"),
            "reference_language": case.get("reference_language"),
            "voice_profile": case.get("voice_profile"),
            "mode": case.get("mode"),
        }
    )


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inventory, corpus = read_rows(args.inventory), read_rows(args.corpus)
    if len({r["language"] for r in inventory}) != len(inventory):
        raise ValueError("Product route inventory is not unique")
    known = {r["language"]: r for r in inventory}
    run_dirs = [p.resolve() for p in args.run_dir]
    if args.runs_root:
        run_dirs += [
            p.parent.resolve() for p in args.runs_root.rglob("quality-manifest.json")
        ]
    run_dirs = sorted(set(run_dirs))
    probe_paths = {p.resolve() for p in getattr(args, "probe_manifest", [])}
    http_run_dirs = []
    for directory in run_dirs:
        parent_metadata = directory.parent / "metadata.json"
        recognized_probe = (
            not (directory / "quality-asr-ready.json").exists()
            and parent_metadata.exists()
            and json.loads(parent_metadata.read_text()).get("prototype_sha256")
        )
        if directory / "quality-manifest.json" in probe_paths or recognized_probe:
            probe_paths.add(directory / "quality-manifest.json")
        else:
            http_run_dirs.append(directory)
    run_dirs = http_run_dirs
    arms, logical, sources, errors, hashed_paths = {}, [], [], [], {}
    logical_ids = set()
    unique = {}

    def add_row(row):
        if row["id"] in logical_ids:
            raise ValueError(f"Duplicate namespaced request ID {row['id']}")
        logical_ids.add(row["id"])
        row["asr_input_id"] = None
        path = Path(row["path"]) if row.get("path") else None
        if path:
            try:
                resolved = path.resolve(strict=True)
                stat = resolved.stat()
                cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
                actual = hashed_paths.get(cache_key)
                if actual is None:
                    actual = hashed_paths[cache_key] = digest(resolved)
                row["actual_sha256"] = actual
                row["hash_verified"] = actual == row.get("sha256")
                if not row["hash_verified"]:
                    row["issues"].append("audio_hash_mismatch")
            except OSError as exc:
                row["issues"].append(f"audio_file_unavailable:{type(exc).__name__}")
        eligible = (
            row.get("declared_asr_ready")
            and row.get("hash_verified")
            and row.get("omnilingual_language")
            and "route_absent_from_inventory" not in row["issues"]
            and "ASR_language_differs_from_inventory" not in row["issues"]
        )
        if eligible:
            key = (row["actual_sha256"], row["omnilingual_language"])
            uid = "audio-" + object_digest(list(key))
            row["asr_input_id"] = uid
            if key not in unique:
                unique[key] = {
                    "id": uid,
                    "path": row["path"],
                    "sha256": key[0],
                    "text": row["expected"],
                    "expected": row["expected"],
                    "language": row["language"],
                    "lang_id": key[1],
                    "omnilingual_language": key[1],
                    "representative_logical_id": row["id"],
                    "run": row["arm"],
                    "phase": row["phase"],
                    "deduplication": "verified original WAV SHA256 plus exact ASR language/script",
                    "logical_ids": [],
                }
            unique[key]["logical_ids"].append(row["id"])
        logical.append(row)

    for run_dir in run_dirs:
        arm = (
            str(run_dir.relative_to(args.runs_root.resolve()))
            if args.runs_root
            else run_dir.name
        )
        if arm in arms:
            raise ValueError(
                f"Duplicate arm name {arm}; use --runs-root for unique relative paths"
            )
        paths = {
            name: run_dir / name
            for name in [
                "quality-manifest.json",
                "quality-asr-ready.json",
                "requests.jsonl",
                "metadata.json",
            ]
        }
        for path in paths.values():
            if path.exists():
                sources.append(provenance(path))
        all_quality = read_rows(paths["quality-manifest.json"])
        ready = {r["id"]: r for r in read_rows(paths["quality-asr-ready.json"])}
        requests = (
            read_rows(paths["requests.jsonl"])
            if paths["requests.jsonl"].exists()
            else []
        )
        if len({r["id"] for r in all_quality}) != len(all_quality):
            raise ValueError(f"Duplicate quality IDs in {run_dir}")
        if len({r["id"] for r in requests}) != len(requests):
            raise ValueError(f"Duplicate request IDs in {run_dir}")
        quality_by_id = {r["id"]: r for r in all_quality}
        if not set(ready).issubset(quality_by_id):
            raise ValueError(
                f"ASR-ready entries absent from full quality manifest: {run_dir}"
            )
        for identity, row in ready.items():
            if any(
                row.get(k) != quality_by_id[identity].get(k)
                for k in [
                    "path",
                    "sha256",
                    "language",
                    "omnilingual_language",
                    "expected",
                ]
            ):
                raise ValueError(
                    f"ASR-ready/full manifest disagreement for {run_dir}/{identity}"
                )
        metadata = (
            json.loads(paths["metadata.json"].read_text())
            if paths["metadata.json"].exists()
            else {}
        )
        arms[arm] = {
            "run_dir": str(run_dir),
            "hardware": metadata.get("hardware", {}),
            "hardware_label": hardware_label(metadata.get("hardware", {})),
            "source_ref": metadata.get("source_ref"),
            "source": metadata.get("source"),
            "profile": metadata.get("profile"),
            "startup_seed": metadata.get("requested_args", {}).get("startup_seed"),
            "metadata": metadata,
            "quality_rows": len(all_quality),
        }
        request_ids = {r["id"] for r in requests}
        entries = requests + [
            {
                "id": r["id"],
                "status": 200,
                "case": r.get("case", r),
                "quality_only": True,
            }
            for r in all_quality
            if r["id"] not in request_ids
        ]
        if not requests:
            errors.append(
                {
                    "arm": arm,
                    "issue": "requests.jsonl missing/empty; HTTP failure denominator unknown",
                }
            )
        for request in entries:
            q = quality_by_id.get(request["id"], {})
            case = q.get("case") or request.get("case", {})
            language = q.get("language", case.get("language"))
            headers = {str(k).lower(): v for k, v in request.get("headers", {}).items()}
            issues = []
            if language not in known:
                issues.append("route_absent_from_inventory")
            if request.get("status") == 200 and not q:
                issues.append("HTTP200_missing_quality_manifest_row")
            if q and request.get("status") != 200:
                issues.append("quality_row_for_non200_request")
            if request.get("quality_only"):
                issues.append("request_metadata_missing")
            if q and q.get("audio", {}).get("signal_valid") is False:
                issues.append("recorded_audio_signal_invalid")
            lang_id = q.get("omnilingual_language", case.get("omnilingual_language"))
            if (
                language in known
                and lang_id
                and lang_id != known[language].get("omnilingual_language")
            ):
                issues.append("ASR_language_differs_from_inventory")
            row = {
                "id": f"{arm}::{request['id']}",
                "original_id": request["id"],
                "kind": "request",
                "arm": arm,
                "phase": q.get("phase", request.get("phase")),
                "case_id": case.get("id"),
                "language": language,
                "omnilingual_language": lang_id,
                "expected": q.get(
                    "expected",
                    q.get("text", case.get("expected", case.get("text", ""))),
                ),
                "http_status": request.get("status"),
                "error": request.get("error"),
                "path": str((run_dir / q["path"]).resolve()) if q.get("path") else None,
                "sha256": q.get("sha256"),
                "actual_sha256": None,
                "hash_verified": False,
                "declared_asr_ready": request["id"] in ready
                and request.get("status") == 200,
                "voice_profile": q.get("voice_profile", case.get("voice_profile")),
                "reference_id": case.get("reference_id"),
                "reference_language": q.get(
                    "reference_language", case.get("reference_language")
                ),
                "asr_mapping_is_proxy": bool(
                    q.get("asr_mapping_is_proxy", case.get("asr_mapping_is_proxy"))
                ),
                "asr_mapping_note": q.get(
                    "asr_mapping_note", case.get("asr_mapping_note")
                ),
                "translation_status": q.get(
                    "translation_status", case.get("translation_status")
                ),
                "length": case.get("length"),
                "source": q.get("source", case.get("source")),
                "audio": q.get("audio", {}),
                "semantic_input_sha256": semantic_identity(case, request),
                "effective_input_sha256": request.get("effective_input_sha256"),
                "response_source": headers.get(
                    "x-omnivoice-response-source", "unreported"
                ),
                "inference_executed": headers.get(
                    "x-omnivoice-inference-executed", "unreported"
                ),
                "origin_request_id": headers.get("x-omnivoice-origin-request-id"),
                "http_request_id": headers.get("x-omnivoice-request-id"),
                "complete_ms": request.get("complete_ms"),
                "case": case,
                "issues": issues,
            }
            add_row(row)

    corpus_by_id = {case["id"]: case for case in corpus}
    for manifest in sorted(probe_paths):
        directory = manifest.parent
        try:
            arm = (
                str(directory.relative_to(args.runs_root.resolve()))
                if args.runs_root
                else f"probe/{directory.parent.name}/{directory.name}"
            )
        except ValueError:
            arm = f"probe/{directory.parent.name}/{directory.name}"
        if arm in arms:
            raise ValueError(f"Duplicate probe/HTTP arm name: {arm}")
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            metadata_path = directory.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        if metadata_path.exists():
            sources.append(provenance(metadata_path))
        sources.append(provenance(manifest))
        rows = read_rows(manifest)
        arms[arm] = {
            "run_dir": str(directory),
            "kind": "model_probe",
            "hardware": metadata.get("hardware", {}),
            "hardware_label": hardware_label(metadata.get("hardware", {})),
            "source_ref": metadata.get("source_ref"),
            "profile": metadata.get("profile"),
            "metadata": metadata,
            "quality_rows": len(rows),
            "availability_denominator": "not an HTTP run; only retained model outputs are present; failure denominator unknown",
        }
        for q in rows:
            issues = ["non_HTTP_model_probe_failure_denominator_unknown"]
            case_id = q.get("case_id")
            if case_id is None:
                match = re.fullmatch(r"(.+)-b\d+-\d+", q["id"])
                case_id = match[1] if match else None
            published = corpus_by_id.get(case_id, {})
            expected = q.get("expected", q.get("text", ""))
            language = q.get("language")
            if published and (
                published["text"] != expected or published["language"] != language
            ):
                issues.append("probe_corpus_identity_mismatch")
                published = {}
            if not published:
                issues.append("probe_text_source_not_joined")
            lang_id = q.get("omnilingual_language")
            if language not in known:
                issues.append("route_absent_from_inventory")
            elif lang_id != known[language].get("omnilingual_language"):
                issues.append("ASR_language_differs_from_inventory")
            reference_language = q.get("reference_language")
            voice_profile = (
                "probe_target_language_reference"
                if reference_language == language
                else "probe_cross_language_english_reference"
                if reference_language == "en"
                else "probe_reference_unreported"
            )
            # Corpus provenance does not establish a probe's actual reference.
            case = {
                **published,
                "id": case_id,
                "text": expected,
                "language": language,
                "reference_language": reference_language,
                "reference_id": q.get("reference_id"),
                "voice_profile": voice_profile,
            }
            signal_valid = q.get("signal_valid") is True
            if not signal_valid:
                issues.append("probe_signal_invalid_or_unverified")
            add_row(
                {
                    "id": f"{arm}::{q['id']}",
                    "original_id": q["id"],
                    "kind": "model_probe",
                    "arm": arm,
                    "phase": q.get("phase", "model-probe"),
                    "case_id": case_id,
                    "language": language,
                    "omnilingual_language": lang_id,
                    "expected": expected,
                    "http_status": None,
                    "path": str((directory / q["path"]).resolve()),
                    "sha256": q["sha256"],
                    "actual_sha256": None,
                    "hash_verified": False,
                    "declared_asr_ready": signal_valid and bool(lang_id),
                    "asr_readiness_evidence": "Existing probe signal_valid flag plus hash; no HTTP availability inferred",
                    "reference_id": q.get("reference_id"),
                    "reference_language": reference_language,
                    "voice_profile": voice_profile,
                    "asr_mapping_is_proxy": bool(published.get("asr_mapping_is_proxy")),
                    "asr_mapping_note": published.get("asr_mapping_note"),
                    "translation_status": published.get("translation_status"),
                    "length": published.get("length"),
                    "source": published.get("source"),
                    "audio": {**q.get("signal", {}), "signal_valid": signal_valid},
                    "case": case,
                    "semantic_input_sha256": semantic_identity(
                        case,
                        {
                            "parameters": q.get(
                                "profile", {"steps": metadata.get("steps")}
                            )
                        },
                    ),
                    "response_source": "model_probe_not_HTTP",
                    "inference_executed": "model_probe_retained_output",
                    "origin_request_id": None,
                    "http_request_id": None,
                    "issues": issues,
                }
            )

    if args.references:
        sources.append(provenance(args.references))
        for ref in read_rows(args.references):
            name = ref.get("reference_id", ref["id"])
            add_row(
                {
                    "id": f"reference-controls::{name}",
                    "original_id": name,
                    "kind": "reference_control",
                    "arm": "reference-controls",
                    "phase": "reference-controls",
                    "case_id": name,
                    "language": ref.get("language", ref.get("lang")),
                    "omnilingual_language": ref.get("omnilingual_language"),
                    "expected": ref.get("expected", ref.get("text", "")),
                    "http_status": None,
                    "path": str(
                        (
                            args.references.parent / ref.get("file", ref.get("path"))
                        ).resolve()
                    ),
                    "sha256": ref["sha256"],
                    "actual_sha256": None,
                    "hash_verified": False,
                    "declared_asr_ready": True,
                    "voice_profile": "existing_synthetic_native_reference_control",
                    "reference_id": name,
                    "reference_language": ref.get("language", ref.get("lang")),
                    "asr_mapping_is_proxy": False,
                    "asr_mapping_note": None,
                    "translation_status": ref.get("transcript_status"),
                    "length": "reference-control",
                    "source": ref,
                    "audio": {},
                    "issues": [],
                    "generation_hardware": "Historical Gemini service hardware and VRAM unverified; CPU-only asset inspection here",
                }
            )

    origin_groups = defaultdict(list)
    for row in logical:
        if row.get("origin_request_id"):
            origin_groups[(row["arm"], row["origin_request_id"])].append(row)
    origins = []
    for (arm, origin), rows in origin_groups.items():
        hashes = sorted({r["actual_sha256"] for r in rows if r.get("hash_verified")})
        semantics = sorted({r["semantic_input_sha256"] for r in rows})
        origins.append(
            {
                "arm": arm,
                "origin_request_id": origin,
                "logical_ids": [r["id"] for r in rows],
                "sha256s": hashes,
                "semantic_input_sha256s": semantics,
                "issues": (
                    ["same_origin_different_WAV_hashes"] if len(hashes) > 1 else []
                )
                + (
                    ["same_origin_different_semantic_inputs"]
                    if len(semantics) > 1
                    else []
                ),
            }
        )
    coverage = route_coverage(inventory, logical)
    for route in coverage:
        route["hardware_by_arm"] = {
            arm: arms[arm]["hardware_label"] for arm in route["arms"]
        }
    summary = {
        "created_at": now(),
        "status": "awaiting_ASR_outputs"
        if unique
        else "awaiting_TTS_outputs_or_valid_audio",
        "postprocessing_hardware": cpu_metadata(),
        "inventory_routes": len(inventory),
        "corpus_cases": len(corpus),
        "inventory": provenance(args.inventory),
        "corpus": provenance(args.corpus),
        "arms": arms,
        "input_sources": sources,
        "logical_rows": len(logical),
        "request_rows": sum(r["kind"] == "request" for r in logical),
        "model_probe_rows": sum(r["kind"] == "model_probe" for r in logical),
        "reference_controls": sum(r["kind"] == "reference_control" for r in logical),
        "unique_ASR_inputs": len(unique),
        "hash_reads": len(hashed_paths),
        "hash_verified_rows": sum(bool(r.get("hash_verified")) for r in logical),
        "integrity_issues": errors,
        "caveats": CAVEATS,
    }
    write_json(output / "inventory.json", inventory)
    write_json(output / "corpus.json", corpus)
    write_json(output / "asr-input-manifest.json", list(unique.values()))
    write_jsonl(output / "logical-cases.jsonl", logical)
    write_json(output / "origin-groups.json", origins)
    write_json(output / "coverage.json", coverage)
    write_json(output / "preparation.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "logical_rows": len(logical),
                "unique_ASR_inputs": len(unique),
                "routes": len(inventory),
                "corpus_cases": len(corpus),
            }
        )
    )
    return summary


def normalize(text):
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(
        " " if unicodedata.category(c).startswith(("P", "S")) else c for c in value
    )
    value = "".join(
        c
        for c in value
        if c != "\u0640" and not "\u064b" <= c <= "\u065f" and c != "\u0670"
    )
    value = value.translate(
        str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
    )
    return " ".join(value.split())


def edit_counts(reference, hypothesis):
    """Unit-cost Levenshtein; tied paths prefer substitution, deletion, insertion.

    Return S,D,I,H using O(len(hypothesis)) memory. This has the same minimal
    distance as standard WER/CER. Tied S/D/I decomposition may differ from jiwer.
    """
    previous = [(j, 0, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            diag = previous[j - 1]
            if ref == hyp:
                value = (diag[0], diag[1], diag[2], diag[3], diag[4] + 1)
            else:
                delete, insert = previous[j], current[j - 1]
                candidates = [
                    (diag[0] + 1, diag[1] + 1, diag[2], diag[3], diag[4]),
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3], delete[4]),
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1, insert[4]),
                ]
                value = min(candidates, key=lambda x: x[0])
            current.append(value)
        previous = current
    edits, substitutions, deletions, insertions, hits = previous[-1]
    return {
        "edits": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "hits": hits,
    }


def text_metrics(expected, transcript):
    ref, hyp = normalize(expected), normalize(transcript)
    words, characters = (
        edit_counts(ref.split(), hyp.split()),
        edit_counts(ref.replace(" ", ""), hyp.replace(" ", "")),
    )
    reference_words, reference_chars = len(ref.split()), len(ref.replace(" ", ""))
    return {
        "normalized_expected": ref,
        "normalized_transcript": hyp,
        **{k: v for k, v in words.items() if k != "edits"},
        "word_edits": words["edits"],
        "character_edits": characters["edits"],
        "reference_words": reference_words,
        "reference_characters": reference_chars,
        "wer": words["edits"] / reference_words
        if reference_words
        else (0.0 if not hyp else None),
        "cer": characters["edits"] / reference_chars
        if reference_chars
        else (0.0 if not hyp else None),
        "exact_normalized": ref == hyp,
        "spacing_only_difference": ref != hyp
        and ref.replace(" ", "") == hyp.replace(" ", ""),
        "exact_characters_ignoring_whitespace": ref.replace(" ", "")
        == hyp.replace(" ", ""),
        "numeric_representation_differs": re.findall(r"\d+", ref)
        != re.findall(r"\d+", hyp),
    }


def calibration_status(logical, models):
    result = {}
    for row in logical:
        if row["kind"] != "reference_control":
            continue
        judges = row.get("judges", {})
        status = (
            "control_ASR_outputs_missing"
            if len(judges) != len(models)
            else "control_ASR_screen_matches"
            if all(j["exact_normalized"] for j in judges.values())
            else "control_ASR_mismatch_review"
        )
        result[row["language"]] = {
            "status": status,
            "id": row["id"],
            "asr_mapping": row["omnilingual_language"],
            "note": "Existing synthetic reference transcript control only; not independent native certification",
        }
    return result


def route_coverage(inventory, logical, models=()):
    result = []
    for route in inventory:
        rows = [
            r
            for r in logical
            if r["language"] == route["language"] and r["kind"] == "request"
        ]
        controls = [
            r
            for r in logical
            if r["language"] == route["language"] and r["kind"] == "reference_control"
        ]
        probes = [
            r
            for r in logical
            if r["language"] == route["language"] and r["kind"] == "model_probe"
        ]
        record = {
            "language": route["language"],
            "label": route.get("label"),
            "omnilingual_language": route.get("omnilingual_language"),
            "asr_mapping_status": route.get("asr_mapping_status"),
            "asr_mapping_note": route.get("asr_mapping_note"),
            "local_native_reference_available": route.get(
                "local_native_reference_available"
            ),
            "status": "not_run" if not rows else "awaiting_ASR_outputs",
            "requests": len(rows),
            "http_status_counts": dict(
                Counter(str(r.get("http_status")) for r in rows)
            ),
            "retained_HTTP200": sum(
                r.get("http_status") == 200 and bool(r.get("path")) for r in rows
            ),
            "hash_verified": sum(bool(r.get("hash_verified")) for r in rows),
            "asr_ready_logical": sum(bool(r.get("asr_input_id")) for r in rows),
            "unique_ASR_audio_language_inputs": len(
                {r["asr_input_id"] for r in rows if r.get("asr_input_id")}
            ),
            "unique_audio_expected_observations": len(
                {
                    (r.get("asr_input_id"), r["expected"])
                    for r in rows
                    if r.get("asr_input_id")
                }
            ),
            "voice_profiles": dict(
                Counter(r.get("voice_profile", "unreported") for r in rows)
            ),
            "reference_languages": dict(
                Counter(r.get("reference_language", "unreported") for r in rows)
            ),
            "response_sources": dict(
                Counter(r.get("response_source", "unreported") for r in rows)
            ),
            "proxy_mapping_requests": sum(
                bool(r.get("asr_mapping_is_proxy")) for r in rows
            ),
            "reference_controls": len(controls),
            "model_probe_outputs": len(probes),
            "model_probe_ASR_ready": sum(bool(r.get("asr_input_id")) for r in probes),
            "model_probe_screen_status_counts": dict(
                Counter(r.get("screen_status", "awaiting_ASR_outputs") for r in probes)
            ),
            "model_probe_voice_profiles": dict(
                Counter(r.get("voice_profile", "unreported") for r in probes)
            ),
            "model_probe_failure_denominator": "unknown; model-stage probe, not HTTP"
            if probes
            else None,
            "model_probe_metrics": {},
            "arms": sorted({r["arm"] for r in rows + probes}),
            "reference_calibration": rows[0].get(
                "reference_calibration", "awaiting_ASR_outputs"
            )
            if rows
            else "not_run",
            "quality_certified": False,
            "metrics": {},
        }
        if models and rows:
            record["screen_status_counts"] = dict(
                Counter(r.get("screen_status", "not_scored") for r in rows)
            )
            record["status"] = (
                "screen_outputs_available"
                if any(r.get("judges") for r in rows)
                else "awaiting_ASR_outputs"
            )
            for model in models:
                selected = [r for r in rows if model in r.get("judges", {})]
                unique_rows = {}
                for row in selected:
                    unique_rows[(row["asr_input_id"], row["expected"])] = row
                record["metrics"][model] = {
                    "logical_request_weighted": aggregate_metrics(selected, model),
                    "unique_audio_expected": aggregate_metrics(
                        list(unique_rows.values()), model
                    ),
                }
        if models and probes:
            for model in models:
                selected = [r for r in probes if model in r.get("judges", {})]
                unique_rows = {(r["asr_input_id"], r["expected"]): r for r in selected}
                record["model_probe_metrics"][model] = {
                    "logical_probe_output_weighted": aggregate_metrics(selected, model),
                    "unique_audio_expected": aggregate_metrics(
                        list(unique_rows.values()), model
                    ),
                }
        result.append(record)
    return result


def aggregate_metrics(rows, model):
    values = [r["judges"][model] for r in rows]
    words = sum(r["reference_words"] for r in values)
    chars = sum(r["reference_characters"] for r in values)
    return {
        "observations": len(values),
        "exact_normalized": sum(r["exact_normalized"] for r in values),
        "spacing_only": sum(r["spacing_only_difference"] for r in values),
        "reference_words": words,
        "word_edits": sum(r["word_edits"] for r in values),
        "reference_characters": chars,
        "character_edits": sum(r["character_edits"] for r in values),
        "aggregate_wer": sum(r["word_edits"] for r in values) / words
        if words
        else None,
        "aggregate_cer": sum(r["character_edits"] for r in values) / chars
        if chars
        else None,
    }


def load_asr(asr_dirs, models, unique, logical):
    results, issues, sources, metadata = {}, [], [], {}
    prepared_audio = {}
    ids = {r["id"]: (r["sha256"], r["omnilingual_language"]) for r in unique}
    ids.update(
        {
            r["id"]: (r["actual_sha256"], r["omnilingual_language"])
            for r in logical
            if r.get("asr_input_id")
        }
    )
    for directory in asr_dirs:
        prepared = directory / "prepared.json"
        if prepared.exists():
            sources.append(provenance(prepared))
            for row in read_rows(prepared):
                ids[row["id"]] = (
                    row["sha256"],
                    row.get("lang_id", row.get("omnilingual_language")),
                )
                prepared_audio[ids[row["id"]]] = {
                    "audio": row.get("audio", {}),
                    "duration_s": row.get("duration_s"),
                    "chunks": len(row.get("chunks", [])),
                }
    valid_keys = {(r["sha256"], r["omnilingual_language"]) for r in unique}
    by_hash = defaultdict(set)
    for sha, lang in valid_keys:
        by_hash[sha].add(lang)
    for model in models:
        found, conflicts, observed_rows = {}, set(), 0
        reported_inference_rows, reported_reuse_rows, reported_asr_wall_s = 0, 0, 0.0
        for directory in asr_dirs:
            path = directory / f"{model}.jsonl"
            meta = directory / f"{model}-metadata.json"
            if meta.exists():
                sources.append(provenance(meta))
                metadata[str(meta)] = json.loads(meta.read_text())
            if not path.exists():
                continue
            sources.append(provenance(path))
            for row in read_rows(path):
                observed_rows += 1
                reported_reuse_rows += bool(row.get("asr_reused_from_id"))
                reported_inference_rows += not bool(row.get("asr_reused_from_id"))
                reported_asr_wall_s += row.get("asr_wall_s", 0) or 0
                key = ids.get(row.get("id"))
                explicit_lang = row.get("lang_id", row.get("omnilingual_language"))
                if key is None and row.get("sha256"):
                    langs = by_hash[row["sha256"]]
                    lang = explicit_lang or (
                        next(iter(langs)) if len(langs) == 1 else None
                    )
                    key = (row["sha256"], lang) if lang else None
                if (
                    key not in valid_keys
                    or row.get("sha256") != key[0]
                    or (explicit_lang and explicit_lang != key[1])
                ):
                    issues.append(
                        {
                            "model": model,
                            "id": row.get("id"),
                            "issue": "unmatched_or_hash_language_mismatch",
                            "source": str(path),
                        }
                    )
                    continue
                if row.get("model_card", model) != model or not isinstance(
                    row.get("transcript"), str
                ):
                    issues.append(
                        {
                            "model": model,
                            "id": row.get("id"),
                            "issue": "invalid_model_or_transcript",
                            "source": str(path),
                        }
                    )
                    continue
                if key in found and normalize(found[key]["transcript"]) != normalize(
                    row["transcript"]
                ):
                    conflicts.add(key)
                    issues.append(
                        {
                            "model": model,
                            "id": row.get("id"),
                            "issue": "conflicting_transcripts_for_same_audio_and_language",
                            "source": str(path),
                        }
                    )
                elif key not in found:
                    found[key] = {
                        **row,
                        "asr_output_source": str(path),
                        "asr_input_validation": prepared_audio.get(key, {}),
                    }
        for key in conflicts:
            found.pop(key, None)
        results[model] = found
        metadata[f"{model}:coverage"] = {
            "observed_output_rows": observed_rows,
            "reported_actual_ASR_inference_rows": reported_inference_rows,
            "reported_reused_ASR_rows": reported_reuse_rows,
            "reported_ASR_wall_s": reported_asr_wall_s,
            "usable_unique_audio_language_inputs": len(found),
            "conflicting_inputs_excluded": len(conflicts),
        }
    return results, issues, sources, metadata


def paired_rows(logical, pairs, models):
    result = []
    for pair in pairs:
        before, after = pair.split("=", 1)
        groups = {arm: defaultdict(list) for arm in [before, after]}
        for row in logical:
            if row["kind"] in {"request", "model_probe"} and row["arm"] in groups:
                key = (row["phase"], row["case_id"], row["semantic_input_sha256"])
                groups[row["arm"]][key].append(row)
        for key in sorted(set(groups[before]) | set(groups[after]), key=str):
            left, right = groups[before][key], groups[after][key]
            for index in range(max(len(left), len(right))):
                a = left[index] if index < len(left) else None
                b = right[index] if index < len(right) else None
                comparable = bool(
                    a and b and a.get("hash_verified") and b.get("hash_verified")
                )
                row = {
                    "pair": pair,
                    "phase": key[0],
                    "case_id": key[1],
                    "semantic_input_sha256": key[2],
                    "occurrence": index,
                    "before_id": a["id"] if a else None,
                    "after_id": b["id"] if b else None,
                    "language": (a or b)["language"],
                    "voice_profile": (a or b).get("voice_profile"),
                    "before_http_status": a.get("http_status") if a else None,
                    "after_http_status": b.get("http_status") if b else None,
                    "before_sha256": a.get("actual_sha256") if a else None,
                    "after_sha256": b.get("actual_sha256") if b else None,
                    "hash_comparable": comparable,
                    "hash_equal": a["actual_sha256"] == b["actual_sha256"]
                    if comparable
                    else None,
                    "judge_deltas": {},
                }
                if comparable:
                    for model in models:
                        x, y = (
                            a.get("judges", {}).get(model),
                            b.get("judges", {}).get(model),
                        )
                        if x and y:
                            row["judge_deltas"][model] = {
                                "wer_delta": y["wer"] - x["wer"]
                                if x["wer"] is not None and y["wer"] is not None
                                else None,
                                "cer_delta": y["cer"] - x["cer"]
                                if x["cer"] is not None and y["cer"] is not None
                                else None,
                                "new_mismatch_screen_flag": x["exact_normalized"]
                                and not y["exact_normalized"],
                            }
                result.append(row)
    return result


def score(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((args.prepared / "preparation.json").read_text())
    inventory = read_rows(args.prepared / "inventory.json")
    logical = read_rows(args.prepared / "logical-cases.jsonl")
    unique = read_rows(args.prepared / "asr-input-manifest.json")
    transcripts, issues, sources, asr_metadata = load_asr(
        args.asr_dir, args.models, unique, logical
    )
    metrics_cache = {}
    for row in logical:
        judges, flags = {}, list(row["issues"])
        if row.get("asr_input_id"):
            key = (row["actual_sha256"], row["omnilingual_language"])
            for model in args.models:
                transcript = transcripts[model].get(key)
                if transcript:
                    metric_key = (row["expected"], transcript["transcript"])
                    if metric_key not in metrics_cache:
                        metrics_cache[metric_key] = text_metrics(*metric_key)
                    judges[model] = {
                        **transcript,
                        **metrics_cache[metric_key],
                        "propagated_to_logical_id": row["id"],
                        "evidence_key": [key[0], key[1]],
                    }
        if len(judges) != len(args.models):
            flags.append("missing_ASR_output")
        if len({j["normalized_transcript"] for j in judges.values()}) > 1:
            flags.append("asr_disagreement")
        if any(not j["exact_normalized"] for j in judges.values()):
            flags.append("word_mismatch_review")
        if any(j["spacing_only_difference"] for j in judges.values()):
            flags.append("spacing_only_difference_present")
        if any(j["numeric_representation_differs"] for j in judges.values()):
            flags.append("numeric_representation_review_not_proven_TTS_error")
        if any(
            j.get("asr_input_validation", {}).get("chunks", 0) > 1
            or len(j.get("chunk_transcripts", [])) > 1
            for j in judges.values()
        ):
            flags.append("ASR_chunk_boundaries_review")
        if any(
            j.get("asr_input_validation", {})
            .get("audio", {})
            .get("clipped_sample_fraction", 0)
            > 0
            for j in judges.values()
        ):
            flags.append("clipped_samples_review")
        if row.get("asr_mapping_is_proxy"):
            flags.append("proxy_language_mapping_unqualified")
        if row.get("length") == "critical":
            flags.append("critical_content_native_review_required")
        row["judges"], row["flags"] = judges, sorted(set(flags))
        row["screen_status"] = (
            "not_scored"
            if not judges
            else "partial_ASR_outputs"
            if len(judges) != len(args.models)
            else "review_required"
            if flags
            else "ASR_screen_matches"
        )
        row["quality_certified"] = False
    calibration = calibration_status(logical, args.models)
    for row in logical:
        if row["kind"] not in {"request", "model_probe"}:
            continue
        row["reference_calibration"] = {
            "target_language": calibration.get(
                row["language"], {"status": "target_language_control_not_available"}
            ),
            "reference_language": calibration.get(
                row.get("reference_language"),
                {"status": "reference_language_control_not_available"},
            ),
        }
        target_status = row["reference_calibration"]["target_language"]["status"]
        if target_status != "control_ASR_screen_matches":
            row["flags"].append("target_language_ASR_uncalibrated_or_control_review")
            if row["screen_status"] == "ASR_screen_matches":
                row["screen_status"] = "review_required"
    coverage = route_coverage(inventory, logical, args.models)
    for route in coverage:
        route["hardware_by_arm"] = {
            arm: preparation["arms"][arm]["hardware_label"] for arm in route["arms"]
        }
    pairs = paired_rows(logical, args.pair, args.models)
    groups = read_rows(args.prepared / "origin-groups.json")
    scored_by_id = {r["id"]: r for r in logical}
    for group in groups:
        rows = [scored_by_id[identity] for identity in group["logical_ids"]]
        group["screen_status_counts"] = dict(Counter(r["screen_status"] for r in rows))
        group["unique_ASR_inputs"] = sorted(
            {r["asr_input_id"] for r in rows if r.get("asr_input_id")}
        )
    summary = {
        "created_at": now(),
        "status": "ASR_outputs_available"
        if any(transcripts.values())
        else "awaiting_ASR_outputs",
        "postprocessing_hardware": cpu_metadata(),
        "inventory_routes": len(inventory),
        "corpus_cases": preparation["corpus_cases"],
        "inventory": preparation["inventory"],
        "corpus": preparation["corpus"],
        "preparation": provenance(args.prepared / "preparation.json"),
        "arms": preparation["arms"],
        "ASR_sources": sources,
        "ASR_hardware_and_output_metadata": asr_metadata,
        "ASR_issues": issues,
        "models_required": args.models,
        "reference_calibration": calibration,
        "logical_request_screen_statuses": dict(
            Counter(r["screen_status"] for r in logical if r["kind"] == "request")
        ),
        "logical_model_probe_screen_statuses": dict(
            Counter(r["screen_status"] for r in logical if r["kind"] == "model_probe")
        ),
        "routes_with_requests": sum(r["requests"] > 0 for r in coverage),
        "routes_with_both_ASR_outputs": sum(
            any(
                len(r["judges"]) == len(args.models)
                for r in logical
                if r["kind"] == "request" and r["language"] == x["language"]
            )
            for x in coverage
        ),
        "unique_ASR_inputs": len(unique),
        "logical_rows": len(logical),
        "paired": {
            p: {
                "logical_pairs": sum(r["pair"] == p for r in pairs),
                "hash_comparable": sum(
                    r["pair"] == p and r["hash_comparable"] for r in pairs
                ),
                "hash_identical": sum(
                    r["pair"] == p and r["hash_equal"] is True for r in pairs
                ),
            }
            for p in args.pair
        },
        "quality_certified": False,
        "caveats": CAVEATS,
    }
    write_jsonl(output / "scored-logical-cases.jsonl", logical)
    write_json(output / "per-route.json", coverage)
    write_jsonl(output / "paired-before-after.jsonl", pairs)
    write_json(output / "origin-propagation.json", groups)
    write_json(output / "summary.json", summary)
    with (output / "per-route.csv").open("w") as stream:
        fields = [
            "language",
            "label",
            "omnilingual_language",
            "status",
            "requests",
            "retained_HTTP200",
            "hash_verified",
            "asr_ready_logical",
            "unique_ASR_audio_language_inputs",
            "unique_audio_expected_observations",
            "local_native_reference_available",
            "proxy_mapping_requests",
            "quality_certified",
            "model_probe_outputs",
            "model_probe_ASR_ready",
            "arms",
            "hardware_by_arm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(coverage)
    lines = [
        "# Multilingual complete-WAV quality evidence",
        "",
        "CPU-only postprocessing; no GPU inference was run by this helper.",
        f"Status: **{summary['status']}**. Inventory: {len(inventory)} routes; source corpus: {preparation['corpus_cases']} cases.",
        "",
        "## Hardware attribution",
        "",
    ]
    for arm, data in preparation["arms"].items():
        lines += [f"- TTS arm `{arm}`: {data['hardware_label']}"]
    if not preparation["arms"]:
        lines += ["No TTS outputs supplied; GPU/VRAM/runtime/co-load unverified."]
    for path, data in asr_metadata.items():
        if not path.endswith(":coverage"):
            lines += [f"- ASR `{path}`: {hardware_label(data.get('hardware', data))}"]
    for model in args.models:
        if not any(
            model in path and not path.endswith(":coverage") for path in asr_metadata
        ):
            lines += [
                f"- ASR `{model}`: GPU, physical/visible VRAM, runtime and co-load unverified; metadata not supplied."
            ]
    lines += [
        "",
        f"Routes with requests: {summary['routes_with_requests']}/{len(inventory)}. Routes with both requested ASR outputs: {summary['routes_with_both_ASR_outputs']}/{len(inventory)}.",
        "",
        "Only verified audio hashes and exact ASR language/script identifiers propagate transcripts. Every logical request is scored against its own expected text. Cache hits and coalesced copies remain logical request outcomes; unique audio plus expected text is also reported separately.",
        "",
        "## Interpretation limits",
        "",
    ] + [f"- {x}" for x in CAVEATS]
    lines += [
        "",
        "See `per-route.json` for every route, `scored-logical-cases.jsonl` for transcripts/flags and provenance, `paired-before-after.jsonl` for matched inputs, and `origin-propagation.json` for cache/coalescing origins.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "routes_with_both_ASR_outputs": summary["routes_with_both_ASR_outputs"],
                "routes": len(inventory),
                "quality_certified": False,
            }
        )
    )
    return summary


def export_samples(args):
    """Export exactly one real short pair per route after complete runs exist."""
    report = args.report.resolve()
    summary = json.loads((report / "summary.json").read_text())
    rows = {r["id"]: r for r in read_rows(report / "scored-logical-cases.jsonl")}
    inventory = read_rows(args.inventory)
    if summary.get("inventory", {}).get("sha256") != digest(args.inventory):
        raise ValueError(
            "Export inventory differs from the inventory used to prepare/score these runs"
        )
    pairs = [
        p
        for p in read_rows(report / "paired-before-after.jsonl")
        if p["pair"] == args.pair and p["phase"] == args.phase
    ]
    before_arm, after_arm = args.pair.split("=", 1)
    for arm in [before_arm, after_arm]:
        hardware = summary.get("arms", {}).get(arm, {}).get("hardware", {})
        if "CPU-only" in hardware.get("measurement_kind", ""):
            raise ValueError(
                "Cannot export fake/CPU-only HTTP samples as GPU benchmark audio"
            )
        evidence = str(
            hardware.get("nvidia_smi_gpu", hardware.get("physical_gpu_nvidia_smi", ""))
        )
        if args.expected_gpu.casefold() not in evidence.casefold():
            raise ValueError(
                f"Physical GPU evidence for {arm} does not establish {args.expected_gpu!r}"
            )
    selected = {}
    for pair in pairs:
        a, b = rows.get(pair["before_id"]), rows.get(pair["after_id"])
        if not a or not b or a.get("length") != "short" or b.get("length") != "short":
            continue
        if (
            not pair["hash_comparable"]
            or a.get("http_status") != 200
            or b.get("http_status") != 200
        ):
            continue
        if a["language"] in selected:
            raise ValueError(
                f"Multiple short sample pairs for {a['language']}; choose a unique phase"
            )
        selected[a["language"]] = (pair, a, b)
    missing = sorted({r["language"] for r in inventory} - set(selected))
    if missing:
        raise ValueError(
            f"Complete {len(inventory)}-route real short-pair evidence required; missing: {missing}"
        )
    # Check all original bytes again before writing anything to the export target.
    for _, a, b in selected.values():
        for row in [a, b]:
            path = Path(row["path"])
            if path.stat().st_size > args.max_file_bytes:
                raise ValueError(f"Audio exceeds export limit: {path}")
            if digest(path) != row["actual_sha256"]:
                raise ValueError(f"Audio changed since preparation: {path}")
    output = args.output.resolve()
    if args.fixtures_output and args.fixtures_output.exists():
        raise FileExistsError(args.fixtures_output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "before").mkdir()
    (output / "after").mkdir()
    manifest = []
    for route in inventory:
        language = route["language"]
        if not re.fullmatch(r"[A-Za-z0-9-]+", language):
            raise ValueError(f"Unsafe route filename: {language}")
        pair, a, b = selected[language]
        entry = {
            "language": language,
            "language_label": route.get("label"),
            "api_language": a["case"].get("api_language"),
            "omnilingual_language": a["omnilingual_language"],
            "expected": a["expected"],
            "case_id": a["case_id"],
            "phase": pair["phase"],
            "semantic_input_sha256": pair["semantic_input_sha256"],
            "hash_equal": pair["hash_equal"],
            "voice_profile": a.get("voice_profile"),
            "reference_id": a.get("reference_id"),
            "reference_language": a.get("reference_language"),
            "asr_mapping_is_proxy": a.get("asr_mapping_is_proxy"),
            "asr_mapping_note": a.get("asr_mapping_note"),
            "source": a.get("source"),
            "translation_status": a.get("translation_status"),
            "quality_certified": False,
            "caveats": CAVEATS,
        }
        for label, row in [("before", a), ("after", b)]:
            relative = f"{label}/{language}.wav"
            shutil.copyfile(row["path"], output / relative)
            entry[label] = {
                "file": relative,
                "sha256": row["actual_sha256"],
                "original_path": row["path"],
                "request_id": row["id"],
                "arm": row["arm"],
                "audio": row.get("audio"),
                "response_source": row.get("response_source"),
                "flags": row.get("flags"),
                "judges": row.get("judges"),
                "source_ref": summary["arms"][row["arm"]].get("source_ref"),
                "hardware": summary["arms"][row["arm"]].get("hardware", {}),
            }
        manifest.append(entry)
    write_json(output / "paired-manifest.json", manifest)
    write_json(
        output / "export-provenance.json",
        {
            "created_at": now(),
            "postprocessing_hardware": cpu_metadata(),
            "report_summary": provenance(report / "summary.json"),
            "pair": args.pair,
            "routes": len(manifest),
            "files": len(manifest) * 2,
            "expected_gpu": args.expected_gpu,
            "original_full_299_and_load_audio_retention": "Original paths remain in paired manifest and full run directories; this is only the 104 short-pair review subset",
            "arms": summary["arms"],
            "caveats": CAVEATS,
        },
    )
    if args.fixtures_output:
        fixtures = args.fixtures_output.resolve()
        fixtures.mkdir(parents=True, exist_ok=False)
        names = [
            "active-languages.json",
            "corpus.json",
            "load-pool.json",
            "README.md",
            "corpus-source-coverage.json",
            "inventory-summary.json",
            "load-pool-summary.json",
            "dataset-readme.md",
        ]
        copies = []
        for name in names:
            source = args.fixtures_source / name
            shutil.copyfile(source, fixtures / name)
            copies.append(provenance(source) | {"packaged_file": name})
        source = args.fixtures_source / "flores200_dataset" / "README"
        shutil.copyfile(source, fixtures / "FLORES-README.txt")
        copies.append(provenance(source) | {"packaged_file": "FLORES-README.txt"})
        write_json(
            fixtures / "fixture-provenance.json",
            {
                "created_at": now(),
                "postprocessing_hardware": cpu_metadata(),
                "files": copies,
                "licenses": {
                    "FLORES200": "CC-BY-SA-4.0",
                    "Meta Omnilingual corpus": "CC-BY-4.0",
                    "Wikipedia excerpts": "CC-BY-SA-4.0",
                },
                "attribution": "Every corpus/load-pool record retains source dataset/page, revision/hash, sample/segment identifiers and license; read README.md for source-specific caveats.",
            },
        )
    print(
        json.dumps(
            {
                "status": "exported_real_short_pairs",
                "routes": len(manifest),
                "output": str(output),
                "quality_certified": False,
            }
        )
    )


def command_export(sub):
    exporter = sub.add_parser(
        "export-samples",
        help="Export all104 real short before/after pairs; refuses fake/partial runs",
    )
    exporter.add_argument("--report", type=Path, required=True)
    exporter.add_argument("--pair", required=True, metavar="BEFORE=AFTER")
    exporter.add_argument("--phase", default="qa-short-all-languages")
    exporter.add_argument(
        "--inventory", type=Path, default=LANGUAGES / "active-languages.json"
    )
    exporter.add_argument("--expected-gpu", default="RTX 4070")
    exporter.add_argument("--max-file-bytes", type=int, default=10_000_000)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--fixtures-output", type=Path)
    exporter.add_argument("--fixtures-source", type=Path, default=LANGUAGES)
    exporter.set_defaults(func=export_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser(
        "prepare", help="CPU-only verified-hash deduplication; no inference"
    )
    prep.add_argument("--run-dir", type=Path, action="append", default=[])
    prep.add_argument(
        "--probe-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional non-HTTP model-stage quality manifest; probe counts never enter HTTP availability",
    )
    prep.add_argument(
        "--runs-root",
        type=Path,
        help="Discover quality manifests; do not point this at fake smoke-run directories",
    )
    prep.add_argument(
        "--inventory", type=Path, default=LANGUAGES / "active-languages.json"
    )
    prep.add_argument("--corpus", type=Path, default=LANGUAGES / "corpus.json")
    prep.add_argument(
        "--references",
        "--reference-manifest",
        dest="references",
        type=Path,
        help="Optional original portable reference manifest; historical generation hardware unverified",
    )
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=prepare)
    scored = sub.add_parser(
        "score", help="CPU-only scoring of existing blind-ASR JSONL files"
    )
    scored.add_argument("--prepared", type=Path, required=True)
    scored.add_argument(
        "--asr-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory with <model>.jsonl and optional prepared.json/<model>-metadata.json",
    )
    scored.add_argument("--models", nargs="+", default=MODELS)
    scored.add_argument("--pair", action="append", default=[], metavar="BEFORE=AFTER")
    scored.add_argument("--output", type=Path, required=True)
    scored.set_defaults(func=score)
    command_export(sub)
    args = parser.parse_args()
    if args.action == "score":
        if any(x.count("=") != 1 for x in args.pair):
            parser.error("--pair must be BEFORE=AFTER")
        if len(args.models) != 2 or len(set(args.models)) != 2:
            parser.error("Exactly two distinct judge model cards are required")
    args.func(args)


if __name__ == "__main__":
    main()
