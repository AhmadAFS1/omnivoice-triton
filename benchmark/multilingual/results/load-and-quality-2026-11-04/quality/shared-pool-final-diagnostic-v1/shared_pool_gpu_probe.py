"""Finite shared-pool neural gate; CPU plan by default, never auto-executes.

Based on the v1 RoPE model gate's tracing/serialization conventions. Changes
only model.forward storage ownership, never neural math, shapes or sampling.
An explicit root lease matching this exact prototype SHA is required. The
outer supervisor owns its child process group and enforces a300s total bound.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_REF = "ce48b253e66b951d87ea42919f945828c62e96a4"
SOURCE_TREE = "e3e5aa9fb61edc8f24bd62b430651265a2816695eb6567da0ae7fcd20f6f98f7"
MODEL_PIN = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
UUID = "GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33"


class ForwardTap:
    """One stable outer forward identity; CPU recording is toggled by a flag."""

    def __init__(self, target, tensor_type, identify, fixture_store):
        self.target, self.tensor_type, self.identify = target, tensor_type, identify
        self.fixture_store = fixture_store
        self.active, self.rows, self.fixture_key = False, [], None

    def __call__(self, *pos, **kwargs):
        output = self.target(*pos, **kwargs)
        if self.active:
            names = (
                "input_ids",
                "audio_mask",
                "labels",
                "attention_mask",
                "document_ids",
                "position_ids",
            )
            supplied = dict(zip(names, pos)) | kwargs
            step = len(self.rows)
            self.rows.append(
                {
                    "step": step,
                    "logits": self.identify(output.logits),
                    "inputs": {
                        k: self.identify(v)
                        for k, v in supplied.items()
                        if isinstance(v, self.tensor_type)
                    },
                }
            )
            if self.fixture_key and step < 2:
                stored = {
                    k: v.detach().cpu().clone()
                    for k, v in supplied.items()
                    if isinstance(v, self.tensor_type)
                }
                stored["logits"] = output.logits.detach().cpu().clone()
                self.fixture_store[(self.fixture_key, step)] = stored
        return output


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def smi(query):
    return subprocess.check_output(
        ["nvidia-smi", "--query-" + query, "--format=csv,noheader"],
        text=True,
        timeout=5,
    ).strip()


def prepare(args):
    corpus = read(args.corpus)
    corpus = corpus["cases"] if isinstance(corpus, dict) else corpus
    cases = [
        next(x for x in corpus if x["id"] == key)
        for key in ("en-short", "es-short", "ar-short")
    ]
    refs = read(args.references)
    refs = refs["references"] if isinstance(refs, dict) else refs
    by_id = {x.get("id", x.get("reference_id")): x for x in refs}
    for case in cases:
        reference = by_id[case["reference_id"]]
        path = Path(reference.get("path", reference.get("file")))
        path = path if path.is_absolute() else args.references.parent / path
        assert sha(path) == reference["sha256"]
    return cases, by_id


def plan(args, cases):
    return {
        "measurement_kind": "CPU plan unless execute+exactSHA fresh root lease; notHTTPcapacity",
        "source_ref": SOURCE_REF,
        "source_tree_sha256": SOURCE_TREE,
        "model_revision": MODEL_PIN,
        "cases": cases,
        "batch_sizes": [1, 8],
        "members_per_arm": 27,
        "steps": 16,
        "speed": 1,
        "postprocess": "off",
        "runner": "hybridFP16",
        "KV_cache": False,
        "split": False,
        "sequence_bucket": False,
        "fused_RoPE": False,
        "codec_batch": 1,
        "graphs": 8,
        "graph_memory_row_tokens": 0,
        "private_reference_kind": "retainedprivate8"
        if args.retain_private_shapes
        else "private8configured;fenced/clearedperjob;resident1semanticreferenceonly",
        "shared_kind": "retainonly6observedshapes(max8),known2/3B1first;max3separateevictionepoch",
        "headroom": "publicfree>=max4GiB/qualifiedshapeestimatebeforeeveryprivate/sharedcapture;noOOMretry/fallback",
        "strict_gate": "all16full-logit/inputSHA,allfinaltokens,floatWAVbits,PCMbodySHA;separateall-tensorold-returnaliaschecks",
        "timing": "separatecleanwarm synchronizedfullgeneration+codec;parityhashingandserializationexcluded",
        "bounds": {
            "total_supervisor_s": 300,
            "child_soft_s": 260,
            "capture_shapes": 8,
            "CFG_B_max": 16,
            "S_max": 768,
        },
        "prototype_sha256": sha(Path(__file__).with_name("shared_pool.py")),
        "script_sha256": sha(Path(__file__)),
        "GPU_executed_by_plan": False,
    }


def validate_lease(args):
    release = read(args.gpu_release)
    assert release["gpu_released"] and release["shared_pool_model_gate"]
    assert release["prototype_sha256"] == sha(
        Path(__file__).with_name("shared_pool.py")
    )
    assert release["probe_sha256"] == sha(Path(__file__))
    assert release["source_ref"] == SOURCE_REF and release["GPU_UUID"] == UUID
    age = time.time() - dt.datetime.fromisoformat(release["at"]).timestamp()
    assert 0 <= age < 600, "Fresh explicit lease required"
    return release


def validate_source(args):
    tree = hashlib.sha256()
    for path in sorted((args.source_root / "src").rglob("*.py")):
        tree.update(
            str(path.relative_to(args.source_root)).encode()
            + b"\0"
            + sha(path).encode()
            + b"\n"
        )
    assert tree.hexdigest() == SOURCE_TREE
    assert (
        sha(args.model / "config.json")
        == "5e359117e13b420c5e0c925d4aba650d624767131f1d1746928f8b850d5dc372"
    )
    integrity = read(args.model_integrity)
    assert integrity["revision"] == MODEL_PIN
    for entry in integrity["files"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert sha(args.model / entry["path"]) == entry["sha256"]


def supervise(args, proposal):
    started = time.monotonic()
    release = validate_lease(args)
    validate_source(args)
    assert not smi("compute-apps=pid,process_name,used_memory")
    assert UUID in smi("gpu=name,uuid,memory.total,driver_version")
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "supervisor.json",
        {
            "at": now(),
            "release": release,
            "plan": proposal,
            "source_integrity_sha256": sha(args.model_integrity),
            "GPU_work_authorized_only_by_explicit_lease": True,
        },
    )
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(ROOT / "hf-cache"),
    )
    env["PYTHONPATH"] = (
        str(args.source_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"]
    status, child, code = "failed", None, None
    previous_handlers = {}

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt("Supervisor interrupted; terminate only owned child")

    for number in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[number] = signal.signal(number, interrupt)
    try:
        remaining = 280 - (time.monotonic() - started)
        assert remaining > 60, "Preflight exhausted the bounded gate"
        with (args.output / "child.log").open("w") as log:
            child = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            write(args.output / "child-pid.json", {"at": now(), "pid": child.pid})
            try:
                code = child.wait(timeout=remaining)
                status = "completed" if code == 0 else "child_failed"
            except subprocess.TimeoutExpired:
                status = "timed_out"
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait(timeout=5)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        idle_error = None
        try:
            active = smi("compute-apps=pid,process_name,used_memory")
        except (OSError, subprocess.SubprocessError) as error:
            active, idle_error = "idle proof unavailable", repr(error)
        if active:
            status = "GPU_not_idle_after_child"
        write(
            args.output / "supervisor-outcome.json",
            {
                "at": now(),
                "status": status,
                "exit_code": code,
                "elapsed_s": time.monotonic() - started,
                "compute_processes": active,
                "GPU_idle": not bool(active),
                "idle_error": idle_error,
                "quality_or_production_certified": False,
            },
        )
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    assert status == "completed", status


def child_main(args, cases, by_id, proposal):
    # Validate before importing Torch/model code; CPU plan never reaches here.
    release = validate_lease(args)
    assert (args.output / "supervisor.json").exists()
    assert not smi("compute-apps=pid,process_name,used_memory")
    assert UUID in smi("gpu=name,uuid,memory.total,driver_version")
    sys.path.insert(0, str(args.source_root / "src"))
    import numpy as np
    import omnivoice_triton
    import soundfile as sf
    import torch
    from omnivoice import OmniVoiceGenerationConfig
    from omnivoice_triton import create_runner
    from run_split_cfg_probe import hardware, seed
    from shared_pool import (
        PIN,
        SharedPoolAdmissionError,
        SharedPoolPatch,
        capture_headroom_bytes,
    )

    assert (
        Path(omnivoice_triton.__file__).resolve().parents[2]
        == args.source_root.resolve()
    )
    assert (
        torch.__version__ == "2.8.0+cu128" and not torch.backends.cuda.matmul.allow_tf32
    )
    deadline = time.monotonic() + 260
    write(
        args.output / "metadata.json",
        {
            "at": now(),
            "hardware": hardware(),
            "plan": proposal,
            "release": release,
            "source_ref": SOURCE_REF,
            "model_revision": PIN,
            "CPU_helper_SHA256": sha(
                Path(__file__).with_name("run_split_cfg_probe.py")
            ),
            "FP16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "TF32": torch.backends.cuda.matmul.allow_tf32,
        },
    )
    runner = None
    status, records, saved, alias_records = "failed", [], [], []
    token_capture, traced_fixtures = [], {}

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt(
            "Supervisor termination; preserve files and exit process"
        )

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def check_budget():
        assert time.monotonic() < deadline, "Finite child budget exhausted"

    def tensor_identity(value):
        raw = value.detach().cpu().contiguous().numpy()
        assert np.isfinite(raw).all()
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }

    def array_identity(value):
        raw = np.ascontiguousarray(value)
        assert np.isfinite(raw).all()
        return {
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }

    def snapshot():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "public_free_bytes": free,
            "device_total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    try:
        runner = create_runner(
            "hybrid",
            device="cuda",
            model_id=str(args.model),
            dtype="fp16",
            disable_unused_kv_cache=True,
            split_cfg=False,
            max_cuda_graphs=8,
            max_cuda_graph_row_tokens=0,
            audio_decode_batch_size=1,
            enable_sage_attention=False,
        )
        runner.load_model()
        model = runner.model
        model.eval()
        private = runner._graph_forward
        assert model.forward is private and runner._graph_memory is None
        original_capture, original_iterative = (
            private._capture,
            model._generate_iterative,
        )
        admission_records = []

        def guarded_capture(ids, *pos, **kwargs):
            check_budget()
            shape = tuple(ids.shape)
            assert (
                len(shape) == 3
                and shape[0] <= 16
                and shape[0] % 2 == 0
                and shape[1] == 8
                and shape[2] <= 768
            )
            free = torch.cuda.mem_get_info(device=ids.device)[0]
            required = capture_headroom_bytes(shape)
            admission_records.append(
                {"shape": shape, "public_free": free, "required": required}
            )
            if free < required:
                raise SharedPoolAdmissionError(
                    "Private capture headroom rejected; no fallback"
                )
            return original_capture(ids, *pos, **kwargs)

        private._capture = guarded_capture

        def iterative(*pos, **kwargs):
            result = original_iterative(*pos, **kwargs)
            token_capture[:] = result
            return result

        model._generate_iterative = iterative
        config = OmniVoiceGenerationConfig(
            num_step=16,
            class_temperature=0.0,
            position_temperature=5.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=False,
        )
        config.postprocess_mode = "off"

        def clear_private():
            torch.cuda.synchronize()
            private.clear()
            torch.cuda.empty_cache()

        def generate(case, prompt, batch, call_seed):
            check_budget()
            seed(call_seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            waves = model.generate(
                text=[case["text"]] * batch,
                language=[case["api_language"]] * batch,
                voice_clone_prompt=[prompt] * batch,
                speed=[1.0] * batch,
                generation_config=config,
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            return (
                waves,
                [x.detach().cpu().numpy().copy() for x in token_capture],
                elapsed,
            )

        def save(case, batch, arm, phase, waves, tokens):
            stem = f"{case['id']}-B{batch}-{arm}-{phase}"
            np.savez_compressed(
                args.output / (stem + "-tokens.npz"),
                **{f"item{i}": x for i, x in enumerate(tokens)},
            )
            np.savez_compressed(
                args.output / (stem + "-float-audio.npz"),
                **{f"item{i}": x for i, x in enumerate(waves)},
            )
            hashes = []
            for index, wave in enumerate(waves):
                raw = np.asarray(wave, dtype=np.float32)
                assert raw.size and np.isfinite(raw).all()
                path = args.output / (stem + f"-{index:03d}.wav")
                sf.write(path, raw, int(model.sampling_rate), subtype="PCM_16")
                hashes.append(sha(path))
                saved.append(
                    {
                        "id": path.stem,
                        "path": path.name,
                        "sha256": hashes[-1],
                        "language": case["language"],
                        "omnilingual_language": case["omnilingual_language"],
                        "expected": case["text"],
                        "reference_id": case["reference_id"],
                        "arm": arm,
                        "phase": phase,
                        "quality_certified": False,
                    }
                )
            write(args.output / "quality-manifest.json", saved)
            return hashes

        prompts, jobs, reference_results = {}, [], {}
        with torch.inference_mode():
            for case in cases:
                reference = by_id[case["reference_id"]]
                path = Path(reference.get("path", reference.get("file")))
                path = path if path.is_absolute() else args.references.parent / path
                prompts[case["id"]] = model.create_voice_clone_prompt(
                    ref_audio=str(path),
                    ref_text=reference["text"],
                    preprocess_prompt=True,
                )
            for batch in (1, 8):
                for index, case in enumerate(cases):
                    jobs.append(
                        (
                            f"{case['id']}-B{batch}",
                            case,
                            batch,
                            20261104 + index * 100 + batch,
                        )
                    )
            tap = ForwardTap(private, torch.Tensor, tensor_identity, traced_fixtures)
            model.forward = tap
            for key, case, batch, call_seed in jobs:
                if not args.retain_private_shapes:
                    clear_private()
                torch.cuda.reset_peak_memory_stats()
                tap.active, tap.rows, tap.fixture_key = True, [], key
                waves, tokens, _ = generate(case, prompts[case["id"]], batch, call_seed)
                tap.active = False
                assert len(tap.rows) == 16 and len(waves) == len(tokens) == batch
                result = {
                    "logits": tap.rows,
                    "tokens": [array_identity(x) for x in tokens],
                    "float_waves": [array_identity(x) for x in waves],
                    "PCM": save(case, batch, "private", "parity", waves, tokens),
                }
                reference_results[key] = result
                write(args.output / (key + "-private-full-logit-hashes.json"), tap.rows)
                timed_waves, timed_tokens, elapsed = generate(
                    case, prompts[case["id"]], batch, call_seed
                )
                assert [array_identity(x) for x in timed_tokens] == result[
                    "tokens"
                ] and [array_identity(x) for x in timed_waves] == result["float_waves"]
                assert (
                    save(case, batch, "private", "timed", timed_waves, timed_tokens)
                    == result["PCM"]
                )
                records.append(
                    {
                        "arm": "private",
                        "job": key,
                        "full_model_warm_ms": elapsed,
                        "memory": snapshot(),
                        "graphs": private.get_metrics(),
                        "reference_kind": proposal["private_reference_kind"],
                    }
                )
                write(args.output / "measurements.json", records)
            model.forward = private
            clear_private()
            shapes = tuple(
                dict.fromkeys(
                    tuple(traced_fixtures[(key, 0)]["input_ids"].shape)
                    for key, *_ in jobs
                )
            )
            assert 2 <= len(shapes) <= 6
            write(
                args.output / "actual-allowed-shapes.json",
                {
                    "shapes": shapes,
                    "derived_from": "Actual private first-step tensors, no padding or estimated shapes",
                },
            )
            torch.save(
                traced_fixtures, args.output / "two-step-input-logit-fixtures.pt"
            )
            patch = SharedPoolPatch(
                model,
                private,
                enabled=True,
                allowed_shapes=shapes,
                max_graphs=8,
                model_revision=PIN,
            )
            patch.enable()
            shared = patch.forward
            tap = ForwardTap(shared, torch.Tensor, tensor_identity, traced_fixtures)
            model.forward = tap
            for key, case, batch, call_seed in jobs:
                torch.cuda.reset_peak_memory_stats()
                tap.active, tap.rows, tap.fixture_key = True, [], None
                waves, tokens, _ = generate(case, prompts[case["id"]], batch, call_seed)
                tap.active = False
                result = {
                    "logits": tap.rows,
                    "tokens": [array_identity(x) for x in tokens],
                    "float_waves": [array_identity(x) for x in waves],
                    "PCM": save(case, batch, "shared", "parity", waves, tokens),
                }
                write(args.output / (key + "-shared-full-logit-hashes.json"), tap.rows)
                exact = result == reference_results[key]
                write(
                    args.output / (key + "-parity.json"),
                    {"all16logit_input_token_float_PCM_equal": exact},
                )
                assert exact, "Strict neural parity failed; no fallback/promotion"
                timed_waves, timed_tokens, elapsed = generate(
                    case, prompts[case["id"]], batch, call_seed
                )
                assert [array_identity(x) for x in timed_tokens] == result[
                    "tokens"
                ] and [array_identity(x) for x in timed_waves] == result["float_waves"]
                assert (
                    save(case, batch, "shared", "timed", timed_waves, timed_tokens)
                    == result["PCM"]
                )
                records.append(
                    {
                        "arm": "shared",
                        "job": key,
                        "full_model_warm_ms": elapsed,
                        "memory": snapshot(),
                        "graphs": shared.metrics(),
                        "exact": True,
                    }
                )
                write(args.output / "measurements.json", records)

            def aliases(forward, sequence, label):
                retained = []
                for key, step in sequence:
                    check_budget()
                    fixture = traced_fixtures[(key, step)]
                    supplied = {
                        k: v.to("cuda") for k, v in fixture.items() if k != "logits"
                    }
                    output = forward(**supplied)
                    observed = output.logits.detach().cpu()
                    expected = fixture["logits"]
                    exact = torch.equal(
                        observed.view(torch.int16), expected.view(torch.int16)
                    )
                    record = {
                        "label": label,
                        "job": key,
                        "step": step,
                        "exact_full_logits": exact,
                        "max_abs_error": float(
                            (observed.float() - expected.float()).abs().max()
                        ),
                        "data_ptr": output.logits.data_ptr(),
                    }
                    alias_records.append(record)
                    write(args.output / "alias-records.json", alias_records)
                    assert exact, "Alias/replay full-tensor gate failed"
                    retained.append((output.logits, expected))
                    for old, wanted in retained:
                        assert torch.equal(
                            old.detach().cpu().view(torch.int16),
                            wanted.view(torch.int16),
                        ), "Old returned global tensor changed"
                return retained

            key_by_shape = {}
            for job_key, *_ in jobs:
                key_by_shape.setdefault(
                    tuple(traced_fixtures[(job_key, 0)]["input_ids"].shape), job_key
                )
            keys = list(key_by_shape.values())
            assert len(keys) >= 3, (
                "Three distinct observed shapes required for the alias gate"
            )
            retained_epoch = aliases(
                shared,
                [
                    (keys[i], step)
                    for i, step in [(0, 0), (1, 0), (2, 0), (1, 1), (0, 1)]
                ],
                "ABCBA",
            )
            aliases(
                shared,
                [(key, 1) for key in reversed(keys)],
                "reverse-and-updated-valid-inputs",
            )
            model.forward = shared
            patch.disable()
            eviction_limit = min(3, len(keys) - 1)
            patch = SharedPoolPatch(
                model,
                private,
                enabled=True,
                allowed_shapes=shapes,
                max_graphs=eviction_limit,
                model_revision=PIN,
            )
            patch.enable()
            eviction_shared = patch.forward
            model.forward = ForwardTap(
                eviction_shared, torch.Tensor, tensor_identity, traced_fixtures
            )
            sequence = [(keys[i], 0) for i in range(eviction_limit)] + [
                (keys[1], 1),
                (keys[0], 1),
                (keys[eviction_limit], 0),
                (keys[1], 0),
            ]
            aliases(
                eviction_shared, sequence, "newcapture-eviction-with-old-global-returns"
            )
            assert eviction_shared.evictions > 0
            final_metrics = eviction_shared.metrics()
            model.forward = eviction_shared
            patch.disable()
            for old, wanted in retained_epoch:
                assert torch.equal(
                    old.detach().cpu().view(torch.int16), wanted.view(torch.int16)
                ), "Old global return changed across pool clear/new epoch"
            del private._capture
            model._generate_iterative = original_iterative
            write(args.output / "private-admission.json", admission_records)
            write(args.output / "final-shared-metrics.json", final_metrics)
            status = "completed"
        runner.unload_model()
        write(
            args.output / "outcome.json",
            {
                "at": now(),
                "status": status,
                "pairs_checked": 6,
                "saved_WAVs": len(saved),
                "alias_rows": len(alias_records),
                "runner_unload_called": True,
                "GPU_idle_requires_supervisor_proof": True,
                "quality_or_production_certified": False,
            },
        )
    except BaseException as error:  # noqa: BLE001 -- fatal/termination must exit without CUDA cleanup
        write(
            args.output / "failure.json",
            {
                "at": now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "records": len(records),
                "saved_WAVs": len(saved),
                "policy": "No eager/OOMretry or CUDA cleanup after failure; owned child exits and supervisor verifies GPU release",
            },
        )
        write(
            args.output / "outcome.json",
            {
                "at": now(),
                "status": "failed",
                "saved_WAVs": len(saved),
                "quality_or_production_certified": False,
            },
        )
        # Keep all CUDA owners alive until process teardown; do not unwind into
        # graph/model destructors after a possibly invalid capture/context.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--model-integrity", type=Path)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "candidate-v8-audio-guard"
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "languages/corpus.json")
    parser.add_argument(
        "--references", type=Path, default=ROOT / "languages/references/manifest.json"
    )
    parser.add_argument("--gpu-release", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retain-private-shapes", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases, refs = prepare(args)
    proposal = plan(args, cases)
    if not args.execute:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))
        return
    assert args.gpu_release and args.model_integrity and args.output
    if args.child:
        child_main(args, cases, refs, proposal)
    else:
        supervise(args, proposal)


if __name__ == "__main__":
    main()
