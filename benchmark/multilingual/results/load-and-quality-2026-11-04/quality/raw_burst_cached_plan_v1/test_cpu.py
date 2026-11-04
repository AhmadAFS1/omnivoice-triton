"""CPU-only cached/coalesced proof contracts; no model/HTTP API calls."""

import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cached_assay", HERE / "run.py")
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)


def row(source):
    return {
        "status": 200,
        "full_body_received": True,
        "headers": {
            "x-omnivoice-response-source": source,
            "x-omnivoice-inference-executed": "true"
            if source == "generated"
            else "false",
        },
    }


def delta(count):
    return {
        "batches_started": count,
        "members_started": count,
        "members_succeeded": count,
        "members_failed": 0,
    }


class ProofTests(unittest.TestCase):
    def test_swapped_generated_and_coalesced_inference_flags_fail(self):
        generated, coalesced = row("generated"), row("coalesced")
        generated["headers"]["x-omnivoice-inference-executed"] = "false"
        coalesced["headers"]["x-omnivoice-inference-executed"] = "true"
        proof = assay.response_proof(
            [generated, coalesced], {0: b"same", 1: b"same"}, 2, delta(1), 1
        )
        self.assertEqual(
            proof["actual_inference_header_counts"], {"true": 1, "false": 1}
        )
        self.assertEqual(proof["source_counts"], {"generated": 1, "coalesced": 1})
        self.assertFalse(proof["per_response_source_inference_mapping_pass"])
        self.assertFalse(proof["gate_pass"])

    def test_new_child_guard_stops_server_after_early_metadata_error(self):
        class Server:
            pid = 12345
            returncode = None

            def __init__(self):
                self.signals = []

            def poll(self):
                return self.returncode

            def send_signal(self, number):
                self.signals.append(number)

            def wait(self, timeout):
                self.returncode = 0
                return 0

        server = Server()
        core = SimpleNamespace(
            import_file=lambda *args: SimpleNamespace(
                build_command=lambda args: [sys.executable, "seed-placeholder"]
            ),
            build_args=lambda *args: SimpleNamespace(),
        )
        original = assay.write

        def fail(path, value):
            if path.name == "server-pid.json":
                raise OSError("earlymetadata")
            return original(path, value)

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), source=Path(folder))
            with (
                patch.object(assay.subprocess, "Popen", return_value=server),
                patch.object(assay, "write", side_effect=fail),
                self.assertRaisesRegex(OSError, "earlymetadata"),
            ):
                assay.child(args, {}, core, {}, None, None)
            self.assertEqual(server.signals, [signal.SIGKILL])
            self.assertEqual(server.returncode, 0)

    def test_cold_one_generation_allows_coalesced_and_cache_tail(self):
        proof = assay.response_proof(
            [row("generated"), row("coalesced"), row("cache")],
            {0: b"same", 1: b"same", 2: b"same"},
            3,
            delta(1),
            1,
        )
        self.assertTrue(proof["gate_pass"])
        self.assertEqual(
            proof["source_counts"], {"generated": 1, "coalesced": 1, "cache": 1}
        )

    def test_second_neural_generation_or_partial_body_never_passes(self):
        for rows, count in (
            ([row("generated"), row("coalesced")], 2),
            ([row("generated"), dict(row("cache"), full_body_received=False)], 1),
        ):
            proof = assay.response_proof(
                rows, {0: b"same", 1: b"same"}, 2, delta(count), 1
            )
            self.assertFalse(proof["gate_pass"])

    def test_hot_requires_zero_inference_cache_headers_and_cold_hash(self):
        wanted = hashlib.sha256(b"same").hexdigest()
        proof = assay.response_proof(
            [row("cache")] * 5, {i: b"same" for i in range(5)}, 5, delta(0), 0, wanted
        )
        self.assertTrue(proof["gate_pass"])
        proof = assay.response_proof(
            [row("cache")] * 5,
            {i: b"changed" for i in range(5)},
            5,
            delta(0),
            0,
            wanted,
        )
        self.assertFalse(proof["gate_pass"])
        proof = assay.response_proof(
            [row("coalesced")] * 5,
            {i: b"same" for i in range(5)},
            5,
            delta(0),
            0,
            wanted,
        )
        self.assertFalse(proof["gate_pass"])

    def test_missing_legacy_headers_cannot_be_invented_as_cache_proof(self):
        proof = assay.response_proof(
            [{"status": 200, "full_body_received": True}], {0: b"same"}, 1, delta(0), 0
        )
        self.assertFalse(proof["gate_pass"])
        self.assertEqual(proof["actual_inference_header_counts"], {"unreported": 1})

    def test_default_dry_validates_pinned_core_and_disjoint_keys_without_neural_import(
        self,
    ):
        code = """
import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0]in{'torch','numpy','soundfile','omnivoice','omnivoice_triton'}:
  raise RuntimeError('CPU dry attempted neural import:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(HERE / "run.py")],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        output = json.loads(result.stdout)
        self.assertFalse(output["GPU_executed"])
        self.assertNotEqual(
            output["warmup_case"]["text"], output["target_case"]["text"]
        )
        self.assertEqual(output["target_case"]["id"], "en-short")
        self.assertEqual(
            [p["requests"] for p in output["config"]["phases"]], [1000, 5000]
        )


if __name__ == "__main__":
    unittest.main()
