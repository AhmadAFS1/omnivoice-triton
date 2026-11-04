"""Pure CPU fake-pipeline contract checks; no ASR/CUDA imports or audio read."""

import unittest

import batch_prototype as prototype


class FakePipeline:
    def __init__(self):
        self.calls = []

    def transcribe(self, paths, *, lang, batch_size):
        assert len(paths) == len(lang)
        assert all(
            isinstance(path, str) and "NEVERPASSGOLD" not in path for path in paths
        )
        self.calls.append({"paths": paths, "lang": lang, "batch_size": batch_size})
        return ["heard-" + path.rsplit("/", 1)[-1] for path in paths]


class BatchProtocol(unittest.TestCase):
    def cases(self):
        return [
            {
                "id": "a",
                "sha256": "a" * 64,
                "lang_id": "arb_Arab",
                "text": "NEVERPASSGOLD",
                "source": {"run": "before-all299"},
                "chunks": [
                    {"path": "/a0.wav", "sha256": "1", "start_s": 0, "end_s": 12},
                    {"path": "/a1.wav", "sha256": "2", "start_s": 12, "end_s": 13},
                ],
            },
            {
                "id": "b",
                "sha256": "b" * 64,
                "lang_id": "eng_Latn",
                "text": "NEVERPASSGOLD",
                "source": {"run": "experimental-all299"},
                "chunks": [{"path": "/b.wav", "sha256": "3", "start_s": 0, "end_s": 2}],
            },
            {
                "id": "c",
                "sha256": "c" * 64,
                "lang_id": "cmn_Hans",
                "text": "NEVERPASSGOLD",
                "source": {"run": "reference-controls"},
                "chunks": [{"path": "/c.wav", "sha256": "4", "start_s": 0, "end_s": 5}],
            },
        ]

    def test_language_alignment_and_original_chunk_order(self):
        pipeline = FakePipeline()
        cases = self.cases()
        rows, calls = prototype.transcribe(cases, pipeline, 2)
        self.assertEqual(calls[0]["group"][0], 0)
        result = {r["id"]: r for r in rows}
        self.assertEqual(result["a"]["transcript"], "heard-a0.wav heard-a1.wav")
        self.assertTrue(any(call["mixed_language_count"] == 2 for call in calls))
        seen = {
            path: language
            for call in pipeline.calls
            for path, language in zip(call["paths"], call["lang"])
        }
        self.assertEqual(
            seen,
            {
                "/a0.wav": "arb_Arab",
                "/a1.wav": "arb_Arab",
                "/b.wav": "eng_Latn",
                "/c.wav": "cmn_Hans",
            },
        )
        self.assertAlmostEqual(
            sum(row["asr_wall_s"] for row in rows),
            sum(call["transcribe_call_wall_s"] for call in calls),
        )

    def test_configs_have_independent_ids_and_no_cache(self):
        identifiers = [
            prototype.protocol_id(batch, "bfloat16", "pipeline", "weights")[0]
            for batch in [1, 2, 4]
        ]
        self.assertEqual(len(set(identifiers)), 3)
        for batch in [1, 2, 4]:
            pipeline = FakePipeline()
            rows, _ = prototype.transcribe(self.cases(), pipeline, batch)
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(len(call["paths"]) for call in pipeline.calls), 4)
            self.assertTrue(all(call["batch_size"] == batch for call in pipeline.calls))

    def test_unchanged_chunk_limit(self):
        cases = self.cases()
        cases[0]["chunks"][0]["end_s"] = 31
        with self.assertRaises(AssertionError):
            prototype.schedule(cases)


if __name__ == "__main__":
    unittest.main()
