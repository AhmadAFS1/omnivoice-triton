"""CPU-only selection contracts; no inference, WAVs or transcript scores."""

import unittest

from select_future_priority import choose


class FutureSelectionTests(unittest.TestCase):
    def fixture(self):
        inventory = [{"language": code} for code in ("ar", "en", "so")]
        unique = [{"id": f"audio-{code}"} for code in ("ar", "en", "control")]
        logical = [
            {
                "id": code,
                "original_id": code,
                "kind": "request",
                "arm": "candidate-step-frontier-16",
                "language": code,
                "asr_input_id": None if code == "so" else f"audio-{code}",
                "http_status": 200,
                "issues": ["zero_frame"] if code == "so" else [],
            }
            for code in ("ar", "en", "so")
        ]
        logical += [
            {
                "id": "control",
                "original_id": "ref",
                "kind": "reference_control",
                "arm": "reference",
                "language": "en",
                "asr_input_id": "audio-control",
            }
        ]
        # An unrelated HTTP cached alias must not create another ASR evaluation.
        logical += [
            {
                "id": "cached",
                "original_id": "cached",
                "kind": "request",
                "arm": "cache-hot",
                "language": "ar",
                "asr_input_id": "audio-ar",
            }
        ]
        return logical, unique, inventory

    def test_invalid_route_remains_selected_but_not_an_asr_input(self):
        selected, chosen, summary = choose(*self.fixture(), "priority104-step16")
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(summary["request_routes"], ["ar", "en", "so"])
        self.assertEqual(summary["unscorable_logical_outcomes"][0]["id"], "so")
        selected_all, _, _ = choose(*self.fixture(), "all")
        self.assertEqual(len(selected_all), 3)

    def test_missing_route_is_not_silently_replaced(self):
        logical, unique, inventory = self.fixture()
        logical = [row for row in logical if row["id"] != "en"]
        with self.assertRaisesRegex(AssertionError, "every inventory route"):
            choose(logical, unique, inventory, "priority104-step16")


if __name__ == "__main__":
    unittest.main()
