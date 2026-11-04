"""CPU-only synthetic timings; no HTTP or inference."""

import unittest

from arrival_report import phase_report


class ArrivalAccounting(unittest.TestCase):
    def row(self, status, index, complete):
        return {
            "id": str(index),
            "status": status,
            "complete_ms": complete,
            "scheduled_offset_ms": index * 1000,
            "dispatched_from_barrier_ms": index * 1000,
            "completion_from_barrier_ms": index * 1000 + complete,
            "effective_input_sha256": str(index),
            "trace": {},
            "headers": {},
        }

    def test_rejection_cannot_meet_latency_target(self):
        rows = [self.row(200, 0, 100), self.row(429, 1, 1)]
        result = phase_report({"id": "fake", "kind": "staggered"}, rows, 2)
        self.assertEqual(result["per_request_full_body_ms"]["p95"], 100)
        self.assertFalse(result["all_offers_http200_and_p95_under_500ms"])
        self.assertFalse(result["all_offers_http200_and_cohort_under_2s"])
        self.assertIsNone(result["all200_last_scheduled_arrival_to_last_completion_ms"])

    def test_each_response_and_cohort_targets_are_different(self):
        rows = [self.row(200, 0, 1500), self.row(200, 1, 1500)]
        result = phase_report({"id": "fake", "kind": "staggered"}, rows, 2)
        self.assertTrue(result["all_offers_http200_and_each_under_2s"])
        self.assertFalse(result["all_offers_http200_and_cohort_under_2s"])
        self.assertEqual(
            result["all200_last_scheduled_arrival_to_last_completion_ms"], 1500
        )
        self.assertEqual(result["scheduled_arrival_span_ms"], 1000)


if __name__ == "__main__":
    unittest.main()
