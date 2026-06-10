from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ui.results_browser import ResultRepository, infer_score


class ResultRepositoryTests(unittest.TestCase):
    def test_loads_reports_and_ignores_summaries_and_old_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active = root / "active"
            active.mkdir()
            payload = {
                "model": "model-a",
                "benchmark": "benchmark-a",
                "report": {
                    "results": [
                        {"index": 1, "correct": True, "stats": {"success": True}},
                        {"index": 2, "correct": False, "stats": {"success": True}},
                    ],
                    "stats": {
                        "number_of_benchmark_samples_completed": 2,
                        "success_count": 2,
                        "failure_count": 0,
                        "wall_clock_time_per_sample_seconds_mean": 1.25,
                    },
                },
            }
            (active / "model-a_benchmark-a.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (active / "run_summary.json").write_text("[]", encoding="utf-8")
            old = root / "old_results"
            old.mkdir()
            (old / "old.json").write_text(json.dumps(payload), encoding="utf-8")

            runs = ResultRepository(root).load()

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].group, "active")
            self.assertEqual(runs[0].score_label, "Accuracy")
            self.assertEqual(runs[0].score, 0.5)
            self.assertEqual(runs[0].seconds_per_sample, 1.25)

    def test_skips_invalid_json_and_non_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "broken.json").write_text("{", encoding="utf-8")
            (root / "other.json").write_text('{"value": 1}', encoding="utf-8")
            self.assertEqual(ResultRepository(root).load(), [])


class ScoreInferenceTests(unittest.TestCase):
    def test_prefers_detection_f1(self) -> None:
        label, value, is_error = infer_score(
            [{"f1": 0.25, "correct": False}, {"f1": 0.75, "correct": True}],
            {},
        )
        self.assertEqual(label, "F1")
        self.assertEqual(value, 0.5)
        self.assertFalse(is_error)

    def test_uses_mae_as_lower_is_better(self) -> None:
        label, value, is_error = infer_score(
            [{"absolute_error": 2}, {"absolute_error": 4}],
            {},
        )
        self.assertEqual(label, "MAE")
        self.assertEqual(value, 3.0)
        self.assertTrue(is_error)


if __name__ == "__main__":
    unittest.main()
