import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_feasibility import attach_reviews, contained, review_errors, summarize, union_ms


def windows():
    return pd.DataFrame(dict(window_id=["a", "b"], start_timestamp_ms=[0., 4000.],
        end_timestamp_ms=[8000., 12000.], imu_severity=["stationary", "mild"],
        ppg_candidate=[True, True], signal_eligible=[True, False],
        classifier_accept=pd.Series([True, pd.NA], dtype="boolean"),
        reviewed_label=["motion_corrupted", "clean"]))


class FeasibilityTests(unittest.TestCase):
    def test_overlap_union_clipped_to_interval(self):
        self.assertEqual(union_ms([(0, 8), (4, 12), (15, 20)], 2, 17), 12)

    def test_time_partition_does_not_double_count_windows(self):
        result = summarize(windows(), 0, 16000)
        self.assertEqual(result["still_percent"], 37.5)
        self.assertEqual(result["moving_percent"], 37.5)
        self.assertEqual(result["unknown_percent"], 25)
        self.assertEqual(result["ppg_candidate_coverage_percent"], 75)
        self.assertEqual(result["signal_eligible_coverage_percent"], 50)

    def test_guard_excludes_crossing_windows(self):
        self.assertEqual(contained(windows(), 1000, 12000).window_id.tolist(), ["b"])
        result = summarize(windows(), 1000, 11000)
        self.assertEqual(result["windows"], 0)
        self.assertEqual(result["unknown_percent"], 100)

    def test_contact_flags_integrate_timestamps_not_rows(self):
        ppg = pd.DataFrame({"timestamp_ms": [0., 1000., 4000., 12000.]})
        result = summarize(windows(), 0, 12000, ppg, {"clipping_flag": np.array([False, True, False, False])})
        self.assertEqual(result["clipping_flag_time_percent"], 25)

    def test_review_denominators_include_unscored_counts(self):
        result = review_errors(windows(), "classifier_accept")
        self.assertEqual(result["false_acceptance"]["rate"], 1)
        self.assertEqual(result["false_rejection"]["unscored_windows"], 1)
        self.assertIsNone(result["false_rejection"]["rate"])
        frame = windows()
        frame["reviewed_label"] = "uncertain"
        self.assertEqual(review_errors(frame, "classifier_accept")["false_acceptance"]["reviewed_windows"], 0)

    def test_review_join_requires_identity_and_reviewer(self):
        frame = windows().drop(columns="reviewed_label")
        reviews = pd.DataFrame(dict(window_id=["a"], reviewed_label=["clean"], reviewer=["reviewer"]))
        self.assertEqual(len(attach_reviews(frame, reviews)), 2)
        for invalid in (pd.concat([reviews, reviews]), reviews.assign(window_id="other"),
                        reviews.assign(reviewer=""), reviews.assign(reviewed_label="125/65")):
            with self.assertRaises(ValueError):
                attach_reviews(frame, invalid)

    def test_invalid_reporting_interval(self):
        for start, end in [(1, 1), (2, 1), (0, float("nan"))]:
            with self.assertRaises(ValueError):
                summarize(windows(), start, end)


if __name__ == "__main__":
    unittest.main()
