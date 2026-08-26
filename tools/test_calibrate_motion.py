import math
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_motion import (
    CLEAN_STILL_INTERVALS,
    MOVEMENT_INTERVALS,
    acceptance_gates,
    calculate_threshold,
    causal_activity,
    classify_trial,
    in_intervals,
    main,
    percentile,
    rolling_rms,
)


def synthetic_trial(still_value=0.005, motion_value=0.100, transition_value=None):
    trial = []
    for index in range(9000):
        time_s = index / 100.0
        value = motion_value if in_intervals(time_s, MOVEMENT_INTERVALS) else still_value
        if transition_value is not None and in_intervals(
            time_s, ((18.0, 20.0), (30.0, 32.0), (45.0, 47.0), (55.0, 57.0), (70.0, 72.0), (80.0, 82.0))
        ):
            value = transition_value
        trial.append((time_s, value))
    return trial


class MotionMathTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([0.0, 1.0, 2.0], 0.25), 0.5)

    def test_rolling_rms_is_causal_and_drops_oldest_sample(self):
        prefix = [1.0] * 100
        full = rolling_rms(prefix + [9.0], window_samples=100)

        self.assertEqual(full[:100], rolling_rms(prefix, window_samples=100))
        self.assertAlmostEqual(full[99], 1.0)
        self.assertAlmostEqual(full[100], math.sqrt(1.08))

    def test_threshold_search_selects_lowest_passing_integer_mg(self):
        trial = synthetic_trial()

        result = calculate_threshold([trial, trial, trial])

        self.assertEqual(result["config_motion_threshold_mg"], 8)
        self.assertTrue(result["passes_all_gates"])
        self.assertGreaterEqual(result["stationary_still_fraction"], 0.95)
        self.assertEqual(result["movement_blocks_detected"], 9)

    def test_guarded_margins_do_not_change_threshold(self):
        clean = synthetic_trial()
        noisy_margins = synthetic_trial(transition_value=0.300)

        clean_result = calculate_threshold([clean, clean, clean])
        noisy_result = calculate_threshold([noisy_margins, noisy_margins, noisy_margins])

        self.assertEqual(
            noisy_result["config_motion_threshold_mg"], clean_result["config_motion_threshold_mg"]
        )

    def test_no_threshold_is_returned_when_motion_is_indistinguishable(self):
        trial = synthetic_trial(still_value=0.010, motion_value=0.010)

        result = calculate_threshold([trial, trial, trial])

        self.assertIsNone(result["config_motion_threshold_mg"])
        self.assertFalse(result["passes_all_gates"])

    def test_nine_of_ten_blocks_passes_movement_gate(self):
        self.assertTrue(acceptance_gates(0.95, 9, 10)["passes_all_gates"])
        self.assertFalse(acceptance_gates(0.95, 8, 10)["passes_movement_gate"])

    def test_hysteresis_uses_moving_and_still_dwell(self):
        activity = [(index / 100.0, 0.01) for index in range(300)]
        activity.extend((3.0 + index / 100.0, 0.10) for index in range(30))
        activity.extend((3.3 + index / 100.0, 0.01) for index in range(120))

        classified = classify_trial(activity, moving_threshold_g=0.05)

        self.assertEqual(next(status for time_s, status in classified if time_s >= 3.21), "moving")
        self.assertEqual(classified[-1][1], "still")

    def test_causal_activity_reacts_to_axis_motion(self):
        rows = []
        for index in range(300):
            x_raw = 100 if 200 <= index < 250 and index % 2 == 0 else 0
            rows.append({"timestamp_ms": index * 10, "x_raw": x_raw, "y_raw": 0, "z_raw": 256})

        activity = causal_activity(rows)

        self.assertGreater(max(value for _, value in activity[200:260]), max(value for _, value in activity[:100]))

    def test_python_activity_matches_firmware_algorithm_reference(self):
        rows = [
            {
                "timestamp_ms": index * 10,
                "x_raw": (index % 17) - 8,
                "y_raw": (index % 11) - 5,
                "z_raw": 256 + (index % 7) - 3,
            }
            for index in range(250)
        ]

        python_values = [value for _, value in causal_activity(rows)]
        gravity = None
        window = [0.0] * 100
        window_sum = 0.0
        window_index = 0
        window_count = 0
        firmware_values = []
        for row in rows:
            axes = [row[f"{axis}_raw"] * 0.0039 for axis in ("x", "y", "z")]
            if gravity is None:
                gravity = list(axes)
            gravity = [old + 0.01 * (new - old) for old, new in zip(gravity, axes)]
            squared = sum((new - old) ** 2 for new, old in zip(axes, gravity))
            if window_count == 100:
                window_sum -= window[window_index]
            else:
                window_count += 1
            window[window_index] = squared
            window_sum += squared
            window_index = (window_index + 1) % 100
            firmware_values.append(math.sqrt(max(window_sum / window_count, 0.0)))

        for python_value, firmware_value in zip(python_values, firmware_values):
            self.assertAlmostEqual(python_value, firmware_value, places=12)

    def test_clean_stationary_intervals_exclude_recovery_margins(self):
        self.assertFalse(in_intervals(31.0, CLEAN_STILL_INTERVALS))
        self.assertTrue(in_intervals(34.0, CLEAN_STILL_INTERVALS))


class MotionCliTests(unittest.TestCase):
    def test_requires_three_trials(self):
        with TemporaryDirectory() as tmpdir:
            with redirect_stdout(StringIO()):
                result = main([str(Path(tmpdir) / "missing.csv")])
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
