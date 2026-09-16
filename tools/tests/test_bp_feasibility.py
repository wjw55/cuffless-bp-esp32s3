import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bp_core.datasets import Recording
from bp_core.feasibility import (
    assess_gap_admission,
    find_gap_events,
    process_gap_tolerant_frames,
)
from test_bp_pipeline import minimal_config


def policy():
    return {
        "schema_version": 1,
        "policy_name": "test_feasibility",
        "gap_detection_threshold_ms": 40.0,
        "maximum_gap_events": 1,
        "maximum_gap_span_ms": 200.0,
        "gap_synchronization_tolerance_ms": 50.0,
        "gap_guard_seconds": 1.0,
        "minimum_accepted_windows_per_occasion": 3,
        "minimum_unique_clean_coverage_seconds": 24.0,
        "require_upper_arm_analyzer_acceptance": False,
        "recovery_protocol": {
            "trial_name_contains": "recovery",
            "planned_movement_start_seconds": 90.0,
            "planned_movement_end_seconds": 110.0,
            "movement_guard_seconds": 1.0,
        },
    }


def recording(name="stationary_001", sbp=120, dbp=70):
    return Recording(
        "local_upper_arm", "P001", "session", name, f"occasion:{name}", "1",
        "upper_arm", 100.0, "", "", sbp, dbp, 70.0, "omron", "after_ppg", "reject"
    )


def frames(seconds=70, gap_at=None, gap_samples=10, imu_gap_at=None):
    count = int(seconds * 100) + 1
    ppg_seq = np.arange(count)
    ppg_time = np.arange(count) * 10.0
    imu_seq = np.arange(count)
    imu_time = np.arange(count) * 10.0
    if gap_at is not None:
        ppg_seq[gap_at:] += gap_samples
        ppg_time[gap_at:] += gap_samples * 10.0
    if imu_gap_at is None:
        imu_gap_at = gap_at
    if imu_gap_at is not None:
        imu_seq[imu_gap_at:] += gap_samples
        imu_time[imu_gap_at:] += gap_samples * 10.0
    t = ppg_time / 1000.0
    ppg = pd.DataFrame({
        "sample_seq": ppg_seq,
        "timestamp_ms": ppg_time,
        "red": 80000 + 1000 * np.sin(2 * np.pi * 1.2 * t),
        "ir": 100000 + 2000 * np.sin(2 * np.pi * 1.2 * t),
    })
    imu = pd.DataFrame({
        "imu_seq": imu_seq,
        "timestamp_ms": imu_time,
        "x_raw": 0,
        "y_raw": 0,
        "z_raw": 256,
    })
    return ppg, imu


def metadata():
    return {
        "firmware_i2c_error_count": 0,
        "firmware_fifo_overflow_count": 0,
        "imu_firmware_i2c_error_count": 0,
        "imu_firmware_fifo_overflow_count": 0,
        "firmware_motion_updates": [
            {"timestamp_ms": value, "status": "still"} for value in range(0, 210001, 1000)
        ],
    }


class FeasibilityPolicyTests(unittest.TestCase):
    def config(self):
        config = minimal_config(Path("."))
        config["quality"].update(
            minimum_beats_per_window=4,
            local_contact_threshold_counts=50000,
            motion_margin_seconds=1,
            contact_margin_seconds=2,
        )
        return config

    def test_brief_synchronized_gap_is_admitted_and_never_crossed(self):
        ppg, imu = frames(gap_at=3500)
        result = process_gap_tolerant_frames(recording(), ppg, imu, metadata(), self.config(), policy())
        self.assertFalse(result.admission_reasons)
        self.assertEqual(result.occasion["gap_event_count"], 1)
        gap = result.gaps[result.gaps.stream == "ppg"].iloc[0]
        crossing = result.segments[
            (result.segments.start_s * 1000 < gap.after_timestamp_ms)
            & (result.segments.end_s * 1000 > gap.before_timestamp_ms)
        ]
        self.assertTrue(crossing.empty)

    def test_overlapping_windows_do_not_inflate_clean_coverage(self):
        ppg, imu = frames(seconds=32)
        result = process_gap_tolerant_frames(recording(), ppg, imu, metadata(), self.config(), policy())
        accepted = result.segments[result.segments.accepted == True]  # noqa: E712
        summed = float((accepted.end_s - accepted.start_s).sum())
        self.assertLessEqual(result.occasion["unique_clean_coverage_s"], 32.0)
        self.assertLess(result.occasion["unique_clean_coverage_s"], summed)

    def test_unsynchronized_or_long_gap_rejects(self):
        ppg, imu = frames(gap_at=3000, imu_gap_at=3200)
        _, _, reasons, _ = assess_gap_admission(ppg, imu, metadata(), policy())
        self.assertTrue(any("unsynchronized_gap_time" in reason for reason in reasons))
        ppg, imu = frames(gap_at=3000, gap_samples=30)
        _, _, reasons, _ = assess_gap_admission(ppg, imu, metadata(), policy())
        self.assertTrue(any("gap_too_long" in reason for reason in reasons))

    def test_health_error_rejects_but_missing_counter_is_warning(self):
        ppg, imu = frames()
        broken = metadata(); broken["firmware_i2c_error_count"] = 1
        _, _, reasons, _ = assess_gap_admission(ppg, imu, broken, policy())
        self.assertIn("sensor_health_error:firmware_i2c_error_count=1", reasons)
        incomplete = metadata(); incomplete.pop("firmware_i2c_error_count")
        _, _, reasons, warnings = assess_gap_admission(ppg, imu, incomplete, policy())
        self.assertFalse(reasons)
        self.assertIn("missing_sensor_health_counter:firmware_i2c_error_count", warnings)

    def test_non_monotonic_timestamps_reject(self):
        ppg, _ = frames()
        ppg.loc[20, "timestamp_ms"] = ppg.loc[19, "timestamp_ms"]
        with self.assertRaisesRegex(ValueError, "non_monotonic_ppg_timestamps"):
            find_gap_events(ppg, stream="ppg", sequence_column="sample_seq", threshold_ms=40)

    def test_empty_recording_is_rejected_cleanly(self):
        ppg, imu = frames()
        result = process_gap_tolerant_frames(
            recording(), ppg.iloc[0:0], imu, metadata(), self.config(), policy()
        )
        self.assertFalse(result.occasion["occasion_usable"])
        self.assertIn("insufficient_ppg_or_imu_samples", result.admission_reasons)

    def test_recovery_windows_exclude_planned_movement(self):
        ppg, imu = frames(seconds=140)
        result = process_gap_tolerant_frames(recording("recovery_001"), ppg, imu, metadata(), self.config(), policy())
        self.assertFalse(((result.segments.start_s < 111) & (result.segments.end_s > 89)).any())
        self.assertEqual(set(result.segments.protocol_phase), {"still_baseline", "still_recovery"})

    def test_quality_decision_is_independent_of_bp_label_and_strict_config_is_unchanged(self):
        ppg, imu = frames(seconds=32)
        config = self.config(); before = copy.deepcopy(config)
        first = process_gap_tolerant_frames(recording(sbp=90, dbp=55), ppg, imu, metadata(), config, policy())
        second = process_gap_tolerant_frames(recording(sbp=160, dbp=100), ppg, imu, metadata(), config, policy())
        self.assertEqual(first.occasion["occasion_usable"], second.occasion["occasion_usable"])
        self.assertEqual(first.occasion["occasion_rejection_reasons"], second.occasion["occasion_rejection_reasons"])
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
