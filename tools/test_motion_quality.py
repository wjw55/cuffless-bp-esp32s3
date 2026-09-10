import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion_quality import (  # noqa: E402
    MotionTrial,
    REVIEW_COLUMNS,
    _causal_imu,
    _health_reasons,
    _protocol_context,
    _window_features,
    file_sha256,
    finalize_dataset,
    load_config,
    prepare_trial,
)
from motion_study_protocol import MOTION_QUALITY_V1  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "motion_quality_v1.json"


def annotation_frame(participant="P001", session="motion_quality_v1", trial="motion_quality_001"):
    return pd.DataFrame(
        [
            {
                "participant_id": participant,
                "session_id": session,
                "trial_id": trial,
                "protocol": MOTION_QUALITY_V1.name,
                "block_index": index,
                "block_name": block.block_name,
                "scheduled_start_s": block.start_s,
                "scheduled_end_s": block.end_s,
                "activity_label": block.activity_label,
                "completion_status": "complete",
            }
            for index, block in enumerate(MOTION_QUALITY_V1.blocks, start=1)
        ]
    )


def write_trial(root: Path, *, health_error=False, sequence_error=False) -> MotionTrial:
    participant, session, trial_id = "P001", "motion_quality_v1", "motion_quality_001"
    prefix = root / f"{participant}_{session}_{trial_id}"
    ppg_time = 10_000.0 + np.arange(24_001) * 10.0
    seconds = (ppg_time - ppg_time[0]) / 1000.0
    ir = 60_000.0 + 1_200.0 * np.sin(2 * np.pi * 1.2 * seconds)
    red = 55_000.0 + 900.0 * np.sin(2 * np.pi * 1.2 * seconds + 0.1)
    ppg_seq = np.arange(len(ppg_time))
    if sequence_error:
        ppg_seq[100:] += 1
    ppg = pd.DataFrame({"sample_seq": ppg_seq, "timestamp_ms": ppg_time, "red": red, "ir": ir})

    # Offset and unequal sample rate deliberately exercise timestamp-based alignment.
    imu_time = 9_992.0 + np.arange(30_004) * 8.0
    imu_seconds = (imu_time - ppg_time[0]) / 1000.0
    x = np.zeros(len(imu_time))
    y = np.zeros(len(imu_time))
    z = np.full(len(imu_time), 256.0)
    moving = ((imu_seconds >= 30) & (imu_seconds < 60)) | ((imu_seconds >= 150) & (imu_seconds < 180))
    x[moving] = 30.0 * np.sin(2 * np.pi * 1.5 * imu_seconds[moving])
    imu = pd.DataFrame(
        {"imu_seq": np.arange(len(imu_time)), "timestamp_ms": imu_time, "x_raw": x, "y_raw": y, "z_raw": z}
    )

    ppg_path = prefix.with_name(prefix.name + "_ppg.csv")
    imu_path = prefix.with_name(prefix.name + "_imu.csv")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    annotation_path = prefix.with_name(prefix.name + "_activity_annotations.csv")
    ppg.to_csv(ppg_path, index=False)
    imu.to_csv(imu_path, index=False)
    annotation_frame(participant, session, trial_id).to_csv(annotation_path, index=False)
    metadata = {
        "subject_id": participant,
        "session_id": session,
        "trial_id": trial_id,
        "motion_study_protocol": MOTION_QUALITY_V1.name,
        "firmware_i2c_error_count": 1 if health_error else 0,
        "firmware_fifo_overflow_count": 0,
        "imu_firmware_i2c_error_count": 0,
        "imu_firmware_fifo_overflow_count": 0,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return MotionTrial(participant, session, trial_id, ppg_path, imu_path, metadata_path, annotation_path, metadata)


class ProtocolGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)
        self.annotations = annotation_frame()

    def test_edges_transitions_and_recovery_are_excluded(self):
        self.assertIn("recording_edge_guard", _protocol_context(0, 8, self.annotations, self.config)[1])
        self.assertEqual(_protocol_context(20, 28, self.annotations, self.config)[1], [])
        self.assertIn("cue_transition_guard", _protocol_context(24, 32, self.annotations, self.config)[1])
        self.assertIn("early_recovery_guard", _protocol_context(60, 68, self.annotations, self.config)[1])
        self.assertEqual(_protocol_context(68, 76, self.annotations, self.config)[1], [])

    def test_activity_is_context_only(self):
        context, reasons = _protocol_context(32, 40, self.annotations, self.config)
        self.assertEqual(reasons, [])
        self.assertEqual(context["activity_label"], "gentle_arm_motion")
        self.assertNotIn("activity_label", _window_features.__annotations__)


class FeatureAndSynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)

    def test_causal_activity_distinguishes_motion(self):
        count = 1600
        time = np.arange(count) * 10.0
        still = pd.DataFrame({"timestamp_ms": time, "x_raw": 0, "y_raw": 0, "z_raw": 256})
        moving = still.copy()
        moving["x_raw"] = 40 * np.sin(2 * np.pi * 1.5 * time / 1000.0)
        still_activity = _causal_imu(still, self.config)["activity"]
        moving_activity = _causal_imu(moving, self.config)["activity"]
        self.assertLess(float(np.median(still_activity[-800:])), 0.005)
        self.assertGreater(float(np.median(moving_activity[-800:])), 0.05)

    def test_contact_step_and_clipping_features_are_exposed(self):
        time = np.arange(800) * 0.01
        ir = 60_000.0 + 1000.0 * np.sin(2 * np.pi * 1.2 * time)
        ir[400:] += 5000.0
        ir[500] = (1 << 18) - 1
        ppg = pd.DataFrame(
            {"timestamp_ms": time * 1000, "ir": ir, "red": 55_000 + 800 * np.sin(2 * np.pi * 1.2 * time)}
        )
        imu_time = np.arange(-1, 801) * 0.01
        imu = pd.DataFrame(
            {"timestamp_ms": imu_time * 1000, "x_raw": 0, "y_raw": 0, "z_raw": 256}
        )
        features, _, _ = _window_features(
            ppg, imu, _causal_imu(imu, self.config), np.arange(800), np.arange(802), self.config
        )
        self.assertGreater(features["ppg_max_step_counts"], 100_000)
        self.assertGreater(features["ppg_clipping_fraction"], 0)

    def test_every_sensor_health_counter_is_rejected(self):
        keys = (
            "firmware_i2c_error_count",
            "firmware_fifo_overflow_count",
            "firmware_fifo_overflow_recovery_count",
            "imu_firmware_i2c_error_count",
            "imu_firmware_fifo_overflow_count",
        )
        for key in keys:
            with self.subTest(key=key):
                reasons = _health_reasons({key: 1})
                self.assertEqual(reasons, [f"sensor_health_error:{key}=1"])

    def test_interpolation_refuses_extrapolation(self):
        time = np.arange(800) * 0.01
        ppg = pd.DataFrame(
            {"timestamp_ms": time * 1000, "ir": 60_000 + 1000 * np.sin(2 * np.pi * time), "red": 55_000 + 800 * np.sin(2 * np.pi * time)}
        )
        imu = pd.DataFrame(
            {"timestamp_ms": 10 + time * 1000, "x_raw": 0, "y_raw": 0, "z_raw": 256}
        )
        derived = _causal_imu(imu, self.config)
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            _window_features(ppg, imu, derived, np.arange(800), np.arange(800), self.config)

    def test_deterministic_windows_and_unequal_rate_alignment(self):
        with TemporaryDirectory() as directory:
            rows, report, _ = prepare_trial(write_trial(Path(directory)), self.config)
        self.assertEqual(len(rows), 59)
        self.assertEqual(rows["start_s"].tolist(), [float(value) for value in range(0, 233, 4)])
        self.assertEqual(report["recording_rejection_reasons"], [])
        self.assertEqual(report["reviewable_window_count"], 35)
        self.assertAlmostEqual(report["ppg_rate_hz"], 100.0)
        self.assertAlmostEqual(report["imu_rate_hz"], 125.0)
        self.assertTrue((rows.loc[rows["reviewable"], "window_rejection_reasons"] == "").all())

    def test_sequence_and_sensor_health_faults_reject_recording(self):
        with TemporaryDirectory() as directory:
            trial = write_trial(Path(directory), health_error=True, sequence_error=True)
            rows, report, _ = prepare_trial(trial, self.config)
        self.assertIn("ppg_missing_or_non_monotonic_sequence", report["recording_rejection_reasons"])
        self.assertTrue(any(reason.startswith("sensor_health_error") for reason in report["recording_rejection_reasons"]))
        self.assertFalse(rows["reviewable"].any())


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)

    def _write_run(self, root: Path, reviews: list[dict[str, str]]) -> None:
        features = pd.DataFrame(
            {
                "window_id": ["w001", "w002", "w003", "excluded"],
                "reviewable": [True, True, True, False],
                "activity_label": ["still", "gentle_arm_motion", "still", "still"],
                "ppg_ac_rms": [1.0, 2.0, 3.0, 4.0],
            }
        )
        feature_path = root / "window_features.csv"
        features.to_csv(feature_path, index=False)
        pd.DataFrame(reviews, columns=REVIEW_COLUMNS).to_csv(root / "window_review.csv", index=False)
        (root / "study_report.json").write_text(
            json.dumps({"window_features_sha256": file_sha256(feature_path)}), encoding="utf-8"
        )

    def test_finalization_maps_labels_without_using_activity(self):
        reviews = [
            {"window_id": "w001", "reviewed_label": "clean", "reviewer": "A", "review_notes": ""},
            {"window_id": "w002", "reviewed_label": "motion_corrupted", "reviewer": "A", "review_notes": ""},
            {"window_id": "w003", "reviewed_label": "uncertain", "reviewer": "A", "review_notes": ""},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, reviews)
            report = finalize_dataset(self.config, root)
            result = pd.read_csv(root / "reviewed_windows.csv")
        self.assertEqual(report["supervised_training_window_count"], 2)
        self.assertEqual(result.loc[result["window_id"] == "w001", "usable"].iloc[0], 1)
        self.assertEqual(result.loc[result["window_id"] == "w002", "usable"].iloc[0], 0)
        self.assertTrue(np.isnan(result.loc[result["window_id"] == "w003", "usable"].iloc[0]))
        self.assertEqual(result.loc[result["window_id"] == "w002", "reviewed_label"].iloc[0], "motion_corrupted")
        self.assertEqual(result.loc[result["window_id"] == "w002", "artifact_subtype"].iloc[0], "motion")

    def test_finalization_rejects_bad_reviews_and_feature_changes(self):
        valid = [
            {"window_id": "w001", "reviewed_label": "clean", "reviewer": "A", "review_notes": ""},
            {"window_id": "w002", "reviewed_label": "contact_corrupted", "reviewer": "A", "review_notes": ""},
            {"window_id": "w003", "reviewed_label": "uncertain", "reviewer": "A", "review_notes": ""},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "duplicate": [valid[0], valid[0], valid[2]],
                "missing": valid[:2],
                "invalid": [valid[0], valid[1], {**valid[2], "reviewed_label": "bad"}],
            }
            for name, reviews in cases.items():
                with self.subTest(name=name):
                    self._write_run(root, reviews)
                    with self.assertRaises(ValueError):
                        finalize_dataset(self.config, root)
            self._write_run(root, valid)
            with (root / "window_features.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                finalize_dataset(self.config, root)


if __name__ == "__main__":
    unittest.main()
