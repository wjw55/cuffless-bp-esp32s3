import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bp_core.datasets import Recording, SignalData, audit_datasets, discover_recordings
from bp_core.features import (
    SignalQualityError,
    aggregate_recording_features,
    build_occasion_features,
    extract_window_features,
    process_recording,
    process_signal,
)
from bp_core.models import (
    build_personalized_examples,
    evaluate_personalized_models,
    evaluate_single_subject_models,
    prepare_single_subject_split,
)


def minimal_config(root: Path) -> dict:
    return {
        "schema_version": 1,
        "random_seed": 7,
        "output_root": str(root / "processed"),
        "datasets": {
            "one_month_wrist": {
                "enabled": True,
                "root": str(root / "one_month"),
                "sensor_site": "wrist",
            },
            "ppg_bp": {
                "enabled": False,
                "root": str(root / "ppg_bp"),
                "optional": True,
            },
            "local_upper_arm": {
                "enabled": True,
                "raw_dir": str(root / "raw"),
                "labels_dir": str(root / "labels"),
                "required_profile": "upper_arm_experimental",
                "optional": True,
            },
        },
        "signal": {
            "window_seconds": 8.0,
            "window_step_seconds": 4.0,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 8.0,
            "filter_order": 4,
            "template_samples": 128,
            "minimum_hr_bpm": 40.0,
            "maximum_hr_bpm": 180.0,
        },
        "quality": {
            "minimum_sample_completeness": 0.995,
            "minimum_beats_per_window": 4,
            "maximum_interval_cv": 0.2,
            "minimum_template_correlation": 0.6,
            "maximum_clipped_fraction": 0.005,
            "minimum_accepted_windows_per_occasion": 3,
            "minimum_unique_clean_coverage_seconds": 16.0,
            "require_upper_arm_analyzer_acceptance": False,
            "local_contact_threshold_counts": 50000.0,
            "motion_margin_seconds": 1.0,
            "contact_margin_seconds": 2.0,
        },
        "models": {
            "outer_evaluation": "leave_one_participant_out",
            "inner_max_splits": 3,
            "bootstrap_iterations": 10,
            "ridge_alphas": [1.0],
            "elastic_net_alphas": [0.01],
            "elastic_net_l1_ratios": [0.5],
            "hist_gradient_boosting_learning_rates": [0.1],
            "hist_gradient_boosting_max_leaf_nodes": [5],
            "hist_gradient_boosting_min_samples_leaf": [2],
        },
    }


def synthetic_ppg(duration_s=24.0, sample_rate_hz=100.0, bpm=72.0):
    time_s = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
    phase = 2.0 * np.pi * (bpm / 60.0) * time_s
    ir = 90000.0 + 6000.0 * (np.sin(phase) + 0.25 * np.sin(2.0 * phase))
    red = 70000.0 + 3000.0 * (np.sin(phase) + 0.20 * np.sin(2.0 * phase))
    return time_s, ir, red


def write_local_trial(root: Path, subject: str, trial: str, sbp: float, dbp: float, moving=False):
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    time_s, ir, red = synthetic_ppg()
    prefix = f"{subject}_session_{trial}"
    ppg_path = raw / f"{prefix}_ppg.csv"
    pd.DataFrame(
        {
            "sample_seq": np.arange(len(time_s)),
            "timestamp_ms": np.round(time_s * 1000).astype(int),
            "red": red,
            "ir": ir,
        }
    ).to_csv(ppg_path, index=False)
    updates = []
    if moving:
        updates = [
            {"timestamp_ms": 5000, "status": "moving"},
            {"timestamp_ms": 9000, "status": "still"},
        ]
    metadata = {
        "subject_id": subject,
        "session_id": "session",
        "trial_id": trial,
        "ppg_profile": "upper_arm_experimental",
        "sensor_location": "upper_arm",
        "recording_start_time": f"2026-01-{int(trial) + 1:02d}T09:00:00",
        "output_csv_path": str(ppg_path),
        "systolic_mmHg": sbp,
        "diastolic_mmHg": dbp,
        "approximate_sampling_rate_hz": 100.0,
        "firmware_motion_updates": updates,
    }
    (raw / f"{prefix}_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def single_subject_occasions(count=17, calibration_index=0):
    rows = []
    for occasion in range(count):
        rows.append(
            {
                "dataset_id": "local_upper_arm",
                "participant_id": "test",
                "session_id": "personal_development",
                "label_group_id": f"test:o{occasion:02d}",
                "chronological_order": f"2026-01-{occasion + 1:02d}T09:00:00+08:00",
                "sensor_site": "upper_arm",
                "sbp": 110.0 + 2.0 * occasion,
                "dbp": 70.0 + occasion,
                "reference_hr": 60.0 + occasion,
                "calibration_occasion": occasion == calibration_index,
                "occasion_usable": True,
                "median__feature__pulse_rate_bpm": 60.0 + occasion,
                "median__feature__median_ibi_s": 60.0 / (60.0 + occasion),
                "median__feature__rise_time_s": 0.20 + 0.01 * occasion,
                "iqr__feature__rise_time_s": 0.01 + 0.001 * occasion,
            }
        )
    return pd.DataFrame(rows)


class BPPipelineTests(unittest.TestCase):
    @staticmethod
    def quality_recording() -> Recording:
        return Recording(
            dataset_id="local_upper_arm",
            participant_id="P001",
            session_id="quality",
            recording_id="quality_001",
            label_group_id="local_upper_arm:P001:quality:quality_001",
            chronological_order="2026-01-01T09:00:00",
            sensor_site="upper_arm",
            sample_rate_hz=100.0,
            ppg_path="",
            imu_path=None,
            sbp=120.0,
            dbp=80.0,
            reference_hr=70.0,
            label_source="omron",
            label_timing="after_ppg",
            quality_status="good",
        )

    @staticmethod
    def quality_signal(duration_s=70.0, metadata=None) -> SignalData:
        time_s, ir, red = synthetic_ppg(duration_s)
        return SignalData(
            time_s=time_s,
            ir=ir,
            red=red,
            acceleration_m_s2=np.column_stack(
                [np.zeros(len(time_s)), np.zeros(len(time_s)), np.full(len(time_s), 9.80665)]
            ),
            sample_sequence=np.arange(len(time_s), dtype=float),
            metadata=dict(metadata or {}),
        )

    def test_extracts_normalized_morphology_from_synthetic_ppg(self):
        with TemporaryDirectory() as tmp:
            config = minimal_config(Path(tmp))
            time_s, ir, red = synthetic_ppg(duration_s=8.0)

            features, diagnostics = extract_window_features(
                time_s, ir, red, 100.0, config["signal"], config["quality"]
            )

            self.assertIsNotNone(features)
            self.assertAlmostEqual(features["pulse_rate_bpm"], 72.0, delta=1.0)
            self.assertGreater(features["template_correlation"], 0.95)
            self.assertIn(diagnostics["polarity"], {-1, 1})

    def test_one_month_adapter_uses_raw_headers_and_audits_processed_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = minimal_config(root)
            folder = root / "one_month" / "A1" / "week1" / "session1" / "2026_01_01_09_00_00"
            folder.mkdir(parents=True)
            time_s, ir, red = synthetic_ppg(duration_s=10.0, sample_rate_hz=100.0)
            pd.DataFrame(
                {
                    "Time [sec]": time_s,
                    "Acceleration X [m/sec2]": np.zeros(len(time_s)),
                    "Acceleration Y [m/sec2]": np.zeros(len(time_s)),
                    "Acceleration Z [m/sec2]": np.full(len(time_s), 9.80665),
                    "PPG Red [-]": red,
                    "PPG IR [-]": ir,
                    "SBP [mmHg]": np.full(len(time_s), 120),
                    "DBP [mmHg]": np.full(len(time_s), 78),
                    "HR [bpm]": np.full(len(time_s), 72),
                }
            ).to_csv(folder / "ESP00001(PPG-IMU).csv", index=False)
            values = list(range(250)) + [72, 120, 78]
            (folder / "processed.csv").write_text(",".join(map(str, values)) + "\n", encoding="utf-8")
            (folder / "processed_feature.csv").write_text(",".join(map(str, values)) + "\n", encoding="utf-8")

            recordings, statuses = discover_recordings(config)
            audit = audit_datasets(config, recordings, statuses)

            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0].sbp, 120)
            self.assertAlmostEqual(recordings[0].sample_rate_hz, 100.0)
            self.assertEqual(audit["processed_schema_checks"]["processed.csv"]["column_count"], 253)
            self.assertFalse(audit["processed_schema_checks"]["processed_feature.csv"]["training_allowed"])
            self.assertTrue(audit["processed_schema_checks"]["processed_label_order"]["matches_hr_sbp_dbp"])

    def test_conflicting_local_label_is_fatal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = minimal_config(root)
            (root / "one_month").mkdir()
            write_local_trial(root, "P001", "001", 120, 78)
            labels = root / "labels"
            labels.mkdir()
            pd.DataFrame(
                [{"subject": "P001", "session": "session", "trial_id": "001", "sbp": 130, "dbp": 78}]
            ).to_csv(labels / "session_labels.csv", index=False)

            _, statuses = discover_recordings(config)

            local = next(item for item in statuses if item["dataset_id"] == "local_upper_arm")
            self.assertEqual(local["status"], "error")
            self.assertIn("Conflicting local systolic_mmHg", local["error"])

    def test_motion_updates_reject_overlapping_windows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = minimal_config(root)
            (root / "one_month").mkdir()
            write_local_trial(root, "P001", "001", 120, 78, moving=True)
            recordings, _ = discover_recordings(config)

            rows = process_recording(recordings[0], config)

            self.assertTrue(any("motion" in row["rejection_reason"] for row in rows))
            self.assertTrue(any(row["accepted"] for row in rows))

    def test_recording_rejects_missing_and_non_monotonic_sequences_and_timestamps(self):
        config = minimal_config(Path("."))
        cases = []
        missing_sequence = self.quality_signal()
        missing_sequence.sample_sequence[100:] += 1
        cases.append((missing_sequence, "missing_ppg_sequences"))
        backwards_sequence = self.quality_signal()
        backwards_sequence.sample_sequence[100] = backwards_sequence.sample_sequence[99]
        cases.append((backwards_sequence, "non_monotonic_ppg_sequences"))
        missing_timestamp = self.quality_signal()
        missing_timestamp.time_s[100:] += 0.02
        cases.append((missing_timestamp, "missing_ppg_timestamps"))
        backwards_timestamp = self.quality_signal()
        backwards_timestamp.time_s[100] = backwards_timestamp.time_s[99]
        cases.append((backwards_timestamp, "non_monotonic_ppg_timestamps"))

        for signal, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(SignalQualityError, reason):
                process_signal(self.quality_recording(), signal, config)

    def test_recording_rejects_every_ppg_and_imu_health_counter(self):
        config = minimal_config(Path("."))
        counters = (
            "firmware_i2c_error_count",
            "firmware_fifo_overflow_count",
            "imu_firmware_i2c_error_count",
            "imu_firmware_fifo_overflow_count",
        )
        for counter in counters:
            with self.subTest(counter=counter), self.assertRaisesRegex(SignalQualityError, counter):
                process_signal(
                    self.quality_recording(),
                    self.quality_signal(metadata={counter: 1}),
                    config,
                )

    def test_local_upper_arm_waveform_must_pass_stricter_analyzer(self):
        config = minimal_config(Path("."))
        config["quality"]["require_upper_arm_analyzer_acceptance"] = True
        first = self.quality_signal(duration_s=45.0)
        second = self.quality_signal(duration_s=45.0)
        second_time, second_ir, second_red = synthetic_ppg(duration_s=45.0, bpm=102.0)
        second.time_s = second_time + first.time_s[-1] + 0.01
        second.ir = second_ir
        second.red = second_red
        second.sample_sequence += len(first.time_s)
        signal = SignalData(
            time_s=np.concatenate([first.time_s, second.time_s]),
            ir=np.concatenate([first.ir, second.ir]),
            red=np.concatenate([first.red, second.red]),
            acceleration_m_s2=np.vstack([first.acceleration_m_s2, second.acceleration_m_s2]),
            sample_sequence=np.concatenate([first.sample_sequence, second.sample_sequence]),
            metadata={},
        )

        with self.assertRaisesRegex(SignalQualityError, "poor_waveform_quality"):
            process_signal(self.quality_recording(), signal, config)

    def test_occasion_coverage_uses_interval_union_and_minimum_window_gate(self):
        config = minimal_config(Path("."))
        config["quality"]["minimum_unique_clean_coverage_seconds"] = 17.0
        segments = pd.DataFrame(
            [
                {"recording_id": "r1", "start_s": 0.0, "end_s": 8.0, "accepted": True, "rejection_reason": ""},
                {"recording_id": "r1", "start_s": 4.0, "end_s": 12.0, "accepted": True, "rejection_reason": ""},
                {"recording_id": "r1", "start_s": 8.0, "end_s": 16.0, "accepted": True, "rejection_reason": ""},
            ]
        )
        rejected = aggregate_recording_features(self.quality_recording(), segments, config)
        self.assertEqual(rejected["unique_clean_coverage_s"], 16.0)
        self.assertFalse(rejected["occasion_usable"])
        self.assertIn("insufficient_unique_clean_coverage", rejected["occasion_rejection_reasons"])

        config["quality"]["minimum_unique_clean_coverage_seconds"] = 16.0
        accepted = aggregate_recording_features(self.quality_recording(), segments, config)
        self.assertTrue(accepted["occasion_usable"])
        config["quality"]["minimum_accepted_windows_per_occasion"] = 4
        too_few = aggregate_recording_features(self.quality_recording(), segments, config)
        self.assertIn("insufficient_accepted_windows", too_few["occasion_rejection_reasons"])

    def test_unresolved_motion_and_contact_reasons_are_explicit(self):
        config = minimal_config(Path("."))
        config["quality"]["minimum_unique_clean_coverage_seconds"] = 60.0
        segments = pd.DataFrame(
            [
                {"recording_id": "r1", "start_s": 0.0, "end_s": 8.0, "accepted": True, "rejection_reason": ""},
                {"recording_id": "r1", "start_s": 4.0, "end_s": 12.0, "accepted": False, "rejection_reason": "motion"},
                {"recording_id": "r1", "start_s": 8.0, "end_s": 16.0, "accepted": False, "rejection_reason": "contact_step"},
            ]
        )
        occasion = aggregate_recording_features(self.quality_recording(), segments, config)
        self.assertFalse(occasion["occasion_usable"])
        self.assertIn("unresolved_motion_artifact", occasion["occasion_rejection_reasons"])
        self.assertIn("unresolved_contact_artifact", occasion["occasion_rejection_reasons"])

    def test_failed_recording_remains_as_explicitly_rejected_occasion(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = minimal_config(root)
            config["quality"]["minimum_unique_clean_coverage_seconds"] = 16.0
            (root / "one_month").mkdir()
            write_local_trial(root, "P001", "001", 120, 78)
            metadata_path = root / "raw" / "P001_session_001_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["firmware_i2c_error_count"] = 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            recordings, _ = discover_recordings(config)

            _, occasions, errors = build_occasion_features(recordings, config)

            self.assertEqual(len(occasions), 1)
            self.assertFalse(bool(occasions.iloc[0]["occasion_usable"]))
            self.assertIn("firmware_i2c_error_count", occasions.iloc[0]["occasion_rejection_reasons"])
            self.assertEqual(errors, [])

    def test_quality_decisions_do_not_depend_on_bp_labels(self):
        config = minimal_config(Path("."))
        signal = self.quality_signal()
        first = self.quality_recording()
        second = self.quality_recording()
        second.sbp = 180.0
        second.dbp = 105.0

        first_rows = process_signal(first, signal, config)
        second_rows = process_signal(second, signal, config)

        self.assertEqual(
            [(row["accepted"], row["rejection_reason"]) for row in first_rows],
            [(row["accepted"], row["rejection_reason"]) for row in second_rows],
        )

    def test_aggregates_windows_before_creating_personalized_examples(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = minimal_config(root)
            (root / "one_month").mkdir()
            write_local_trial(root, "P001", "001", 120, 78)
            write_local_trial(root, "P001", "002", 124, 80)
            recordings, _ = discover_recordings(config)

            segments, occasions, errors = build_occasion_features(recordings, config)
            examples = build_personalized_examples(occasions)

            self.assertEqual(errors, [])
            self.assertGreater(len(segments), len(occasions))
            self.assertEqual(len(occasions), 2)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples.iloc[0]["delta_sbp"], 4)
            self.assertEqual(examples.iloc[0]["delta_dbp"], 2)

    def test_leave_one_participant_out_predictions_exclude_calibration_rows(self):
        with TemporaryDirectory() as tmp:
            config = minimal_config(Path(tmp))
            rows = []
            for participant_index in range(4):
                for occasion in range(3):
                    rows.append(
                        {
                            "dataset_id": "one_month_wrist",
                            "participant_id": f"P{participant_index}",
                            "session_id": f"s{occasion}",
                            "label_group_id": f"P{participant_index}:s{occasion}",
                            "chronological_order": f"2026-01-{occasion + 1:02d}",
                            "sensor_site": "wrist",
                            "sbp": 110 + participant_index * 3 + occasion,
                            "dbp": 70 + participant_index + occasion,
                            "reference_hr": 65 + occasion,
                            "calibration_occasion": occasion == 0,
                            "occasion_usable": True,
                            "median__feature__pulse_rate_bpm": 65 + participant_index + occasion,
                            "median__feature__rise_time_s": 0.2 + occasion * 0.01,
                        }
                    )
            examples = build_personalized_examples(pd.DataFrame(rows))

            predictions, parameters, _ = evaluate_personalized_models(
                examples, config, "one_month_wrist"
            )

            self.assertEqual(examples["participant_id"].nunique(), 4)
            self.assertFalse(predictions.empty)
            self.assertEqual(set(predictions["participant_id"]), {"P0", "P1", "P2", "P3"})
            self.assertFalse(
                predictions["label_group_id"].str.endswith(":s0").any(),
                "Calibration occasions must never be prediction targets",
            )
            self.assertTrue(any(item.get("held_out_participant") == "P0" for item in parameters))

    def test_single_subject_split_is_chronological_deterministic_and_excludes_calibration(self):
        config = minimal_config(Path("."))
        occasions = single_subject_occasions(count=18, calibration_index=2).sample(
            frac=1.0, random_state=19
        )

        calibration, development, test, split = prepare_single_subject_split(
            occasions, "test"
        )
        _, development_again, test_again, split_again = prepare_single_subject_split(
            occasions, "test"
        )

        self.assertEqual(calibration["label_group_id"], "test:o02")
        self.assertEqual(split["excluded_pre_calibration_occasion_ids"], ["test:o00", "test:o01"])
        self.assertNotIn(calibration["label_group_id"], set(development["label_group_id"]))
        self.assertNotIn(calibration["label_group_id"], set(test["label_group_id"]))
        self.assertEqual(len(test), 5)
        self.assertLess(
            max(pd.to_datetime(development["chronological_order"])),
            min(pd.to_datetime(test["chronological_order"])),
        )
        self.assertEqual(split, split_again)
        pd.testing.assert_frame_equal(development, development_again)
        pd.testing.assert_frame_equal(test, test_again)
        self.assertEqual(config["models"]["outer_evaluation"], "leave_one_participant_out")

    def test_single_subject_split_fails_cleanly_when_five_test_occasions_cannot_be_locked(self):
        with self.assertRaisesRegex(ValueError, "Insufficient post-calibration occasions"):
            prepare_single_subject_split(single_subject_occasions(count=10), "test")

    def test_single_subject_evaluation_uses_development_only_and_computes_baseline(self):
        config = minimal_config(Path("."))
        _, development, test, _ = prepare_single_subject_split(
            single_subject_occasions(), "test"
        )

        predictions, cv_results, models, metrics = evaluate_single_subject_models(
            development, test, config
        )
        zero = predictions[predictions["model"] == "zero_change"]
        self.assertTrue(np.allclose(zero["predicted_bp"], zero["baseline_bp"]))
        self.assertNotIn("hist_gradient_boosting", set(predictions["model"]))
        self.assertEqual(set(models), {"sbp", "dbp"})
        self.assertEqual(len(test), 5)
        self.assertTrue(set(development["label_group_id"]).isdisjoint(test["label_group_id"]))
        self.assertIn("maximum_absolute_error", metrics["targets"]["sbp"]["selected_model_metrics"])
        self.assertTrue(cv_results["selected_model"].any())

        altered_test = test.copy()
        altered_test["true_sbp"] += 100.0
        altered_test["delta_sbp"] += 100.0
        altered_predictions, _, _, _ = evaluate_single_subject_models(
            development, altered_test, config
        )
        original_sbp = predictions[predictions["target"] == "sbp"].sort_values(
            ["model", "label_group_id"]
        )
        altered_sbp = altered_predictions[altered_predictions["target"] == "sbp"].sort_values(
            ["model", "label_group_id"]
        )
        np.testing.assert_allclose(original_sbp["predicted_bp"], altered_sbp["predicted_bp"])


if __name__ == "__main__":
    unittest.main()
