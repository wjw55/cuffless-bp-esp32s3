import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_quality_classifier import (  # noqa: E402
    evaluate_frozen_model,
    evaluate_candidates,
    file_sha256,
    grouped_trial_splits,
    intensity_band_metrics,
    load_training_data,
    train_and_evaluate,
)


def classifier_settings(trees=20):
    return {
        "schema_version": 1,
        "random_seed": 42,
        "minimum_trials": 3,
        "decision_threshold": 0.5,
        "acceptance": {
            "minimum_balanced_accuracy": 0.6,
            "minimum_usable_recall": 0.6,
            "minimum_unusable_recall": 0.6,
        },
        "logistic_l2": {"c": 0.1, "maximum_iterations": 1000},
        "random_forest": {
            "trees": trees,
            "maximum_depth": 3,
            "minimum_leaf_samples": 1,
            "maximum_features": "sqrt",
        },
    }


def synthetic_frame():
    rows = []
    for trial_number in range(3):
        for index in range(12):
            usable = 1 if index < 5 else 0
            rows.append(
                {
                    "window_id": f"t{trial_number}_w{index}",
                    "participant_id": "P001",
                    "session_id": "motion_quality_v1",
                    "trial_id": f"trial_{trial_number}",
                    "start_s": float(index * 4),
                    "end_s": float(index * 4 + 8),
                    "activity_label": "deliberately_correlated_context" if usable == 0 else "still",
                    "reviewed_label": "clean" if usable else "motion_corrupted",
                    "reviewer": "tester",
                    "review_notes": "",
                    "usable": usable,
                    "supervised_training_eligible": True,
                    "ppg_ac_rms": 1.0 + (1 - usable) * 5.0 + trial_number * 0.1,
                    "imu_dynamic_rms_g": 0.01 + (1 - usable) * 0.2 + index * 0.0001,
                    "imu_activity_mean_g": 0.01 + (1 - usable) * 0.2 + index * 0.0001,
                    "imu_activity_above_threshold_fraction": float(1 - usable),
                }
            )
    rows.append(
        {
            "window_id": "uncertain",
            "participant_id": "P001",
            "session_id": "motion_quality_v1",
            "trial_id": "trial_0",
            "start_s": 100.0,
            "end_s": 108.0,
            "activity_label": "still",
            "reviewed_label": "uncertain",
            "reviewer": "tester",
            "review_notes": "",
            "usable": np.nan,
            "supervised_training_eligible": False,
            "ppg_ac_rms": 100.0,
            "imu_dynamic_rms_g": 10.0,
            "imu_activity_mean_g": 10.0,
            "imu_activity_above_threshold_fraction": 1.0,
        }
    )
    return pd.DataFrame(rows)


def write_finalized_run(root: Path, frame=None, features=None):
    frame = synthetic_frame() if frame is None else frame
    features = features or ["ppg_ac_rms", "imu_dynamic_rms_g", "imu_activity_above_threshold_fraction"]
    path = root / "reviewed_windows.csv"
    frame.to_csv(path, index=False)
    report = {
        "schema_version": 1,
        "bp_labels_used": False,
        "activity_labels_are_model_features": False,
        "model_feature_columns": features,
        "reviewed_window_count": len(frame),
        "supervised_training_window_count": int(frame["supervised_training_eligible"].sum()),
        "reviewed_windows_sha256": file_sha256(path),
    }
    (root / "finalization_report.json").write_text(json.dumps(report), encoding="utf-8")


class TrainingDataTests(unittest.TestCase):
    def test_uncertain_is_excluded_and_context_is_not_a_feature(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_finalized_run(root)
            frame, features, _ = load_training_data(root)
        self.assertEqual(len(frame), 36)
        self.assertNotIn("uncertain", frame["reviewed_label"].tolist())
        self.assertNotIn("activity_label", features)
        self.assertNotIn("trial_id", features)

    def test_forbidden_feature_and_changed_dataset_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_finalized_run(root, features=["activity_label"])
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                load_training_data(root)
            write_finalized_run(root)
            with (root / "reviewed_windows.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_training_data(root)


class LeakageAndEvaluationTests(unittest.TestCase):
    def test_leave_one_trial_out_has_no_overlap(self):
        frame = synthetic_frame().iloc[:-1].copy()
        splits = grouped_trial_splits(frame, 3)
        self.assertEqual(len(splits), 3)
        for train, test in splits:
            self.assertFalse(set(frame.iloc[train]["trial_id"]) & set(frame.iloc[test]["trial_id"]))

    def test_insufficient_trials_fail_cleanly(self):
        frame = synthetic_frame().iloc[:-1].copy()
        frame = frame[frame["trial_id"] != "trial_2"]
        with self.assertRaisesRegex(ValueError, "At least 3 trials"):
            grouped_trial_splits(frame, 3)

    def test_evaluation_is_deterministic(self):
        frame = synthetic_frame().iloc[:-1].copy()
        features = ["ppg_ac_rms", "imu_dynamic_rms_g", "imu_activity_above_threshold_fraction"]
        first = evaluate_candidates(frame, features, classifier_settings())
        second = evaluate_candidates(frame, features, classifier_settings())
        pd.testing.assert_frame_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        self.assertTrue(first[3])

    def test_per_band_metrics_keep_missing_classes_explicit(self):
        frame = synthetic_frame().iloc[:-1].copy()
        frame["motion_intensity_band"] = np.where(frame["usable"] == 1, "stationary", "mild")
        predictions, summaries, _, _ = evaluate_candidates(
            frame,
            ["ppg_ac_rms", "imu_dynamic_rms_g", "imu_activity_above_threshold_fraction"],
            classifier_settings(),
        )
        rows = intensity_band_metrics(
            predictions,
            [str(item["model"]) for item in summaries],
            ["stationary", "mild", "moderate", "severe"],
        )
        logistic = {row["motion_intensity_band"]: row for row in rows if row["model"] == "logistic_l2"}
        self.assertEqual(logistic["moderate"]["window_count"], 0)
        self.assertIsNone(logistic["stationary"]["balanced_accuracy"])
        self.assertIsNone(logistic["mild"]["balanced_accuracy"])
        self.assertEqual(logistic["stationary"]["usable_recall"], 1.0)
        self.assertEqual(logistic["mild"]["unusable_recall"], 1.0)

    def test_per_band_time_coverage_does_not_double_count_overlapping_windows(self):
        predictions = pd.DataFrame(
            {
                "model": ["model", "model"],
                "participant_id": ["P001", "P001"],
                "session_id": ["s", "s"],
                "trial_id": ["t", "t"],
                "start_s": [0.0, 4.0],
                "end_s": [8.0, 12.0],
                "motion_intensity_band": ["mild", "mild"],
                "true_unusable": [0, 1],
                "predicted_unusable": [0, 1],
                "unusable_probability": [0.1, 0.9],
            }
        )
        metrics = intensity_band_metrics(predictions, ["model"], ["mild"])[0]
        self.assertEqual(metrics["supported_unique_seconds"], 12.0)
        self.assertEqual(metrics["true_usable_unique_seconds"], 8.0)
        self.assertAlmostEqual(metrics["true_usable_time_coverage"], 2.0 / 3.0)


class OutputPackageTests(unittest.TestCase):
    def test_outputs_are_explicitly_development_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            output_dir = run_dir / "classifier_v1"
            run_dir.mkdir()
            write_finalized_run(run_dir)
            config_path = root / "config.json"
            config = {"classifier": classifier_settings()}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report = train_and_evaluate(config, config_path, run_dir, output_dir)
            package = joblib.load(output_dir / "motion_quality_classifier.joblib")
            expected_outputs_exist = all(
                (output_dir / name).exists()
                for name in (
                    "fold_predictions.csv",
                    "model_metrics.csv",
                    "feature_importance.csv",
                    "confusion_matrix.png",
                )
            )
        self.assertTrue(report["single_subject_development"])
        self.assertFalse(report["independent_validation"])
        self.assertFalse(report["deployment_eligible"])
        self.assertFalse(package["deployment_eligible"])
        self.assertTrue(expected_outputs_exist)

    def test_frozen_validation_does_not_refit_or_modify_package(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development"
            validation = root / "validation"
            model_dir = development / "classifier_v1"
            output_dir = validation / "classifier_validation_v1"
            development.mkdir()
            validation.mkdir()
            write_finalized_run(development)
            validation_frame = synthetic_frame()
            validation_frame["trial_id"] = validation_frame["trial_id"].map(
                lambda value: "validation_" + str(value)
            )
            validation_frame["session_id"] = "motion_quality_validation_v1"
            validation_frame["window_id"] = validation_frame["window_id"].map(
                lambda value: "validation_" + str(value)
            )
            write_finalized_run(validation, frame=validation_frame)
            config_path = root / "config.json"
            config = {"classifier": classifier_settings()}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            train_and_evaluate(config, config_path, development, model_dir)
            package_path = model_dir / "motion_quality_classifier.joblib"
            package_hash = file_sha256(package_path)

            report = evaluate_frozen_model(
                config,
                config_path,
                validation,
                package_path,
                output_dir,
            )

            self.assertEqual(file_sha256(package_path), package_hash)
            self.assertTrue(report["model_frozen"])
            self.assertFalse(report["model_refit_on_validation"])
            self.assertFalse(report["validation_data_used_for_training"])
            self.assertFalse(report["training_validation_trial_overlap"])
            self.assertTrue((output_dir / "validation_predictions.csv").exists())
            self.assertTrue((output_dir / "validation_report.json").exists())

    def test_frozen_validation_rejects_training_trial_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development"
            model_dir = development / "classifier_v1"
            development.mkdir()
            write_finalized_run(development)
            config_path = root / "config.json"
            config = {"classifier": classifier_settings()}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            train_and_evaluate(config, config_path, development, model_dir)

            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_frozen_model(
                    config,
                    config_path,
                    development,
                    model_dir / "motion_quality_classifier.joblib",
                    development / "invalid_validation",
                )


if __name__ == "__main__":
    unittest.main()
