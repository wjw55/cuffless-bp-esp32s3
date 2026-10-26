import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bp_core.combined import (
    episode_examples, evaluate_combined, extract_selected, fit_variant, forward_folds,
    identity, load_combined, predict_combined, predict_occasion, train_combined, validate_study,
)
from bp_core.datasets import Recording, _discover_local
from bp_core.feasibility import process_gap_tolerant_frames
from bp_core.inference import ModelCompatibilityError, load_model_bundle
from bp_pipeline import parse_args
from test_bp_pipeline import minimal_config
from test_bp_feasibility import policy, frames, metadata


def oid(episode, trial):
    return f"local_upper_arm:P001:{episode}:{trial}"


def study():
    return {"schema_version": 1, "participant_id": "P001", "target_episode_id": "new",
            "training_variant": "combined", "minimum_development_occasions": 5, "minimum_test_occasions": 5,
            "episodes": [{"episode_id": episode, "setup_id": episode, "sensor_location": "arm",
                          "calibration_occasion_id": oid(episode, "cal"),
                          "development_occasion_ids": [oid(episode, f"d{i}") for i in range(count)],
                          "test_occasion_ids": [oid(episode, f"t{i}") for i in range(5)] if episode == "new" else []}
                         for episode, count in (("old", 6), ("new", 4))]}


def occasions():
    rows = []
    for episode, month, count, base_sbp, base_dbp in (("old", 1, 6, 108, 63), ("new", 2, 4, 103, 73)):
        for i in range(count+1):
            trial = "cal" if i == 0 else f"d{i-1}"
            date = f"2026-{month:02d}-{i+1:02d}T12:00:00+08:00"
            rows.append({"dataset_id": "local_upper_arm", "participant_id": "P001",
                         "session_id": episode, "recording_id": trial, "label_group_id": oid(episode, trial),
                         "chronological_order": date, "_time": pd.Timestamp(date), "sensor_site": "arm",
                         "episode_id": episode, "setup_id": episode,
                         "role": "calibration" if i == 0 else "development",
                         "calibration_occasion": i == 0, "occasion_usable": True,
                         "sbp": base_sbp + i*2, "dbp": base_dbp + i,
                         "median__feature__rise_time_s": .20 + i*.01,
                         "median__feature__pulse_rate_bpm": 60 + i*2,
                         "iqr__feature__pulse_rate_bpm": 1 + i*.1})
    return pd.DataFrame(rows)


class ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class CombinedDevelopmentTests(unittest.TestCase):
    def test_explicit_mapping_rejects_participant_and_role_leakage(self):
        self.assertEqual(len(validate_study(study())), 17)
        for invalid in ([], None, {**study(), "episodes": [1, 2]}):
            with self.assertRaises(ValueError):
                validate_study(invalid)
        for mutation in ("duplicate", "participant", "calibration", "few_tests", "target", "unsafe"):
            s = study()
            if mutation == "duplicate":
                s["episodes"][1]["test_occasion_ids"][0] = s["episodes"][0]["development_occasion_ids"][0]
            elif mutation == "participant":
                s["episodes"][0]["development_occasion_ids"][0] = "local_upper_arm:P002:old:d0"
            elif mutation == "calibration":
                s["episodes"][0]["calibration_occasion_id"] = s["episodes"][0]["development_occasion_ids"][0]
            elif mutation == "few_tests":
                s["episodes"][1]["test_occasion_ids"] = []
            elif mutation == "target":
                s["target_episode_id"] = "missing"
            else:
                s["episodes"][0]["development_occasion_ids"][0] = "local_upper_arm:P001:../old:d0"
            with self.assertRaises(ValueError, msg=mutation):
                validate_study(s)

    def test_episode_specific_calibration_bp_and_features(self):
        examples, calibrations = episode_examples(occasions(), study())
        self.assertEqual(len(examples), 10)
        new = examples[examples.episode_id == "new"].iloc[0]
        old = examples[examples.episode_id == "old"].iloc[0]
        self.assertEqual((new.baseline_sbp, new.baseline_dbp), (103, 73))
        self.assertEqual((old.baseline_sbp, old.baseline_dbp), (108, 63))
        self.assertEqual(new.delta_sbp, 2)
        self.assertAlmostEqual(new["change__median__rise_time_s"], .01)
        self.assertNotIn(calibrations["new"]["label_group_id"], set(examples.label_group_id))

    def test_rejected_calibration_never_falls_back(self):
        frame = occasions()
        frame.loc[frame.label_group_id == oid("new", "cal"), "occasion_usable"] = False
        with self.assertRaisesRegex(ValueError, "no fallback"):
            episode_examples(frame, study())

    def test_pre_calibration_occasions_are_errors(self):
        frame = occasions()
        frame.loc[frame.label_group_id == oid("new", "d0"), "_time"] = pd.Timestamp("2025-01-01", tz="UTC")
        with self.assertRaisesRegex(ValueError, "does not follow"):
            episode_examples(frame, study())

    def test_forward_folds_only_score_target_and_use_earlier_training(self):
        examples, _ = episode_examples(occasions(), study())
        folds = forward_folds(examples, "new", "combined", 5, 3)
        for train, val in folds:
            self.assertLess(examples.iloc[train]._time.max(), examples.iloc[val]._time.min())
            self.assertTrue(examples.iloc[val].episode_id.eq("new").all())
            self.assertFalse(set(examples.iloc[train].label_group_id) & set(examples.iloc[val].label_group_id))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            forward_folds(examples, "new", "new_housing_only", 5, 3)

    def test_ridge_only_fitting_and_deterministic_selection(self):
        examples, _ = episode_examples(occasions(), study())
        config = minimal_config(Path("."))
        packages, scores, rows, folds = fit_variant(examples, config, study(), "combined")
        self.assertEqual(set(packages), {"sbp", "dbp"})
        self.assertEqual({row["model"] for row in scores}, {"ridge"})
        self.assertEqual({row["model"] for row in rows}, {"ridge", "zero_change", "mean_change", "hr_only"})
        for package in packages.values():
            self.assertNotIn("episode_id", package["feature_columns"])
            self.assertNotIn("sbp", package["feature_columns"])
            self.assertEqual(type(package["estimator"].named_steps["regressor"]).__name__, "Ridge")
        other = fit_variant(examples, config, study(), "combined")
        self.assertEqual(scores, other[1])
        self.assertEqual(rows, other[2])
        self.assertEqual(folds, other[3])

    def make_run(self, root, prepare=False):
        root = Path(root)
        config = minimal_config(root)
        for name, data in (("config.json", config), ("policy.json", policy()), ("study.json", study())):
            (root/name).write_text(json.dumps(data), encoding="utf-8")
        output = root / "run"
        with patch("bp_core.combined.extract_selected", return_value=(occasions(), pd.DataFrame(), [])):
            report = train_combined(root/"config.json", root/"policy.json", root/"study.json", output, prepare)
        return output, report

    def test_saved_research_package_fails_closed_for_live_viewer(self):
        with TemporaryDirectory() as tmp:
            root, report = self.make_run(tmp)
            self.assertEqual(report["status"], "development_model_fitted")
            self.assertFalse(report["viewer_eligible"])
            self.assertFalse((root/"model_manifest.json").exists())
            with self.assertRaises(ModelCompatibilityError):
                load_model_bundle(root)
            manifest, *_ = load_combined(root)
            self.assertEqual(manifest["calibration"]["sbp"], 103)
            self.assertEqual(report["variants"]["new_housing_only"]["status"], "insufficient_or_failed")

    def test_prepare_mode_has_no_model_and_refuses_overwrite(self):
        with TemporaryDirectory() as tmp:
            root, report = self.make_run(tmp, prepare=True)
            self.assertEqual(report["status"], "prepared_only")
            self.assertFalse(list(root.glob("*.joblib")))
            with self.assertRaises(FileExistsError):
                train_combined(Path(tmp)/"config.json", Path(tmp)/"policy.json", Path(tmp)/"study.json", root)

    def test_corrupt_snapshots_calibration_and_model_fail_cleanly(self):
        for name in ("config_snapshot.json", "policy_snapshot.json", "study_snapshot.json", "split.json",
                     "calibration_snapshot.json", "research_sbp_ridge.joblib", "manifest_calibration", "manifest_participant"):
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                root, _ = self.make_run(tmp)
                if name.startswith("manifest"):
                    path = root/"combined_model_manifest.json"
                    manifest = json.loads(path.read_text())
                    if name.endswith("calibration"):
                        manifest["calibration"]["sbp"] = 200
                    else:
                        manifest["participant_id"] = "P999"
                    path.write_text(json.dumps(manifest))
                else:
                    (root/name).write_bytes(b"corrupt")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    load_combined(root)

    def test_offline_input_construction_matches_training_predictions(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.make_run(tmp)
            manifest, _, _, s, _, packages = load_combined(root)
            frame = occasions()
            current = frame[frame.label_group_id == oid("new", "d3")].iloc[0].to_dict()
            result = predict_occasion(manifest, packages, current)
            examples, _ = episode_examples(frame, s)
            example = examples[examples.label_group_id == current["label_group_id"]]
            for target, package in packages.items():
                expected = manifest["calibration"][target] + package["estimator"].predict(example[package["feature_columns"]])[0]
                self.assertAlmostEqual(result[target], expected, places=10)
            self.assertEqual(result, predict_occasion(manifest, packages, {**current, "sbp": 250, "dbp": 200}))

    def test_invalid_model_output_is_hidden_not_clipped(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.make_run(tmp)
            manifest, _, _, _, _, packages = load_combined(root)
            current = occasions().iloc[-1].to_dict()
            for bad in (float("nan"), -200):
                modified = copy.deepcopy(packages)
                modified["sbp"]["estimator"] = ConstantPredictor(bad)
                result = predict_occasion(manifest, modified, current)
                self.assertEqual(result["status"], "invalid_model_output")
                self.assertIsNone(result["sbp"])
            self.assertEqual(predict_occasion(manifest, packages, {**current, "occasion_usable": False})["status"], "signal_rejected")

    def test_test_evaluation_never_refits_and_is_one_time(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.make_run(tmp)
            tests = occasions().query("episode_id == 'new' and role == 'development'").copy()
            extra = tests.iloc[-1:].copy()
            tests = pd.concat([tests, extra], ignore_index=True)
            for i in range(5):
                tests.loc[i, "label_group_id"] = oid("new", f"t{i}")
                tests.loc[i, "_time"] = pd.Timestamp(f"2026-03-{i+1:02d}", tz="UTC")
            before = (root/"research_sbp_ridge.joblib").read_bytes()
            with patch("bp_core.combined.extract_selected", return_value=(tests, pd.DataFrame(), [])), \
                 patch("bp_core.combined.fit_variant", side_effect=AssertionError("Must not refit")):
                result = evaluate_combined(root)
            self.assertEqual(result["test_occasions"], 5)
            self.assertFalse(result["model_refitted"])
            self.assertFalse(result["viewer_eligible"])
            self.assertEqual(before, (root/"research_sbp_ridge.joblib").read_bytes())
            with self.assertRaises(FileExistsError):
                evaluate_combined(root)

    def test_test_chronology_and_missing_future_recordings_rejected(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.make_run(tmp)
            with self.assertRaisesRegex(ValueError, "Missing"):
                evaluate_combined(root)
            with patch("bp_core.combined.extract_selected", return_value=(occasions().iloc[:2], pd.DataFrame(), [])):
                with self.assertRaisesRegex(ValueError, "strictly later"):
                    evaluate_combined(root)
            self.assertFalse((root/"prospective_evaluation").exists())

    def test_selected_discovery_ignores_other_participants_and_test_label_conflicts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"raw").mkdir()
            (root/"labels").mkdir()
            meta = {"subject_id": "P001", "session_id": "new", "trial_id": "d0",
                    "ppg_profile": "upper_arm_experimental", "systolic_mmHg": 120, "diastolic_mmHg": 70}
            (root/"raw/P001_new_d0_metadata.json").write_text(json.dumps(meta))
            (root/"labels/new_labels.csv").write_text("subject,session,trial_id,sbp,dbp\nP001,new,d0,120,70\nP001,new,t0,garbage,garbage\nP002,new,d0,90,60\nP002,new,d0,99,62\n")
            cfg = minimal_config(root)["datasets"]["local_upper_arm"]
            rows = _discover_local(cfg, selected_identities={("P001", "new", "d0")})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sbp, 120)
            with self.assertRaises(ValueError):
                _discover_local(cfg)

    def test_training_discovery_selects_no_reserved_test_ids(self):
        s = study()
        with patch("bp_core.combined._discover_local", return_value=[]) as discovery:
            with self.assertRaises(ValueError):
                extract_selected(s, minimal_config(Path(".")), policy(), {"calibration", "development"})
        selected = discovery.call_args.kwargs["selected_identities"]
        self.assertTrue(all(not trial.startswith("t") for _, _, trial in selected))

    def test_real_raw_training_and_offline_inference_share_processing(self):
        from bp_core.datasets import sha256_file
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, labels = root/"raw", root/"labels"
            raw.mkdir()
            labels.mkdir()
            s, config = study(), minimal_config(root)
            label_rows = []
            for index, row in occasions().iterrows():
                _, participant, session, trial = row.label_group_id.split(":")
                prefix = f"{participant}_{session}_{trial}"
                ppg, imu = frames(70)
                ppg["timestamp_ms"] += index * 100000
                imu["timestamp_ms"] += index * 100000
                ppg.to_csv(raw/f"{prefix}_ppg.csv", index=False)
                imu.to_csv(raw/f"{prefix}_imu.csv", index=False)
                meta = {**metadata(), "subject_id": participant, "session_id": session, "trial_id": trial,
                        "ppg_profile": "upper_arm_experimental", "sensor_location": "arm",
                        "recording_start_time": row.chronological_order}
                (raw/f"{prefix}_metadata.json").write_text(json.dumps(meta))
                label_rows.append({"subject": participant, "session": session, "trial_id": trial,
                                   "sbp": row.sbp, "dbp": row.dbp})
            pd.DataFrame(label_rows).to_csv(labels/"study_labels.csv", index=False)
            # Reserved files must not even enter metadata parsing during fitting.
            (raw/"P001_new_t0_metadata.json").write_text("invalid future file: do not read")
            for name, value in (("config.json", config), ("policy.json", policy()), ("study.json", s)):
                (root/name).write_text(json.dumps(value))
            before = {path.name: sha256_file(path) for path in raw.iterdir()}
            output = root/"run"
            result = train_combined(root/"config.json", root/"policy.json", root/"study.json", output)
            self.assertEqual(result["status"], "development_model_fitted")
            self.assertEqual(before, {path.name: sha256_file(path) for path in raw.iterdir()})
            ppg, imu, meta = (raw/f"P001_new_d3_{suffix}" for suffix in ("ppg.csv", "imu.csv", "metadata.json"))
            pending = predict_combined(output, ppg, imu, meta, "new")
            self.assertEqual(pending["status"], "model_validation_pending")
            self.assertIsNone(pending["sbp"])
            predicted = predict_combined(output, ppg, imu, meta, "new", allow_unvalidated=True)
            self.assertEqual(predicted["status"], "unvalidated_development_estimate")
            manifest, cfg, pol, _, _, packages = load_combined(output)
            current = next(rec for rec in _discover_local(config["datasets"]["local_upper_arm"],
                         selected_identities={("P001", "new", "d3")}) if rec.recording_id == "d3")
            from bp_core.feasibility import process_gap_tolerant_recording
            same = predict_occasion(manifest, packages, process_gap_tolerant_recording(current, cfg, pol).occasion)
            self.assertAlmostEqual(predicted["sbp"], same["sbp"], places=10)
            self.assertAlmostEqual(predicted["dbp"], same["dbp"], places=10)
            with self.assertRaisesRegex(ValueError, "setup"):
                predict_combined(output, ppg, imu, meta, "old", allow_unvalidated=True)
            changed = json.loads(meta.read_text())
            changed["trial_id"] = "t0"
            meta.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Reserved test"):
                predict_combined(output, ppg, imu, meta, "new", allow_unvalidated=True)

    def test_processing_source_changes_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.make_run(tmp)
            with patch("bp_core.combined._code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    load_combined(root)

    def test_corrupted_test_occasion_is_retained_in_coverage(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.make_run(tmp)
            tests = pd.concat([occasions().iloc[-1:]] * 5, ignore_index=True)
            for i in range(5):
                tests.loc[i, "label_group_id"] = oid("new", f"t{i}")
                tests.loc[i, "_time"] = pd.Timestamp(f"2026-03-{i+1:02d}", tz="UTC")
            tests.loc[0, "occasion_usable"] = False
            tests.loc[0, "occasion_rejection_reasons"] = "sensor_health:imu_i2c_error"
            with patch("bp_core.combined.extract_selected", return_value=(tests, pd.DataFrame(), [])):
                result = evaluate_combined(root)
            self.assertEqual(result["coverage"], .8)
            self.assertEqual(result["accepted_occasions"], 4)
            self.assertFalse(result["beats_zero_change_both_targets"])

    def test_cli_keeps_existing_commands_and_adds_separate_modes(self):
        self.assertEqual(parse_args(["single-subject", "--participant-id", "P001", "--run-dir", "x"]).minimum_test_occasions, 5)
        self.assertEqual(parse_args(["combined-subject", "--study-config", "x", "--output-dir", "y"]).command, "combined-subject")
        self.assertEqual(parse_args(["combined-evaluate", "--run-dir", "x"]).command, "combined-evaluate")


if __name__ == "__main__":
    unittest.main()
