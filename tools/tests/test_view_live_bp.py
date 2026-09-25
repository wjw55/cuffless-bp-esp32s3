import hashlib
import json
import sys
import unittest
from argparse import Namespace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bp_core.inference import (
    ModelCompatibilityError,
    load_model_bundle,
    make_short_window_bundle,
    predict_frame,
)
from view_live_bp import (
    BPInferenceResult,
    BPViewerState,
    LastValidatedBP,
    LiveMotionIntensityState,
    ViewerContext,
    buffer_duration_s,
    build_validation_record,
    live_motion_intensity,
    maybe_predict,
    parse_args,
    render_screen,
    run_viewer,
    update_state_from_line,
)


def config():
    return {
        "schema_version": 1,
        "random_seed": 7,
        "datasets": {},
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
            "minimum_unique_clean_coverage_seconds": 60.0,
            "require_upper_arm_analyzer_acceptance": True,
            "local_contact_threshold_counts": 50000.0,
            "motion_margin_seconds": 1.0,
            "contact_margin_seconds": 2.0,
        },
        "models": {"bootstrap_iterations": 10},
    }


def intensity_config():
    return {
        "schema_version": 2,
        "window_seconds": 8.0,
        "window_step_seconds": 4.0,
        "timing": {"minimum_completeness": 0.995},
        "imu": {
            "scale_g_per_lsb": 0.0039,
            "gravity_alpha": 0.01,
            "activity_window_samples": 100,
            "firmware_motion_threshold_g": 0.05,
        },
        "motion_intensity": {
            "source_feature": "imu_activity_mean_g",
            "boundaries_g": [0.02, 0.08, 0.2],
            "labels": ["stationary", "mild", "moderate", "severe"],
            "severe_policy": "unavailable",
        },
    }


def ppg_frame(duration_s=86.0, bpm=72.0):
    time_s = np.arange(int(duration_s * 100) + 1) / 100.0
    phase = 2 * np.pi * bpm / 60.0 * time_s
    return pd.DataFrame(
        {
            "sample_seq": np.arange(len(time_s)),
            "timestamp_ms": np.round(time_s * 1000).astype(int),
            "red": 80000 + 3000 * (np.sin(phase) + 0.2 * np.sin(2 * phase)),
            "ir": 140000 + 6000 * (np.sin(phase) + 0.25 * np.sin(2 * phase)),
        }
    )


def write_model_dir(root: Path, eligible=True):
    cfg = root / "config_snapshot.json"
    cfg.write_text(json.dumps(config()), encoding="utf-8")
    checksum = hashlib.sha256(cfg.read_bytes()).hexdigest()
    models = root / "models"
    models.mkdir()
    columns = [
        "baseline_sbp",
        "baseline_dbp",
        "current__median__pulse_rate_bpm",
        "calibration__median__pulse_rate_bpm",
        "change__median__pulse_rate_bpm",
    ]
    entries = {}
    for target, constant in (("sbp", -2.0), ("dbp", 1.0)):
        estimator = DummyRegressor(strategy="constant", constant=constant)
        estimator.fit(pd.DataFrame([[116, 72, 72, 72, 0]], columns=columns), [constant])
        filename = f"single_subject_{target}.joblib"
        joblib.dump(
            {
                "estimator": estimator,
                "feature_columns": columns,
                "target": target,
                "participant_id": "P001",
                "calibration_label_group_id": "calibration:001",
                "model_manifest_schema_version": 1,
                "config_sha256": checksum,
            },
            models / filename,
        )
        entries[target] = {
            "file": f"models/{filename}",
            "feature_count": len(columns),
            "feature_columns": columns,
            "beats_zero_change_on_locked_test": eligible,
        }
    manifest = {
        "schema_version": 1,
        "config_sha256": checksum,
        "participant_id": "P001",
        "viewer_eligible": eligible,
        "passes_both_targets": eligible,
        "calibration": {
            "label_group_id": "calibration:001",
            "sbp": 116,
            "dbp": 72,
            "features": {"median__feature__pulse_rate_bpm": 72.0},
        },
        "models": entries,
    }
    (root / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def add_still_data(state: BPViewerState, duration_s=86.0):
    update_state_from_line(
        state, "# motion timestamp_ms=0 status=still activity_g=0.010 threshold_g=0.050", 0.0
    )
    for row in ppg_frame(duration_s).itertuples(index=False):
        update_state_from_line(
            state, f"{row.sample_seq},{row.timestamp_ms},{int(row.red)},{int(row.ir)}", row.timestamp_ms / 1000
        )
    update_state_from_line(
        state,
        f"# motion timestamp_ms={int(duration_s * 1000)} status=still activity_g=0.010 threshold_g=0.050",
        duration_s,
    )


class BPInferenceTests(unittest.TestCase):
    def test_short_window_bundle_changes_only_in_memory_quality_policy(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root, eligible=True)
            bundle = load_model_bundle(root)

            short = make_short_window_bundle(bundle, 30)

            self.assertEqual(
                short.config["quality"]["minimum_accepted_windows_per_occasion"], 3
            )
            self.assertEqual(
                short.config["quality"]["minimum_unique_clean_coverage_seconds"], 24.0
            )
            self.assertFalse(short.config["quality"]["require_upper_arm_analyzer_acceptance"])
            self.assertEqual(
                bundle.config["quality"]["minimum_unique_clean_coverage_seconds"], 60.0
            )
            self.assertTrue(short.viewer_eligible)
            self.assertFalse(short.allow_unvalidated)

    def test_eligible_model_produces_quality_gated_prediction(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root, eligible=True)
            bundle = load_model_bundle(root, expected_participant_id="P001")

            result = predict_frame(bundle, ppg_frame(), {"firmware_motion_updates": []})

            self.assertEqual(result.status, "prediction_ready")
            self.assertAlmostEqual(result.sbp, 114.0)
            self.assertAlmostEqual(result.dbp, 73.0)
            self.assertGreaterEqual(result.accepted_windows, 3)

    def test_unvalidated_model_is_hidden_unless_explicitly_allowed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root, eligible=False)
            hidden = predict_frame(load_model_bundle(root), ppg_frame(), {})
            shown = predict_frame(load_model_bundle(root, allow_unvalidated=True), ppg_frame(), {})

            self.assertEqual(hidden.status, "model_validation_failed")
            self.assertFalse(hidden.numeric_available)
            self.assertEqual(shown.status, "unvalidated_estimate")
            self.assertTrue(shown.numeric_available)

    def test_offline_inference_uses_shared_sequence_and_health_gates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root, eligible=True)
            bundle = load_model_bundle(root)
            broken = ppg_frame()
            broken.loc[100:, "sample_seq"] += 1

            sequence_result = predict_frame(bundle, broken, {})
            health_result = predict_frame(bundle, ppg_frame(), {"imu_firmware_fifo_overflow_count": 1})

            self.assertEqual(sequence_result.status, "invalid_timing")
            self.assertIn("missing_ppg_sequences", sequence_result.reason)
            self.assertEqual(health_result.status, "invalid_timing")
            self.assertIn("imu_firmware_fifo_overflow_count", health_result.reason)

    def test_insufficient_unique_clean_coverage_hides_prediction(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root, eligible=True)
            bundle = load_model_bundle(root)

            result = predict_frame(bundle, ppg_frame(duration_s=40.0), {})

            self.assertFalse(result.numeric_available)
            self.assertLess(result.clean_coverage_s, 60.0)
            self.assertIn("insufficient_unique_clean_coverage", result.reason)

    def test_invalid_model_output_is_never_displayable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root, eligible=True)
            bundle = load_model_bundle(root)
            bundle.packages["sbp"]["estimator"] = Mock(predict=Mock(return_value=[-60.0]))

            result = predict_frame(bundle, ppg_frame(), {})

            self.assertEqual(result.status, "invalid_model_output")
            self.assertFalse(result.numeric_available)

    def test_model_loader_rejects_participant_and_checksum_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root)
            with self.assertRaisesRegex(ModelCompatibilityError, "does not match requested"):
                load_model_bundle(root, expected_participant_id="P002")
            (root / "config_snapshot.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ModelCompatibilityError, "checksum"):
                load_model_bundle(root)

    def test_model_loader_rejects_tampered_feature_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root)
            manifest_path = root / "model_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"]["sbp"]["feature_columns"][0] = "wrong_column"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ModelCompatibilityError, "feature columns"):
                load_model_bundle(root)

    def test_model_loader_rejects_missing_corrupt_and_wrong_calibration_packages(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            missing.mkdir()
            with self.assertRaisesRegex(ModelCompatibilityError, "must contain"):
                load_model_bundle(missing)

            corrupt = Path(tmp) / "corrupt"
            corrupt.mkdir()
            (corrupt / "model_manifest.json").write_text("not-json", encoding="utf-8")
            (corrupt / "config_snapshot.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ModelCompatibilityError, "cannot read"):
                load_model_bundle(corrupt)

            wrong_calibration = Path(tmp) / "wrong_calibration"
            wrong_calibration.mkdir()
            write_model_dir(wrong_calibration)
            package_path = wrong_calibration / "models" / "single_subject_sbp.joblib"
            package = joblib.load(package_path)
            package["calibration_label_group_id"] = "different:calibration"
            joblib.dump(package, package_path)
            with self.assertRaisesRegex(ModelCompatibilityError, "calibration"):
                load_model_bundle(wrong_calibration)


class BPViewerTests(unittest.TestCase):
    def test_live_motion_intensity_uses_causal_eight_second_imu_history(self):
        state = BPViewerState(started_at=0.0)
        state.motion_intensity = LiveMotionIntensityState(intensity_config())
        for index in range(801):
            update_state_from_line(
                state,
                f"imu,{index},{index * 10},0,0,256",
                index / 100.0,
            )

        band, activity = live_motion_intensity(state, 8.0)

        self.assertEqual(band, "stationary")
        self.assertIsNotNone(activity)
        self.assertLess(activity, 0.02)

    def test_live_motion_intensity_is_unknown_during_warmup_or_when_stale(self):
        state = BPViewerState(started_at=0.0)
        state.motion_intensity = LiveMotionIntensityState(intensity_config())
        update_state_from_line(state, "imu,0,0,0,0,256", 0.0)

        self.assertEqual(live_motion_intensity(state, 0.0), ("unknown", None))
        self.assertEqual(live_motion_intensity(state, 4.0), ("unknown", None))

    def test_motion_intensity_display_does_not_replace_binary_bp_gate(self):
        state = BPViewerState(started_at=0.0)
        state.motion_intensity = LiveMotionIntensityState(intensity_config())
        state.motion_intensity.band = "mild"
        state.motion_intensity.mean_activity_g = 0.04
        state.motion_intensity.last_update_at = 10.0
        update_state_from_line(
            state,
            "# motion timestamp_ms=10000 status=moving activity_g=0.04 threshold_g=0.05",
            10.0,
        )
        context = ViewerContext("P001", 116, 72, config())

        screen = render_screen(state, context, 10.0, "COM5", 115200)

        self.assertIn("Motion detected", screen)
        self.assertIn("Motion intensity: Mild", screen)
        self.assertIn("Display only; Still/Moving remains the BP gate", screen)

    def test_default_mode_still_requires_85_seconds(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state, duration_s=31.0)
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        predictor = Mock()

        self.assertFalse(maybe_predict(state, context, 31.0, predictor=predictor))
        predictor.assert_not_called()
        self.assertEqual(state.result.status, "warming_up")
        self.assertIn("31.0/85 s", state.result.reason)

    def test_opt_in_fast_mode_uses_only_trailing_30_seconds(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state, duration_s=35.0)
        standard_bundle = Mock(viewer_eligible=True)
        fast_bundle = Mock(viewer_eligible=True)
        context = ViewerContext(
            "P001",
            116,
            72,
            config(),
            bundle=standard_bundle,
            experimental_fast_window_seconds=30.0,
            experimental_fast_bundle=fast_bundle,
        )
        observed = {}

        def predictor(bundle, frame, metadata):
            observed["bundle"] = bundle
            observed["span"] = (frame.timestamp_ms.iloc[-1] - frame.timestamp_ms.iloc[0]) / 1000
            observed["motion_updates"] = metadata["firmware_motion_updates"]
            return BPInferenceResult(
                "prediction_ready", "accepted", sbp=114, dbp=73, delta_sbp=-2, delta_dbp=1
            )

        self.assertTrue(maybe_predict(state, context, 35.0, predictor=predictor))
        self.assertIs(observed["bundle"], fast_bundle)
        self.assertLessEqual(observed["span"], 30.0)
        self.assertEqual(state.result.status, "experimental_fast_estimate")
        screen = render_screen(state, context, 35.0, "COM5", 115200)
        self.assertIn("EXPERIMENTAL FAST ESTIMATE", screen)
        self.assertIn("Standard fallback: 85 s", screen)

    def test_fast_mode_switches_back_to_standard_policy_at_85_seconds(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state, duration_s=86.0)
        standard_bundle = Mock(viewer_eligible=True)
        fast_bundle = Mock(viewer_eligible=True)
        context = ViewerContext(
            "P001",
            116,
            72,
            config(),
            bundle=standard_bundle,
            experimental_fast_window_seconds=30.0,
            experimental_fast_bundle=fast_bundle,
        )
        observed = {}

        def predictor(bundle, frame, _metadata):
            observed["bundle"] = bundle
            observed["span"] = (frame.timestamp_ms.iloc[-1] - frame.timestamp_ms.iloc[0]) / 1000
            return BPInferenceResult("prediction_ready", "accepted", sbp=114, dbp=73)

        self.assertTrue(maybe_predict(state, context, 86.0, predictor=predictor))
        self.assertIs(observed["bundle"], standard_bundle)
        self.assertGreater(observed["span"], 85.0)
        self.assertEqual(state.result.status, "prediction_ready")

    def test_pending_mode_never_calls_predictor_or_displays_bp(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext("P001", 116, 72, config())
        predictor = Mock()

        self.assertFalse(maybe_predict(state, context, 86.0, predictor=predictor))
        predictor.assert_not_called()
        screen = render_screen(state, context, 86.0, "COM5", 115200)
        self.assertIn("Estimated BP: --/--", screen)
        self.assertIn("Model validation pending", screen)

    def test_last_validated_result_is_held_during_motion(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        result = BPInferenceResult(
            "prediction_ready", "accepted", sbp=114, dbp=73, delta_sbp=-2, delta_dbp=1
        )
        maybe_predict(state, context, 86.0, predictor=lambda *_args: result)
        self.assertIn("Estimated BP: 114/73", render_screen(state, context, 86.1, "COM5", 115200))

        update_state_from_line(
            state, "# motion timestamp_ms=87000 status=moving activity_g=0.2 threshold_g=0.05", 87.0
        )
        self.assertEqual(buffer_duration_s(state), 0.0)
        screen = render_screen(state, context, 87.1, "COM5", 115200)
        self.assertIn("Last validated BP: 114/73", screen)
        self.assertIn("Last validated estimate (not a new measurement)", screen)
        self.assertIn("Current status: Motion detected", screen)
        self.assertIn("HELD VALUE", screen)

    def test_last_validated_result_expires(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext(
            "P001",
            116,
            72,
            config(),
            bundle=Mock(viewer_eligible=True),
            last_validated_max_age_seconds=0.5,
        )
        result = BPInferenceResult("prediction_ready", "accepted", sbp=114, dbp=73)
        maybe_predict(state, context, 86.0, predictor=lambda *_args: result)
        update_state_from_line(
            state, "# motion timestamp_ms=87000 status=moving activity_g=0.2 threshold_g=0.05", 87.0
        )

        screen = render_screen(state, context, 87.1, "COM5", 115200)

        self.assertIn("Estimated BP: --/--", screen)
        self.assertIn("last validated estimate expired", screen)

    def test_new_accepted_estimate_replaces_held_snapshot(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        first = BPInferenceResult("prediction_ready", "first", sbp=114, dbp=73)
        second = BPInferenceResult("prediction_ready", "second", sbp=119, dbp=76)
        maybe_predict(state, context, 86.0, predictor=lambda *_args: first)
        update_state_from_line(
            state, "# motion timestamp_ms=91000 status=still activity_g=0.01 threshold_g=0.05", 91.1
        )
        maybe_predict(state, context, 91.1, predictor=lambda *_args: second)
        update_state_from_line(
            state, "# motion timestamp_ms=92000 status=moving activity_g=0.2 threshold_g=0.05", 92.0
        )

        screen = render_screen(state, context, 92.1, "COM5", 115200)

        self.assertIn("Last validated BP: 119/76", screen)
        self.assertNotIn("Last validated BP: 114/73", screen)

    def test_held_snapshot_is_not_reused_for_another_participant(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        bundle = Mock(viewer_eligible=True)
        context = ViewerContext("P001", 116, 72, config(), bundle=bundle)
        result = BPInferenceResult("prediction_ready", "accepted", sbp=114, dbp=73)
        maybe_predict(state, context, 86.0, predictor=lambda *_args: result)
        update_state_from_line(
            state, "# motion timestamp_ms=87000 status=moving activity_g=0.2 threshold_g=0.05", 87.0
        )
        other_context = ViewerContext("P002", 120, 80, config(), bundle=bundle)

        screen = render_screen(state, other_context, 87.1, "COM5", 115200)

        self.assertIn("Estimated BP: --/--", screen)
        self.assertNotIn("Last validated BP", screen)

    def test_shadow_unusable_prediction_does_not_gate_bp(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        shadow = Mock()
        shadow.result = Mock(prediction="unusable", status="unusable", unusable_probability=0.93)
        state.motion_quality_shadow = shadow
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        result = BPInferenceResult(
            "prediction_ready", "accepted", sbp=114, dbp=73, delta_sbp=-2, delta_dbp=1
        )

        self.assertTrue(maybe_predict(state, context, 86.0, predictor=lambda *_args: result))
        screen = render_screen(state, context, 86.0, "COM5", 115200)

        self.assertIn("Estimated BP: 114/73", screen)
        self.assertIn("Prediction: Unusable", screen)
        self.assertIn("Control effect: None", screen)

    def test_numeric_result_is_hidden_when_serial_data_is_stale(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        result = BPInferenceResult("prediction_ready", "accepted", sbp=114, dbp=73)
        maybe_predict(state, context, 86.0, predictor=lambda *_args: result)
        # Isolate the serial-data stale gate from the separate motion-status
        # stale gate, which is covered by the existing HR/viewer tests.
        state.last_motion_at = 91.0

        screen = render_screen(state, context, 92.0, "COM5", 115200)

        self.assertIn("Estimated BP: --/--", screen)
        self.assertIn("serial data is stale", screen)

    def test_numeric_result_is_hidden_after_sensor_timing_fault(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        result = BPInferenceResult("prediction_ready", "accepted", sbp=114, dbp=73)
        maybe_predict(state, context, 86.0, predictor=lambda *_args: result)
        update_state_from_line(state, "9000,90000,80000,140000", 87.0)

        screen = render_screen(state, context, 87.1, "COM5", 115200)

        self.assertIn("Estimated BP: --/--", screen)
        self.assertIn("PPG continuity fault", screen)
        self.assertNotIn("Last validated BP", screen)

    def test_sequence_gap_and_new_health_error_restart_clean_buffer(self):
        state = BPViewerState(started_at=0.0)
        update_state_from_line(
            state, "# motion timestamp_ms=0 status=still activity_g=0.01 threshold_g=0.05", 0.0
        )
        update_state_from_line(state, "0,0,80000,140000", 0.0)
        update_state_from_line(state, "2,20,80000,140000", 0.02)
        self.assertEqual(len(state.ppg_samples), 1)
        self.assertEqual(state.result.status, "invalid_timing")

        update_state_from_line(
            state,
            "# stats samples=3 rate_hz=100 ovf=1 i2c_errors=0",
            0.03,
        )
        self.assertEqual(len(state.ppg_samples), 0)
        update_state_from_line(state, "3,30,80000,140000", 0.03)
        update_state_from_line(
            state,
            "# stats samples=4 rate_hz=100 ovf=1 i2c_errors=0",
            0.04,
        )
        self.assertEqual(len(state.ppg_samples), 1)

    def test_rolling_buffer_is_bounded_and_malformed_rows_are_ignored(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state, duration_s=95.0)

        self.assertLessEqual(buffer_duration_s(state), 90.0)
        sample_count = len(state.ppg_samples)
        self.assertFalse(update_state_from_line(state, "malformed,serial,row", 95.1))
        self.assertEqual(len(state.ppg_samples), sample_count)

    def test_validation_record_contains_model_and_sensor_fields(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        state.ppg_stats = {"rate_hz": 100.0, "i2c_errors": 0, "ovf": 0}
        context = ViewerContext("P001", 116, 72, config())
        maybe_predict(state, context, 86.0)
        row = build_validation_record(state, context, 86.0, 86.0)
        self.assertEqual(row["status"], "model_pending")
        self.assertEqual(row["display_mode"], "unavailable")
        self.assertEqual(row["current_status"], "model_pending")
        self.assertEqual(row["ppg_rate_hz"], 100.0)
        self.assertIsNone(row["sbp"])

    def test_validation_record_distinguishes_held_from_current_status(self):
        state = BPViewerState(started_at=0.0)
        add_still_data(state)
        context = ViewerContext("P001", 116, 72, config(), bundle=Mock(viewer_eligible=True))
        result = BPInferenceResult("prediction_ready", "accepted", sbp=114, dbp=73)
        maybe_predict(state, context, 86.0, predictor=lambda *_args: result)
        update_state_from_line(
            state, "# motion timestamp_ms=87000 status=moving activity_g=0.2 threshold_g=0.05", 87.0
        )

        row = build_validation_record(state, context, 87.1, 87.1)

        self.assertEqual(row["display_mode"], "held")
        self.assertEqual(row["status"], "last_validated_estimate")
        self.assertEqual(row["current_status"], "motion_detected")
        self.assertEqual(row["sbp"], 114)
        self.assertAlmostEqual(row["estimate_age_s"], 1.1)

    def test_disconnected_transport_hides_held_estimate(self):
        state = BPViewerState(started_at=0.0, transport_connected=False)
        state.result = BPInferenceResult(
            "analysis_stale", "BLE disconnected; reconnecting"
        )
        context = ViewerContext("P001", 116, 72, config())
        state.last_validated_bp = LastValidatedBP(
            result=BPInferenceResult("prediction_ready", "accepted", sbp=118, dbp=74),
            measured_at=5.0,
            sensor_timestamp_ms=5000,
            participant_id="P001",
            model_identity="no-model",
        )

        screen = render_screen(
            state,
            context,
            10.0,
            "PPG-LOGGER-A1B2C3",
            115200,
            transport="ble",
        )

        self.assertIn("--/-- mmHg", screen)
        self.assertIn("Disconnected; reconnecting", screen)

    def test_cli_requires_calibration_only_without_model(self):
        args = parse_args(
            ["--port", "COM5", "--participant-id", "P001", "--calibration-sbp", "116", "--calibration-dbp", "72"]
        )
        self.assertEqual(args.calibration_sbp, 116)
        self.assertEqual(args.last_validated_max_age, 300.0)
        with self.assertRaises(SystemExit), patch("sys.stderr", StringIO()):
            parse_args(["--port", "COM5", "--participant-id", "P001"])

    def test_cli_accepts_last_validated_max_age_override(self):
        args = parse_args(
            [
                "--port", "COM5",
                "--participant-id", "P001",
                "--calibration-sbp", "116",
                "--calibration-dbp", "72",
                "--last-validated-max-age", "120",
            ]
        )
        self.assertEqual(args.last_validated_max_age, 120.0)

    def test_cli_accepts_ble_without_serial_port(self):
        args = parse_args(
            [
                "--transport", "ble",
                "--ble-device", "PPG-LOGGER-A1B2C3",
                "--participant-id", "P001",
                "--calibration-sbp", "116",
                "--calibration-dbp", "72",
            ]
        )

        self.assertEqual(args.transport, "ble")
        self.assertEqual(args.ble_device, "PPG-LOGGER-A1B2C3")
        self.assertIsNone(args.port)

    def test_cli_accepts_optional_motion_quality_shadow_model(self):
        args = parse_args(
            [
                "--port", "COM5",
                "--participant-id", "P001",
                "--calibration-sbp", "116",
                "--calibration-dbp", "72",
                "--motion-quality-shadow-model", "model.joblib",
            ]
        )
        self.assertEqual(args.motion_quality_shadow_model, "model.joblib")

    def test_cli_fast_window_is_opt_in_and_requires_model(self):
        args = parse_args(
            [
                "--port", "COM5",
                "--participant-id", "P001",
                "--model-dir", "model",
                "--experimental-fast-window", "30",
            ]
        )
        self.assertEqual(args.experimental_fast_window, 30)
        with self.assertRaises(SystemExit), patch("sys.stderr", StringIO()):
            parse_args(
                [
                    "--port", "COM5",
                    "--participant-id", "P001",
                    "--calibration-sbp", "116",
                    "--calibration-dbp", "72",
                    "--experimental-fast-window", "30",
                ]
            )


class FakeSerialException(Exception):
    pass


class InterruptingPort:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def set_buffer_size(self, **_kwargs):
        pass

    def reset_input_buffer(self):
        pass

    def readline(self):
        raise KeyboardInterrupt


class FakeSerialModule:
    SerialException = FakeSerialException

    @staticmethod
    def Serial(*_args, **_kwargs):
        return InterruptingPort()


class FailingSerialModule:
    class SerialException(Exception):
        pass

    @classmethod
    def Serial(cls, *_args, **_kwargs):
        raise cls.SerialException("port busy")


class BPViewerRuntimeTests(unittest.TestCase):
    def test_ctrl_c_exits_cleanly(self):
        args = Namespace(port="COM5", baud=115200, refresh=1.0)
        context = ViewerContext("P001", 116, 72, config())
        output = StringIO()
        with patch("sys.stdout", output):
            result = run_viewer(args, context, FakeSerialModule, clock=lambda: 0.0, sleep=lambda _x: None)
        self.assertEqual(result, 0)
        self.assertIn("No data was saved", output.getvalue())

    def test_serial_port_error_fails_cleanly(self):
        args = Namespace(port="COM5", baud=115200, refresh=1.0)
        context = ViewerContext("P001", 116, 72, config())
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            result = run_viewer(
                args,
                context,
                FailingSerialModule,
                clock=lambda: 0.0,
                sleep=lambda _x: None,
            )
        self.assertEqual(result, 1)
        self.assertIn("port busy", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
