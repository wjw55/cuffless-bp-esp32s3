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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bp_core.inference import ModelCompatibilityError, load_model_bundle, predict_frame
from view_live_bp import (
    BPInferenceResult,
    BPViewerState,
    ViewerContext,
    buffer_duration_s,
    build_validation_record,
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
            "local_contact_threshold_counts": 50000.0,
            "motion_margin_seconds": 1.0,
            "contact_margin_seconds": 2.0,
        },
        "models": {"bootstrap_iterations": 10},
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

    def test_numeric_result_is_hidden_immediately_on_motion(self):
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
        self.assertIn("Estimated BP: --/--", render_screen(state, context, 87.1, "COM5", 115200))

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
        self.assertEqual(row["ppg_rate_hz"], 100.0)
        self.assertIsNone(row["sbp"])

    def test_cli_requires_calibration_only_without_model(self):
        args = parse_args(
            ["--port", "COM5", "--participant-id", "P001", "--calibration-sbp", "116", "--calibration-dbp", "72"]
        )
        self.assertEqual(args.calibration_sbp, 116)
        with self.assertRaises(SystemExit), patch("sys.stderr", StringIO()):
            parse_args(["--port", "COM5", "--participant-id", "P001"])


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
