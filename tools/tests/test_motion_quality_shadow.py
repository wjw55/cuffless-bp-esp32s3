import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_quality_shadow import (
    MotionQualityShadowState,
    ShadowModelCompatibilityError,
    build_shadow_record,
    load_shadow_bundle,
    maybe_score_shadow,
)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "motion_quality_v1.json"


def write_model(path: Path, config_path: Path = CONFIG_PATH) -> Path:
    model = LogisticRegression(random_state=7).fit(
        pd.DataFrame({"imu_dynamic_rms_g": [0.0, 0.01, 0.2, 0.3]}),
        [0, 0, 1, 1],
    )
    package = {
        "model": model,
        "model_name": "test_logistic",
        "feature_columns": ["imu_dynamic_rms_g"],
        "decision_threshold": 0.5,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    joblib.dump(package, path)
    return path


def add_signal(state: MotionQualityShadowState, duration_s: float) -> None:
    count = int(duration_s * 100)
    for index in range(count):
        timestamp = index * 10
        phase = 2 * np.pi * 1.2 * index / 100.0
        state.add_ppg((index, timestamp, int(90000 + 2000 * np.sin(phase)), int(140000 + 5000 * np.sin(phase))))
        state.add_imu((index, timestamp, 0, 0, 256))
        maybe_score_shadow(state)


class MotionQualityShadowTests(unittest.TestCase):
    def test_scores_completed_windows_at_four_second_cadence_without_refitting(self):
        with TemporaryDirectory() as tmp:
            model_path = write_model(Path(tmp) / "model.joblib")
            before = hashlib.sha256(model_path.read_bytes()).hexdigest()
            state = MotionQualityShadowState(load_shadow_bundle(model_path, CONFIG_PATH))

            add_signal(state, 12.0)

            self.assertEqual(state.prediction_count, 2)
            self.assertIn(state.result.prediction, {"usable", "unusable"})
            self.assertEqual(before, hashlib.sha256(model_path.read_bytes()).hexdigest())
            self.assertFalse(build_shadow_record(state, 12.0)["affects_bp_or_hr"])

    def test_waits_until_eight_seconds_of_both_streams(self):
        with TemporaryDirectory() as tmp:
            state = MotionQualityShadowState(
                load_shadow_bundle(write_model(Path(tmp) / "model.joblib"), CONFIG_PATH)
            )
            add_signal(state, 7.9)

            self.assertEqual(state.prediction_count, 0)
            self.assertEqual(state.result.status, "warming_up")

    def test_continuity_fault_clears_shadow_history_only(self):
        with TemporaryDirectory() as tmp:
            state = MotionQualityShadowState(
                load_shadow_bundle(write_model(Path(tmp) / "model.joblib"), CONFIG_PATH)
            )
            state.add_ppg((0, 0, 1, 1))
            state.add_imu((0, 0, 0, 0, 256))
            state.add_ppg((2, 20, 1, 1))

            self.assertEqual(list(state.ppg_samples), [(2, 20, 1, 1)])
            self.assertFalse(state.imu_samples)
            self.assertEqual(state.result.status, "invalid_timing")

    def test_rejects_model_config_checksum_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = write_model(root / "model.joblib")
            changed_config = root / "config.json"
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config["window_seconds"] = 9.0
            changed_config.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ShadowModelCompatibilityError, "checksums"):
                load_shadow_bundle(model_path, changed_config)

    def test_sensor_health_fault_is_latched_and_prevents_scoring(self):
        with TemporaryDirectory() as tmp:
            state = MotionQualityShadowState(
                load_shadow_bundle(write_model(Path(tmp) / "model.joblib"), CONFIG_PATH)
            )
            state.add_health("imu_stats", {"i2c_errors": 1, "fifo_overflows": 0})
            add_signal(state, 12.0)

            self.assertEqual(state.prediction_count, 0)
            self.assertEqual(state.result.status, "invalid_health")
            self.assertIn("i2c_errors=1", state.result.reason)


if __name__ == "__main__":
    unittest.main()
