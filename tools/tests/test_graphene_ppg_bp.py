import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphene_ppg_bp import (
    build_personalized_examples,
    discover_trials,
    extract_finapres_reference,
    extract_trial_observations,
    load_config,
)


def test_config(root: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "test",
        "research_only": True,
        "deployment_eligible": False,
        "dataset": {"root": str(root), "source_url": "test", "version": "1", "license": "test"},
        "observation": {
            "durations_seconds": [20],
            "update_step_seconds": 5,
            "maximum_duration_shortfall_seconds": 0,
            "minimum_clean_fraction": 0.7,
            "minimum_accepted_segments": 2,
        },
        "signal": {
            "window_seconds": 8,
            "window_step_seconds": 4,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 8,
            "filter_order": 4,
            "template_samples": 128,
            "minimum_hr_bpm": 40,
            "maximum_hr_bpm": 180,
            "analysis_sample_rate_hz": 125,
        },
        "quality": {
            "minimum_beats_per_window": 4,
            "maximum_interval_cv": 0.2,
            "minimum_template_correlation": 0.5,
        },
        "reference": {
            "minimum_sbp_mmHg": 70,
            "maximum_sbp_mmHg": 240,
            "minimum_dbp_mmHg": 35,
            "maximum_dbp_mmHg": 150,
            "minimum_pulse_pressure_mmHg": 10,
            "minimum_beats": 5,
            "peak_prominence_mmHg": 5,
            "maximum_interval_cv": 0.25,
            "label_window_seconds": 10,
        },
        "model": {"ridge_alpha": 10, "minimum_participants": 3, "minimum_training_examples": 10},
    }


def write_trial(root: Path, trial_number: int, duration=45.0, pressure_offset=0.0):
    folder = root / "subject1_day1" / "setup01_baseline"
    folder.mkdir(parents=True, exist_ok=True)
    ppg_time = np.arange(int(duration * 250)) / 250 + 1.0
    phase = 2 * np.pi * 1.2 * ppg_time
    ppg = 10 + 2.0 * (np.sin(phase) + 0.25 * np.sin(2 * phase))
    bp_time = np.arange(int(duration * 100)) / 100 + 1.0
    bp_phase = 2 * np.pi * 1.2 * bp_time
    pressure = 95 + pressure_offset + 25 * np.sin(bp_phase)
    pd.DataFrame({"time": ppg_time, "PPG": ppg}).to_csv(
        folder / f"data_trial{trial_number:02d}_ppg.csv", index=False
    )
    pd.DataFrame({"time": bp_time, "FinapresBP": pressure}).to_csv(
        folder / f"data_trial{trial_number:02d}_finapresBP.csv", index=False
    )


class GraphenePPGBPTests(unittest.TestCase):
    def test_config_must_remain_non_deployable(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = test_config(Path(tmp))
            config["deployment_eligible"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deployment_eligible=false"):
                load_config(path)

    def test_discovers_exact_ppg_finapres_pairs_and_missing_pair(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trial(root, 1)
            folder = root / "subject1_day1" / "setup01_baseline"
            pd.DataFrame({"time": [0, 1], "PPG": [1, 2]}).to_csv(
                folder / "data_trial02_ppg.csv", index=False
            )
            trials = discover_trials(root)
            self.assertEqual(len(trials), 2)
            self.assertEqual(trials[0].participant_id, "subject1")
            self.assertEqual(trials[0].discovery_status, "paired")
            self.assertEqual(trials[1].discovery_status, "missing_finapres_pair")

    def test_finapres_reference_uses_pressure_peaks_and_troughs(self):
        settings = test_config(Path("."))["reference"]
        time_s = np.arange(30 * 100) / 100
        pressure = 95 + 25 * np.sin(2 * np.pi * 1.2 * time_s)
        result, reason = extract_finapres_reference(time_s, pressure, settings)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(result["sbp"], 120, delta=0.2)
        self.assertAlmostEqual(result["dbp"], 70, delta=0.2)
        self.assertAlmostEqual(result["reference_hr"], 72, delta=0.5)

    def test_extracts_time_aligned_observations_without_ptt_channels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trial(root, 1)
            trial = discover_trials(root)[0]
            rows = extract_trial_observations(trial, test_config(root))
            self.assertGreater(len(rows), 1)
            self.assertTrue(any(row["accepted"] for row in rows))
            self.assertTrue(all(row["start_s"] >= 1.0 for row in rows))
            self.assertEqual(sum(bool(row["matched_endpoint_row"]) for row in rows), 1)
            forbidden = {"ecg", "bioz", "ptt", "pat"}
            self.assertFalse(any(any(word in key.lower() for word in forbidden) for key in rows[0]))

    def test_bp_values_do_not_change_ppg_quality_or_features(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trial(root, 1, pressure_offset=0)
            trial = discover_trials(root)[0]
            first = extract_trial_observations(trial, test_config(root))
            bp_path = Path(trial.finapres_path)
            bp_frame = pd.read_csv(bp_path)
            bp_frame["FinapresBP"] += 10
            bp_frame.to_csv(bp_path, index=False)
            second = extract_trial_observations(trial, test_config(root))
            for left, right in zip(first, second):
                self.assertEqual(left["signal_usable"], right["signal_usable"])
                self.assertEqual(left["signal_rejection_reason"], right["signal_rejection_reason"])
                self.assertAlmostEqual(
                    left["median__feature__pulse_rate_bpm"],
                    right["median__feature__pulse_rate_bpm"],
                    delta=1e-10,
                )
                if left["reference_usable"] and right["reference_usable"]:
                    self.assertAlmostEqual(right["sbp"] - left["sbp"], 10, delta=0.2)

    def test_entire_calibration_recording_is_excluded_from_examples(self):
        rows = []
        for recording_index in range(3):
            for update in range(2):
                rows.append(
                    {
                        "participant_id": "subject1",
                        "recording_id": f"r{recording_index}",
                        "observation_id": f"r{recording_index}:u{update}",
                        "chronological_order": f"day1:s{recording_index}",
                        "start_s": update * 20,
                        "requested_duration_s": 20,
                        "accepted": True,
                        "model_evaluation_row": True,
                        "matched_endpoint_row": True,
                        "sbp": 120 + recording_index,
                        "dbp": 75 + recording_index,
                        "median__feature__pulse_rate_bpm": 70 + recording_index,
                        "iqr__feature__pulse_rate_bpm": 1,
                    }
                )
        examples, calibrations = build_personalized_examples(pd.DataFrame(rows), 20)
        self.assertEqual(calibrations[0]["calibration_recording_id"], "r0")
        self.assertNotIn("r0", set(examples["recording_id"]))
        self.assertEqual(set(examples["recording_id"]), {"r1", "r2"})


if __name__ == "__main__":
    unittest.main()
