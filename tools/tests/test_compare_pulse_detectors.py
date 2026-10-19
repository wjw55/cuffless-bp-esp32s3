import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import upper_arm_hr as hr
from experiments.compare_pulse_detectors import (
    compare_recording, estimate_guided, main, reference_hr, review_overlaps,
)


def signal_frame(duration=90, fs=100, bpm=60, harmonic=0):
    t = np.arange(int(duration * fs)) / fs
    pulse = np.sin(2 * np.pi * bpm / 60 * t) + harmonic * np.sin(4 * np.pi * bpm / 60 * t)
    return pd.DataFrame({"sample_seq": np.arange(len(t)), "timestamp_ms": t * 1000,
                         "ir": np.rint(100000 + 1000 * pulse).astype(int),
                         "red": np.rint(80000 + 800 * pulse).astype(int)})


def metadata():
    return {"subject_id": "P001", "session_id": "replay", "trial_id": "a",
            "firmware_i2c_error_count": 0, "timing_quality": "good"}


class PulseDetectorComparisonTests(unittest.TestCase):
    def test_clean_sine_and_native_sampling_rates(self):
        for fs in (80, 100, 125):
            frame = signal_frame(20, fs, 72)
            t = frame.timestamp_ms.to_numpy() / 1000
            signal = frame.ir.to_numpy() - 100000
            result = estimate_guided(t, signal, signal, fs, np.arange(len(frame)))
            self.assertEqual(result.status, "accepted")
            self.assertAlmostEqual(result.bpm, 72, delta=1)

    def test_stronger_second_harmonic_not_double_counted(self):
        frame = signal_frame(20, harmonic=2)
        t = frame.timestamp_ms.to_numpy() / 1000
        signal = frame.ir.to_numpy() - 100000
        result = estimate_guided(t, signal, signal, 100, np.arange(len(frame)))
        self.assertEqual(result.status, "accepted")
        self.assertAlmostEqual(result.bpm, 60, delta=1)
        self.assertGreater(result.spectral_bpm, 110)

    def test_fast_pulse_is_not_automatically_halved(self):
        frame = signal_frame(20, bpm=120)
        t = frame.timestamp_ms.to_numpy() / 1000
        signal = frame.ir.to_numpy() - 100000
        result = estimate_guided(t, signal, signal, 100, np.arange(len(frame)))
        self.assertAlmostEqual(result.bpm, 120, delta=2)

    def test_unsupported_subharmonic_is_vetoed(self):
        frame = signal_frame(20)
        signal = frame.ir.to_numpy() - 100000
        with patch("experiments.compare_pulse_detectors.periodicity", return_value=(60., .8, .01)):
            result = estimate_guided(frame.timestamp_ms.to_numpy()/1000, signal, signal, 100, np.arange(len(frame)))
        self.assertIsNone(result.bpm)
        self.assertIn("unsupported_fundamental", result.reason)

    def test_candidate_trial_uses_shared_consensus_and_unique_coverage(self):
        frame = signal_frame(harmonic=2)
        result = hr.analyze_upper_arm_ppg(frame, metadata(), window_estimator=estimate_guided)
        self.assertEqual(result.status, "usable")
        self.assertAlmostEqual(result.bpm, 60, delta=1)
        self.assertGreaterEqual(result.accepted_window_count, 3)
        self.assertLessEqual(result.clean_coverage_s, 80)

    def test_nonfinite_recording_fails_explicitly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a_ppg.csv"
            frame = signal_frame()
            frame.loc[500, "ir"] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                compare_recording(path, metadata())

    def test_flat_noise_nan_and_short_inputs_rejected(self):
        signals = [np.zeros(2000), np.random.default_rng(1).normal(size=2000),
                   np.full(2000, np.nan), np.ones(10)]
        for signal in signals:
            t = np.arange(len(signal)) / 100
            result = estimate_guided(t, signal, signal, 100, np.arange(len(signal)))
            self.assertIsNone(result.bpm)

    def test_original_signal_used_for_template_quality(self):
        frame = signal_frame(20, harmonic=2)
        signal = frame.ir.to_numpy() - 100000
        with patch.object(hr, "_template_correlation", wraps=hr._template_correlation) as template:
            estimate_guided(frame.timestamp_ms.to_numpy()/1000, signal, signal, 100, np.arange(len(frame)))
            np.testing.assert_array_equal(template.call_args.args[0], signal)

    def test_existing_default_and_explicit_original_estimator_identical(self):
        frame = signal_frame()
        a = hr.analyze_upper_arm_ppg(frame, metadata())
        b = hr.analyze_upper_arm_ppg(frame, metadata(), window_estimator=hr.estimate_window)
        self.assertEqual(a.status, "usable")
        self.assertEqual((a.status, a.bpm, a.clean_coverage_s), (b.status, b.bpm, b.clean_coverage_s))
        self.assertEqual([w.as_dict() for w in a.windows], [w.as_dict() for w in b.windows])

    def test_motion_contact_clipping_and_fault_gates_not_bypassed(self):
        for kind in ("motion", "contact", "clipping", "i2c", "fifo", "sequence", "timestamp"):
            frame, meta = signal_frame(), metadata()
            if kind == "motion":
                meta["firmware_motion_updates"] = [{"timestamp_ms": 0, "status": "moving"}]
            elif kind == "contact":
                frame["ir"] = 20000
            elif kind == "clipping":
                frame["ir"] = hr.ADC_MAX
            elif kind == "i2c":
                meta["imu_firmware_i2c_error_count"] = 1
            elif kind == "fifo":
                meta["firmware_fifo_overflow_count"] = 1
            elif kind == "sequence":
                frame.loc[1000:, "sample_seq"] += 1
            elif kind == "timestamp":
                frame.loc[1000, "timestamp_ms"] = frame.loc[999, "timestamp_ms"]
            result = hr.analyze_upper_arm_ppg(frame, meta, window_estimator=estimate_guided)
            self.assertIsNone(result.bpm, kind)
            self.assertNotEqual(result.status, "usable", kind)

    def test_reference_values_cannot_change_waveform_decisions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a_ppg.csv"
            signal_frame(harmonic=2).to_csv(path, index=False)
            a = compare_recording(path, metadata())[:2]
            b = compare_recording(path, {**metadata(), "sbp": 250, "cuff_hr_bpm": 170,
                                         "notes": "must pass", "quality": "good"})[:2]
            self.assertEqual(a, b)

    def test_review_overlap_union_and_uncertain_not_clean(self):
        frame = signal_frame()
        result = hr.analyze_upper_arm_ppg(frame, metadata())
        reviews = [{"start_timestamp_ms": 10000, "end_timestamp_ms": 18000,
                    "window_id": "a", "reviewed_label": "motion_corrupted"},
                   {"start_timestamp_ms": 10000, "end_timestamp_ms": 18000,
                    "window_id": "b", "reviewed_label": "uncertain"}]
        rows = review_overlaps(frame, result, reviews, "candidate", "x")
        self.assertEqual(rows[0]["accepted_overlap_s"], 8)
        self.assertTrue(rows[0]["overlap_warning"])
        self.assertFalse(rows[1]["overlap_warning"])

    def test_reference_conflicts_are_errors(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay_labels.csv"
            path.write_text("subject,session,trial_id,omron_hr\nP001,replay,a,60\n", encoding="utf-8")
            self.assertEqual(reference_hr(metadata(), Path(tmp)), 60)
            with self.assertRaises(ValueError):
                reference_hr({**metadata(), "cuff_hr_bpm": 90}, Path(tmp))

    def test_cli_reproducible_output_and_raw_bytes_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppg, meta = root / "a_ppg.csv", root / "a_metadata.json"
            signal_frame().to_csv(ppg, index=False)
            meta.write_text(json.dumps(metadata()), encoding="utf-8")
            before = (ppg.read_bytes(), meta.read_bytes())
            for name in ("one", "two"):
                self.assertEqual(main(["--input-dir", str(root), "--labels-dir", str(root),
                                       "--session", "replay", "--output-dir", str(root/name)]), 0)
            for file in ("recording_comparison.csv", "window_comparison.csv", "report.json"):
                self.assertEqual((root/"one"/file).read_bytes(), (root/"two"/file).read_bytes())
            self.assertEqual(before, (ppg.read_bytes(), meta.read_bytes()))
            with self.assertRaises(SystemExit):
                main(["--session", "replay", "--output-dir", str(root/"one")])


if __name__ == "__main__":
    unittest.main()
