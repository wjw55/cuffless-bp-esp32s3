import json
import math
import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_trials import (
    build_summary_row,
    classify_analysis_quality,
    discover_trial_pairs,
    estimate_ppg_hr,
    infer_timing_quality,
)


def make_ppg_df(duration_s=20, hr_bpm=72, sample_rate_hz=100, amplitude=10000):
    sample_count = int(duration_s * sample_rate_hz)
    time_s = np.arange(sample_count) / sample_rate_hz
    ir = 90000 + amplitude * np.sin(2 * np.pi * (hr_bpm / 60.0) * time_s)
    red = 70000 + (amplitude / 2) * np.sin(2 * np.pi * (hr_bpm / 60.0) * time_s)

    return pd.DataFrame(
        {
            "sample_seq": np.arange(sample_count),
            "timestamp_ms": (time_s * 1000).astype(int),
            "red": red.astype(int),
            "ir": ir.astype(int),
        }
    )


def make_metadata(**overrides):
    metadata = {
        "subject_id": "test",
        "session_id": "omron_pilot_001",
        "trial_id": "omron_001",
        "systolic_mmHg": 118,
        "diastolic_mmHg": 76,
        "cuff_hr_bpm": 72,
        "cuff_start_time_s": 25.0,
        "cuff_reading_time_s": 55.0,
        "sample_count": 2000,
        "data_duration_seconds": 20.0,
        "approximate_sampling_rate_hz": 100.0,
        "missing_sample_sequences": 0,
        "median_sample_interval_ms": 10.0,
        "mean_sample_interval_ms": 10.0,
        "max_sample_interval_ms": 10.0,
        "p95_sample_interval_ms": 10.0,
        "p99_sample_interval_ms": 10.0,
        "timestamp_gaps_gt_15ms": 0,
        "timestamp_gaps_gt_20ms": 0,
        "non_increasing_timestamp_count": 0,
        "timing_quality": "usable",
        "timing_quality_reason": "No missing samples or >20 ms gaps.",
        "warnings": [],
    }
    metadata.update(overrides)
    return metadata


def write_trial(input_dir: Path, metadata: dict, df=None, write_csv=True, write_metadata=True):
    df = df if df is not None else make_ppg_df()
    prefix = f"{metadata['subject_id']}_{metadata['session_id']}_{metadata['trial_id']}"
    csv_path = input_dir / f"{prefix}_ppg.csv"
    metadata_path = input_dir / f"{prefix}_metadata.json"

    metadata = dict(metadata)
    metadata["output_csv_filename"] = csv_path.name
    metadata["output_csv_path"] = str(csv_path)

    if write_csv:
        df.to_csv(csv_path, index=False)
    if write_metadata:
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    return csv_path, metadata_path


class AnalyzeTrialsTests(unittest.TestCase):
    def test_matching_csv_metadata_pairs(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            csv_path, metadata_path = write_trial(input_dir, make_metadata())

            pairs, problems = discover_trial_pairs(input_dir, "omron_pilot_001")

            self.assertEqual(problems, [])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].csv_path, csv_path)
            self.assertEqual(pairs[0].metadata_path, metadata_path)
            self.assertEqual(pairs[0].metadata["trial_id"], "omron_001")

    def test_summary_row_generation(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            csv_path, metadata_path = write_trial(input_dir, make_metadata())
            pair = discover_trial_pairs(input_dir, "omron_pilot_001")[0][0]

            row = build_summary_row(pair)

            self.assertEqual(row["subject_id"], "test")
            self.assertEqual(row["session_id"], "omron_pilot_001")
            self.assertEqual(row["trial_id"], "omron_001")
            self.assertEqual(row["csv_file"], str(csv_path))
            self.assertEqual(row["metadata_file"], str(metadata_path))
            self.assertGreater(row["ir_span"], 1000)
            self.assertEqual(row["analysis_quality"], "usable")

    def test_usable_timing_ignores_legacy_timestamp_warning(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            legacy_warning = "Timestamps irregular: p99_dt=10.0 ms, max_dt=20.0 ms, gaps_gt_20ms=0"
            write_trial(
                input_dir,
                make_metadata(
                    timing_quality="usable",
                    timing_quality_reason="No missing samples or >20 ms gaps.",
                    warnings=[legacy_warning],
                ),
            )
            pair = discover_trial_pairs(input_dir, "omron_pilot_001")[0][0]

            row = build_summary_row(pair)

            self.assertEqual(row["analysis_quality"], "usable")
            self.assertEqual(row["warnings"], legacy_warning)
            self.assertEqual(row["ignored_legacy_warnings"], legacy_warning)

    def test_usable_timing_keeps_non_timing_warning_borderline(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            warning = "IR signal has repeated flat segments"
            write_trial(
                input_dir,
                make_metadata(
                    timing_quality="usable",
                    warnings=[warning],
                ),
            )
            pair = discover_trial_pairs(input_dir, "omron_pilot_001")[0][0]

            row = build_summary_row(pair)

            self.assertEqual(row["analysis_quality"], "borderline")
            self.assertEqual(row["warnings"], warning)
            self.assertEqual(row["ignored_legacy_warnings"], "")

    def test_hr_estimation_on_synthetic_pulse_like_ir(self):
        df = make_ppg_df(duration_s=30, hr_bpm=72)

        hr_bpm, num_peaks, peak_indices, processed_ir = estimate_ppg_hr(df)

        self.assertIsNotNone(hr_bpm)
        self.assertGreater(num_peaks, 20)
        self.assertLess(abs(hr_bpm - 72), 3)
        self.assertEqual(len(peak_indices), num_peaks)
        self.assertEqual(len(processed_ir), len(df))

    def test_quality_classification_usable_timing(self):
        quality, reason = classify_analysis_quality(
            timing_quality="usable",
            missing_sample_sequences=0,
            num_detected_peaks=20,
            estimated_ppg_hr_bpm=72.0,
            hr_error_vs_cuff_bpm=2.0,
            ir_span=5000,
            metadata_warnings=[],
        )

        self.assertEqual(quality, "usable")
        self.assertIn("usable", reason)

    def test_quality_classification_borderline_timing(self):
        quality, reason = classify_analysis_quality(
            timing_quality="borderline",
            missing_sample_sequences=0,
            num_detected_peaks=20,
            estimated_ppg_hr_bpm=72.0,
            hr_error_vs_cuff_bpm=2.0,
            ir_span=5000,
            metadata_warnings=[],
        )

        self.assertEqual(quality, "borderline")
        self.assertIn("borderline timing", reason)

    def test_quality_classification_reject_timing(self):
        quality, reason = classify_analysis_quality(
            timing_quality="reject",
            missing_sample_sequences=0,
            num_detected_peaks=20,
            estimated_ppg_hr_bpm=72.0,
            hr_error_vs_cuff_bpm=2.0,
            ir_span=5000,
            metadata_warnings=[],
        )

        self.assertEqual(quality, "reject")
        self.assertIn("timing_quality=reject", reason)

    def test_infers_usable_timing_from_csv_stats_without_metadata_quality(self):
        stats = {
            "missing_sample_sequences": 0,
            "non_increasing_timestamp_count": 0,
            "timestamp_gaps_gt_20ms": 0,
            "max_sample_interval_ms": 20.0,
        }

        quality, reason = infer_timing_quality(stats)

        self.assertEqual(quality, "usable")
        self.assertIn("max_dt=20.0", reason)

    def test_missing_file_handling(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            _, metadata_path = write_trial(input_dir, make_metadata(), write_csv=False)

            pairs, problems = discover_trial_pairs(input_dir, "omron_pilot_001")

            self.assertEqual(len(pairs), 1)
            self.assertIsNone(pairs[0].csv_path)
            self.assertEqual(pairs[0].metadata_path, metadata_path)
            self.assertTrue(any("Missing CSV" in problem for problem in problems))

    def test_missing_metadata_file_handling(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            csv_path, _ = write_trial(input_dir, make_metadata(), write_metadata=False)

            pairs, problems = discover_trial_pairs(input_dir, "omron_pilot_001")

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].csv_path, csv_path)
            self.assertIsNone(pairs[0].metadata_path)
            self.assertEqual(pairs[0].metadata["subject_id"], "test")
            self.assertEqual(pairs[0].metadata["trial_id"], "omron_001")
            self.assertTrue(any("Missing metadata" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
