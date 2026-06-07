import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_ppg import build_metadata, parse_args, parse_ppg_row, validate_collection_args


class ParsePpgRowTests(unittest.TestCase):
    def test_parses_sample_sequence_timestamp_red_and_ir(self):
        self.assertEqual(parse_ppg_row("42,12345,50001,60002"), (42, 12345, 50001, 60002))

    def test_ignores_new_csv_header(self):
        self.assertIsNone(parse_ppg_row("sample_seq,timestamp_ms,red,ir"))


def make_args(**overrides):
    defaults = {
        "subject": "test",
        "session": "baseline_001",
        "trial_id": "T01",
        "posture": "seated",
        "sensor_location": "right_index_finger",
        "cuff_arm": "left",
        "ppg_hand": "right",
        "port": "COM3",
        "baud": 115200,
        "duration": 90.0,
        "systolic_mmHg": None,
        "diastolic_mmHg": None,
        "cuff_hr_bpm": None,
        "cuff_start_time_s": None,
        "cuff_reading_time_s": None,
        "cuff_timestamp": "",
        "notes": "",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def make_summary():
    return {
        "data_duration_s": 89.71,
        "sample_count": 8972,
        "sample_sequence_start": 0,
        "sample_sequence_end": 8971,
        "missing_sample_sequences": 0,
        "median_dt_ms": 10.0,
        "estimated_rate_hz": 100.0,
        "warnings": [],
    }


class MetadataTests(unittest.TestCase):
    def test_builds_ppg_only_metadata_without_bp_fields(self):
        metadata = build_metadata(
            make_args(notes="Stable PPG-only check"),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=2,
            zoom_start_s=20.0,
            zoom_end_s=30.0,
        )

        self.assertEqual(metadata["subject_id"], "test")
        self.assertEqual(metadata["session_id"], "baseline_001")
        self.assertEqual(metadata["trial_id"], "T01")
        self.assertEqual(metadata["posture"], "seated")
        self.assertEqual(metadata["sensor_location"], "right_index_finger")
        self.assertEqual(metadata["cuff_arm"], "left")
        self.assertEqual(metadata["ppg_hand"], "right")
        self.assertIsNone(metadata["systolic_mmHg"])
        self.assertIsNone(metadata["diastolic_mmHg"])
        self.assertIsNone(metadata["cuff_hr_bpm"])
        self.assertIsNone(metadata["cuff_start_time_s"])
        self.assertIsNone(metadata["cuff_reading_time_s"])
        self.assertEqual(metadata["notes"], "Stable PPG-only check")

    def test_builds_omron_labeled_metadata_with_bp_fields(self):
        metadata = build_metadata(
            make_args(
                session="omron_pilot_001",
                trial_id="omron_001",
                systolic_mmHg=118,
                diastolic_mmHg=76,
                cuff_hr_bpm=72,
                cuff_start_time_s=25.0,
                cuff_reading_time_s=55.0,
                notes="Omron pilot trial 1",
            ),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=20.0,
            zoom_end_s=30.0,
        )

        self.assertEqual(metadata["session_id"], "omron_pilot_001")
        self.assertEqual(metadata["trial_id"], "omron_001")
        self.assertEqual(metadata["systolic_mmHg"], 118)
        self.assertEqual(metadata["diastolic_mmHg"], 76)
        self.assertEqual(metadata["cuff_hr_bpm"], 72)
        self.assertEqual(metadata["cuff_start_time_s"], 25.0)
        self.assertEqual(metadata["cuff_reading_time_s"], 55.0)
        self.assertEqual(metadata["cuff_arm"], "left")
        self.assertEqual(metadata["ppg_hand"], "right")

    def test_parses_omron_labeled_cli_fields(self):
        args = parse_args([
            "--port",
            "COM3",
            "--duration",
            "90",
            "--subject",
            "test",
            "--session",
            "omron_pilot_001",
            "--trial-id",
            "omron_001",
            "--posture",
            "seated",
            "--sensor-location",
            "right_index_finger",
            "--cuff-arm",
            "left",
            "--ppg-hand",
            "right",
            "--systolic-mmHg",
            "118",
            "--diastolic-mmHg",
            "76",
            "--cuff-hr-bpm",
            "72",
            "--cuff-start-time-s",
            "25",
            "--cuff-reading-time-s",
            "55",
            "--notes",
            "Omron pilot trial 1",
        ])

        self.assertEqual(args.systolic_mmHg, 118)
        self.assertEqual(args.diastolic_mmHg, 76)
        self.assertEqual(args.cuff_hr_bpm, 72)
        self.assertEqual(args.cuff_start_time_s, 25.0)
        self.assertEqual(args.cuff_reading_time_s, 55.0)

    def test_rejects_invalid_bp_values(self):
        invalid_options = [
            ("--systolic-mmHg", "0"),
            ("--diastolic-mmHg", "-1"),
            ("--cuff-hr-bpm", "0"),
        ]

        for option_name, option_value in invalid_options:
            with self.subTest(option_name=option_name):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args([
                            "--port",
                            "COM3",
                            "--duration",
                            "90",
                            "--subject",
                            "test",
                            "--session",
                            "bad_bp",
                            option_name,
                            option_value,
                        ])

    def test_rejects_cuff_time_outside_recording_duration(self):
        invalid_overrides = [
            {"cuff_start_time_s": 91.0},
            {"cuff_reading_time_s": 91.0},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(SystemExit):
                    validate_collection_args(make_args(**overrides))


if __name__ == "__main__":
    unittest.main()
