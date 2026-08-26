import unittest
import csv
from argparse import Namespace
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_ppg import (
    LABEL_COLUMNS,
    add_motion_columns,
    apply_prompt_bp_after,
    build_label_csv_path,
    build_label_row,
    build_metadata,
    build_output_paths,
    create_firmware_diagnostics,
    ensure_output_paths_available,
    parse_args,
    parse_firmware_status_line,
    parse_imu_row,
    parse_ppg_row,
    summarize,
    summarize_imu,
    update_firmware_diagnostics,
    validate_collection_args,
    write_label_row,
)


class ParsePpgRowTests(unittest.TestCase):
    def test_parses_sample_sequence_timestamp_red_and_ir(self):
        self.assertEqual(parse_ppg_row("42,12345,50001,60002"), (42, 12345, 50001, 60002))

    def test_ignores_new_csv_header(self):
        self.assertIsNone(parse_ppg_row("sample_seq,timestamp_ms,red,ir"))


class ParseImuRowTests(unittest.TestCase):
    def test_parses_tagged_signed_raw_axes(self):
        self.assertEqual(parse_imu_row("imu,42,12345,-256,0,257"), (42, 12345, -256, 0, 257))

    def test_rejects_header_and_out_of_range_axis(self):
        self.assertIsNone(parse_imu_row("imu,imu_seq,timestamp_ms,x_raw,y_raw,z_raw"))
        self.assertIsNone(parse_imu_row("imu,1,100,40000,0,0"))

    def test_ppg_parser_ignores_imu_rows(self):
        self.assertIsNone(parse_ppg_row("imu,42,12345,-256,0,257"))


class FirmwareStatusParsingTests(unittest.TestCase):
    def test_parses_firmware_stats_line(self):
        parsed = parse_firmware_status_line(
            "# stats samples=500 captured_samples=500 rate_hz=99.8 "
            "effective_rate_hz=99.9 fifo_avail=2 ovf=0 i2c_errors=0 "
            "timestamp_resyncs=0 timestamp_corrections=0"
        )

        self.assertIsNotNone(parsed)
        status_type, fields = parsed
        self.assertEqual(status_type, "stats")
        self.assertEqual(fields["captured_samples"], 500)
        self.assertEqual(fields["rate_hz"], 99.8)
        self.assertEqual(fields["effective_rate_hz"], 99.9)
        self.assertEqual(fields["timestamp_resyncs"], 0)

    def test_updates_firmware_diagnostics_from_stats_and_warnings(self):
        diagnostics = create_firmware_diagnostics()

        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line(
                "# stats samples=500 captured_samples=500 rate_hz=99.8 "
                "effective_rate_hz=99.9 fifo_avail=2 ovf=0 i2c_errors=0 "
                "timestamp_resyncs=0 timestamp_corrections=0 overflow_recoveries=0"
            ),
        )
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line("# warning event=timestamp_resync sample_seq=1200 lag_us=81000 count=1"),
        )
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line("# warning event=fifo_overflow_recovery count=1 total=1 sample_seq=1200"),
        )

        self.assertEqual(diagnostics["metadata_fields"]["firmware_captured_samples"], 500)
        self.assertEqual(diagnostics["metadata_fields"]["firmware_effective_rate_hz"], 99.9)
        self.assertEqual(diagnostics["metadata_fields"]["firmware_fifo_overflow_count"], 0)
        self.assertEqual(diagnostics["metadata_fields"]["firmware_timestamp_resync_count"], 1)
        self.assertEqual(diagnostics["metadata_fields"]["firmware_fifo_overflow_recovery_count"], 1)
        self.assertEqual(len(diagnostics["warning_events"]), 2)

    def test_tracks_timestamp_lag_warnings_separately_from_resyncs(self):
        diagnostics = create_firmware_diagnostics()

        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line(
                "# stats samples=500 captured_samples=500 rate_hz=99.8 "
                "effective_rate_hz=99.9 fifo_avail=2 ovf=0 i2c_errors=0 "
                "timestamp_resyncs=1 timestamp_corrections=0 "
                "timestamp_lag_warnings=2 overflow_recoveries=1"
            ),
        )
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line("# warning event=timestamp_lag sample_seq=1200 lag_us=60000 count=3"),
        )

        self.assertEqual(diagnostics["metadata_fields"]["firmware_timestamp_resync_count"], 1)
        self.assertEqual(diagnostics["metadata_fields"]["firmware_timestamp_lag_warning_count"], 3)
        self.assertEqual(diagnostics["warning_events"][0]["event"], "timestamp_lag")

    def test_tracks_imu_stats_separately(self):
        diagnostics = create_firmware_diagnostics()
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line(
                "# imu_stats samples=501 rate_hz=100.0 effective_rate_hz=99.8 "
                "fifo_entries=1 fifo_overflows=0 i2c_errors=0 "
                "timestamp_resyncs=0 timestamp_corrections=0 "
                "clock_adjustments=480 clock_adjustment_us=-11840"
            ),
        )

        self.assertEqual(diagnostics["latest_imu_stats"]["samples"], 501)
        self.assertEqual(diagnostics["metadata_fields"]["imu_firmware_interval_rate_hz"], 100.0)
        self.assertEqual(diagnostics["metadata_fields"]["imu_firmware_fifo_overflow_count"], 0)
        self.assertEqual(diagnostics["metadata_fields"]["imu_firmware_clock_adjustment_count"], 480)
        self.assertEqual(diagnostics["metadata_fields"]["imu_firmware_clock_adjustment_total_us"], -11840)

    def test_tracks_live_hr_status_updates_without_treating_them_as_raw_rows(self):
        diagnostics = create_firmware_diagnostics()
        stable_line = "# hr timestamp_ms=20420 bpm=72.4 status=stable beats=6"
        warming_line = "# hr timestamp_ms=10420 bpm=na status=warming_up beats=2"

        update_firmware_diagnostics(diagnostics, parse_firmware_status_line(warming_line))
        update_firmware_diagnostics(diagnostics, parse_firmware_status_line(stable_line))

        self.assertEqual(len(diagnostics["hr_updates"]), 2)
        self.assertEqual(diagnostics["latest_hr"]["bpm"], 72.4)
        self.assertEqual(diagnostics["latest_hr"]["status"], "stable")
        self.assertIsNone(parse_ppg_row(stable_line))
        self.assertIsNone(parse_imu_row(stable_line))

    def test_tracks_motion_status_without_treating_it_as_raw_data(self):
        diagnostics = create_firmware_diagnostics()
        motion_line = "# motion timestamp_ms=20420 status=still activity_g=0.018 threshold_g=0.050"

        update_firmware_diagnostics(diagnostics, parse_firmware_status_line(motion_line))

        self.assertEqual(diagnostics["latest_motion"]["status"], "still")
        self.assertEqual(diagnostics["motion_updates"][0]["activity_g"], 0.018)
        self.assertIsNone(parse_ppg_row(motion_line))
        self.assertIsNone(parse_imu_row(motion_line))


def make_args(**overrides):
    defaults = {
        "subject": "test",
        "session": "baseline_001",
        "trial_id": "T01",
        "posture": "seated",
        "sensor_location": "right_index_finger",
        "ppg_profile": "finger",
        "ppg_orientation": "",
        "mounting_method": "",
        "strap_tension": "",
        "led_current_ma": 7.2,
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
        "prompt_bp_after": False,
        "label_sbp": None,
        "label_dbp": None,
        "label_omron_hr": None,
        "label_omron_timing": "",
        "label_notes": "",
        "prompt_labels": False,
        "labels_dir": "data/labels",
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
        "mean_sample_interval_ms": 10.0,
        "min_sample_interval_ms": 10.0,
        "max_sample_interval_ms": 10.0,
        "p95_sample_interval_ms": 10.0,
        "p99_sample_interval_ms": 10.0,
        "timestamp_gaps_gt_15ms": 0,
        "timestamp_gaps_gt_20ms": 0,
        "non_increasing_timestamp_count": 0,
        "timestamp_irregularity_reason": None,
        "timing_quality": "good",
        "timing_quality_reason": "No missing samples, monotonic timestamps, and all intervals <= 15 ms.",
        "estimated_rate_hz": 100.0,
        "warnings": [],
    }


def make_ppg_df(timestamps_ms, sample_seq=None):
    if sample_seq is None:
        sample_seq = list(range(len(timestamps_ms)))

    return pd.DataFrame(
        {
            "sample_seq": sample_seq,
            "timestamp_ms": timestamps_ms,
            "red": [60000] * len(timestamps_ms),
            "ir": [70000] * len(timestamps_ms),
        }
    )


def make_imu_df(timestamps_ms, x_raw=None):
    sample_count = len(timestamps_ms)
    return pd.DataFrame(
        {
            "imu_seq": list(range(sample_count)),
            "timestamp_ms": timestamps_ms,
            "x_raw": x_raw if x_raw is not None else [0] * sample_count,
            "y_raw": [0] * sample_count,
            "z_raw": [256] * sample_count,
        }
    )


class ImuSummaryTests(unittest.TestCase):
    def test_stationary_100_hz_capture_has_good_timing(self):
        imu_df = make_imu_df(list(range(0, 1000, 10)))

        summary = summarize_imu(imu_df, requested_duration_s=1.0)

        self.assertEqual(summary["sample_count"], 100)
        self.assertEqual(summary["missing_sample_sequences"], 0)
        self.assertEqual(summary["estimated_rate_hz"], 100.0)
        self.assertEqual(summary["timing_quality"], "good")

    def test_motion_spike_is_flagged_by_recording_specific_threshold(self):
        x_raw = [0] * 100
        x_raw[50] = 300
        imu_df = make_imu_df(list(range(0, 1000, 10)), x_raw=x_raw)

        enriched, threshold_g = add_motion_columns(imu_df)

        self.assertGreater(float(enriched.loc[50, "dynamic_accel_g"]), threshold_g)
        self.assertTrue(bool(enriched.loc[50, "motion_candidate"]))


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

    def test_metadata_includes_output_csv_path_when_available(self):
        csv_path = Path("data/raw/test_omron_pilot_001_omron_001_ppg.csv")

        metadata = build_metadata(
            make_args(session="omron_pilot_001", trial_id="omron_001"),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=20.0,
            zoom_end_s=30.0,
            csv_path=csv_path,
        )

        self.assertEqual(metadata["output_csv_filename"], "test_omron_pilot_001_omron_001_ppg.csv")
        self.assertEqual(metadata["output_csv_path"], str(csv_path))

    def test_metadata_records_imu_configuration_and_quality(self):
        imu_summary = summarize_imu(make_imu_df(list(range(0, 1000, 10))), requested_duration_s=1.0)
        imu_csv_path = Path("data/raw/test_baseline_001_T01_imu.csv")
        args = make_args()
        args.imu_location = "right_forearm"
        args.imu_orientation = "x_distal_y_left_z_outward"

        metadata = build_metadata(
            args,
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=0.0,
            zoom_end_s=1.0,
            imu_summary=imu_summary,
            imu_csv_path=imu_csv_path,
        )

        self.assertEqual(metadata["imu_sensor_model"], "ADXL345")
        self.assertEqual(metadata["imu_location"], "right_forearm")
        self.assertEqual(metadata["imu_sample_count"], 100)
        self.assertEqual(metadata["imu_timing_quality"], "good")
        self.assertEqual(metadata["output_imu_csv_path"], str(imu_csv_path))
        self.assertEqual(metadata["sensor_timestamp_timebase"], "esp_timer_monotonic")
        self.assertEqual(metadata["imu_warnings"], [])

    def test_metadata_preserves_live_hr_updates(self):
        diagnostics = create_firmware_diagnostics()
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line("# hr timestamp_ms=9000 bpm=na status=warming_up beats=2"),
        )
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line("# hr timestamp_ms=10000 bpm=71.8 status=stable beats=5"),
        )

        metadata = build_metadata(
            make_args(),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=0.0,
            zoom_end_s=10.0,
            firmware_diagnostics=diagnostics,
        )

        self.assertEqual(metadata["firmware_hr_update_count"], 2)
        self.assertEqual(metadata["firmware_hr_stable_update_count"], 1)
        self.assertEqual(metadata["firmware_latest_hr"]["bpm"], 71.8)

    def test_metadata_preserves_motion_and_upper_arm_mounting_details(self):
        diagnostics = create_firmware_diagnostics()
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line(
                "# motion timestamp_ms=10000 status=moving activity_g=0.120 threshold_g=0.050"
            ),
        )
        metadata = build_metadata(
            make_args(
                sensor_location="right_inner_upper_arm_3cm_above_elbow_crease",
                ppg_profile="upper_arm_experimental",
                ppg_orientation="leds_distal_photodiode_proximal",
                mounting_method="opaque_elastic_strap_dark_foam",
                strap_tension="mark_2",
            ),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=0.0,
            zoom_end_s=10.0,
            firmware_diagnostics=diagnostics,
        )

        self.assertEqual(metadata["ppg_profile"], "upper_arm_experimental")
        self.assertEqual(metadata["ppg_strap_tension"], "mark_2")
        self.assertEqual(metadata["ppg_led_current_ma"], 7.2)
        self.assertEqual(metadata["firmware_motion_update_count"], 1)
        self.assertEqual(metadata["firmware_latest_motion"]["status"], "moving")

    def test_metadata_includes_timestamp_diagnostics(self):
        metadata = build_metadata(
            make_args(),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=20.0,
            zoom_end_s=30.0,
        )

        self.assertEqual(metadata["mean_sample_interval_ms"], 10.0)
        self.assertEqual(metadata["min_sample_interval_ms"], 10.0)
        self.assertEqual(metadata["max_sample_interval_ms"], 10.0)
        self.assertEqual(metadata["p95_sample_interval_ms"], 10.0)
        self.assertEqual(metadata["p99_sample_interval_ms"], 10.0)
        self.assertEqual(metadata["timestamp_gaps_gt_15ms"], 0)
        self.assertEqual(metadata["timestamp_gaps_gt_20ms"], 0)
        self.assertEqual(metadata["non_increasing_timestamp_count"], 0)
        self.assertIsNone(metadata["timestamp_irregularity_reason"])
        self.assertEqual(metadata["timing_quality"], "good")
        self.assertEqual(
            metadata["timing_quality_reason"],
            "No missing samples, monotonic timestamps, and all intervals <= 15 ms.",
        )

    def test_metadata_includes_firmware_diagnostics_when_available(self):
        diagnostics = create_firmware_diagnostics()
        update_firmware_diagnostics(
            diagnostics,
            parse_firmware_status_line(
                "# stats samples=500 captured_samples=500 rate_hz=99.8 "
                "effective_rate_hz=99.9 fifo_avail=2 ovf=0 i2c_errors=0 "
                "timestamp_resyncs=0 timestamp_corrections=0 overflow_recoveries=0"
            ),
        )

        metadata = build_metadata(
            make_args(),
            make_summary(),
            datetime(2026, 6, 7, 20, 0, tzinfo=timezone.utc),
            interrupted=False,
            ignored_lines=0,
            zoom_start_s=20.0,
            zoom_end_s=30.0,
            firmware_diagnostics=diagnostics,
        )

        self.assertEqual(metadata["firmware_captured_samples"], 500)
        self.assertEqual(metadata["firmware_interval_rate_hz"], 99.8)
        self.assertEqual(metadata["firmware_effective_rate_hz"], 99.9)
        self.assertEqual(metadata["firmware_fifo_overflow_count"], 0)
        self.assertEqual(metadata["firmware_i2c_error_count"], 0)
        self.assertEqual(metadata["firmware_timestamp_resync_count"], 0)
        self.assertEqual(metadata["firmware_fifo_overflow_recovery_count"], 0)
        self.assertEqual(metadata["firmware_latest_stats"]["captured_samples"], 500)

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

    def test_prompt_bp_after_blank_inputs_keep_existing_values(self):
        args = make_args(
            systolic_mmHg=118,
            diastolic_mmHg=76,
            cuff_hr_bpm=72,
            cuff_reading_time_s=55.0,
            notes="Omron pilot trial 1",
        )
        responses = iter(["", "", "", "", ""])

        updated = apply_prompt_bp_after(args, input_func=lambda _prompt: next(responses), output_func=lambda _text: None)

        self.assertEqual(updated.systolic_mmHg, 118)
        self.assertEqual(updated.diastolic_mmHg, 76)
        self.assertEqual(updated.cuff_hr_bpm, 72)
        self.assertEqual(updated.cuff_reading_time_s, 55.0)
        self.assertEqual(updated.notes, "Omron pilot trial 1")

    def test_prompt_bp_after_entered_values_override_cli_values(self):
        args = make_args(
            systolic_mmHg=110,
            diastolic_mmHg=70,
            cuff_hr_bpm=65,
            cuff_reading_time_s=50.0,
            notes="Omron pilot trial 1",
        )
        responses = iter(["118", "76", "72", "55", "Reading appeared stable"])

        updated = apply_prompt_bp_after(args, input_func=lambda _prompt: next(responses), output_func=lambda _text: None)

        self.assertEqual(updated.systolic_mmHg, 118)
        self.assertEqual(updated.diastolic_mmHg, 76)
        self.assertEqual(updated.cuff_hr_bpm, 72)
        self.assertEqual(updated.cuff_reading_time_s, 55.0)
        self.assertEqual(updated.notes, "Omron pilot trial 1 | Reading appeared stable")

    def test_prompt_bp_after_reprompts_invalid_values(self):
        args = make_args()
        responses = iter(["0", "118", "", "", "91", "55", ""])
        messages = []

        updated = apply_prompt_bp_after(args, input_func=lambda _prompt: next(responses), output_func=messages.append)

        self.assertEqual(updated.systolic_mmHg, 118)
        self.assertIsNone(updated.diastolic_mmHg)
        self.assertIsNone(updated.cuff_hr_bpm)
        self.assertEqual(updated.cuff_reading_time_s, 55.0)
        self.assertTrue(any("must be greater than 0" in message for message in messages))
        self.assertTrue(any("must be between 0 and recording duration" in message for message in messages))

    def test_parses_prompt_bp_after_flag(self):
        args = parse_args([
            "--port",
            "COM3",
            "--duration",
            "90",
            "--subject",
            "test",
            "--session",
            "omron_pilot_001",
            "--prompt-bp-after",
        ])

        self.assertTrue(args.prompt_bp_after)

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


class LabelCsvTests(unittest.TestCase):
    def test_parses_label_cli_fields(self):
        args = parse_args([
            "--port",
            "COM3",
            "--duration",
            "90",
            "--subject",
            "self",
            "--session",
            "omron_pilot_002",
            "--trial-id",
            "trial_001",
            "--prompt-labels",
            "--sbp",
            "108",
            "--dbp",
            "61",
            "--omron-hr",
            "69",
            "--omron-timing",
            "during_ppg",
            "--label-notes",
            "Omron labelled finger PPG trial",
        ])

        self.assertTrue(args.prompt_labels)
        self.assertEqual(args.label_sbp, 108)
        self.assertEqual(args.label_dbp, 61)
        self.assertEqual(args.label_omron_hr, 69)
        self.assertEqual(args.label_omron_timing, "during_ppg")
        self.assertEqual(args.label_notes, "Omron labelled finger PPG trial")

    def test_builds_label_csv_path_for_session(self):
        labels_path = build_label_csv_path(Path("data/labels"), "omron_pilot_002")

        self.assertEqual(labels_path, Path("data/labels/omron_pilot_002_labels.csv"))

    def test_builds_label_row_without_touching_raw_ppg_columns(self):
        row = build_label_row(
            make_args(
                subject="self",
                session="omron_pilot_002",
                trial_id="trial_001",
                posture="seated",
                sensor_location="right_index_finger",
                ppg_hand="right",
                cuff_arm="left",
                label_sbp=108,
                label_dbp=61,
                label_omron_hr=69,
                label_omron_timing="during_ppg",
                label_notes="Omron labelled finger PPG trial",
            ),
            make_summary(),
            Path("data/raw/self_omron_pilot_002_trial_001_ppg.csv"),
            Path("data/raw/self_omron_pilot_002_trial_001_metadata.json"),
        )

        self.assertEqual(list(row.keys()), LABEL_COLUMNS)
        self.assertEqual(row["session"], "omron_pilot_002")
        self.assertEqual(row["trial_id"], "trial_001")
        self.assertEqual(row["subject"], "self")
        self.assertEqual(row["ppg_csv"], "data/raw/self_omron_pilot_002_trial_001_ppg.csv")
        self.assertEqual(row["metadata_json"], "data/raw/self_omron_pilot_002_trial_001_metadata.json")
        self.assertEqual(row["ppg_location"], "right_index_finger")
        self.assertEqual(row["sbp"], 108)
        self.assertEqual(row["dbp"], 61)
        self.assertEqual(row["omron_hr"], 69)
        self.assertEqual(row["omron_timing"], "during_ppg")
        self.assertEqual(row["timing_quality"], "good")
        self.assertEqual(row["quality"], "good")

    def test_write_label_row_creates_directory_and_skips_duplicates(self):
        with TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "data" / "labels" / "omron_pilot_002_labels.csv"
            messages = []
            row = build_label_row(
                make_args(
                    subject="self",
                    session="omron_pilot_002",
                    trial_id="trial_001",
                    label_sbp=108,
                    label_dbp=61,
                    label_omron_hr=69,
                ),
                make_summary(),
                Path("data/raw/self_omron_pilot_002_trial_001_ppg.csv"),
                Path("data/raw/self_omron_pilot_002_trial_001_metadata.json"),
            )

            first_result = write_label_row(labels_path, row, output_func=messages.append)
            second_result = write_label_row(labels_path, row, output_func=messages.append)

            self.assertEqual(first_result, "appended")
            self.assertEqual(second_result, "duplicate_skipped")
            with labels_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["session"], "omron_pilot_002")
            self.assertTrue(any("already exists" in message for message in messages))

    def test_write_label_row_can_update_duplicate_when_confirmed(self):
        with TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "data" / "labels" / "omron_pilot_002_labels.csv"
            original = build_label_row(
                make_args(session="omron_pilot_002", trial_id="trial_001", label_sbp=108),
                make_summary(),
                Path("old.csv"),
                Path("old.json"),
            )
            updated = dict(original)
            updated["sbp"] = 109
            updated["notes"] = "updated label"

            write_label_row(labels_path, original, output_func=lambda _text: None)
            result = write_label_row(labels_path, updated, update_existing=True, output_func=lambda _text: None)

            self.assertEqual(result, "updated")
            with labels_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sbp"], "109")
            self.assertEqual(rows[0]["notes"], "updated label")


class OutputPathTests(unittest.TestCase):
    def test_different_trial_ids_produce_different_output_filenames(self):
        outdir = Path("data/raw")

        trial_1 = build_output_paths(outdir, "test", "omron_pilot_001", "omron_001")
        trial_2 = build_output_paths(outdir, "test", "omron_pilot_001", "omron_002")

        self.assertEqual(trial_1["csv"].name, "test_omron_pilot_001_omron_001_ppg.csv")
        self.assertEqual(trial_2["csv"].name, "test_omron_pilot_001_omron_002_ppg.csv")
        self.assertNotEqual(trial_1["csv"], trial_2["csv"])
        self.assertEqual(trial_1["metadata"].name, "test_omron_pilot_001_omron_001_metadata.json")
        self.assertEqual(trial_1["plot"].name, "test_omron_pilot_001_omron_001_plot.png")
        self.assertEqual(trial_1["zoom_plot"].name, "test_omron_pilot_001_omron_001_zoom_plot.png")

    def test_existing_output_file_is_not_overwritten_by_default(self):
        with TemporaryDirectory() as tmpdir:
            paths = build_output_paths(Path(tmpdir), "test", "session", "trial_001")
            paths["csv"].write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError) as context:
                ensure_output_paths_available(paths, overwrite=False)

            self.assertIn(str(paths["csv"]), str(context.exception))

    def test_overwrite_allows_existing_output_file(self):
        with TemporaryDirectory() as tmpdir:
            paths = build_output_paths(Path(tmpdir), "test", "session", "trial_001")
            paths["csv"].write_text("existing\n", encoding="utf-8")

            ensure_output_paths_available(paths, overwrite=True)

    def test_parses_overwrite_flag(self):
        args = parse_args([
            "--port",
            "COM3",
            "--duration",
            "90",
            "--subject",
            "test",
            "--session",
            "session",
            "--trial-id",
            "trial_001",
            "--overwrite",
        ])

        self.assertTrue(args.overwrite)


class TimestampDiagnosticsTests(unittest.TestCase):
    def test_nominal_timestamp_cursor_output_is_monotonic_and_good(self):
        timestamps = [1234 + (index * 10) for index in range(100)]
        summary = summarize(make_ppg_df(timestamps), requested_duration_s=0.99)

        self.assertEqual(summary["non_increasing_timestamp_count"], 0)
        self.assertEqual(summary["min_sample_interval_ms"], 10.0)
        self.assertEqual(summary["max_sample_interval_ms"], 10.0)
        self.assertEqual(summary["timing_quality"], "good")

    def test_regular_10ms_timestamps_have_clean_diagnostics(self):
        summary = summarize(make_ppg_df([0, 10, 20, 30, 40, 50]), requested_duration_s=0.05)

        self.assertEqual(summary["median_dt_ms"], 10.0)
        self.assertEqual(summary["mean_sample_interval_ms"], 10.0)
        self.assertEqual(summary["min_sample_interval_ms"], 10.0)
        self.assertEqual(summary["max_sample_interval_ms"], 10.0)
        self.assertEqual(summary["p95_sample_interval_ms"], 10.0)
        self.assertEqual(summary["p99_sample_interval_ms"], 10.0)
        self.assertEqual(summary["timestamp_gaps_gt_15ms"], 0)
        self.assertEqual(summary["timestamp_gaps_gt_20ms"], 0)
        self.assertEqual(summary["non_increasing_timestamp_count"], 0)
        self.assertIsNone(summary["timestamp_irregularity_reason"])
        self.assertEqual(summary["timing_quality"], "good")
        self.assertNotIn("Timestamps", " ".join(summary["warnings"]))

    def test_20ms_timestamp_intervals_are_usable_without_warning(self):
        timestamps = []
        current = 0
        for index in range(80):
            timestamps.append(current)
            current += 20 if index % 3 == 0 else 10

        summary = summarize(make_ppg_df(timestamps), requested_duration_s=current / 1000.0)

        self.assertEqual(summary["max_sample_interval_ms"], 20.0)
        self.assertGreater(summary["timestamp_gaps_gt_15ms"], 0)
        self.assertEqual(summary["timestamp_gaps_gt_20ms"], 0)
        self.assertEqual(summary["non_increasing_timestamp_count"], 0)
        self.assertEqual(summary["timing_quality"], "usable")
        self.assertIn("max_dt=20.0 ms", summary["timing_quality_reason"])
        self.assertFalse(any("Timing" in warning or "Timestamps irregular" in warning for warning in summary["warnings"]))

    def test_rare_large_timestamp_gap_is_borderline(self):
        timestamps = [index * 10 for index in range(100)]
        timestamps[50:] = [timestamp + 30 for timestamp in timestamps[50:]]

        summary = summarize(make_ppg_df(timestamps), requested_duration_s=1.02)

        self.assertEqual(summary["timestamp_gaps_gt_15ms"], 1)
        self.assertEqual(summary["timestamp_gaps_gt_20ms"], 1)
        self.assertEqual(summary["max_sample_interval_ms"], 40.0)
        self.assertEqual(summary["timing_quality"], "borderline")
        self.assertIn("gaps_gt_20ms=1", summary["timing_quality_reason"])
        self.assertIn("Timing borderline", summary["warnings"][-1])

    def test_many_large_timestamp_gaps_are_rejected(self):
        timestamps = [index * 10 for index in range(80)]
        for gap_index in [10, 20, 30, 40, 50, 60]:
            timestamps[gap_index:] = [timestamp + 15 for timestamp in timestamps[gap_index:]]

        summary = summarize(make_ppg_df(timestamps), requested_duration_s=0.88)

        self.assertEqual(summary["timestamp_gaps_gt_20ms"], 6)
        self.assertEqual(summary["timing_quality"], "reject")
        self.assertIn("gaps_gt_20ms=6", summary["timing_quality_reason"])
        self.assertIn("Timing reject", summary["warnings"][-1])

    def test_missing_sample_sequence_is_rejected_even_with_regular_timestamps(self):
        summary = summarize(
            make_ppg_df([0, 10, 20, 30], sample_seq=[0, 1, 3, 4]),
            requested_duration_s=0.03,
        )

        self.assertEqual(summary["missing_sample_sequences"], 1)
        self.assertEqual(summary["timing_quality"], "reject")
        self.assertIn("missing_sequences=1", summary["timing_quality_reason"])

    def test_interval_over_40ms_is_rejected(self):
        summary = summarize(make_ppg_df([0, 10, 20, 70, 80]), requested_duration_s=0.08)

        self.assertEqual(summary["max_sample_interval_ms"], 50.0)
        self.assertEqual(summary["timing_quality"], "reject")
        self.assertIn("max_dt=50.0 ms", summary["timing_quality_reason"])

    def test_non_increasing_timestamps_have_specific_diagnostics(self):
        summary = summarize(make_ppg_df([0, 10, 10, 20, 15, 30]), requested_duration_s=0.03)

        self.assertEqual(summary["non_increasing_timestamp_count"], 2)
        self.assertEqual(summary["min_sample_interval_ms"], -5.0)
        self.assertEqual(summary["timestamp_gaps_gt_15ms"], 0)
        self.assertEqual(summary["timing_quality"], "reject")
        self.assertIn("non_increasing=2", summary["timing_quality_reason"])
        self.assertIn("Timing reject", summary["warnings"][-1])


if __name__ == "__main__":
    unittest.main()
