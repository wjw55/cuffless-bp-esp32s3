"""Collect synchronized MAX30102 PPG and optional ADXL345 IMU rows.

Expected firmware output:
    sample_seq,timestamp_ms,red,ir
    42,12345,48231,53120

This script saves one recording session as:
    data/raw/<subject>_<session>_<trial>_ppg.csv
    data/raw/<subject>_<session>_<trial>_imu.csv
    data/raw/<subject>_<session>_metadata.json
    data/raw/<subject>_<session>_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ADC_MAX_VALUE = 0x3FFFF
SATURATION_MARGIN = 1000
LOW_IR_THRESHOLD = 50000
SERIAL_STARTUP_DELAY_S = 1.0
DEFAULT_ZOOM_START_S = 20.0
DEFAULT_ZOOM_END_S = 30.0
DEFAULT_ZOOM_DURATION_S = 10.0
MAX_FIRMWARE_WARNING_EVENTS = 50
MAX_FIRMWARE_HR_EVENTS = 600
MAX_FIRMWARE_MOTION_EVENTS = 600
ADXL345_SCALE_G_PER_LSB = 0.0039
ADXL345_SAMPLE_RATE_HZ = 100
MOTION_MAD_MULTIPLIER = 6.0

FIRMWARE_STATS_METADATA_KEYS = {
    "samples": "firmware_status_sample_count",
    "captured_samples": "firmware_captured_samples",
    "rate_hz": "firmware_interval_rate_hz",
    "effective_rate_hz": "firmware_effective_rate_hz",
    "fifo_avail": "firmware_latest_fifo_available",
    "ovf": "firmware_fifo_overflow_count",
    "i2c_errors": "firmware_i2c_error_count",
    "timestamp_resyncs": "firmware_timestamp_resync_count",
    "timestamp_corrections": "firmware_timestamp_correction_count",
    "timestamp_lag_warnings": "firmware_timestamp_lag_warning_count",
    "overflow_recoveries": "firmware_fifo_overflow_recovery_count",
}

IMU_FIRMWARE_STATS_METADATA_KEYS = {
    "samples": "imu_firmware_sample_count",
    "rate_hz": "imu_firmware_interval_rate_hz",
    "effective_rate_hz": "imu_firmware_effective_rate_hz",
    "fifo_entries": "imu_firmware_latest_fifo_entries",
    "fifo_overflows": "imu_firmware_fifo_overflow_count",
    "i2c_errors": "imu_firmware_i2c_error_count",
    "timestamp_resyncs": "imu_firmware_timestamp_resync_count",
    "timestamp_corrections": "imu_firmware_timestamp_correction_count",
    "clock_adjustments": "imu_firmware_clock_adjustment_count",
    "clock_adjustment_us": "imu_firmware_clock_adjustment_total_us",
}

LABEL_COLUMNS = [
    "session",
    "trial_id",
    "subject",
    "ppg_csv",
    "metadata_json",
    "posture",
    "ppg_location",
    "ppg_hand",
    "cuff_arm",
    "sbp",
    "dbp",
    "omron_hr",
    "omron_timing",
    "timing_quality",
    "quality",
    "notes",
]

OMRON_TIMING_CHOICES = ("before_ppg", "during_ppg", "after_ppg", "unknown")


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")

    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")

    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")

    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synchronized PPG and optional IMU rows from ESP32 serial output."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM3")
    parser.add_argument("--duration", required=True, type=positive_float, help="Recording duration in seconds")
    parser.add_argument("--subject", required=True, help="Subject ID, for example S01")
    parser.add_argument("--session", required=True, help="Session name, for example test_001")
    parser.add_argument("--trial-id", default="", help="Trial ID, for example T01")
    parser.add_argument("--posture", default="", help="Subject posture, for example seated")
    parser.add_argument("--sensor-location", default="", help="PPG sensor location, for example index_finger")
    parser.add_argument(
        "--ppg-profile",
        choices=("finger", "upper_arm_experimental"),
        default="finger",
        help="Optical placement profile; upper-arm remains experimental",
    )
    parser.add_argument("--ppg-orientation", default="", help="Optical module orientation on the skin")
    parser.add_argument("--mounting-method", default="", help="How the optical sensor is secured")
    parser.add_argument("--strap-tension", default="", help="Repeatable strap-tension mark or setting")
    parser.add_argument(
        "--led-current-ma",
        type=positive_float,
        default=7.2,
        help="Configured MAX30102 LED current in mA, default 7.2",
    )
    parser.add_argument("--cuff-arm", default="", help="Arm used for cuff BP reference, for example left")
    parser.add_argument("--ppg-hand", default="", help="Hand used for PPG sensor, for example right")
    parser.add_argument("--imu-location", default="", help="ADXL345 mounting location, for example right_forearm")
    parser.add_argument(
        "--imu-orientation",
        default="",
        help="ADXL345 axis orientation, for example x_distal_y_left_z_outward",
    )
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate, default 115200")
    parser.add_argument("--notes", default="", help="Optional notes saved into metadata")
    parser.add_argument(
        "--systolic-mmhg",
        "--systolic-mmHg",
        dest="systolic_mmHg",
        type=positive_int,
        help="Optional cuff systolic BP",
    )
    parser.add_argument(
        "--diastolic-mmhg",
        "--diastolic-mmHg",
        dest="diastolic_mmHg",
        type=positive_int,
        help="Optional cuff diastolic BP",
    )
    parser.add_argument("--cuff-hr-bpm", type=positive_int, help="Optional cuff heart rate")
    parser.add_argument(
        "--cuff-start-time-s",
        type=nonnegative_float,
        help="Optional time in seconds when the cuff measurement was started relative to PPG recording start",
    )
    parser.add_argument(
        "--cuff-reading-time-s",
        type=nonnegative_float,
        help="Optional time in seconds when the cuff reading appeared relative to PPG recording start",
    )
    parser.add_argument("--cuff-timestamp", default="", help="Optional cuff reading timestamp, ideally ISO 8601")
    parser.add_argument(
        "--prompt-bp-after",
        action="store_true",
        help="Prompt for Omron BP values after recording finishes and before metadata is saved",
    )
    parser.add_argument("--outdir", default="data/raw", help="Output folder, default data/raw")
    parser.add_argument("--labels-dir", default="data/labels", help="Label CSV folder, default data/labels")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files")
    parser.add_argument("--plot-start", type=nonnegative_float, help="Zoom plot start time in seconds")
    parser.add_argument("--plot-end", type=nonnegative_float, help="Zoom plot end time in seconds")
    parser.add_argument(
        "--prompt-labels",
        action="store_true",
        help="Prompt for Omron labels after recording and append them to data/labels/<session>_labels.csv",
    )
    parser.add_argument("--sbp", dest="label_sbp", type=positive_int, help="Optional Omron systolic BP label")
    parser.add_argument("--dbp", dest="label_dbp", type=positive_int, help="Optional Omron diastolic BP label")
    parser.add_argument("--omron-hr", dest="label_omron_hr", type=positive_int, help="Optional Omron HR label")
    parser.add_argument(
        "--omron-timing",
        dest="label_omron_timing",
        choices=OMRON_TIMING_CHOICES,
        default="",
        help="When the Omron label was taken relative to PPG",
    )
    parser.add_argument("--label-notes", default="", help="Optional notes saved only in the labels CSV")
    parser.add_argument(
        "--live-upper-arm-validation",
        action="store_true",
        help=(
            "Show and save PC rolling upper-arm HR/quality updates while preserving the normal "
            "raw PPG, IMU, metadata, and plot outputs"
        ),
    )
    parser.add_argument(
        "--live-bp-model-dir",
        help="Show and save quality-gated experimental BP updates using a saved single-subject model directory",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Allow an explicitly labelled unvalidated BP model during live BP validation",
    )
    args = parser.parse_args(argv)
    return validate_collection_args(args, parser)


def validate_collection_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    cuff_time_fields = {
        "cuff_start_time_s": "--cuff-start-time-s",
        "cuff_reading_time_s": "--cuff-reading-time-s",
    }

    for field_name, option_name in cuff_time_fields.items():
        value = getattr(args, field_name, None)
        if value is not None and value > args.duration:
            message = f"{option_name} must be between 0 and recording duration ({args.duration:g} s)"
            if parser is not None:
                parser.error(message)
            raise SystemExit(f"ERROR: {message}")

    if getattr(args, "live_upper_arm_validation", False) and args.ppg_profile != "upper_arm_experimental":
        message = "--live-upper-arm-validation requires --ppg-profile upper_arm_experimental"
        if parser is not None:
            parser.error(message)
        raise SystemExit(f"ERROR: {message}")

    live_bp_model_dir = getattr(args, "live_bp_model_dir", None)
    if live_bp_model_dir and args.ppg_profile != "upper_arm_experimental":
        message = "--live-bp-model-dir requires --ppg-profile upper_arm_experimental"
        if parser is not None:
            parser.error(message)
        raise SystemExit(f"ERROR: {message}")
    if live_bp_model_dir and getattr(args, "live_upper_arm_validation", False):
        message = "--live-bp-model-dir cannot be combined with --live-upper-arm-validation"
        if parser is not None:
            parser.error(message)
        raise SystemExit(f"ERROR: {message}")
    if getattr(args, "allow_unvalidated", False) and not live_bp_model_dir:
        message = "--allow-unvalidated requires --live-bp-model-dir"
        if parser is not None:
            parser.error(message)
        raise SystemExit(f"ERROR: {message}")
    if live_bp_model_dir and args.duration < 85.0:
        message = "--live-bp-model-dir requires --duration of at least 85 seconds"
        if parser is not None:
            parser.error(message)
        raise SystemExit(f"ERROR: {message}")

    return args


def optional_prompt_suffix(value) -> str:
    if value is None or value == "":
        return ""

    return f" [{value}]"


def prompt_optional_positive_int(
    label: str,
    current_value: int | None,
    input_func=input,
    output_func=print,
) -> int | None:
    while True:
        raw_value = input_func(f"{label}{optional_prompt_suffix(current_value)}: ").strip()
        if raw_value == "":
            return current_value

        try:
            return positive_int(raw_value)
        except argparse.ArgumentTypeError as exc:
            output_func(f"Invalid {label}: {exc}")


def prompt_optional_recording_time(
    label: str,
    current_value: float | None,
    duration_s: float,
    input_func=input,
    output_func=print,
) -> float | None:
    while True:
        raw_value = input_func(f"{label}{optional_prompt_suffix(current_value)}: ").strip()
        if raw_value == "":
            return current_value

        try:
            parsed = nonnegative_float(raw_value)
        except argparse.ArgumentTypeError as exc:
            output_func(f"Invalid {label}: {exc}")
            continue

        if parsed > duration_s:
            output_func(f"Invalid {label}: must be between 0 and recording duration ({duration_s:g} s)")
            continue

        return parsed


def append_prompt_notes(existing_notes: str, prompted_notes: str) -> str:
    cleaned_existing = existing_notes.strip()
    cleaned_prompted = prompted_notes.strip()

    if not cleaned_prompted:
        return existing_notes

    if cleaned_existing:
        return f"{cleaned_existing} | {cleaned_prompted}"

    return cleaned_prompted


def apply_prompt_bp_after(args: argparse.Namespace, input_func=input, output_func=print) -> argparse.Namespace:
    output_func("\nEnter Omron BP values. Leave blank to keep existing values or omit optional fields.")

    args.systolic_mmHg = prompt_optional_positive_int(
        "Systolic mmHg",
        args.systolic_mmHg,
        input_func,
        output_func,
    )
    args.diastolic_mmHg = prompt_optional_positive_int(
        "Diastolic mmHg",
        args.diastolic_mmHg,
        input_func,
        output_func,
    )
    args.cuff_hr_bpm = prompt_optional_positive_int(
        "Omron HR bpm",
        args.cuff_hr_bpm,
        input_func,
        output_func,
    )
    args.cuff_reading_time_s = prompt_optional_recording_time(
        "Cuff reading time s",
        args.cuff_reading_time_s,
        args.duration,
        input_func,
        output_func,
    )

    prompted_notes = input_func(f"Additional notes{optional_prompt_suffix(args.notes)}: ")
    args.notes = append_prompt_notes(args.notes, prompted_notes)

    return args


def prompt_optional_choice(
    label: str,
    current_value: str,
    choices: tuple[str, ...],
    input_func=input,
    output_func=print,
) -> str:
    choices_text = "/".join(choices)
    while True:
        raw_value = input_func(f"{label} ({choices_text}){optional_prompt_suffix(current_value)}: ").strip()
        if raw_value == "":
            return current_value
        if raw_value in choices:
            return raw_value
        output_func(f"Invalid {label}: choose one of {choices_text}")


def label_inputs_present(args: argparse.Namespace) -> bool:
    return any(
        [
            getattr(args, "label_sbp", None) is not None,
            getattr(args, "label_dbp", None) is not None,
            getattr(args, "label_omron_hr", None) is not None,
            bool(getattr(args, "label_omron_timing", "")),
            bool(getattr(args, "label_notes", "")),
        ]
    )


def apply_prompt_labels(args: argparse.Namespace, input_func=input, output_func=print) -> bool:
    has_existing_label_values = label_inputs_present(args)
    output_func("\nEnter Omron label values. Leave SBP blank to skip label entry.")

    while True:
        raw_sbp = input_func(f"Omron SBP mmHg{optional_prompt_suffix(args.label_sbp)}: ").strip()
        if raw_sbp == "":
            if not has_existing_label_values:
                output_func("Label entry skipped.")
                return False
            break

        try:
            args.label_sbp = positive_int(raw_sbp)
            break
        except argparse.ArgumentTypeError as exc:
            output_func(f"Invalid Omron SBP mmHg: {exc}")

    args.label_dbp = prompt_optional_positive_int(
        "Omron DBP mmHg",
        args.label_dbp,
        input_func,
        output_func,
    )
    args.label_omron_hr = prompt_optional_positive_int(
        "Omron HR bpm",
        args.label_omron_hr,
        input_func,
        output_func,
    )
    args.label_omron_timing = prompt_optional_choice(
        "Omron label timing",
        args.label_omron_timing,
        OMRON_TIMING_CHOICES,
        input_func,
        output_func,
    )

    prompted_notes = input_func(f"Label notes{optional_prompt_suffix(args.label_notes)}: ")
    args.label_notes = append_prompt_notes(args.label_notes, prompted_notes)

    return label_inputs_present(args)


def get_firmware_git_commit() -> str | None:
    """Return the current repo commit when this script is run from a Git checkout."""
    project_root = Path(__file__).resolve().parents[1]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def import_dependencies():
    missing = []

    try:
        import serial  # type: ignore
    except ImportError:
        serial = None
        missing.append("pyserial")

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None
        missing.append("pandas")

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        plt = None
        missing.append("matplotlib")

    if missing:
        print(
            "ERROR: Missing Python package(s): "
            + ", ".join(missing)
            + "\nInstall them with:\n  python -m pip install pyserial pandas matplotlib",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return serial, pd, plt


def safe_name(value: str) -> str:
    """Make metadata values safe for Windows filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def build_output_paths(outdir: Path, subject: str, session: str, trial_id: str) -> dict[str, Path]:
    prefix = f"{safe_name(subject)}_{safe_name(session)}_{safe_name(trial_id)}"

    return {
        "csv": outdir / f"{prefix}_ppg.csv",
        "imu_csv": outdir / f"{prefix}_imu.csv",
        "live_hr_csv": outdir / f"{prefix}_live_hr.csv",
        "live_bp_csv": outdir / f"{prefix}_live_bp.csv",
        "metadata": outdir / f"{prefix}_metadata.json",
        "plot": outdir / f"{prefix}_plot.png",
        "zoom_plot": outdir / f"{prefix}_zoom_plot.png",
        "motion_plot": outdir / f"{prefix}_motion_plot.png",
    }


def build_label_csv_path(labels_dir: Path, session: str) -> Path:
    return labels_dir / f"{safe_name(session)}_labels.csv"


def ensure_output_paths_available(paths: dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return

    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")


def parse_ppg_row(line: str) -> tuple[int, int, int, int] | None:
    """Return a valid PPG row, or None for headers, boot logs, and debug lines."""
    text = line.strip()
    if not text:
        return None

    if text.lower() in {"sample_seq,timestamp_ms,red,ir", "timestamp_ms,red,ir"}:
        return None

    if text.startswith("#"):
        return None

    parts = text.split(",")
    if len(parts) != 4:
        return None

    try:
        sample_seq, timestamp_ms, red, ir = (int(part.strip()) for part in parts)
    except ValueError:
        return None

    if sample_seq < 0 or timestamp_ms < 0 or red < 0 or ir < 0:
        return None

    return sample_seq, timestamp_ms, red, ir


def parse_imu_row(line: str) -> tuple[int, int, int, int, int] | None:
    """Return a valid tagged ADXL345 row without confusing it with legacy PPG rows."""
    parts = line.strip().split(",")
    if len(parts) != 6 or parts[0].strip().lower() != "imu":
        return None

    try:
        imu_seq, timestamp_ms, x_raw, y_raw, z_raw = (int(part.strip()) for part in parts[1:])
    except ValueError:
        return None

    if imu_seq < 0 or timestamp_ms < 0:
        return None
    if any(value < -32768 or value > 32767 for value in (x_raw, y_raw, z_raw)):
        return None
    return imu_seq, timestamp_ms, x_raw, y_raw, z_raw


def parse_firmware_status_value(value: str):
    cleaned = value.strip().strip(",")

    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)

    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        return float(cleaned)

    return cleaned


def parse_firmware_status_line(line: str) -> tuple[str, dict] | None:
    """Parse comment-prefixed firmware status lines without touching CSV rows."""
    text = line.strip()
    if not text.startswith("#"):
        return None

    body = text[1:].strip()
    if not body:
        return None

    normalized = body.replace(",", " ")
    parts = normalized.split()
    if not parts:
        return None

    status_type = parts[0]
    fields = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = parse_firmware_status_value(value)

    return status_type, fields


def create_firmware_diagnostics() -> dict:
    return {
        "latest_stats": {},
        "latest_imu_stats": {},
        "latest_hr": {},
        "hr_updates": [],
        "latest_motion": {},
        "motion_updates": [],
        "metadata_fields": {},
        "warning_events": [],
    }


def update_firmware_diagnostics(diagnostics: dict, parsed_status: tuple[str, dict] | None) -> None:
    if parsed_status is None:
        return

    status_type, fields = parsed_status
    metadata_fields = diagnostics["metadata_fields"]

    if status_type == "stats":
        diagnostics["latest_stats"] = fields
        for status_key, metadata_key in FIRMWARE_STATS_METADATA_KEYS.items():
            if status_key in fields:
                metadata_fields[metadata_key] = fields[status_key]
        return

    if status_type == "imu_stats":
        diagnostics["latest_imu_stats"] = fields
        for status_key, metadata_key in IMU_FIRMWARE_STATS_METADATA_KEYS.items():
            if status_key in fields:
                metadata_fields[metadata_key] = fields[status_key]
        return

    if status_type == "hr":
        diagnostics["latest_hr"] = fields
        if len(diagnostics["hr_updates"]) < MAX_FIRMWARE_HR_EVENTS:
            diagnostics["hr_updates"].append(fields)
        return

    if status_type == "motion":
        diagnostics["latest_motion"] = fields
        if len(diagnostics["motion_updates"]) < MAX_FIRMWARE_MOTION_EVENTS:
            diagnostics["motion_updates"].append(fields)
        return

    if status_type != "warning":
        return

    if len(diagnostics["warning_events"]) < MAX_FIRMWARE_WARNING_EVENTS:
        diagnostics["warning_events"].append(fields)

    event = fields.get("event")
    if event == "fifo_overflow" and "total" in fields:
        metadata_fields["firmware_fifo_overflow_count"] = fields["total"]
    elif event == "fifo_read_failed" and "i2c_errors" in fields:
        metadata_fields["firmware_i2c_error_count"] = fields["i2c_errors"]
    elif event == "fifo_overflow_recovery" and "total" in fields:
        metadata_fields["firmware_fifo_overflow_recovery_count"] = fields["total"]
    elif event == "timestamp_resync" and "count" in fields:
        metadata_fields["firmware_timestamp_resync_count"] = fields["count"]
    elif event == "timestamp_correction" and "count" in fields:
        metadata_fields["firmware_timestamp_correction_count"] = fields["count"]
    elif event == "timestamp_lag" and "count" in fields:
        metadata_fields["firmware_timestamp_lag_warning_count"] = fields["count"]
    elif event == "imu_fifo_overflow" and "count" in fields:
        metadata_fields["imu_firmware_fifo_overflow_count"] = fields["count"]
    elif event in {"imu_fifo_read_failed", "imu_fifo_status_failed"} and "i2c_errors" in fields:
        metadata_fields["imu_firmware_i2c_error_count"] = fields["i2c_errors"]


def classify_timing_quality(
    missing_sample_sequences: int,
    non_increasing_timestamp_count: int,
    timestamp_gaps_gt_15ms: int,
    timestamp_gaps_gt_20ms: int,
    max_sample_interval_ms: float | None,
) -> tuple[str, str]:
    if max_sample_interval_ms is None:
        return "reject", "Fewer than 2 timestamped samples; cannot evaluate timing quality."

    if (
        missing_sample_sequences > 0
        or non_increasing_timestamp_count > 0
        or timestamp_gaps_gt_20ms > 5
        or max_sample_interval_ms > 40
    ):
        return (
            "reject",
            f"missing_sequences={missing_sample_sequences}, non_increasing={non_increasing_timestamp_count}, "
            f"max_dt={max_sample_interval_ms:.1f} ms, gaps_gt_15ms={timestamp_gaps_gt_15ms}, "
            f"gaps_gt_20ms={timestamp_gaps_gt_20ms}",
        )

    if (
        missing_sample_sequences == 0
        and non_increasing_timestamp_count == 0
        and timestamp_gaps_gt_15ms == 0
        and max_sample_interval_ms <= 15
    ):
        return "good", "No missing samples, monotonic timestamps, and all intervals <= 15 ms."

    if (
        missing_sample_sequences == 0
        and non_increasing_timestamp_count == 0
        and timestamp_gaps_gt_20ms == 0
        and max_sample_interval_ms <= 20
    ):
        return (
            "usable",
            f"No missing samples or >20 ms gaps; max_dt={max_sample_interval_ms:.1f} ms, "
            f"gaps_gt_15ms={timestamp_gaps_gt_15ms}.",
        )

    return (
        "borderline",
        f"No missing samples, but timing has larger gaps; max_dt={max_sample_interval_ms:.1f} ms, "
        f"gaps_gt_15ms={timestamp_gaps_gt_15ms}, gaps_gt_20ms={timestamp_gaps_gt_20ms}.",
    )


def summarize(df, requested_duration_s: float) -> dict:
    sample_count = int(len(df))
    data_duration_s = None
    median_dt_ms = None
    mean_sample_interval_ms = None
    min_sample_interval_ms = None
    max_sample_interval_ms = None
    p95_sample_interval_ms = None
    p99_sample_interval_ms = None
    estimated_rate_hz = None
    red_min = red_max = ir_min = ir_max = None
    sample_sequence_start = sample_sequence_end = None
    missing_sample_sequences = 0
    timestamp_gaps_gt_15ms = 0
    timestamp_gaps_gt_20ms = 0
    non_increasing_timestamp_count = 0
    timestamp_irregularity_reason = None
    timing_quality = None
    timing_quality_reason = None
    warnings = []

    if sample_count == 0:
        warnings.append("No valid CSV samples were recorded.")
    else:
        red_min = int(df["red"].min())
        red_max = int(df["red"].max())
        ir_min = int(df["ir"].min())
        ir_max = int(df["ir"].max())
        sample_sequence_start = int(df["sample_seq"].iloc[0])
        sample_sequence_end = int(df["sample_seq"].iloc[-1])

        sequence_delta = df["sample_seq"].diff().dropna()
        if bool((sequence_delta <= 0).any()):
            warnings.append("Sample sequence numbers are not strictly increasing.")

        forward_gaps = sequence_delta[sequence_delta > 1]
        if len(forward_gaps) > 0:
            missing_sample_sequences = int((forward_gaps - 1).sum())
            warnings.append(f"Sample sequence gaps detected: {missing_sample_sequences} missing sequence number(s).")

        if ir_max < LOW_IR_THRESHOLD:
            warnings.append("IR signal looks low: check finger contact and sensor alignment.")

        saturation_level = ADC_MAX_VALUE - SATURATION_MARGIN
        if red_max >= saturation_level or ir_max >= saturation_level:
            warnings.append("Signal is near ADC saturation: reduce LED current or ambient light.")

    if sample_count >= 2:
        data_duration_s = float((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0)
        dt_ms = df["timestamp_ms"].diff().dropna()
        median_dt_ms = float(dt_ms.median())
        mean_sample_interval_ms = float(dt_ms.mean())
        min_sample_interval_ms = float(dt_ms.min())
        max_sample_interval_ms = float(dt_ms.max())
        p95_sample_interval_ms = float(dt_ms.quantile(0.95))
        p99_sample_interval_ms = float(dt_ms.quantile(0.99))
        timestamp_gaps_gt_15ms = int((dt_ms > 15).sum())
        timestamp_gaps_gt_20ms = int((dt_ms > 20).sum())
        non_increasing_timestamp_count = int((dt_ms <= 0).sum())

        if median_dt_ms > 0:
            estimated_rate_hz = 1000.0 / median_dt_ms

        timing_quality, timing_quality_reason = classify_timing_quality(
            missing_sample_sequences,
            non_increasing_timestamp_count,
            timestamp_gaps_gt_15ms,
            timestamp_gaps_gt_20ms,
            max_sample_interval_ms,
        )

        if timing_quality in {"borderline", "reject"}:
            timestamp_irregularity_reason = (
                f"p99_dt={p99_sample_interval_ms:.1f} ms, max_dt={max_sample_interval_ms:.1f} ms, "
                f"gaps_gt_15ms={timestamp_gaps_gt_15ms}, gaps_gt_20ms={timestamp_gaps_gt_20ms}, "
                f"non_increasing={non_increasing_timestamp_count}"
            )
            warnings.append(f"Timing {timing_quality}: {timing_quality_reason}")
    else:
        timing_quality, timing_quality_reason = classify_timing_quality(
            missing_sample_sequences,
            non_increasing_timestamp_count,
            timestamp_gaps_gt_15ms,
            timestamp_gaps_gt_20ms,
            max_sample_interval_ms,
        )
        warnings.append(f"Timing {timing_quality}: {timing_quality_reason}")

    return {
        "sample_count": sample_count,
        "requested_duration_s": requested_duration_s,
        "data_duration_s": data_duration_s,
        "median_dt_ms": median_dt_ms,
        "mean_sample_interval_ms": mean_sample_interval_ms,
        "min_sample_interval_ms": min_sample_interval_ms,
        "max_sample_interval_ms": max_sample_interval_ms,
        "p95_sample_interval_ms": p95_sample_interval_ms,
        "p99_sample_interval_ms": p99_sample_interval_ms,
        "estimated_rate_hz": estimated_rate_hz,
        "red_min": red_min,
        "red_max": red_max,
        "ir_min": ir_min,
        "ir_max": ir_max,
        "sample_sequence_start": sample_sequence_start,
        "sample_sequence_end": sample_sequence_end,
        "missing_sample_sequences": missing_sample_sequences,
        "timestamp_gaps_gt_15ms": timestamp_gaps_gt_15ms,
        "timestamp_gaps_gt_20ms": timestamp_gaps_gt_20ms,
        "non_increasing_timestamp_count": non_increasing_timestamp_count,
        "timestamp_irregularity_reason": timestamp_irregularity_reason,
        "timing_quality": timing_quality,
        "timing_quality_reason": timing_quality_reason,
        "warnings": warnings,
    }


def add_motion_columns(imu_df):
    """Convert raw acceleration to g and derive a data-adaptive motion flag."""
    result = imu_df.copy()
    for axis in ("x", "y", "z"):
        result[f"{axis}_g"] = result[f"{axis}_raw"].astype(float) * ADXL345_SCALE_G_PER_LSB

    result["accel_magnitude_g"] = (
        result["x_g"] ** 2 + result["y_g"] ** 2 + result["z_g"] ** 2
    ) ** 0.5
    gravity_window_samples = ADXL345_SAMPLE_RATE_HZ + 1
    result["gravity_estimate_g"] = result["accel_magnitude_g"].rolling(
        gravity_window_samples,
        center=True,
        min_periods=1,
    ).median()
    result["dynamic_accel_g"] = (result["accel_magnitude_g"] - result["gravity_estimate_g"]).abs()

    median_dynamic = float(result["dynamic_accel_g"].median()) if len(result) else 0.0
    mad = (
        float((result["dynamic_accel_g"] - median_dynamic).abs().median())
        if len(result)
        else 0.0
    )
    threshold_g = median_dynamic + MOTION_MAD_MULTIPLIER * 1.4826 * mad
    result["motion_candidate"] = result["dynamic_accel_g"] > threshold_g
    return result, threshold_g


def summarize_imu(imu_df, requested_duration_s: float) -> dict:
    """Summarize raw IMU timing and exploratory, recording-specific motion flags."""
    sample_count = int(len(imu_df))
    summary = {
        "sample_count": sample_count,
        "requested_duration_s": requested_duration_s,
        "data_duration_s": None,
        "sample_sequence_start": None,
        "sample_sequence_end": None,
        "missing_sample_sequences": 0,
        "median_dt_ms": None,
        "mean_sample_interval_ms": None,
        "min_sample_interval_ms": None,
        "max_sample_interval_ms": None,
        "p95_sample_interval_ms": None,
        "p99_sample_interval_ms": None,
        "estimated_rate_hz": None,
        "timestamp_gaps_gt_15ms": 0,
        "timestamp_gaps_gt_20ms": 0,
        "non_increasing_timestamp_count": 0,
        "timing_quality": "reject",
        "timing_quality_reason": "No IMU samples were recorded.",
        "motion_threshold_g": None,
        "motion_candidate_samples": 0,
        "motion_candidate_fraction": None,
        "warnings": [],
    }
    if sample_count == 0:
        summary["warnings"].append("No ADXL345 samples were recorded; check wiring and I2C address 0x53.")
        return summary

    summary["sample_sequence_start"] = int(imu_df["imu_seq"].iloc[0])
    summary["sample_sequence_end"] = int(imu_df["imu_seq"].iloc[-1])
    sequence_delta = imu_df["imu_seq"].diff().dropna()
    forward_gaps = sequence_delta[sequence_delta > 1]
    summary["missing_sample_sequences"] = int((forward_gaps - 1).sum()) if len(forward_gaps) else 0
    if bool((sequence_delta <= 0).any()):
        summary["warnings"].append("IMU sequence numbers are not strictly increasing.")

    if sample_count >= 2:
        dt_ms = imu_df["timestamp_ms"].diff().dropna()
        summary["data_duration_s"] = float(
            (imu_df["timestamp_ms"].iloc[-1] - imu_df["timestamp_ms"].iloc[0]) / 1000.0
        )
        summary["median_dt_ms"] = float(dt_ms.median())
        summary["mean_sample_interval_ms"] = float(dt_ms.mean())
        summary["min_sample_interval_ms"] = float(dt_ms.min())
        summary["max_sample_interval_ms"] = float(dt_ms.max())
        summary["p95_sample_interval_ms"] = float(dt_ms.quantile(0.95))
        summary["p99_sample_interval_ms"] = float(dt_ms.quantile(0.99))
        summary["timestamp_gaps_gt_15ms"] = int((dt_ms > 15).sum())
        summary["timestamp_gaps_gt_20ms"] = int((dt_ms > 20).sum())
        summary["non_increasing_timestamp_count"] = int((dt_ms <= 0).sum())
        if summary["median_dt_ms"] > 0:
            summary["estimated_rate_hz"] = 1000.0 / summary["median_dt_ms"]
        summary["timing_quality"], summary["timing_quality_reason"] = classify_timing_quality(
            summary["missing_sample_sequences"],
            summary["non_increasing_timestamp_count"],
            summary["timestamp_gaps_gt_15ms"],
            summary["timestamp_gaps_gt_20ms"],
            summary["max_sample_interval_ms"],
        )

    enriched, threshold_g = add_motion_columns(imu_df)
    motion_samples = int(enriched["motion_candidate"].sum())
    summary["motion_threshold_g"] = threshold_g
    summary["motion_candidate_samples"] = motion_samples
    summary["motion_candidate_fraction"] = motion_samples / sample_count
    if summary["missing_sample_sequences"]:
        summary["warnings"].append(
            f"IMU sequence gaps detected: {summary['missing_sample_sequences']} missing sequence number(s)."
        )
    if summary["timing_quality"] in {"borderline", "reject"}:
        summary["warnings"].append(f"IMU timing {summary['timing_quality']}: {summary['timing_quality_reason']}")
    return summary


def resolve_plot_window(df, plot_start: float | None, plot_end: float | None) -> tuple[float, float]:
    """Choose the zoom window in seconds relative to the first recorded sample."""
    if plot_start is not None and plot_end is not None:
        start_s = plot_start
        end_s = plot_end
    elif plot_start is not None:
        start_s = plot_start
        end_s = plot_start + DEFAULT_ZOOM_DURATION_S
    elif plot_end is not None:
        end_s = plot_end
        start_s = max(0.0, plot_end - DEFAULT_ZOOM_DURATION_S)
    elif len(df) >= 2:
        data_duration_s = float((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0)
        if data_duration_s >= DEFAULT_ZOOM_END_S:
            start_s = DEFAULT_ZOOM_START_S
            end_s = DEFAULT_ZOOM_END_S
        else:
            window_s = min(DEFAULT_ZOOM_DURATION_S, data_duration_s)
            start_s = max(0.0, (data_duration_s - window_s) / 2.0)
            end_s = start_s + window_s
    else:
        start_s = 0.0
        end_s = DEFAULT_ZOOM_DURATION_S

    if end_s <= start_s:
        raise ValueError("--plot-end must be greater than --plot-start")

    return start_s, end_s


def baseline_removed(series):
    """Remove the large DC level by subtracting the mean of the plotted window."""
    return series - series.mean()


def save_plot(df, plot_path: Path, plt) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    if len(df) > 0:
        time_s = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
        axes[0].plot(time_s, df["ir"], linewidth=0.8)
        axes[1].plot(time_s, df["red"], linewidth=0.8, color="tab:red")
    else:
        axes[0].text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=axes[0].transAxes)

    axes[0].set_ylabel("IR")
    axes[1].set_ylabel("Red")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title("Raw PPG quick-look")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def save_zoom_plot(df, zoom_plot_path: Path, plt, start_s: float, end_s: float) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    if len(df) > 0:
        time_s = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
        window = df[(time_s >= start_s) & (time_s <= end_s)]
        window_time_s = time_s.loc[window.index]

        if len(window) > 0:
            # This is only for visualization: subtract the window mean so the pulse is easier to see.
            axes[0].plot(window_time_s, baseline_removed(window["ir"]), linewidth=0.9)
            axes[1].plot(window_time_s, baseline_removed(window["red"]), linewidth=0.9, color="tab:red")
        else:
            axes[0].text(
                0.5,
                0.5,
                "No samples in selected zoom window",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
    else:
        axes[0].text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=axes[0].transAxes)

    axes[0].set_ylabel("IR - mean(IR)")
    axes[1].set_ylabel("Red - mean(Red)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(f"Baseline-removed PPG zoom ({start_s:.2f}-{end_s:.2f} s)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(zoom_plot_path, dpi=150)
    plt.close(fig)


def save_motion_plot(ppg_df, imu_df, motion_plot_path: Path, plt) -> None:
    """Plot both streams against their shared ESP timer and highlight motion candidates."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    first_timestamps = []
    if len(ppg_df):
        first_timestamps.append(float(ppg_df["timestamp_ms"].iloc[0]))
    if len(imu_df):
        first_timestamps.append(float(imu_df["timestamp_ms"].iloc[0]))
    origin_ms = min(first_timestamps) if first_timestamps else 0.0

    if len(ppg_df):
        ppg_time_s = (ppg_df["timestamp_ms"].astype(float) - origin_ms) / 1000.0
        axes[0].plot(ppg_time_s, baseline_removed(ppg_df["ir"]), linewidth=0.75, color="tab:blue")
    else:
        axes[0].text(0.5, 0.5, "No PPG samples", ha="center", va="center", transform=axes[0].transAxes)

    if len(imu_df):
        enriched, threshold_g = add_motion_columns(imu_df)
        imu_time_s = (enriched["timestamp_ms"].astype(float) - origin_ms) / 1000.0
        axes[1].plot(imu_time_s, enriched["accel_magnitude_g"], linewidth=0.75, color="tab:green")
        axes[2].plot(imu_time_s, enriched["dynamic_accel_g"], linewidth=0.75, color="tab:orange")
        axes[2].axhline(threshold_g, color="tab:red", linestyle="--", linewidth=0.9, label="data-adaptive threshold")
        axes[2].fill_between(
            imu_time_s,
            0,
            enriched["dynamic_accel_g"],
            where=enriched["motion_candidate"],
            color="tab:red",
            alpha=0.3,
            label="motion candidate",
        )
        axes[2].legend(loc="upper right")
    else:
        axes[1].text(0.5, 0.5, "No IMU samples", ha="center", va="center", transform=axes[1].transAxes)

    axes[0].set_ylabel("IR - mean")
    axes[1].set_ylabel("|accel| (g)")
    axes[2].set_ylabel("Dynamic (g)")
    axes[2].set_xlabel("Shared ESP timer (s)")
    axes[0].set_title("Synchronized PPG and general body/arm motion diagnostic")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(motion_plot_path, dpi=150)
    plt.close(fig)


def print_summary(
    summary: dict,
    imu_summary: dict,
    csv_path: Path,
    imu_csv_path: Path,
    metadata_path: Path,
    plot_path: Path,
    zoom_plot_path: Path,
    motion_plot_path: Path,
    zoom_start_s: float,
    zoom_end_s: float,
) -> None:
    def format_optional(value, digits: int = 2) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    print("\nRecording summary")
    print(f"  sample count: {summary['sample_count']}")
    print(f"  requested duration: {summary['requested_duration_s']:.2f} s")
    print(f"  data duration: {format_optional(summary['data_duration_s'])} s")
    print(f"  median dt: {format_optional(summary['median_dt_ms'])} ms")
    print(f"  mean dt: {format_optional(summary['mean_sample_interval_ms'])} ms")
    print(
        "  dt min/p95/p99/max: "
        f"{format_optional(summary['min_sample_interval_ms'])} / "
        f"{format_optional(summary['p95_sample_interval_ms'])} / "
        f"{format_optional(summary['p99_sample_interval_ms'])} / "
        f"{format_optional(summary['max_sample_interval_ms'])} ms"
    )
    print(f"  estimated sampling rate: {format_optional(summary['estimated_rate_hz'])} Hz")
    print(f"  red min/max: {format_optional(summary['red_min'], 0)} / {format_optional(summary['red_max'], 0)}")
    print(f"  ir min/max: {format_optional(summary['ir_min'], 0)} / {format_optional(summary['ir_max'], 0)}")
    print(
        "  sample seq: "
        f"{format_optional(summary['sample_sequence_start'], 0)}-{format_optional(summary['sample_sequence_end'], 0)}"
    )
    print(f"  missing sample seq: {summary['missing_sample_sequences']}")
    print(
        "  timestamp gaps: "
        f">15 ms={summary['timestamp_gaps_gt_15ms']}, "
        f">20 ms={summary['timestamp_gaps_gt_20ms']}, "
        f"non-increasing={summary['non_increasing_timestamp_count']}"
    )
    print(f"  timestamp irregularity: {summary['timestamp_irregularity_reason'] or 'none'}")
    print(f"  timing quality: {summary['timing_quality']} - {summary['timing_quality_reason']}")

    if summary["warnings"]:
        print("  warnings:")
        for warning in summary["warnings"]:
            print(f"    - {warning}")
    else:
        print("  warnings: none")

    print("\nIMU summary")
    print(f"  sample count: {imu_summary['sample_count']}")
    print(f"  estimated sampling rate: {format_optional(imu_summary['estimated_rate_hz'])} Hz")
    print(f"  missing sample seq: {imu_summary['missing_sample_sequences']}")
    print(f"  timing quality: {imu_summary['timing_quality']} - {imu_summary['timing_quality_reason']}")
    print(f"  exploratory motion threshold: {format_optional(imu_summary['motion_threshold_g'], 4)} g")
    print(f"  candidate motion fraction: {format_optional(imu_summary['motion_candidate_fraction'], 4)}")

    print("\nSaved files")
    print(f"  CSV: {csv_path}")
    print(f"  IMU CSV: {imu_csv_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  plot: {plot_path}")
    print(f"  zoom plot: {zoom_plot_path}")
    print(f"  motion plot: {motion_plot_path}")
    print(f"  zoom window: {zoom_start_s:.2f}-{zoom_end_s:.2f} s")


def build_label_row(
    args: argparse.Namespace,
    summary: dict,
    csv_path: Path,
    metadata_path: Path,
) -> dict:
    timing_quality = summary.get("timing_quality") or ""
    row = {
        "session": args.session,
        "trial_id": args.trial_id,
        "subject": args.subject,
        "ppg_csv": csv_path.as_posix(),
        "metadata_json": metadata_path.as_posix(),
        "posture": args.posture,
        "ppg_location": args.sensor_location,
        "ppg_hand": args.ppg_hand,
        "cuff_arm": args.cuff_arm,
        "sbp": args.label_sbp if args.label_sbp is not None else "",
        "dbp": args.label_dbp if args.label_dbp is not None else "",
        "omron_hr": args.label_omron_hr if args.label_omron_hr is not None else "",
        "omron_timing": args.label_omron_timing or "unknown",
        "timing_quality": timing_quality,
        "quality": timing_quality,
        "notes": args.label_notes,
    }

    return {column: row[column] for column in LABEL_COLUMNS}


def write_label_row(labels_path: Path, row: dict, update_existing: bool = False, output_func=print) -> str:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if labels_path.exists():
        with labels_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    duplicate_index = next(
        (
            index
            for index, existing_row in enumerate(rows)
            if existing_row.get("session") == row["session"] and existing_row.get("trial_id") == row["trial_id"]
        ),
        None,
    )

    if duplicate_index is not None:
        if not update_existing:
            output_func(
                f"WARNING: Label row already exists for session={row['session']} "
                f"trial_id={row['trial_id']}; not appending duplicate."
            )
            return "duplicate_skipped"
        rows[duplicate_index] = row
        result = "updated"
    else:
        rows.append(row)
        result = "appended"

    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    output_func(f"Label row {result}: {labels_path}")
    return result


def build_metadata(
    args: argparse.Namespace,
    summary: dict,
    recording_start: datetime,
    interrupted: bool,
    ignored_lines: int,
    zoom_start_s: float,
    zoom_end_s: float,
    csv_path: Path | None = None,
    firmware_diagnostics: dict | None = None,
    imu_summary: dict | None = None,
    imu_csv_path: Path | None = None,
    live_hr_csv_path: Path | None = None,
    live_hr_records: list[dict] | None = None,
) -> dict:
    firmware_diagnostics = firmware_diagnostics or create_firmware_diagnostics()
    imu_summary = imu_summary or {
        "sample_count": 0,
        "data_duration_s": None,
        "sample_sequence_start": None,
        "sample_sequence_end": None,
        "missing_sample_sequences": 0,
        "median_dt_ms": None,
        "mean_sample_interval_ms": None,
        "min_sample_interval_ms": None,
        "max_sample_interval_ms": None,
        "p95_sample_interval_ms": None,
        "p99_sample_interval_ms": None,
        "estimated_rate_hz": None,
        "timestamp_gaps_gt_15ms": 0,
        "timestamp_gaps_gt_20ms": 0,
        "non_increasing_timestamp_count": 0,
        "timing_quality": "reject",
        "timing_quality_reason": "No IMU samples were recorded.",
        "motion_threshold_g": None,
        "motion_candidate_samples": 0,
        "motion_candidate_fraction": None,
        "warnings": [],
    }
    firmware_metadata = {
        metadata_key: firmware_diagnostics["metadata_fields"].get(metadata_key)
        for metadata_key in (
            list(FIRMWARE_STATS_METADATA_KEYS.values()) + list(IMU_FIRMWARE_STATS_METADATA_KEYS.values())
        )
    }
    imu_warnings = list(imu_summary["warnings"])
    if imu_summary["sample_count"] > 0:
        if not getattr(args, "imu_location", "").strip():
            imu_warnings.append("IMU samples were recorded without --imu-location metadata.")
        if not getattr(args, "imu_orientation", "").strip():
            imu_warnings.append("IMU samples were recorded without --imu-orientation metadata.")

    live_hr_records = live_hr_records or []
    stable_live_records = [
        record
        for record in live_hr_records
        if record.get("status") == "stable" and isinstance(record.get("bpm"), (int, float))
    ]
    metadata = {
        "subject_id": args.subject,
        "session_id": args.session,
        "trial_id": args.trial_id,
        "output_csv_filename": csv_path.name if csv_path is not None else None,
        "output_csv_path": str(csv_path) if csv_path is not None else None,
        "output_imu_csv_filename": imu_csv_path.name if imu_csv_path is not None else None,
        "output_imu_csv_path": str(imu_csv_path) if imu_csv_path is not None else None,
        "output_live_hr_csv_filename": live_hr_csv_path.name if live_hr_csv_path is not None else None,
        "output_live_hr_csv_path": str(live_hr_csv_path) if live_hr_csv_path is not None else None,
        "posture": args.posture,
        "sensor_location": args.sensor_location,
        "ppg_profile": getattr(args, "ppg_profile", "finger"),
        "ppg_orientation": getattr(args, "ppg_orientation", ""),
        "ppg_mounting_method": getattr(args, "mounting_method", ""),
        "ppg_strap_tension": getattr(args, "strap_tension", ""),
        "ppg_led_current_ma": getattr(args, "led_current_ma", None),
        "cuff_arm": args.cuff_arm,
        "ppg_hand": args.ppg_hand,
        "imu_sensor_model": "ADXL345",
        "imu_configured_i2c_address": "0x53",
        "imu_detected": imu_summary["sample_count"] > 0,
        "imu_i2c_address": "0x53" if imu_summary["sample_count"] > 0 else None,
        "imu_location": getattr(args, "imu_location", ""),
        "imu_orientation": getattr(args, "imu_orientation", ""),
        "imu_range_g": 4,
        "imu_full_resolution": True,
        "imu_nominal_sample_rate_hz": ADXL345_SAMPLE_RATE_HZ,
        "imu_scale_g_per_lsb": ADXL345_SCALE_G_PER_LSB,
        "imu_role": "general_body_arm_motion_quality_flag",
        "imu_local_finger_motion_limitation": True,
        "sensor_timestamp_timebase": "esp_timer_monotonic",
        "port": args.port,
        "baud_rate": args.baud,
        "duration_seconds": args.duration,
        "data_duration_seconds": summary["data_duration_s"],
        "recording_start_time": recording_start.isoformat(timespec="seconds"),
        "firmware_git_commit": get_firmware_git_commit(),
        "sample_count": summary["sample_count"],
        "sample_sequence_start": summary["sample_sequence_start"],
        "sample_sequence_end": summary["sample_sequence_end"],
        "missing_sample_sequences": summary["missing_sample_sequences"],
        "median_sample_interval_ms": summary["median_dt_ms"],
        "mean_sample_interval_ms": summary["mean_sample_interval_ms"],
        "min_sample_interval_ms": summary["min_sample_interval_ms"],
        "max_sample_interval_ms": summary["max_sample_interval_ms"],
        "p95_sample_interval_ms": summary["p95_sample_interval_ms"],
        "p99_sample_interval_ms": summary["p99_sample_interval_ms"],
        "timestamp_gaps_gt_15ms": summary["timestamp_gaps_gt_15ms"],
        "timestamp_gaps_gt_20ms": summary["timestamp_gaps_gt_20ms"],
        "non_increasing_timestamp_count": summary["non_increasing_timestamp_count"],
        "timestamp_irregularity_reason": summary["timestamp_irregularity_reason"],
        "timing_quality": summary["timing_quality"],
        "timing_quality_reason": summary["timing_quality_reason"],
        "approximate_sampling_rate_hz": summary["estimated_rate_hz"],
        "systolic_mmHg": args.systolic_mmHg,
        "diastolic_mmHg": args.diastolic_mmHg,
        "cuff_hr_bpm": args.cuff_hr_bpm,
        "cuff_start_time_s": args.cuff_start_time_s,
        "cuff_reading_time_s": args.cuff_reading_time_s,
        "cuff_timestamp": args.cuff_timestamp or None,
        "prompt_bp_after": args.prompt_bp_after,
        "notes": args.notes,
        "pc_upper_arm_live_validation_enabled": bool(
            getattr(args, "live_upper_arm_validation", False)
        ),
        "pc_upper_arm_live_updates": live_hr_records,
        "pc_upper_arm_live_update_count": len(live_hr_records),
        "pc_upper_arm_live_stable_update_count": len(stable_live_records),
        "pc_upper_arm_live_first_stable_elapsed_s": (
            stable_live_records[0].get("elapsed_s") if stable_live_records else None
        ),
        "interrupted": interrupted,
        "ignored_non_csv_lines": ignored_lines,
        "zoom_plot_start_seconds": zoom_start_s,
        "zoom_plot_end_seconds": zoom_end_s,
        "warnings": summary["warnings"],
        "imu_sample_count": imu_summary["sample_count"],
        "imu_data_duration_seconds": imu_summary["data_duration_s"],
        "imu_sample_sequence_start": imu_summary["sample_sequence_start"],
        "imu_sample_sequence_end": imu_summary["sample_sequence_end"],
        "imu_missing_sample_sequences": imu_summary["missing_sample_sequences"],
        "imu_median_sample_interval_ms": imu_summary["median_dt_ms"],
        "imu_mean_sample_interval_ms": imu_summary["mean_sample_interval_ms"],
        "imu_min_sample_interval_ms": imu_summary["min_sample_interval_ms"],
        "imu_max_sample_interval_ms": imu_summary["max_sample_interval_ms"],
        "imu_p95_sample_interval_ms": imu_summary["p95_sample_interval_ms"],
        "imu_p99_sample_interval_ms": imu_summary["p99_sample_interval_ms"],
        "imu_timestamp_gaps_gt_15ms": imu_summary["timestamp_gaps_gt_15ms"],
        "imu_timestamp_gaps_gt_20ms": imu_summary["timestamp_gaps_gt_20ms"],
        "imu_non_increasing_timestamp_count": imu_summary["non_increasing_timestamp_count"],
        "imu_timing_quality": imu_summary["timing_quality"],
        "imu_timing_quality_reason": imu_summary["timing_quality_reason"],
        "imu_approximate_sampling_rate_hz": imu_summary["estimated_rate_hz"],
        "imu_motion_threshold_method": "median_plus_6_scaled_mad_of_dynamic_acceleration",
        "imu_motion_threshold_g": imu_summary["motion_threshold_g"],
        "imu_motion_candidate_samples": imu_summary["motion_candidate_samples"],
        "imu_motion_candidate_fraction": imu_summary["motion_candidate_fraction"],
        "imu_warnings": imu_warnings,
    }

    metadata.update(firmware_metadata)
    metadata["firmware_latest_stats"] = firmware_diagnostics["latest_stats"] or None
    metadata["imu_firmware_latest_stats"] = firmware_diagnostics["latest_imu_stats"] or None
    metadata["firmware_latest_hr"] = firmware_diagnostics["latest_hr"] or None
    metadata["firmware_hr_updates"] = firmware_diagnostics["hr_updates"]
    metadata["firmware_hr_update_count"] = len(firmware_diagnostics["hr_updates"])
    metadata["firmware_hr_stable_update_count"] = sum(
        update.get("status") == "stable" and isinstance(update.get("bpm"), (int, float))
        for update in firmware_diagnostics["hr_updates"]
    )
    metadata["firmware_latest_motion"] = firmware_diagnostics["latest_motion"] or None
    metadata["firmware_motion_updates"] = firmware_diagnostics["motion_updates"]
    metadata["firmware_motion_update_count"] = len(firmware_diagnostics["motion_updates"])
    metadata["firmware_warning_events"] = firmware_diagnostics["warning_events"]

    return metadata


def main() -> int:
    args = parse_args()

    if args.plot_start is not None and args.plot_end is not None and args.plot_end <= args.plot_start:
        print("ERROR: --plot-end must be greater than --plot-start", file=sys.stderr)
        return 1

    if args.plot_start is None and args.plot_end is not None and args.plot_end <= 0:
        print("ERROR: --plot-end must be greater than 0", file=sys.stderr)
        return 1

    serial, pd, plt = import_dependencies()
    live_viewer = None
    live_bp_viewer = None
    live_bp_context = None
    if args.live_upper_arm_validation:
        try:
            import view_live_upper_arm_hr as live_viewer
        except ImportError as exc:
            print(
                "ERROR: Live upper-arm validation requires NumPy, pandas, and SciPy in this "
                f"Python environment.\nDetails: {exc}",
                file=sys.stderr,
            )
            return 2
    elif args.live_bp_model_dir:
        try:
            import view_live_bp as live_bp_viewer

            from bp_core.inference import load_model_bundle

            bundle = load_model_bundle(
                args.live_bp_model_dir,
                expected_participant_id=args.subject,
                allow_unvalidated=bool(args.allow_unvalidated),
            )
            live_bp_context = live_bp_viewer.ViewerContext(
                participant_id=bundle.participant_id,
                calibration_sbp=bundle.calibration_sbp,
                calibration_dbp=bundle.calibration_dbp,
                config=bundle.config,
                bundle=bundle,
                allow_unvalidated=bool(args.allow_unvalidated),
            )
        except (ImportError, ValueError, OSError) as exc:
            print(f"ERROR: Live BP validation could not load the model.\nDetails: {exc}", file=sys.stderr)
            return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_paths = build_output_paths(outdir, args.subject, args.session, args.trial_id)
    try:
        ensure_output_paths_available(output_paths, args.overwrite)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    csv_path = output_paths["csv"]
    imu_csv_path = output_paths["imu_csv"]
    live_hr_csv_path = output_paths["live_hr_csv"]
    live_bp_csv_path = output_paths["live_bp_csv"]
    metadata_path = output_paths["metadata"]
    plot_path = output_paths["plot"]
    zoom_plot_path = output_paths["zoom_plot"]
    motion_plot_path = output_paths["motion_plot"]

    rows: list[tuple[int, int, int, int]] = []
    imu_rows: list[tuple[int, int, int, int, int]] = []
    ignored_lines = 0
    interrupted = False
    firmware_diagnostics = create_firmware_diagnostics()
    live_hr_records: list[dict] = []
    live_bp_records: list[dict] = []
    live_state = None

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            active_viewer = live_viewer or live_bp_viewer
            if active_viewer is not None:
                try:
                    ser.set_buffer_size(rx_size=active_viewer.SERIAL_RECEIVE_BUFFER_BYTES)
                except (AttributeError, NotImplementedError, OSError):
                    pass
            # Many ESP32 boards reset when the serial port opens. Give boot text time to pass.
            time.sleep(SERIAL_STARTUP_DELAY_S)
            ser.reset_input_buffer()

            recording_start = datetime.now().astimezone()
            recording_start_monotonic = time.monotonic()
            deadline = recording_start_monotonic + args.duration
            next_live_update = recording_start_monotonic
            if live_viewer is not None:
                live_state = live_viewer.UpperArmViewerState(started_at=recording_start_monotonic)
            elif live_bp_viewer is not None:
                live_state = live_bp_viewer.BPViewerState(started_at=recording_start_monotonic)
            print(f"Recording for {args.duration:.2f} s. Press Ctrl+C to stop early and save.")

            while time.monotonic() < deadline:
                raw_line = ser.readline()
                now = time.monotonic()
                if not raw_line:
                    if live_viewer is not None and live_state is not None and now >= next_live_update:
                        live_viewer.maybe_analyze(live_state, now)
                        live_hr_records.append(
                            live_viewer.build_validation_record(
                                live_state,
                                now,
                                now - recording_start_monotonic,
                            )
                        )
                        live_viewer.clear_and_render(
                            live_viewer.render_screen(live_state, now, args.port, args.baud, saving=True)
                        )
                        next_live_update = now + live_viewer.DEFAULT_REFRESH_SECONDS
                    elif live_bp_viewer is not None and live_state is not None and live_bp_context is not None and now >= next_live_update:
                        live_bp_viewer.maybe_predict(live_state, live_bp_context, now)
                        live_bp_records.append(
                            live_bp_viewer.build_validation_record(
                                live_state, live_bp_context, now, now - recording_start_monotonic
                            )
                        )
                        live_bp_viewer.clear_and_render(
                            live_bp_viewer.render_screen(
                                live_state, live_bp_context, now, args.port, args.baud, saving=True
                            )
                        )
                        next_live_update = now + live_bp_viewer.DEFAULT_REFRESH_SECONDS
                    continue

                line = raw_line.decode("utf-8", errors="replace")
                if live_viewer is not None and live_state is not None:
                    live_viewer.update_state_from_line(live_state, line, now)
                elif live_bp_viewer is not None and live_state is not None:
                    live_bp_viewer.update_state_from_line(live_state, line, now)
                update_firmware_diagnostics(firmware_diagnostics, parse_firmware_status_line(line))
                imu_row = parse_imu_row(line)
                if imu_row is not None:
                    imu_rows.append(imu_row)
                else:
                    row = parse_ppg_row(line)
                    if row is None:
                        ignored_lines += 1
                    else:
                        rows.append(row)

                if live_viewer is not None and live_state is not None and now >= next_live_update:
                    live_viewer.maybe_analyze(live_state, now)
                    live_hr_records.append(
                        live_viewer.build_validation_record(
                            live_state,
                            now,
                            now - recording_start_monotonic,
                        )
                    )
                    live_viewer.clear_and_render(
                        live_viewer.render_screen(live_state, now, args.port, args.baud, saving=True)
                    )
                    next_live_update = now + live_viewer.DEFAULT_REFRESH_SECONDS
                elif live_bp_viewer is not None and live_state is not None and live_bp_context is not None and now >= next_live_update:
                    live_bp_viewer.maybe_predict(live_state, live_bp_context, now)
                    live_bp_records.append(
                        live_bp_viewer.build_validation_record(
                            live_state, live_bp_context, now, now - recording_start_monotonic
                        )
                    )
                    live_bp_viewer.clear_and_render(
                        live_bp_viewer.render_screen(
                            live_state, live_bp_context, now, args.port, args.baud, saving=True
                        )
                    )
                    next_live_update = now + live_bp_viewer.DEFAULT_REFRESH_SECONDS

    except KeyboardInterrupt:
        interrupted = True
        recording_start = locals().get("recording_start", datetime.now().astimezone())
        print("\nCtrl+C received. Saving collected samples...")
    except serial.SerialException as exc:
        print(
            f"ERROR: Could not open or read from serial port {args.port} at {args.baud} baud.\n"
            f"Details: {exc}\n"
            "Close ESP-IDF monitor or any other app using the COM port, then try again.",
            file=sys.stderr,
        )
        return 1

    df = pd.DataFrame(rows, columns=["sample_seq", "timestamp_ms", "red", "ir"])
    imu_df = pd.DataFrame(imu_rows, columns=["imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"])
    df.to_csv(csv_path, index=False)
    imu_df.to_csv(imu_csv_path, index=False)
    if live_viewer is not None:
        live_hr_df = pd.DataFrame(live_hr_records, columns=live_viewer.VALIDATION_COLUMNS)
        live_hr_df.to_csv(live_hr_csv_path, index=False)
    if live_bp_viewer is not None:
        live_bp_df = pd.DataFrame(live_bp_records, columns=live_bp_viewer.VALIDATION_COLUMNS)
        live_bp_df.to_csv(live_bp_csv_path, index=False)

    summary = summarize(df, args.duration)
    imu_summary = summarize_imu(imu_df, args.duration)
    try:
        zoom_start_s, zoom_end_s = resolve_plot_window(df, args.plot_start, args.plot_end)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.prompt_bp_after:
        args = apply_prompt_bp_after(args)

    metadata = build_metadata(
        args,
        summary,
        recording_start,
        interrupted,
        ignored_lines,
        zoom_start_s,
        zoom_end_s,
        csv_path,
        firmware_diagnostics,
        imu_summary,
        imu_csv_path,
        live_hr_csv_path if live_viewer is not None else None,
        live_hr_records,
    )
    if live_bp_viewer is not None and live_bp_context is not None:
        metadata.update(
            {
                "output_live_bp_csv_filename": live_bp_csv_path.name,
                "output_live_bp_csv_path": str(live_bp_csv_path),
                "pc_live_bp_validation_enabled": True,
                "pc_live_bp_update_count": len(live_bp_records),
                "pc_live_bp_numeric_update_count": sum(
                    record.get("sbp") is not None and record.get("dbp") is not None
                    for record in live_bp_records
                ),
                "pc_live_bp_model_dir": str(Path(args.live_bp_model_dir).resolve()),
                "pc_live_bp_model_eligible": bool(
                    live_bp_context.bundle and live_bp_context.bundle.viewer_eligible
                ),
                "pc_live_bp_allow_unvalidated": bool(args.allow_unvalidated),
                "pc_live_bp_model_participant_id": (
                    live_bp_context.bundle.participant_id if live_bp_context.bundle else None
                ),
                "pc_live_bp_calibration_id": (
                    live_bp_context.bundle.calibration_id if live_bp_context.bundle else None
                ),
                "pc_live_bp_model_manifest_schema_version": (
                    live_bp_context.bundle.manifest.get("schema_version")
                    if live_bp_context.bundle
                    else None
                ),
                "pc_live_bp_config_sha256": (
                    live_bp_context.bundle.manifest.get("config_sha256")
                    if live_bp_context.bundle
                    else None
                ),
                "pc_live_bp_selected_models": (
                    {
                        target: live_bp_context.bundle.manifest.get("models", {})
                        .get(target, {})
                        .get("model")
                        for target in ("sbp", "dbp")
                    }
                    if live_bp_context.bundle
                    else None
                ),
            }
        )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_plot(df, plot_path, plt)
    save_zoom_plot(df, zoom_plot_path, plt, zoom_start_s, zoom_end_s)
    save_motion_plot(df, imu_df, motion_plot_path, plt)

    should_write_label = label_inputs_present(args)
    if args.prompt_labels:
        should_write_label = apply_prompt_labels(args)

    if should_write_label:
        labels_path = build_label_csv_path(Path(args.labels_dir), args.session)
        label_row = build_label_row(args, summary, csv_path, metadata_path)
        label_result = write_label_row(labels_path, label_row)
        if label_result == "duplicate_skipped" and args.prompt_labels:
            answer = input("Update existing label row? [y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                write_label_row(labels_path, label_row, update_existing=True)

    print(f"Zoomed baseline-removed plot saved to: {zoom_plot_path}")
    print_summary(
        summary,
        imu_summary,
        csv_path,
        imu_csv_path,
        metadata_path,
        plot_path,
        zoom_plot_path,
        motion_plot_path,
        zoom_start_s,
        zoom_end_s,
    )
    if live_viewer is not None:
        stable_update_count = sum(record.get("status") == "stable" for record in live_hr_records)
        print(f"  live upper-arm HR CSV: {live_hr_csv_path}")
        print(f"  live upper-arm updates: {len(live_hr_records)} ({stable_update_count} stable)")
    if live_bp_viewer is not None:
        numeric_count = sum(
            record.get("sbp") is not None and record.get("dbp") is not None
            for record in live_bp_records
        )
        print(f"  live BP CSV: {live_bp_csv_path}")
        print(f"  live BP updates: {len(live_bp_records)} ({numeric_count} numeric)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
