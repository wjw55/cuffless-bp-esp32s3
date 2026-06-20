"""Collect raw MAX30102 PPG rows from an ESP32 serial port.

Expected firmware output:
    sample_seq,timestamp_ms,red,ir
    42,12345,48231,53120

This script saves one recording session as:
    data/raw/<subject>_<session>_ppg.csv
    data/raw/<subject>_<session>_metadata.json
    data/raw/<subject>_<session>_plot.png
"""

from __future__ import annotations

import argparse
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
        description="Collect raw PPG CSV rows from ESP32/MAX30102 serial output."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM3")
    parser.add_argument("--duration", required=True, type=positive_float, help="Recording duration in seconds")
    parser.add_argument("--subject", required=True, help="Subject ID, for example S01")
    parser.add_argument("--session", required=True, help="Session name, for example test_001")
    parser.add_argument("--trial-id", default="", help="Trial ID, for example T01")
    parser.add_argument("--posture", default="", help="Subject posture, for example seated")
    parser.add_argument("--sensor-location", default="", help="PPG sensor location, for example index_finger")
    parser.add_argument("--cuff-arm", default="", help="Arm used for cuff BP reference, for example left")
    parser.add_argument("--ppg-hand", default="", help="Hand used for PPG sensor, for example right")
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
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files")
    parser.add_argument("--plot-start", type=nonnegative_float, help="Zoom plot start time in seconds")
    parser.add_argument("--plot-end", type=nonnegative_float, help="Zoom plot end time in seconds")
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
        "metadata": outdir / f"{prefix}_metadata.json",
        "plot": outdir / f"{prefix}_plot.png",
        "zoom_plot": outdir / f"{prefix}_zoom_plot.png",
    }


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


def print_summary(
    summary: dict,
    csv_path: Path,
    metadata_path: Path,
    plot_path: Path,
    zoom_plot_path: Path,
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

    print("\nSaved files")
    print(f"  CSV: {csv_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  plot: {plot_path}")
    print(f"  zoom plot: {zoom_plot_path}")
    print(f"  zoom window: {zoom_start_s:.2f}-{zoom_end_s:.2f} s")


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
) -> dict:
    firmware_diagnostics = firmware_diagnostics or create_firmware_diagnostics()
    firmware_metadata = {
        metadata_key: firmware_diagnostics["metadata_fields"].get(metadata_key)
        for metadata_key in FIRMWARE_STATS_METADATA_KEYS.values()
    }

    metadata = {
        "subject_id": args.subject,
        "session_id": args.session,
        "trial_id": args.trial_id,
        "output_csv_filename": csv_path.name if csv_path is not None else None,
        "output_csv_path": str(csv_path) if csv_path is not None else None,
        "posture": args.posture,
        "sensor_location": args.sensor_location,
        "cuff_arm": args.cuff_arm,
        "ppg_hand": args.ppg_hand,
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
        "interrupted": interrupted,
        "ignored_non_csv_lines": ignored_lines,
        "zoom_plot_start_seconds": zoom_start_s,
        "zoom_plot_end_seconds": zoom_end_s,
        "warnings": summary["warnings"],
    }

    metadata.update(firmware_metadata)
    metadata["firmware_latest_stats"] = firmware_diagnostics["latest_stats"] or None
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_paths = build_output_paths(outdir, args.subject, args.session, args.trial_id)
    try:
        ensure_output_paths_available(output_paths, args.overwrite)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    csv_path = output_paths["csv"]
    metadata_path = output_paths["metadata"]
    plot_path = output_paths["plot"]
    zoom_plot_path = output_paths["zoom_plot"]

    rows: list[tuple[int, int, int, int]] = []
    ignored_lines = 0
    interrupted = False
    firmware_diagnostics = create_firmware_diagnostics()

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            # Many ESP32 boards reset when the serial port opens. Give boot text time to pass.
            time.sleep(SERIAL_STARTUP_DELAY_S)
            ser.reset_input_buffer()

            recording_start = datetime.now().astimezone()
            deadline = time.monotonic() + args.duration
            print(f"Recording for {args.duration:.2f} s. Press Ctrl+C to stop early and save.")

            while time.monotonic() < deadline:
                raw_line = ser.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace")
                update_firmware_diagnostics(firmware_diagnostics, parse_firmware_status_line(line))
                row = parse_ppg_row(line)
                if row is None:
                    ignored_lines += 1
                    continue

                rows.append(row)

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
    df.to_csv(csv_path, index=False)

    summary = summarize(df, args.duration)
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
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_plot(df, plot_path, plt)
    save_zoom_plot(df, zoom_plot_path, plt, zoom_start_s, zoom_end_s)
    print(f"Zoomed baseline-removed plot saved to: {zoom_plot_path}")
    print_summary(summary, csv_path, metadata_path, plot_path, zoom_plot_path, zoom_start_s, zoom_end_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
