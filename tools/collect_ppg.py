"""Collect raw MAX30102 PPG rows from an ESP32 serial port.

Expected firmware output:
    timestamp_ms,red,ir
    12345,48231,53120

This script saves one recording session as:
    data/raw/<subject>_<session>_ppg.csv
    data/raw/<subject>_<session>_metadata.json
    data/raw/<subject>_<session>_plot.png
"""

from __future__ import annotations

import argparse
import json
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw PPG CSV rows from ESP32/MAX30102 serial output."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM3")
    parser.add_argument("--duration", required=True, type=positive_float, help="Recording duration in seconds")
    parser.add_argument("--subject", required=True, help="Subject ID, for example S01")
    parser.add_argument("--session", required=True, help="Session name, for example test_001")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate, default 115200")
    parser.add_argument("--notes", default="", help="Optional notes saved into metadata")
    parser.add_argument("--outdir", default="data/raw", help="Output folder, default data/raw")
    parser.add_argument("--plot-start", type=nonnegative_float, help="Zoom plot start time in seconds")
    parser.add_argument("--plot-end", type=nonnegative_float, help="Zoom plot end time in seconds")
    return parser.parse_args()


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
    """Make subject/session values safe for Windows filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def parse_ppg_row(line: str) -> tuple[int, int, int] | None:
    """Return a valid PPG row, or None for headers, boot logs, and debug lines."""
    text = line.strip()
    if not text:
        return None

    if text.lower() == "timestamp_ms,red,ir":
        return None

    if text.startswith("#"):
        return None

    parts = text.split(",")
    if len(parts) != 3:
        return None

    try:
        timestamp_ms, red, ir = (int(part.strip()) for part in parts)
    except ValueError:
        return None

    if timestamp_ms < 0 or red < 0 or ir < 0:
        return None

    return timestamp_ms, red, ir


def summarize(df, requested_duration_s: float) -> dict:
    sample_count = int(len(df))
    data_duration_s = None
    median_dt_ms = None
    estimated_rate_hz = None
    red_min = red_max = ir_min = ir_max = None
    warnings = []

    if sample_count == 0:
        warnings.append("No valid CSV samples were recorded.")
    else:
        red_min = int(df["red"].min())
        red_max = int(df["red"].max())
        ir_min = int(df["ir"].min())
        ir_max = int(df["ir"].max())

        if ir_max < LOW_IR_THRESHOLD:
            warnings.append("IR signal looks low: check finger contact and sensor alignment.")

        saturation_level = ADC_MAX_VALUE - SATURATION_MARGIN
        if red_max >= saturation_level or ir_max >= saturation_level:
            warnings.append("Signal is near ADC saturation: reduce LED current or ambient light.")

    if sample_count >= 2:
        data_duration_s = float((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0)
        dt_ms = df["timestamp_ms"].diff().dropna()
        median_dt = float(dt_ms.median())
        median_dt_ms = median_dt

        if median_dt > 0:
            estimated_rate_hz = 1000.0 / median_dt
        else:
            warnings.append("Timestamps are not increasing.")

        irregular_limit_ms = max(5.0, median_dt * 0.30)
        irregular_fraction = float(((dt_ms - median_dt).abs() > irregular_limit_ms).mean())
        if irregular_fraction > 0.05 or bool((dt_ms <= 0).any()):
            warnings.append("Timestamps look irregular: check serial drops or firmware timing.")

    return {
        "sample_count": sample_count,
        "requested_duration_s": requested_duration_s,
        "data_duration_s": data_duration_s,
        "median_dt_ms": median_dt_ms,
        "estimated_rate_hz": estimated_rate_hz,
        "red_min": red_min,
        "red_max": red_max,
        "ir_min": ir_min,
        "ir_max": ir_max,
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
    print(f"  estimated sampling rate: {format_optional(summary['estimated_rate_hz'])} Hz")
    print(f"  red min/max: {format_optional(summary['red_min'], 0)} / {format_optional(summary['red_max'], 0)}")
    print(f"  ir min/max: {format_optional(summary['ir_min'], 0)} / {format_optional(summary['ir_max'], 0)}")

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

    prefix = f"{safe_name(args.subject)}_{safe_name(args.session)}"
    csv_path = outdir / f"{prefix}_ppg.csv"
    metadata_path = outdir / f"{prefix}_metadata.json"
    plot_path = outdir / f"{prefix}_plot.png"
    zoom_plot_path = outdir / f"{prefix}_zoom_plot.png"

    rows: list[tuple[int, int, int]] = []
    ignored_lines = 0
    interrupted = False

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

    df = pd.DataFrame(rows, columns=["timestamp_ms", "red", "ir"])
    df.to_csv(csv_path, index=False)

    summary = summarize(df, args.duration)
    try:
        zoom_start_s, zoom_end_s = resolve_plot_window(df, args.plot_start, args.plot_end)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "subject": args.subject,
        "session": args.session,
        "port": args.port,
        "baud_rate": args.baud,
        "duration_seconds": args.duration,
        "data_duration_seconds": summary["data_duration_s"],
        "recording_start_time": recording_start.isoformat(timespec="seconds"),
        "sample_count": summary["sample_count"],
        "median_sample_interval_ms": summary["median_dt_ms"],
        "approximate_sampling_rate_hz": summary["estimated_rate_hz"],
        "notes": args.notes,
        "interrupted": interrupted,
        "ignored_non_csv_lines": ignored_lines,
        "zoom_plot_start_seconds": zoom_start_s,
        "zoom_plot_end_seconds": zoom_end_s,
        "warnings": summary["warnings"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_plot(df, plot_path, plt)
    save_zoom_plot(df, zoom_plot_path, plt, zoom_start_s, zoom_end_s)
    print(f"Zoomed baseline-removed plot saved to: {zoom_plot_path}")
    print_summary(summary, csv_path, metadata_path, plot_path, zoom_plot_path, zoom_start_s, zoom_end_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
