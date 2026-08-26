"""Display live heart rate and sensor health without saving a recording."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from collect_ppg import parse_firmware_status_line


DEFAULT_BAUD_RATE = 115200
DEFAULT_REFRESH_SECONDS = 1.0
SERIAL_STARTUP_DELAY_SECONDS = 1.0
HR_STALE_SECONDS = 3.0
MOTION_STALE_SECONDS = 3.0
CONNECTION_STALE_SECONDS = 5.0
MAX_RECENT_WARNINGS = 3

STATUS_LABELS = {
    "warming_up": "Warming up",
    "stable": "Stable",
    "poor_signal": "Poor signal",
    "no_finger": "No finger",
    "insufficient_beats": "Insufficient beats",
}

MOTION_LABELS = {
    "calibrating": "Calibrating",
    "still": "Still",
    "moving": "Moving",
    "imu_unavailable": "IMU unavailable",
}


@dataclass
class ViewerState:
    started_at: float
    last_line_at: float | None = None
    last_hr_at: float | None = None
    last_motion_at: float | None = None
    hr: dict = field(default_factory=dict)
    motion: dict = field(default_factory=dict)
    ppg_stats: dict = field(default_factory=dict)
    imu_stats: dict = field(default_factory=dict)
    warnings: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_WARNINGS))


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
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
        description="Show a clean live-BPM screen without saving sensor data."
    )
    parser.add_argument("--port", required=True, help="ESP32 serial port, for example COM3")
    parser.add_argument("--baud", type=positive_int, default=DEFAULT_BAUD_RATE)
    parser.add_argument(
        "--refresh",
        type=positive_float,
        default=DEFAULT_REFRESH_SECONDS,
        help="Screen refresh period in seconds, default 1",
    )
    return parser.parse_args(argv)


def format_warning(fields: dict) -> str:
    event = str(fields.get("event", "unknown"))
    details = [f"{key}={value}" for key, value in fields.items() if key != "event"]
    return event if not details else f"{event}: {' '.join(details)}"


def update_state_from_line(state: ViewerState, line: str, now: float) -> bool:
    """Update display state from a firmware status line; raw rows return False."""
    if line.strip():
        state.last_line_at = now

    parsed = parse_firmware_status_line(line)
    if parsed is None:
        return False

    status_type, fields = parsed
    if status_type == "hr":
        state.hr = fields
        state.last_hr_at = now
    elif status_type == "motion":
        state.motion = fields
        state.last_motion_at = now
    elif status_type == "stats":
        state.ppg_stats = fields
    elif status_type == "imu_stats":
        state.imu_stats = fields
    elif status_type == "warning":
        state.warnings.append(format_warning(fields))
    else:
        return False
    return True


def age_text(last_update: float | None, now: float) -> str:
    if last_update is None:
        return "waiting"
    return f"{max(0.0, now - last_update):.1f} s"


def connection_status(state: ViewerState, now: float) -> str:
    if state.last_line_at is None:
        return "Waiting for ESP32" if (now - state.started_at) <= CONNECTION_STALE_SECONDS else "No serial data"
    age = now - state.last_line_at
    return "Receiving" if age <= CONNECTION_STALE_SECONDS else f"Stale ({age:.1f} s without data)"


def hr_display(state: ViewerState, now: float) -> tuple[str, str, str]:
    status_key = str(state.hr.get("status", "waiting"))
    status = STATUS_LABELS.get(status_key, "Waiting for heart-rate status")
    beats = str(state.hr.get("beats", "--"))
    bpm = state.hr.get("bpm")

    if state.last_hr_at is None:
        return "--", status, beats
    if (now - state.last_hr_at) > HR_STALE_SECONDS:
        return "--", "Heart-rate update stale", beats
    if status_key != "stable" or not isinstance(bpm, (int, float)):
        return "--", status, beats
    return f"{float(bpm):.1f}", status, beats


def motion_display(state: ViewerState, now: float) -> tuple[str, str]:
    if state.last_motion_at is None:
        return "Waiting", "--"
    if (now - state.last_motion_at) > MOTION_STALE_SECONDS:
        return "Motion update stale", "--"

    status_key = str(state.motion.get("status", "waiting"))
    status = MOTION_LABELS.get(status_key, "Unknown")
    activity = state.motion.get("activity_g")
    activity_text = f"{float(activity):.3f} g" if isinstance(activity, (int, float)) else "--"
    return status, activity_text


def sensor_health_line(label: str, stats: dict, rate_key: str, overflow_key: str) -> str:
    rate = stats.get(rate_key, "--")
    i2c_errors = stats.get("i2c_errors", "--")
    overflows = stats.get(overflow_key, "--")
    rate_text = f"{float(rate):.1f} Hz" if isinstance(rate, (int, float)) else "-- Hz"
    return f"{label:<4} {rate_text:<10} | I2C errors: {i2c_errors:<4} | FIFO overflows: {overflows}"


def render_screen(state: ViewerState, now: float, port: str, baud: int) -> str:
    bpm, status, beats = hr_display(state, now)
    motion, activity = motion_display(state, now)
    lines = [
        "LIVE HEART-RATE VIEWER",
        "=" * 54,
        "",
        f"             BPM: {bpm}",
        f"          Status: {status}",
        f"      Beats used: {beats}",
        f"   HR update age: {age_text(state.last_hr_at, now)}",
        f"          Motion: {motion}",
        f" Motion activity: {activity}",
        "",
        "SENSOR HEALTH",
        sensor_health_line("PPG", state.ppg_stats, "rate_hz", "ovf"),
        sensor_health_line("IMU", state.imu_stats, "rate_hz", "fifo_overflows"),
        "",
        f"Serial: {port} at {baud} baud | {connection_status(state, now)}",
        "",
        "RECENT WARNINGS",
    ]
    if state.warnings:
        lines.extend(f"- {warning}" for warning in state.warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Experimental preview only; offline analysis remains authoritative.",
            "Press Ctrl+C to exit. No data is being saved.",
        ]
    )
    return "\n".join(lines)


def clear_and_render(screen: str, output=sys.stdout) -> None:
    output.write("\x1b[2J\x1b[H")
    output.write(screen)
    output.write("\n")
    output.flush()


def import_serial():
    try:
        import serial  # type: ignore
    except ImportError:
        print(
            "ERROR: pyserial is required. Install it with:\n  python -m pip install pyserial",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return serial


def run_viewer(args: argparse.Namespace, serial_module, clock=time.monotonic, sleep=time.sleep) -> int:
    state = ViewerState(started_at=clock())
    try:
        with serial_module.Serial(args.port, args.baud, timeout=0.2) as serial_port:
            sleep(SERIAL_STARTUP_DELAY_SECONDS)
            serial_port.reset_input_buffer()
            next_refresh = clock()

            while True:
                raw_line = serial_port.readline()
                now = clock()
                if raw_line:
                    line = raw_line.decode("utf-8", errors="replace")
                    update_state_from_line(state, line, now)

                if now >= next_refresh:
                    clear_and_render(render_screen(state, now, args.port, args.baud))
                    next_refresh = now + args.refresh
    except KeyboardInterrupt:
        print("\nLive viewer stopped. No data was saved.")
        return 0
    except serial_module.SerialException as exc:
        print(
            f"ERROR: Could not open or read {args.port} at {args.baud} baud.\n"
            f"Details: {exc}\n"
            "Close ESP-IDF monitor, the collector, or any other program using the port.",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_viewer(args, import_serial())


if __name__ == "__main__":
    raise SystemExit(main())
