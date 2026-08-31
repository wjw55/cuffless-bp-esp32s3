"""Show a conservative rolling upper-arm heart-rate preview from serial PPG data.

The ESP32 firmware's live HR calculation is finger-specific.  This viewer ignores
those firmware HR updates and applies the validated offline upper-arm profile to a
bounded rolling buffer on the PC.  It is presentation-only and saves no data.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from collect_ppg import parse_firmware_status_line, parse_imu_row, parse_ppg_row
from upper_arm_hr import EDGE_GUARD_S, UpperArmResult, analyze_upper_arm_ppg


DEFAULT_BAUD_RATE = 115200
DEFAULT_REFRESH_SECONDS = 1.0
SERIAL_STARTUP_DELAY_SECONDS = 1.0
ROLLING_BUFFER_SECONDS = 60.0
MINIMUM_ANALYSIS_SECONDS = 40.0
ANALYSIS_PERIOD_SECONDS = 5.0
ANALYSIS_STALE_SECONDS = 12.0
MOTION_STALE_SECONDS = 3.0
CONNECTION_STALE_SECONDS = 5.0
RECENT_WINDOW_TOLERANCE_SECONDS = 1.5
MAX_RECENT_WARNINGS = 3
SERIAL_RECEIVE_BUFFER_BYTES = 65_536

VALIDATION_COLUMNS = [
    "elapsed_s",
    "sensor_timestamp_ms",
    "analysis_timestamp_ms",
    "bpm",
    "status",
    "reason",
    "analysis_age_s",
    "still_buffer_s",
    "accepted_windows",
    "clean_coverage_s",
    "beats",
    "motion_status",
    "motion_activity_g",
    "ppg_rate_hz",
    "imu_rate_hz",
    "ppg_i2c_errors",
    "ppg_fifo_overflows",
    "imu_i2c_errors",
    "imu_fifo_overflows",
]

STATUS_LABELS = {
    "waiting": "Waiting for PPG",
    "warming_up": "Warming up",
    "stable": "Stable",
    "motion_detected": "Motion detected",
    "calibrating": "IMU calibrating",
    "imu_unavailable": "IMU unavailable",
    "motion_stale": "Motion update stale",
    "insufficient_clean_data": "Insufficient clean data",
    "motion_contaminated": "Motion contaminated",
    "contact_artifact": "Contact artifact",
    "poor_waveform_quality": "Poor waveform quality",
    "ambiguous_hr": "Ambiguous heart rate",
    "invalid_timing": "Invalid sensor timing",
    "recent_window_rejected": "Recent signal rejected",
    "analysis_error": "Analysis error",
    "analysis_stale": "Analysis update stale",
}

MOTION_LABELS = {
    "calibrating": "Calibrating",
    "still": "Still",
    "moving": "Moving",
    "imu_unavailable": "IMU unavailable",
}


@dataclass
class PreviewResult:
    status: str = "waiting"
    bpm: float | None = None
    reason: str = "waiting for upper-arm PPG samples"
    beats: int = 0
    accepted_windows: int = 0
    clean_coverage_s: float = 0.0


@dataclass
class UpperArmViewerState:
    started_at: float
    last_line_at: float | None = None
    last_motion_at: float | None = None
    last_analysis_at: float | None = None
    last_analysis_sensor_ms: int | None = None
    latest_motion: dict = field(default_factory=dict)
    ppg_stats: dict = field(default_factory=dict)
    imu_stats: dict = field(default_factory=dict)
    ppg_samples: deque[tuple[int, int, int, int]] = field(default_factory=deque)
    motion_updates: deque[dict] = field(default_factory=deque)
    warnings: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_WARNINGS))
    preview: PreviewResult = field(default_factory=PreviewResult)


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
        description="Show rolling upper-arm HR on the PC without saving sensor data."
    )
    parser.add_argument("--port", required=True, help="ESP32 serial port, for example COM5")
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


def buffer_duration_s(state: UpperArmViewerState) -> float:
    if len(state.ppg_samples) < 2:
        return 0.0
    return max(0.0, (state.ppg_samples[-1][1] - state.ppg_samples[0][1]) / 1000.0)


def reset_ppg_buffer(state: UpperArmViewerState, status: str, reason: str) -> None:
    state.ppg_samples.clear()
    state.motion_updates.clear()
    state.last_analysis_at = None
    state.last_analysis_sensor_ms = None
    state.preview = PreviewResult(status=status, reason=reason)


def _prune_rolling_data(state: UpperArmViewerState) -> None:
    if not state.ppg_samples:
        return
    cutoff_ms = state.ppg_samples[-1][1] - int(ROLLING_BUFFER_SECONDS * 1000.0)
    while state.ppg_samples and state.ppg_samples[0][1] < cutoff_ms:
        state.ppg_samples.popleft()
    while state.motion_updates:
        timestamp_ms = state.motion_updates[0].get("timestamp_ms")
        if not isinstance(timestamp_ms, (int, float)) or timestamp_ms >= cutoff_ms:
            break
        state.motion_updates.popleft()


def update_state_from_line(state: UpperArmViewerState, line: str, now: float) -> bool:
    """Consume one serial line while keeping all raw rows out of the display."""
    if line.strip():
        state.last_line_at = now

    ppg = parse_ppg_row(line)
    if ppg is not None:
        if state.ppg_samples:
            previous = state.ppg_samples[-1]
            if ppg[0] <= previous[0] or ppg[1] <= previous[1]:
                reset_ppg_buffer(state, "warming_up", "sensor timestamps restarted")
        state.ppg_samples.append(ppg)
        _prune_rolling_data(state)
        return True

    if parse_imu_row(line) is not None:
        return True

    parsed = parse_firmware_status_line(line)
    if parsed is None:
        return False

    status_type, fields = parsed
    if status_type == "motion":
        previous_status = str(state.latest_motion.get("status", ""))
        current_status = str(fields.get("status", ""))
        state.latest_motion = fields
        state.last_motion_at = now

        if current_status == "moving" and previous_status != "moving":
            reset_ppg_buffer(state, "motion_detected", "movement detected; collecting will restart when still")
        elif current_status == "still" and previous_status != "still":
            reset_ppg_buffer(state, "warming_up", "collecting fresh still upper-arm data")
        elif current_status == "calibrating" and previous_status != "calibrating":
            reset_ppg_buffer(state, "calibrating", "waiting for IMU calibration")
        elif current_status == "imu_unavailable" and previous_status != "imu_unavailable":
            reset_ppg_buffer(state, "imu_unavailable", "motion quality cannot be verified")

        state.motion_updates.append(fields)
        _prune_rolling_data(state)
        return True

    if status_type == "stats":
        state.ppg_stats = fields
        return True
    if status_type == "imu_stats":
        state.imu_stats = fields
        return True
    if status_type == "warning":
        state.warnings.append(format_warning(fields))
        return True
    if status_type == "hr":
        # Firmware live HR remains the validated finger profile and is intentionally ignored.
        return True
    return False


def _counter(stats: dict, key: str) -> int:
    value = stats.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_analysis_inputs(state: UpperArmViewerState) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(
        list(state.ppg_samples),
        columns=["sample_seq", "timestamp_ms", "red", "ir"],
    )
    metadata = {
        "ppg_profile": "upper_arm_experimental",
        "firmware_motion_updates": list(state.motion_updates),
        "firmware_fifo_overflow_count": _counter(state.ppg_stats, "ovf"),
        "firmware_i2c_error_count": _counter(state.ppg_stats, "i2c_errors"),
        "imu_firmware_fifo_overflow_count": _counter(state.imu_stats, "fifo_overflows"),
        "imu_firmware_i2c_error_count": _counter(state.imu_stats, "i2c_errors"),
    }
    return df, metadata


def motion_gate(state: UpperArmViewerState, now: float) -> tuple[str | None, str | None]:
    if state.last_motion_at is None:
        return "calibrating", "waiting for the first IMU motion status"
    if now - state.last_motion_at > MOTION_STALE_SECONDS:
        return "motion_stale", "IMU motion status is stale"

    motion_status = str(state.latest_motion.get("status", ""))
    if motion_status == "moving":
        return "motion_detected", "movement detected"
    if motion_status == "calibrating":
        return "calibrating", "waiting for IMU calibration"
    if motion_status == "imu_unavailable":
        return "imu_unavailable", "motion quality cannot be verified"
    if motion_status != "still":
        return "calibrating", "waiting for a valid Still status"
    return None, None


def maybe_analyze(
    state: UpperArmViewerState,
    now: float,
    analyzer: Callable[[pd.DataFrame, dict], UpperArmResult] = analyze_upper_arm_ppg,
) -> bool:
    """Update the rolling estimate when enough fresh, stationary data is available."""
    gate_status, gate_reason = motion_gate(state, now)
    if gate_status is not None:
        state.preview = PreviewResult(status=gate_status, reason=str(gate_reason))
        return False

    duration_s = buffer_duration_s(state)
    if duration_s < MINIMUM_ANALYSIS_SECONDS:
        state.preview = PreviewResult(
            status="warming_up",
            reason=f"collecting still data: {duration_s:.1f}/{MINIMUM_ANALYSIS_SECONDS:.0f} s",
        )
        return False

    if state.last_analysis_at is not None and now - state.last_analysis_at < ANALYSIS_PERIOD_SECONDS:
        return False

    state.last_analysis_at = now
    state.last_analysis_sensor_ms = state.ppg_samples[-1][1] if state.ppg_samples else None
    try:
        df, metadata = build_analysis_inputs(state)
        result = analyzer(df, metadata)
    except Exception as exc:  # Keep the presentation viewer alive on malformed/transient input.
        state.preview = PreviewResult(status="analysis_error", reason=str(exc))
        return True

    accepted = [window for window in result.windows if window.status == "accepted"]
    latest_accepted_end = max((window.end_s for window in accepted), default=None)
    most_recent_analyzable_end = float(result.time_s[-1]) - EDGE_GUARD_S if len(result.time_s) else 0.0
    recent_window_available = (
        latest_accepted_end is not None
        and latest_accepted_end >= most_recent_analyzable_end - RECENT_WINDOW_TOLERANCE_SECONDS
    )

    if result.bpm is not None and result.status == "usable" and recent_window_available:
        state.preview = PreviewResult(
            status="stable",
            bpm=float(result.bpm),
            reason="upper-arm rolling consensus accepted",
            beats=result.detected_peak_count,
            accepted_windows=result.accepted_window_count,
            clean_coverage_s=result.clean_coverage_s,
        )
    elif result.bpm is not None and result.status == "usable":
        state.preview = PreviewResult(
            status="recent_window_rejected",
            reason="older windows passed, but the most recent analyzable window did not",
            accepted_windows=result.accepted_window_count,
            clean_coverage_s=result.clean_coverage_s,
        )
    else:
        state.preview = PreviewResult(
            status=result.status,
            reason=result.status_reason,
            beats=result.detected_peak_count,
            accepted_windows=result.accepted_window_count,
            clean_coverage_s=result.clean_coverage_s,
        )
    return True


def connection_status(state: UpperArmViewerState, now: float) -> str:
    if state.last_line_at is None:
        return "Waiting for ESP32" if now - state.started_at <= CONNECTION_STALE_SECONDS else "No serial data"
    age = now - state.last_line_at
    return "Receiving" if age <= CONNECTION_STALE_SECONDS else f"Stale ({age:.1f} s without data)"


def effective_preview(state: UpperArmViewerState, now: float) -> tuple[str, float | None, str]:
    """Return the quality-gated status and BPM that may be shown or recorded."""
    gate_status, gate_reason = motion_gate(state, now)
    if gate_status is not None:
        return gate_status, None, str(gate_reason)
    if state.preview.status == "stable" and state.last_analysis_at is not None:
        if now - state.last_analysis_at > ANALYSIS_STALE_SECONDS:
            return "analysis_stale", None, "rolling analysis update is stale"
        if state.preview.bpm is not None:
            return "stable", float(state.preview.bpm), state.preview.reason
    return state.preview.status, None, state.preview.reason


def display_hr(state: UpperArmViewerState, now: float) -> tuple[str, str]:
    status_key, bpm, _reason = effective_preview(state, now)
    bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
    status_text = STATUS_LABELS.get(status_key, status_key.replace("_", " ").title())
    return bpm_text, status_text


def build_validation_record(state: UpperArmViewerState, now: float, elapsed_s: float) -> dict:
    """Build one machine-readable rolling HR and signal-quality snapshot."""
    status, bpm, reason = effective_preview(state, now)
    latest_sensor_ms = state.ppg_samples[-1][1] if state.ppg_samples else None
    motion_status = str(state.latest_motion.get("status", "")) or None
    activity = state.latest_motion.get("activity_g")
    analysis_age = None if state.last_analysis_at is None else max(0.0, now - state.last_analysis_at)
    record = {
        "elapsed_s": round(max(0.0, elapsed_s), 3),
        "sensor_timestamp_ms": latest_sensor_ms,
        "analysis_timestamp_ms": state.last_analysis_sensor_ms,
        "bpm": round(bpm, 2) if bpm is not None else None,
        "status": status,
        "reason": reason,
        "analysis_age_s": round(analysis_age, 3) if analysis_age is not None else None,
        "still_buffer_s": round(buffer_duration_s(state), 3),
        "accepted_windows": state.preview.accepted_windows,
        "clean_coverage_s": state.preview.clean_coverage_s,
        "beats": state.preview.beats,
        "motion_status": motion_status,
        "motion_activity_g": activity if isinstance(activity, (int, float)) else None,
        "ppg_rate_hz": state.ppg_stats.get("rate_hz"),
        "imu_rate_hz": state.imu_stats.get("rate_hz"),
        "ppg_i2c_errors": state.ppg_stats.get("i2c_errors"),
        "ppg_fifo_overflows": state.ppg_stats.get("ovf"),
        "imu_i2c_errors": state.imu_stats.get("i2c_errors"),
        "imu_fifo_overflows": state.imu_stats.get("fifo_overflows"),
    }
    return {column: record[column] for column in VALIDATION_COLUMNS}


def motion_display(state: UpperArmViewerState, now: float) -> tuple[str, str]:
    if state.last_motion_at is None:
        return "Waiting", "--"
    if now - state.last_motion_at > MOTION_STALE_SECONDS:
        return "Motion update stale", "--"
    status = MOTION_LABELS.get(str(state.latest_motion.get("status", "")), "Unknown")
    activity = state.latest_motion.get("activity_g")
    activity_text = f"{float(activity):.3f} g" if isinstance(activity, (int, float)) else "--"
    return status, activity_text


def sensor_health_line(label: str, stats: dict, rate_key: str, overflow_key: str) -> str:
    rate = stats.get(rate_key, "--")
    i2c_errors = stats.get("i2c_errors", "--")
    overflows = stats.get(overflow_key, "--")
    rate_text = f"{float(rate):.1f} Hz" if isinstance(rate, (int, float)) else "-- Hz"
    return f"{label:<4} {rate_text:<10} | I2C errors: {i2c_errors:<4} | FIFO overflows: {overflows}"


def age_text(last_update: float | None, now: float) -> str:
    return "waiting" if last_update is None else f"{max(0.0, now - last_update):.1f} s"


def render_screen(
    state: UpperArmViewerState,
    now: float,
    port: str,
    baud: int,
    saving: bool = False,
) -> str:
    status_key, bpm_value, effective_reason = effective_preview(state, now)
    bpm = f"{bpm_value:.1f}" if bpm_value is not None else "--"
    status = STATUS_LABELS.get(status_key, status_key.replace("_", " ").title())
    motion, activity = motion_display(state, now)
    duration_s = buffer_duration_s(state)
    lines = [
        "LIVE UPPER-ARM HEART-RATE PREVIEW",
        "=" * 62,
        "",
        f"                 BPM: {bpm}",
        f"              Status: {status}",
        f"              Reason: {effective_reason}",
        f"       Analysis age: {age_text(state.last_analysis_at, now)}",
        f"       Still buffer: {duration_s:.1f} / {MINIMUM_ANALYSIS_SECONDS:.0f} s minimum",
        f"    Accepted windows: {state.preview.accepted_windows}",
        f"       Clean coverage: {state.preview.clean_coverage_s:.1f} s",
        f"          Beats found: {state.preview.beats}",
        f"               Motion: {motion}",
        f"      Motion activity: {activity}",
        "",
        "SENSOR HEALTH",
        sensor_health_line("PPG", state.ppg_stats, "rate_hz", "ovf"),
        sensor_health_line("IMU", state.imu_stats, "rate_hz", "fifo_overflows"),
        "",
        f"Serial: {port} at {baud} baud | {connection_status(state, now)}",
        "",
        "RECENT WARNINGS",
    ]
    lines.extend(f"- {warning}" for warning in state.warnings) if state.warnings else lines.append("- none")
    save_message = (
        "Validation mode: raw streams and rolling updates are saved; Ctrl+C stops early and saves."
        if saving
        else "Press Ctrl+C to exit. No data is being saved."
    )
    lines.extend(
        [
            "",
            "PC rolling estimate; firmware finger BPM is ignored.",
            "Experimental single-participant feasibility preview, not a medical measurement.",
            save_message,
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
            "ERROR: pyserial is required. Install it in this Python environment with:\n"
            "  python -m pip install pyserial",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return serial


def run_viewer(args: argparse.Namespace, serial_module, clock=time.monotonic, sleep=time.sleep) -> int:
    state = UpperArmViewerState(started_at=clock())
    try:
        with serial_module.Serial(args.port, args.baud, timeout=0.05) as serial_port:
            try:
                serial_port.set_buffer_size(rx_size=SERIAL_RECEIVE_BUFFER_BYTES)
            except (AttributeError, NotImplementedError, OSError):
                # Not every pyserial backend exposes receive-buffer sizing.
                pass
            sleep(SERIAL_STARTUP_DELAY_SECONDS)
            serial_port.reset_input_buffer()
            next_refresh = clock()
            while True:
                raw_line = serial_port.readline()
                now = clock()
                if raw_line:
                    update_state_from_line(state, raw_line.decode("utf-8", errors="replace"), now)
                if now >= next_refresh:
                    maybe_analyze(state, now)
                    clear_and_render(render_screen(state, now, args.port, args.baud))
                    next_refresh = now + args.refresh
    except KeyboardInterrupt:
        print("\nUpper-arm live preview stopped. No data was saved.")
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
