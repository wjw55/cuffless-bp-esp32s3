"""Quality-gated PC-side experimental blood-pressure viewer."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

import pandas as pd

from bp_core.config import load_config
from bp_core.inference import (
    BPInferenceResult,
    BPModelBundle,
    ModelCompatibilityError,
    load_model_bundle,
    predict_frame,
)
from collect_ppg import parse_firmware_status_line, parse_imu_row, parse_ppg_row


DEFAULT_BAUD_RATE = 115200
DEFAULT_REFRESH_SECONDS = 1.0
SERIAL_STARTUP_DELAY_SECONDS = 1.0
ROLLING_BUFFER_SECONDS = 90.0
MINIMUM_ANALYSIS_SECONDS = 85.0
ANALYSIS_PERIOD_SECONDS = 5.0
ANALYSIS_STALE_SECONDS = 12.0
MOTION_STALE_SECONDS = 3.0
CONNECTION_STALE_SECONDS = 5.0
MAX_RECENT_WARNINGS = 3
SERIAL_RECEIVE_BUFFER_BYTES = 65_536

VALIDATION_COLUMNS = [
    "elapsed_s",
    "sensor_timestamp_ms",
    "analysis_timestamp_ms",
    "sbp",
    "dbp",
    "delta_sbp",
    "delta_dbp",
    "status",
    "reason",
    "model_eligible",
    "allow_unvalidated",
    "buffer_s",
    "accepted_windows",
    "total_windows",
    "unique_clean_coverage_s",
    "pulse_rate_bpm",
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
    "warming_up": "Collecting clean PPG",
    "model_pending": "Model validation pending",
    "model_incompatible": "Model incompatible",
    "model_validation_failed": "Model validation failed",
    "prediction_ready": "Experimental estimate",
    "unvalidated_estimate": "UNVALIDATED DEVELOPMENT ESTIMATE",
    "motion_detected": "Motion detected",
    "motion_stale": "Motion update stale",
    "calibrating": "IMU calibrating",
    "imu_unavailable": "IMU unavailable",
    "contact_artifact": "Contact artifact",
    "motion_contaminated": "Motion contaminated",
    "poor_waveform_quality": "Poor waveform quality",
    "insufficient_clean_data": "Insufficient clean data",
    "invalid_timing": "Invalid sensor timing",
    "invalid_model_output": "Invalid model output",
    "analysis_error": "Analysis error",
    "analysis_stale": "Analysis update stale",
}


@dataclass
class ViewerContext:
    participant_id: str
    calibration_sbp: float
    calibration_dbp: float
    config: dict
    bundle: BPModelBundle | None = None
    model_error: str | None = None
    allow_unvalidated: bool = False


@dataclass
class BPViewerState:
    started_at: float
    result: BPInferenceResult = field(
        default_factory=lambda: BPInferenceResult("waiting", "waiting for upper-arm PPG samples")
    )
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


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show quality-gated experimental BP from upper-arm PPG.")
    parser.add_argument("--port", required=True, help="ESP32 serial port, for example COM5")
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--calibration-sbp", type=positive_float)
    parser.add_argument("--calibration-dbp", type=positive_float)
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--config", default="config/bp_pipeline_v1.json")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--refresh", type=positive_float, default=DEFAULT_REFRESH_SECONDS)
    args = parser.parse_args(argv)
    if args.model_dir is None and (args.calibration_sbp is None or args.calibration_dbp is None):
        parser.error("pending mode requires --calibration-sbp and --calibration-dbp")
    if args.model_dir is None and args.allow_unvalidated:
        parser.error("--allow-unvalidated requires --model-dir")
    if args.calibration_sbp is not None and args.calibration_dbp is not None and args.calibration_sbp <= args.calibration_dbp:
        parser.error("calibration SBP must be greater than calibration DBP")
    return args


def load_viewer_context(args: argparse.Namespace) -> ViewerContext:
    config, config_path = load_config(args.config)
    if not args.model_dir:
        return ViewerContext(
            participant_id=str(args.participant_id),
            calibration_sbp=float(args.calibration_sbp),
            calibration_dbp=float(args.calibration_dbp),
            config=config,
        )
    try:
        bundle = load_model_bundle(
            args.model_dir,
            expected_participant_id=str(args.participant_id),
            allow_unvalidated=bool(args.allow_unvalidated),
        )
        expected_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        model_hash = str(bundle.manifest.get("config_sha256", ""))
        if expected_hash != model_hash:
            raise ModelCompatibilityError("--config does not match the model's frozen configuration")
    except ModelCompatibilityError as exc:
        return ViewerContext(
            participant_id=str(args.participant_id),
            calibration_sbp=float(args.calibration_sbp or 0.0),
            calibration_dbp=float(args.calibration_dbp or 0.0),
            config=config,
            model_error=str(exc),
            allow_unvalidated=bool(args.allow_unvalidated),
        )
    return ViewerContext(
        participant_id=bundle.participant_id,
        calibration_sbp=bundle.calibration_sbp,
        calibration_dbp=bundle.calibration_dbp,
        config=bundle.config,
        bundle=bundle,
        allow_unvalidated=bool(args.allow_unvalidated),
    )


def buffer_duration_s(state: BPViewerState) -> float:
    if len(state.ppg_samples) < 2:
        return 0.0
    return max(0.0, (state.ppg_samples[-1][1] - state.ppg_samples[0][1]) / 1000.0)


def reset_buffer(state: BPViewerState, status: str, reason: str) -> None:
    state.ppg_samples.clear()
    state.motion_updates.clear()
    state.last_analysis_at = None
    state.last_analysis_sensor_ms = None
    state.result = BPInferenceResult(status, reason)


def _prune(state: BPViewerState) -> None:
    if not state.ppg_samples:
        return
    cutoff = state.ppg_samples[-1][1] - int(ROLLING_BUFFER_SECONDS * 1000)
    while state.ppg_samples and state.ppg_samples[0][1] < cutoff:
        state.ppg_samples.popleft()
    while state.motion_updates and float(state.motion_updates[0].get("timestamp_ms", cutoff)) < cutoff:
        state.motion_updates.popleft()


def _counter(stats: dict, key: str) -> int:
    try:
        return int(stats.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def format_warning(fields: dict) -> str:
    event = str(fields.get("event", "unknown"))
    details = [f"{key}={value}" for key, value in fields.items() if key != "event"]
    return event if not details else f"{event}: {' '.join(details)}"


def update_state_from_line(state: BPViewerState, line: str, now: float) -> bool:
    if line.strip():
        state.last_line_at = now
    ppg = parse_ppg_row(line)
    if ppg is not None:
        if state.ppg_samples:
            previous = state.ppg_samples[-1]
            sequence_gap = ppg[0] != previous[0] + 1
            timestamp_gap_ms = ppg[1] - previous[1]
            if ppg[0] <= previous[0] or timestamp_gap_ms <= 0:
                reset_buffer(state, "warming_up", "sensor timestamps restarted")
            elif sequence_gap or timestamp_gap_ms > 40:
                reset_buffer(state, "invalid_timing", "PPG continuity fault; clean collection restarted")
        state.ppg_samples.append(ppg)
        _prune(state)
        return True
    if parse_imu_row(line) is not None:
        return True
    parsed = parse_firmware_status_line(line)
    if parsed is None:
        return False
    kind, fields = parsed
    if kind == "motion":
        previous = str(state.latest_motion.get("status", ""))
        current = str(fields.get("status", ""))
        state.latest_motion = fields
        state.last_motion_at = now
        if current == "moving" and previous != "moving":
            reset_buffer(state, "motion_detected", "movement detected; clean collection restarted")
        elif current == "still" and previous not in {"", "still"}:
            reset_buffer(state, "warming_up", "collecting fresh stationary PPG")
        elif current in {"calibrating", "imu_unavailable"} and previous != current:
            reset_buffer(state, current, "motion quality is not currently available")
        state.motion_updates.append(fields)
        _prune(state)
        return True
    if kind == "stats":
        previous = state.ppg_stats
        state.ppg_stats = fields
        if (
            _counter(fields, "i2c_errors") > _counter(previous, "i2c_errors")
            or _counter(fields, "ovf") > _counter(previous, "ovf")
        ):
            reset_buffer(state, "invalid_timing", "PPG health counter reported an error")
        return True
    if kind == "imu_stats":
        previous = state.imu_stats
        state.imu_stats = fields
        if (
            _counter(fields, "i2c_errors") > _counter(previous, "i2c_errors")
            or _counter(fields, "fifo_overflows") > _counter(previous, "fifo_overflows")
        ):
            reset_buffer(state, "invalid_timing", "IMU health counter reported an error")
        return True
    if kind == "warning":
        state.warnings.append(format_warning(fields))
        return True
    return kind == "hr"  # Finger-specific firmware HR is deliberately ignored.


def motion_gate(state: BPViewerState, now: float) -> tuple[str | None, str | None]:
    if state.last_motion_at is None:
        return "calibrating", "waiting for IMU motion status"
    if now - state.last_motion_at > MOTION_STALE_SECONDS:
        return "motion_stale", "IMU motion status is stale"
    status = str(state.latest_motion.get("status", ""))
    if status == "moving":
        return "motion_detected", "movement detected"
    if status == "calibrating":
        return "calibrating", "waiting for IMU calibration"
    if status == "imu_unavailable":
        return "imu_unavailable", "motion quality cannot be verified"
    if status != "still":
        return "calibrating", "waiting for a valid Still status"
    return None, None


def build_analysis_inputs(state: BPViewerState) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(list(state.ppg_samples), columns=["sample_seq", "timestamp_ms", "red", "ir"])
    metadata = {
        "ppg_profile": "upper_arm_experimental",
        "firmware_motion_updates": list(state.motion_updates),
        # Counter increases reset the buffer in update_state_from_line. The
        # samples that remain therefore contain no health faults.
        "firmware_fifo_overflow_count": 0,
        "firmware_i2c_error_count": 0,
        "imu_firmware_fifo_overflow_count": 0,
        "imu_firmware_i2c_error_count": 0,
    }
    return frame, metadata


def maybe_predict(
    state: BPViewerState,
    context: ViewerContext,
    now: float,
    predictor: Callable[[BPModelBundle, pd.DataFrame, dict], BPInferenceResult] = predict_frame,
) -> bool:
    gate, reason = motion_gate(state, now)
    if gate:
        state.result = BPInferenceResult(gate, str(reason))
        return False
    if context.model_error:
        state.result = BPInferenceResult("model_incompatible", context.model_error)
        return False
    if context.bundle is None:
        state.result = BPInferenceResult("model_pending", "viewer is ready; no saved prediction model is connected")
        return False
    duration = buffer_duration_s(state)
    if duration < MINIMUM_ANALYSIS_SECONDS:
        state.result = BPInferenceResult(
            "warming_up", f"collecting stationary PPG: {duration:.1f}/{MINIMUM_ANALYSIS_SECONDS:.0f} s"
        )
        return False
    if state.last_analysis_at is not None and now - state.last_analysis_at < ANALYSIS_PERIOD_SECONDS:
        return False
    state.last_analysis_at = now
    state.last_analysis_sensor_ms = state.ppg_samples[-1][1]
    frame, metadata = build_analysis_inputs(state)
    try:
        state.result = predictor(context.bundle, frame, metadata)
    except Exception as exc:
        state.result = BPInferenceResult("analysis_error", str(exc))
    return True


def effective_result(state: BPViewerState, now: float) -> BPInferenceResult:
    gate, reason = motion_gate(state, now)
    if gate:
        return BPInferenceResult(gate, str(reason))
    if (
        state.result.numeric_available
        and state.last_line_at is not None
        and now - state.last_line_at > CONNECTION_STALE_SECONDS
    ):
        return BPInferenceResult("analysis_stale", "serial data is stale")
    if state.result.numeric_available and state.last_analysis_at is not None:
        if now - state.last_analysis_at > ANALYSIS_STALE_SECONDS:
            return BPInferenceResult("analysis_stale", "BP analysis update is stale")
    return state.result


def connection_status(state: BPViewerState, now: float) -> str:
    if state.last_line_at is None:
        return "Waiting for ESP32" if now - state.started_at <= CONNECTION_STALE_SECONDS else "No serial data"
    age = now - state.last_line_at
    return "Receiving" if age <= CONNECTION_STALE_SECONDS else f"Stale ({age:.1f} s without data)"


def _health_line(label: str, stats: dict, overflow_key: str) -> str:
    rate = stats.get("rate_hz")
    rate_text = f"{float(rate):.1f} Hz" if isinstance(rate, (int, float)) else "-- Hz"
    return f"{label:<4} {rate_text:<10} | I2C errors: {stats.get('i2c_errors', '--'):<4} | FIFO overflows: {stats.get(overflow_key, '--')}"


def render_screen(state: BPViewerState, context: ViewerContext, now: float, port: str, baud: int, saving: bool = False) -> str:
    result = effective_result(state, now)
    numeric = result.numeric_available
    sbp = f"{result.sbp:.0f}" if numeric else "--"
    dbp = f"{result.dbp:.0f}" if numeric else "--"
    delta = f"{result.delta_sbp:+.1f}/{result.delta_dbp:+.1f}" if numeric else "--/--"
    status = STATUS_LABELS.get(result.status, result.status.replace("_", " ").title())
    motion = str(state.latest_motion.get("status", "waiting")).replace("_", " ").title()
    eligibility = "Passed preliminary personal test" if context.bundle and context.bundle.viewer_eligible else "Pending / not passed"
    lines = [
        "EXPERIMENTAL UPPER-ARM PPG-TO-BP VIEWER",
        "=" * 64,
        "",
        f"       Estimated BP: {sbp}/{dbp} mmHg",
        f"    Estimated change: {delta} mmHg",
        f"              Status: {status}",
        f"              Reason: {result.reason}",
        "",
        f"         Participant: {context.participant_id}",
        (
            f"      Calibration BP: {context.calibration_sbp:.0f}/{context.calibration_dbp:.0f} mmHg"
            if context.calibration_sbp > context.calibration_dbp > 0
            else "      Calibration BP: --/-- mmHg"
        ),
        f"   Model eligibility: {eligibility}",
        f"         Still buffer: {buffer_duration_s(state):.1f}/{MINIMUM_ANALYSIS_SECONDS:.0f} s minimum",
        f"      Accepted windows: {result.accepted_windows}/{result.total_windows}",
        f" Unique clean coverage: {result.clean_coverage_s:.1f} s",
        f"       PPG pulse rate: {result.pulse_rate_bpm:.1f} BPM" if result.pulse_rate_bpm is not None else "       PPG pulse rate: -- BPM",
        f"              Motion: {motion}",
        "",
        "SENSOR HEALTH",
        _health_line("PPG", state.ppg_stats, "ovf"),
        _health_line("IMU", state.imu_stats, "fifo_overflows"),
        f"Serial: {port} at {baud} baud | {connection_status(state, now)}",
        "",
        "RECENT WARNINGS",
    ]
    lines.extend(f"- {warning}" for warning in state.warnings) if state.warnings else lines.append("- none")
    if result.status == "unvalidated_estimate":
        lines.extend(["", "WARNING: UNVALIDATED DEVELOPMENT ESTIMATE"])
    lines.extend(
        [
            "",
            "Research feasibility output only; not a medical measurement.",
            "Validation capture is active." if saving else "Press Ctrl+C to exit. No data is being saved.",
        ]
    )
    return "\n".join(lines)


def build_validation_record(state: BPViewerState, context: ViewerContext, now: float, elapsed_s: float) -> dict:
    result = effective_result(state, now)
    activity = state.latest_motion.get("activity_g")
    row = {
        "elapsed_s": round(max(0.0, elapsed_s), 3),
        "sensor_timestamp_ms": state.ppg_samples[-1][1] if state.ppg_samples else None,
        "analysis_timestamp_ms": state.last_analysis_sensor_ms,
        "sbp": round(result.sbp, 2) if result.sbp is not None else None,
        "dbp": round(result.dbp, 2) if result.dbp is not None else None,
        "delta_sbp": round(result.delta_sbp, 2) if result.delta_sbp is not None else None,
        "delta_dbp": round(result.delta_dbp, 2) if result.delta_dbp is not None else None,
        "status": result.status,
        "reason": result.reason,
        "model_eligible": bool(context.bundle and context.bundle.viewer_eligible),
        "allow_unvalidated": context.allow_unvalidated,
        "buffer_s": round(buffer_duration_s(state), 3),
        "accepted_windows": result.accepted_windows,
        "total_windows": result.total_windows,
        "unique_clean_coverage_s": round(result.clean_coverage_s, 3),
        "pulse_rate_bpm": result.pulse_rate_bpm,
        "motion_status": state.latest_motion.get("status"),
        "motion_activity_g": activity if isinstance(activity, (int, float)) else None,
        "ppg_rate_hz": state.ppg_stats.get("rate_hz"),
        "imu_rate_hz": state.imu_stats.get("rate_hz"),
        "ppg_i2c_errors": state.ppg_stats.get("i2c_errors"),
        "ppg_fifo_overflows": state.ppg_stats.get("ovf"),
        "imu_i2c_errors": state.imu_stats.get("i2c_errors"),
        "imu_fifo_overflows": state.imu_stats.get("fifo_overflows"),
    }
    return {column: row[column] for column in VALIDATION_COLUMNS}


def clear_and_render(screen: str, output=sys.stdout) -> None:
    output.write("\x1b[2J\x1b[H" + screen + "\n")
    output.flush()


def import_serial():
    try:
        import serial  # type: ignore
    except ImportError:
        print("ERROR: pyserial is required. Install it with: python -m pip install pyserial", file=sys.stderr)
        raise SystemExit(2)
    return serial


def run_viewer(args: argparse.Namespace, context: ViewerContext, serial_module, clock=time.monotonic, sleep=time.sleep) -> int:
    state = BPViewerState(started_at=clock())
    try:
        with serial_module.Serial(args.port, args.baud, timeout=0.05) as serial_port:
            try:
                serial_port.set_buffer_size(rx_size=SERIAL_RECEIVE_BUFFER_BYTES)
            except (AttributeError, NotImplementedError, OSError):
                pass
            sleep(SERIAL_STARTUP_DELAY_SECONDS)
            serial_port.reset_input_buffer()
            next_refresh = clock()
            while True:
                raw = serial_port.readline()
                now = clock()
                if raw:
                    update_state_from_line(state, raw.decode("utf-8", errors="replace"), now)
                if now >= next_refresh:
                    maybe_predict(state, context, now)
                    clear_and_render(render_screen(state, context, now, args.port, args.baud))
                    next_refresh = now + args.refresh
    except KeyboardInterrupt:
        print("\nExperimental BP viewer stopped. No data was saved.")
        return 0
    except serial_module.SerialException as exc:
        print(
            f"ERROR: Could not open or read {args.port} at {args.baud} baud.\nDetails: {exc}\n"
            "Close the monitor, collector, or other program using the port.",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = load_viewer_context(args)
    return run_viewer(args, context, import_serial())


if __name__ == "__main__":
    raise SystemExit(main())
