"""Quality-gated PC-side experimental blood-pressure viewer."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from bp_core.config import load_config
from bp_core.inference import (
    BPInferenceResult,
    BPModelBundle,
    ModelCompatibilityError,
    load_model_bundle,
    make_short_window_bundle,
    predict_frame,
)
from collect_ppg import parse_firmware_status_line, parse_imu_row, parse_ppg_row
from line_transport import (
    LineTransportError,
    add_transport_arguments,
    create_line_source,
    validate_transport_arguments,
)
from motion_quality_shadow import (
    MotionQualityShadowBundle,
    MotionQualityShadowState,
    ShadowModelCompatibilityError,
    load_shadow_bundle,
    maybe_score_shadow,
)
from motion_quality import (
    classify_motion_intensity,
    load_config as load_motion_intensity_config,
    motion_intensity_settings,
)


DEFAULT_BAUD_RATE = 115200
DEFAULT_REFRESH_SECONDS = 1.0
SERIAL_STARTUP_DELAY_SECONDS = 1.0
ROLLING_BUFFER_SECONDS = 90.0
MINIMUM_ANALYSIS_SECONDS = 85.0
EXPERIMENTAL_FAST_WINDOW_SECONDS = 30.0
ANALYSIS_PERIOD_SECONDS = 5.0
ANALYSIS_STALE_SECONDS = 12.0
MOTION_STALE_SECONDS = 3.0
CONNECTION_STALE_SECONDS = 5.0
DEFAULT_LAST_VALIDATED_MAX_AGE_SECONDS = 300.0
MAX_RECENT_WARNINGS = 3
SERIAL_RECEIVE_BUFFER_BYTES = 65_536

HOLDABLE_LAST_VALIDATED_STATUSES = {
    "warming_up",
    "motion_detected",
    "contact_artifact",
    "motion_contaminated",
    "poor_waveform_quality",
    "insufficient_clean_data",
}

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
    "display_mode",
    "estimate_age_s",
    "estimate_sensor_timestamp_ms",
    "current_status",
    "current_reason",
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
    "experimental_fast_estimate": "EXPERIMENTAL FAST ESTIMATE",
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
    "last_validated_estimate": "Last validated estimate",
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
    motion_quality_bundle: MotionQualityShadowBundle | None = None
    motion_quality_error: str | None = None
    experimental_fast_window_seconds: float | None = None
    experimental_fast_bundle: BPModelBundle | None = None
    motion_intensity_config: dict | None = None
    motion_intensity_error: str | None = None
    last_validated_max_age_seconds: float = DEFAULT_LAST_VALIDATED_MAX_AGE_SECONDS


@dataclass(frozen=True)
class LastValidatedBP:
    result: BPInferenceResult
    measured_at: float
    sensor_timestamp_ms: int | None
    participant_id: str
    model_identity: str


@dataclass(frozen=True)
class BPDisplayResolution:
    result: BPInferenceResult
    current_result: BPInferenceResult
    mode: str
    age_s: float | None = None
    sensor_timestamp_ms: int | None = None
    source_status: str | None = None


@dataclass
class LiveMotionIntensityState:
    """Causal live equivalent of the configured eight-second IMU intensity feature."""

    config: dict
    previous_row: tuple[int, int, int, int, int] | None = None
    gravity_g: np.ndarray | None = None
    dynamic_squared: deque[float] = field(default_factory=deque)
    activity_history: deque[tuple[int, float]] = field(default_factory=deque)
    last_update_at: float | None = None
    mean_activity_g: float | None = None
    band: str = "unknown"

    def reset(self) -> None:
        self.previous_row = None
        self.gravity_g = None
        self.dynamic_squared.clear()
        self.activity_history.clear()
        self.mean_activity_g = None
        self.band = "unknown"

    def add_imu(self, row: tuple[int, int, int, int, int], now: float) -> None:
        if self.previous_row is not None:
            sequence_gap = row[0] != self.previous_row[0] + 1
            timestamp_gap = row[1] - self.previous_row[1]
            if sequence_gap or timestamp_gap <= 0 or timestamp_gap > 40:
                self.reset()
        imu = self.config["imu"]
        axes = np.asarray(row[2:5], dtype=float) * float(imu["scale_g_per_lsb"])
        if self.gravity_g is None:
            self.gravity_g = axes.copy()
        else:
            alpha = float(imu["gravity_alpha"])
            self.gravity_g = self.gravity_g + alpha * (axes - self.gravity_g)
        dynamic_squared = float(np.sum(np.square(axes - self.gravity_g)))
        self.dynamic_squared.append(dynamic_squared)
        window_samples = int(imu["activity_window_samples"])
        while len(self.dynamic_squared) > window_samples:
            self.dynamic_squared.popleft()
        activity_g = math.sqrt(float(np.mean(self.dynamic_squared)))
        self.activity_history.append((int(row[1]), activity_g))
        intensity_window_ms = float(self.config["window_seconds"]) * 1000.0
        cutoff = int(row[1] - intensity_window_ms)
        while self.activity_history and self.activity_history[0][0] < cutoff:
            self.activity_history.popleft()
        self.previous_row = row
        self.last_update_at = now
        if (
            len(self.activity_history) >= 2
            and self.activity_history[-1][0] - self.activity_history[0][0]
            >= intensity_window_ms - 20.0
        ):
            self.mean_activity_g = float(
                np.mean([value for _, value in self.activity_history])
            )
            self.band = classify_motion_intensity(self.mean_activity_g, self.config)
        else:
            self.mean_activity_g = None
            self.band = "unknown"


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
    motion_quality_shadow: MotionQualityShadowState | None = None
    motion_intensity: LiveMotionIntensityState | None = None
    last_validated_bp: LastValidatedBP | None = None
    transport_connected: bool = True


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
    add_transport_arguments(parser, default_baud=DEFAULT_BAUD_RATE)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--calibration-sbp", type=positive_float)
    parser.add_argument("--calibration-dbp", type=positive_float)
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument(
        "--experimental-fast-window",
        type=int,
        choices=[int(EXPERIMENTAL_FAST_WINDOW_SECONDS)],
        metavar="SECONDS",
        help=(
            "opt in to the unvalidated 30-second startup/recovery policy; "
            "the normal 85-second policy remains the fallback"
        ),
    )
    parser.add_argument("--config", default="config/bp_pipeline_v1.json")
    parser.add_argument(
        "--motion-quality-shadow-model",
        help="Optional frozen motion-quality classifier package; observes only and never gates BP",
    )
    parser.add_argument(
        "--motion-quality-config",
        default="config/motion_quality_v1.json",
        help="Configuration paired with --motion-quality-shadow-model",
    )
    parser.add_argument(
        "--motion-intensity-config",
        default="config/motion_quality_v2.json",
        help="Configuration containing the live Stationary/Mild/Moderate/Severe boundaries",
    )
    parser.add_argument("--refresh", type=positive_float, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument(
        "--last-validated-max-age",
        type=positive_float,
        default=DEFAULT_LAST_VALIDATED_MAX_AGE_SECONDS,
        metavar="SECONDS",
        help=(
            "maximum age of a held last-validated BP estimate during motion or temporary "
            f"poor signal (default: {DEFAULT_LAST_VALIDATED_MAX_AGE_SECONDS:.0f} seconds)"
        ),
    )
    args = parser.parse_args(argv)
    validate_transport_arguments(args, parser)
    if args.model_dir is None and (args.calibration_sbp is None or args.calibration_dbp is None):
        parser.error("pending mode requires --calibration-sbp and --calibration-dbp")
    if args.model_dir is None and args.allow_unvalidated:
        parser.error("--allow-unvalidated requires --model-dir")
    if args.model_dir is None and args.experimental_fast_window is not None:
        parser.error("--experimental-fast-window requires --model-dir")
    if args.calibration_sbp is not None and args.calibration_dbp is not None and args.calibration_sbp <= args.calibration_dbp:
        parser.error("calibration SBP must be greater than calibration DBP")
    return args


def load_viewer_context(args: argparse.Namespace) -> ViewerContext:
    config, config_path = load_config(args.config)
    motion_intensity_config = None
    motion_intensity_error = None
    try:
        motion_intensity_config = load_motion_intensity_config(
            Path(args.motion_intensity_config)
        )
        if motion_intensity_settings(motion_intensity_config) is None:
            raise ValueError("motion intensity configuration has no bands")
    except (OSError, ValueError, KeyError) as exc:
        motion_intensity_error = str(exc)
    motion_quality_bundle = None
    motion_quality_error = None
    if args.motion_quality_shadow_model:
        try:
            motion_quality_bundle = load_shadow_bundle(
                args.motion_quality_shadow_model,
                args.motion_quality_config,
            )
        except ShadowModelCompatibilityError as exc:
            motion_quality_error = str(exc)
    if not args.model_dir:
        return ViewerContext(
            participant_id=str(args.participant_id),
            calibration_sbp=float(args.calibration_sbp),
            calibration_dbp=float(args.calibration_dbp),
            config=config,
            motion_quality_bundle=motion_quality_bundle,
            motion_quality_error=motion_quality_error,
            motion_intensity_config=motion_intensity_config,
            motion_intensity_error=motion_intensity_error,
            last_validated_max_age_seconds=float(args.last_validated_max_age),
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
            motion_quality_bundle=motion_quality_bundle,
            motion_quality_error=motion_quality_error,
            motion_intensity_config=motion_intensity_config,
            motion_intensity_error=motion_intensity_error,
            last_validated_max_age_seconds=float(args.last_validated_max_age),
        )
    return ViewerContext(
        participant_id=bundle.participant_id,
        calibration_sbp=bundle.calibration_sbp,
        calibration_dbp=bundle.calibration_dbp,
        config=bundle.config,
        bundle=bundle,
        allow_unvalidated=bool(args.allow_unvalidated),
        motion_quality_bundle=motion_quality_bundle,
        motion_quality_error=motion_quality_error,
        experimental_fast_window_seconds=(
            float(args.experimental_fast_window)
            if args.experimental_fast_window is not None
            else None
        ),
        experimental_fast_bundle=(
            make_short_window_bundle(bundle, float(args.experimental_fast_window))
            if args.experimental_fast_window is not None
            else None
        ),
        motion_intensity_config=motion_intensity_config,
        motion_intensity_error=motion_intensity_error,
        last_validated_max_age_seconds=float(args.last_validated_max_age),
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
        if state.motion_quality_shadow is not None:
            state.motion_quality_shadow.add_ppg(ppg)
        if state.ppg_samples:
            previous = state.ppg_samples[-1]
            sequence_gap = ppg[0] != previous[0] + 1
            timestamp_gap_ms = ppg[1] - previous[1]
            if ppg[0] <= previous[0] or timestamp_gap_ms <= 0:
                reset_buffer(state, "invalid_timing", "sensor timestamps restarted")
            elif sequence_gap or timestamp_gap_ms > 40:
                reset_buffer(state, "invalid_timing", "PPG continuity fault; clean collection restarted")
        state.ppg_samples.append(ppg)
        _prune(state)
        return True
    imu = parse_imu_row(line)
    if imu is not None:
        if state.motion_quality_shadow is not None:
            state.motion_quality_shadow.add_imu(imu)
        if state.motion_intensity is not None:
            state.motion_intensity.add_imu(imu, now)
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
        if state.motion_quality_shadow is not None:
            state.motion_quality_shadow.add_health(kind, fields)
        previous = state.ppg_stats
        state.ppg_stats = fields
        if (
            _counter(fields, "i2c_errors") > _counter(previous, "i2c_errors")
            or _counter(fields, "ovf") > _counter(previous, "ovf")
        ):
            reset_buffer(state, "invalid_timing", "PPG health counter reported an error")
        return True
    if kind == "imu_stats":
        if state.motion_quality_shadow is not None:
            state.motion_quality_shadow.add_health(kind, fields)
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


def trailing_analysis_inputs(
    frame: pd.DataFrame,
    metadata: dict,
    duration_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    """Keep exactly the causal trailing slice and matching motion messages."""
    if frame.empty:
        return frame.copy(), dict(metadata)
    end_ms = float(frame["timestamp_ms"].iloc[-1])
    start_ms = end_ms - float(duration_seconds) * 1000.0
    selected = frame.loc[frame["timestamp_ms"] >= start_ms].copy()
    result_metadata = dict(metadata)
    updates = sorted(
        metadata.get("firmware_motion_updates", []),
        key=lambda update: float(update.get("timestamp_ms", 0.0)),
    )
    before = [update for update in updates if float(update.get("timestamp_ms", 0.0)) < start_ms]
    within = [update for update in updates if float(update.get("timestamp_ms", 0.0)) >= start_ms]
    result_metadata["firmware_motion_updates"] = before[-1:] + within
    return selected, result_metadata


def _model_identity(context: ViewerContext) -> str:
    """Return a stable identity for preventing cross-model held estimates."""

    if context.bundle is None:
        return "no-model"
    bundle = context.bundle
    model_dir = getattr(bundle, "model_dir", "")
    calibration_id = getattr(bundle, "calibration_id", "")
    manifest = getattr(bundle, "manifest", {})
    config_hash = manifest.get("config_sha256", "") if isinstance(manifest, dict) else ""
    return "|".join(
        [
            str(context.participant_id),
            str(calibration_id) if isinstance(calibration_id, (str, int)) else "",
            str(model_dir) if isinstance(model_dir, (str, Path)) else "",
            str(config_hash),
        ]
    )


def _remember_validated_bp(
    state: BPViewerState,
    context: ViewerContext,
    now: float,
) -> None:
    if not state.result.numeric_available:
        return
    state.last_validated_bp = LastValidatedBP(
        result=replace(state.result),
        measured_at=now,
        sensor_timestamp_ms=state.last_analysis_sensor_ms,
        participant_id=context.participant_id,
        model_identity=_model_identity(context),
    )


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
    fast_seconds = context.experimental_fast_window_seconds
    minimum_seconds = fast_seconds if fast_seconds is not None else MINIMUM_ANALYSIS_SECONDS
    if duration < minimum_seconds:
        state.result = BPInferenceResult(
            "warming_up", f"collecting stationary PPG: {duration:.1f}/{minimum_seconds:.0f} s"
        )
        return False
    if state.last_analysis_at is not None and now - state.last_analysis_at < ANALYSIS_PERIOD_SECONDS:
        return False
    state.last_analysis_at = now
    state.last_analysis_sensor_ms = state.ppg_samples[-1][1]
    frame, metadata = build_analysis_inputs(state)
    using_fast_policy = (
        fast_seconds is not None
        and context.experimental_fast_bundle is not None
        and duration < MINIMUM_ANALYSIS_SECONDS
    )
    analysis_bundle = context.experimental_fast_bundle if using_fast_policy else context.bundle
    if using_fast_policy:
        frame, metadata = trailing_analysis_inputs(frame, metadata, fast_seconds)
    try:
        state.result = predictor(analysis_bundle, frame, metadata)
        if using_fast_policy and state.result.numeric_available:
            state.result = replace(
                state.result,
                status="experimental_fast_estimate",
                reason=(
                    f"unvalidated {fast_seconds:.0f}-second startup/recovery policy; "
                    f"{state.result.reason}"
                ),
            )
        _remember_validated_bp(state, context, now)
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


def resolve_display_bp(
    state: BPViewerState,
    context: ViewerContext,
    now: float,
) -> BPDisplayResolution:
    """Resolve current versus held BP without weakening the inference gate."""

    current = effective_result(state, now)
    if current.numeric_available:
        snapshot = state.last_validated_bp
        age = max(0.0, now - snapshot.measured_at) if snapshot is not None else 0.0
        return BPDisplayResolution(
            result=current,
            current_result=current,
            mode="current",
            age_s=age,
            sensor_timestamp_ms=(
                snapshot.sensor_timestamp_ms if snapshot is not None else state.last_analysis_sensor_ms
            ),
            source_status=current.status,
        )

    snapshot = state.last_validated_bp
    if snapshot is None or current.status not in HOLDABLE_LAST_VALIDATED_STATUSES:
        return BPDisplayResolution(result=current, current_result=current, mode="unavailable")
    if (
        snapshot.participant_id != context.participant_id
        or snapshot.model_identity != _model_identity(context)
    ):
        return BPDisplayResolution(result=current, current_result=current, mode="unavailable")

    age = max(0.0, now - snapshot.measured_at)
    if age > context.last_validated_max_age_seconds:
        expired = replace(
            current,
            reason=(
                f"{current.reason}; last validated estimate expired at "
                f"{context.last_validated_max_age_seconds:.0f} s"
            ),
        )
        return BPDisplayResolution(result=expired, current_result=current, mode="unavailable")

    current_label = STATUS_LABELS.get(
        current.status, current.status.replace("_", " ").title()
    )
    held = replace(
        snapshot.result,
        status="last_validated_estimate",
        reason=f"Holding the last accepted estimate while current status is {current_label}: {current.reason}",
    )
    return BPDisplayResolution(
        result=held,
        current_result=current,
        mode="held",
        age_s=age,
        sensor_timestamp_ms=snapshot.sensor_timestamp_ms,
        source_status=snapshot.result.status,
    )


def connection_status(state: BPViewerState, now: float) -> str:
    if not state.transport_connected:
        return "Disconnected; reconnecting"
    if state.last_line_at is None:
        return "Waiting for ESP32" if now - state.started_at <= CONNECTION_STALE_SECONDS else "No serial data"
    age = now - state.last_line_at
    return "Receiving" if age <= CONNECTION_STALE_SECONDS else f"Stale ({age:.1f} s without data)"


def _health_line(label: str, stats: dict, overflow_key: str) -> str:
    rate = stats.get("rate_hz")
    rate_text = f"{float(rate):.1f} Hz" if isinstance(rate, (int, float)) else "-- Hz"
    return f"{label:<4} {rate_text:<10} | I2C errors: {stats.get('i2c_errors', '--'):<4} | FIFO overflows: {stats.get(overflow_key, '--')}"


def live_motion_intensity(
    state: BPViewerState,
    now: float,
) -> tuple[str, float | None]:
    tracker = state.motion_intensity
    if (
        tracker is None
        or tracker.last_update_at is None
        or now - tracker.last_update_at > MOTION_STALE_SECONDS
    ):
        return "unknown", None
    return tracker.band, tracker.mean_activity_g


def _format_estimate_age(age_s: float | None) -> str:
    if age_s is None:
        return "--"
    age_s = max(0.0, age_s)
    if age_s < 60.0:
        return f"{age_s:.0f} s ago"
    minutes = int(age_s // 60.0)
    seconds = int(age_s % 60.0)
    return f"{minutes} min {seconds:02d} s ago"


def render_screen(
    state: BPViewerState,
    context: ViewerContext,
    now: float,
    device: str,
    baud: int,
    saving: bool = False,
    transport: str = "serial",
) -> str:
    display = resolve_display_bp(state, context, now)
    result = display.result
    current = display.current_result
    numeric = result.numeric_available
    sbp = f"{result.sbp:.0f}" if numeric else "--"
    dbp = f"{result.dbp:.0f}" if numeric else "--"
    delta = (
        f"{result.delta_sbp:+.1f}/{result.delta_dbp:+.1f}"
        if numeric and result.delta_sbp is not None and result.delta_dbp is not None
        else "--/--"
    )
    status = STATUS_LABELS.get(result.status, result.status.replace("_", " ").title())
    current_status = STATUS_LABELS.get(
        current.status, current.status.replace("_", " ").title()
    )
    bp_label = "Last validated BP" if display.mode == "held" else "Estimated BP"
    display_mode = {
        "current": "Current clean estimate",
        "held": "Last validated estimate (not a new measurement)",
        "unavailable": "No valid estimate",
    }[display.mode]
    motion = str(state.latest_motion.get("status", "waiting")).replace("_", " ").title()
    intensity_band, intensity_g = live_motion_intensity(state, now)
    intensity = intensity_band.replace("_", " ").title()
    intensity_value = f"{intensity_g:.3f} g" if intensity_g is not None else "warming / unavailable"
    eligibility = "Passed preliminary personal test" if context.bundle and context.bundle.viewer_eligible else "Pending / not passed"
    lines = [
        "EXPERIMENTAL UPPER-ARM PPG-TO-BP VIEWER",
        "=" * 64,
        "",
        f"{bp_label:>20}: {sbp}/{dbp} mmHg",
        f"    Estimated change: {delta} mmHg",
        f"        Display mode: {display_mode}",
        f"        Estimate age: {_format_estimate_age(display.age_s)}",
        f"              Status: {status}",
        f"              Reason: {result.reason}",
        f"      Current status: {current_status}",
        f"      Current reason: {current.reason}",
        "",
        f"         Participant: {context.participant_id}",
        (
            f"      Calibration BP: {context.calibration_sbp:.0f}/{context.calibration_dbp:.0f} mmHg"
            if context.calibration_sbp > context.calibration_dbp > 0
            else "      Calibration BP: --/-- mmHg"
        ),
        f"   Model eligibility: {eligibility}",
        (
            f"         Still buffer: {buffer_duration_s(state):.1f}/"
            f"{context.experimental_fast_window_seconds:.0f} s fast attempt"
            if context.experimental_fast_window_seconds is not None
            else f"         Still buffer: {buffer_duration_s(state):.1f}/{MINIMUM_ANALYSIS_SECONDS:.0f} s minimum"
        ),
        *(
            [f"    Standard fallback: {MINIMUM_ANALYSIS_SECONDS:.0f} s"]
            if context.experimental_fast_window_seconds is not None
            else []
        ),
        f"      Accepted windows: {result.accepted_windows}/{result.total_windows}",
        f" Unique clean coverage: {result.clean_coverage_s:.1f} s",
        f"       PPG pulse rate: {result.pulse_rate_bpm:.1f} BPM" if result.pulse_rate_bpm is not None else "       PPG pulse rate: -- BPM",
        f"              Motion: {motion}",
        f"    Motion intensity: {intensity} ({intensity_value})",
        "  Intensity control: Display only; Still/Moving remains the BP gate",
    ]
    if context.motion_intensity_error:
        lines.append(f"  Intensity warning: {context.motion_intensity_error}")
    if state.motion_quality_shadow is not None:
        shadow = state.motion_quality_shadow.result
        probability = (
            f"{100.0 * shadow.unusable_probability:.1f}%"
            if shadow.unusable_probability is not None
            else "--"
        )
        prediction = (shadow.prediction or shadow.status).replace("_", " ").title()
        lines.extend(
            [
                "",
                "MOTION-QUALITY ML - SHADOW ONLY",
                f"          Prediction: {prediction}",
                f" Unusable probability: {probability}",
                "       Control effect: None (BP/HR behavior unchanged)",
            ]
        )
    elif context.motion_quality_error:
        lines.extend(
            [
                "",
                "MOTION-QUALITY ML - SHADOW ONLY",
                "          Prediction: Unavailable",
                f"              Reason: {context.motion_quality_error}",
                "       Control effect: None (BP/HR behavior unchanged)",
            ]
        )
    lines.extend([
        "",
        "SENSOR HEALTH",
        _health_line("PPG", state.ppg_stats, "ovf"),
        _health_line("IMU", state.imu_stats, "fifo_overflows"),
        (
            f"BLE: {device} | {connection_status(state, now)}"
            if transport == "ble"
            else f"Serial: {device} at {baud} baud | {connection_status(state, now)}"
        ),
        "",
        "RECENT WARNINGS",
    ])
    lines.extend(f"- {warning}" for warning in state.warnings) if state.warnings else lines.append("- none")
    if display.mode == "held":
        lines.extend(
            [
                "",
                "HELD VALUE: This is the last accepted BP estimate; no new BP is",
                "being measured while the current signal is unusable.",
            ]
        )
    if display.source_status in {"unvalidated_estimate", "experimental_fast_estimate"}:
        lines.extend(["", "WARNING: UNVALIDATED DEVELOPMENT ESTIMATE"])
    if context.experimental_fast_window_seconds is not None:
        lines.extend(
            [
                "",
                "FAST MODE: 30-second policy is opt-in development evidence only.",
                "If it fails quality checks, the viewer continues toward the 85-second fallback.",
            ]
        )
    lines.extend(
        [
            "",
            "Research feasibility output only; not a medical measurement.",
            "Validation capture is active." if saving else "Press Ctrl+C to exit. No data is being saved.",
        ]
    )
    return "\n".join(lines)


def build_validation_record(state: BPViewerState, context: ViewerContext, now: float, elapsed_s: float) -> dict:
    display = resolve_display_bp(state, context, now)
    result = display.result
    current = display.current_result
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
        "display_mode": display.mode,
        "estimate_age_s": (
            round(display.age_s, 3) if display.age_s is not None else None
        ),
        "estimate_sensor_timestamp_ms": display.sensor_timestamp_ms,
        "current_status": current.status,
        "current_reason": current.reason,
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


def run_viewer(
    args: argparse.Namespace,
    context: ViewerContext,
    serial_module=None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> int:
    state = BPViewerState(started_at=clock())
    if context.motion_intensity_config is not None:
        state.motion_intensity = LiveMotionIntensityState(context.motion_intensity_config)
    if context.motion_quality_bundle is not None:
        state.motion_quality_shadow = MotionQualityShadowState(context.motion_quality_bundle)
    try:
        source = create_line_source(
            args,
            timeout=0.05,
            reconnect=getattr(args, "transport", "serial") == "ble",
            serial_module=serial_module,
        )
        with source:
            try:
                source.set_buffer_size(rx_size=SERIAL_RECEIVE_BUFFER_BYTES)
            except (AttributeError, NotImplementedError, OSError):
                pass
            if getattr(args, "transport", "serial") == "serial":
                sleep(SERIAL_STARTUP_DELAY_SECONDS)
                source.reset_input_buffer()
            next_refresh = clock()
            while True:
                raw = source.readline()
                now = clock()
                connected = source.connected
                if not connected and state.transport_connected:
                    state.transport_connected = False
                    reset_buffer(
                        state,
                        "analysis_stale",
                        "BLE disconnected; reconnecting and discarding the clean buffer",
                    )
                elif connected and not state.transport_connected:
                    state.transport_connected = True
                    reset_buffer(
                        state,
                        "warming_up",
                        "BLE reconnected; collecting a fresh continuous buffer",
                    )
                if raw:
                    update_state_from_line(state, raw.decode("utf-8", errors="replace"), now)
                if now >= next_refresh:
                    if state.motion_quality_shadow is not None:
                        maybe_score_shadow(state.motion_quality_shadow)
                    maybe_predict(state, context, now)
                    clear_and_render(
                        render_screen(
                            state,
                            context,
                            now,
                            source.display_name,
                            args.baud,
                            transport=getattr(args, "transport", "serial"),
                        )
                    )
                    next_refresh = now + args.refresh
    except KeyboardInterrupt:
        print("\nExperimental BP viewer stopped. No data was saved.")
        return 0
    except LineTransportError as exc:
        transport = getattr(args, "transport", "serial")
        destination = getattr(args, "port", None) or getattr(args, "ble_device", None) or "auto-discovery"
        print(
            f"ERROR: Could not open or read {transport} device {destination}.\nDetails: {exc}\n"
            "Close any other program using the device and check that it is powered.",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = load_viewer_context(args)
    return run_viewer(args, context)


if __name__ == "__main__":
    raise SystemExit(main())
