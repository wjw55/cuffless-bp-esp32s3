"""Shared presentation adapter for the terminal, desktop, and web BP viewers.

This module deliberately contains no BP algorithm of its own. It converts the
existing terminal viewer state into immutable, UI-friendly snapshots.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bp_core.inference import BPInferenceResult, BPModelBundle, predict_frame
from line_transport import LineTransportError, create_line_source
from motion_quality_shadow import MotionQualityShadowState, maybe_score_shadow
from view_live_bp import (
    BPViewerState,
    LiveMotionIntensityState,
    MINIMUM_ANALYSIS_SECONDS,
    SERIAL_RECEIVE_BUFFER_BYTES,
    SERIAL_STARTUP_DELAY_SECONDS,
    STATUS_LABELS,
    ViewerContext,
    buffer_duration_s,
    connection_status,
    live_motion_intensity,
    load_viewer_context,
    maybe_predict,
    parse_args,
    reset_buffer,
    resolve_display_bp,
    update_state_from_line,
)


PLOT_SECONDS = 15.0
MAX_PLOT_POINTS = 750
RESEARCH_DISCLAIMER = "Research feasibility only — not a medical BP measurement."
PRESENTATION_REFRESH_SECONDS = 0.5


@dataclass(frozen=True)
class PresentationSnapshot:
    mode: str
    sbp: float | None
    dbp: float | None
    age_s: float | None
    sensor_timestamp_ms: int | None
    status: str
    current_status: str
    reason: str
    current_reason: str
    source_status: str | None
    buffer_s: float
    target_s: float
    accepted_windows: int
    total_windows: int
    clean_coverage_s: float
    pulse_rate_bpm: float | None
    motion: str
    intensity: str
    intensity_g: float | None
    connection: str
    transport_connected: bool
    device: str
    model_eligibility: str
    experimental_warning: str
    ppg_rate_hz: float | None
    imu_rate_hz: float | None
    ppg_i2c_errors: int | None
    ppg_fifo_overflows: int | None
    imu_i2c_errors: int | None
    imu_fifo_overflows: int | None
    ble_connected: str | None
    ble_subscribed: str | None
    ble_mtu: int | None
    ble_dropped_records: int | None
    ble_notify_errors: int | None
    warnings: tuple[str, ...]
    waveform_x_s: tuple[float, ...]
    waveform_ir: tuple[float, ...]


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _count(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def trailing_waveform(state: BPViewerState) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a display-only, baseline-removed trailing raw-IR trace."""
    if not state.ppg_samples:
        return (), ()
    end_ms = state.ppg_samples[-1][1]
    recent = [row for row in state.ppg_samples if row[1] >= end_ms - PLOT_SECONDS * 1000]
    stride = max(1, math.ceil(len(recent) / MAX_PLOT_POINTS))
    recent = recent[::stride]
    if not recent:
        return (), ()
    ir = np.asarray([row[3] for row in recent], dtype=float)
    baseline = float(np.median(ir))
    return (
        tuple((row[1] - end_ms) / 1000.0 for row in recent),
        tuple(float(value - baseline) for value in ir),
    )


class LiveBPPresentationEngine:
    """Thin adapter around the terminal viewer's state and decision functions."""

    def __init__(
        self,
        context: ViewerContext,
        started_at: float,
        predictor: Callable[[BPModelBundle, object, dict], BPInferenceResult] = predict_frame,
    ):
        self.context = context
        self.predictor = predictor
        self.state = BPViewerState(started_at=started_at)
        if context.motion_intensity_config is not None:
            self.state.motion_intensity = LiveMotionIntensityState(context.motion_intensity_config)
        if context.motion_quality_bundle is not None:
            self.state.motion_quality_shadow = MotionQualityShadowState(context.motion_quality_bundle)

    def set_connected(self, connected: bool) -> None:
        state = self.state
        if not connected and state.transport_connected:
            state.transport_connected = False
            state.last_validated_bp = None
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

    def ingest(self, raw: bytes | str, now: float) -> None:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        try:
            update_state_from_line(self.state, line, now)
        except (ValueError, TypeError, KeyError, IndexError):
            self.state.warnings.append("Malformed telemetry line ignored")

    def snapshot(
        self,
        now: float,
        device: str,
        *,
        transport: str,
        baud: int,
    ) -> PresentationSnapshot:
        state = self.state
        if state.motion_quality_shadow is not None:
            maybe_score_shadow(state.motion_quality_shadow)
        maybe_predict(state, self.context, now, predictor=self.predictor)
        display = resolve_display_bp(state, self.context, now)
        result = display.result
        current = display.current_result
        intensity, intensity_g = live_motion_intensity(state, now)
        x_s, ir = trailing_waveform(state)
        context = self.context
        if context.model_error:
            eligibility = "Model incompatible"
        elif context.bundle is None:
            eligibility = "Model pending"
        elif context.bundle.viewer_eligible:
            eligibility = "Passed preliminary personal test"
        else:
            eligibility = "Not validated"
        warning = ""
        if display.source_status == "unvalidated_estimate" or context.allow_unvalidated:
            warning = "UNVALIDATED DEVELOPMENT ESTIMATE"
        elif display.source_status == "experimental_fast_estimate":
            warning = "EXPERIMENTAL 30-SECOND ESTIMATE"
        elif context.experimental_fast_window_seconds is not None:
            warning = "30-second development mode enabled; standard fallback is 85 seconds"
        connection = connection_status(state, now)
        if transport != "ble":
            connection = f"{connection} at {baud} baud"
        ble = state.ble_stats
        return PresentationSnapshot(
            mode=display.mode,
            sbp=result.sbp if result.numeric_available else None,
            dbp=result.dbp if result.numeric_available else None,
            age_s=display.age_s,
            sensor_timestamp_ms=display.sensor_timestamp_ms,
            status=STATUS_LABELS.get(result.status, result.status.replace("_", " ").title()),
            current_status=STATUS_LABELS.get(
                current.status, current.status.replace("_", " ").title()
            ),
            reason=result.reason,
            current_reason=current.reason,
            source_status=display.source_status,
            buffer_s=buffer_duration_s(state),
            target_s=context.experimental_fast_window_seconds or MINIMUM_ANALYSIS_SECONDS,
            accepted_windows=current.accepted_windows,
            total_windows=current.total_windows,
            clean_coverage_s=current.clean_coverage_s,
            pulse_rate_bpm=current.pulse_rate_bpm,
            motion=str(state.latest_motion.get("status", "waiting")).replace("_", " ").title(),
            intensity=intensity.replace("_", " ").title(),
            intensity_g=intensity_g,
            connection=connection,
            transport_connected=state.transport_connected,
            device=device,
            model_eligibility=eligibility,
            experimental_warning=warning,
            ppg_rate_hz=_number(state.ppg_stats.get("rate_hz")),
            imu_rate_hz=_number(state.imu_stats.get("rate_hz")),
            ppg_i2c_errors=_count(state.ppg_stats.get("i2c_errors")),
            ppg_fifo_overflows=_count(state.ppg_stats.get("ovf")),
            imu_i2c_errors=_count(state.imu_stats.get("i2c_errors")),
            imu_fifo_overflows=_count(state.imu_stats.get("fifo_overflows")),
            ble_connected=(str(ble.get("connected")) if "connected" in ble else None),
            ble_subscribed=(str(ble.get("subscribed")) if "subscribed" in ble else None),
            ble_mtu=_count(ble.get("mtu")),
            ble_dropped_records=_count(ble.get("dropped_records")),
            ble_notify_errors=_count(ble.get("notify_errors")),
            warnings=tuple(state.warnings),
            waveform_x_s=x_s,
            waveform_ir=ir,
        )


def build_connection_args(
    *,
    transport: str,
    device: str,
    participant_id: str,
    model_dir: str = "",
    calibration_sbp: str = "",
    calibration_dbp: str = "",
    fast_window: bool = False,
    allow_unvalidated: bool = False,
) -> argparse.Namespace:
    """Validate UI fields through the existing terminal viewer's CLI policy."""
    participant_id = participant_id.strip()
    device = device.strip()
    model_dir = model_dir.strip()
    if not participant_id:
        raise ValueError("Enter a participant ID.")
    if transport not in {"serial", "ble"}:
        raise ValueError("Choose USB serial or BLE.")
    if transport == "serial" and not device:
        raise ValueError("Choose a COM port.")
    if not model_dir:
        try:
            sbp = float(calibration_sbp)
            dbp = float(calibration_dbp)
        except ValueError as exc:
            raise ValueError("Pending mode needs calibration SBP and DBP.") from exc
        if not (math.isfinite(sbp) and math.isfinite(dbp) and sbp > dbp > 0):
            raise ValueError("Calibration must have positive SBP greater than DBP.")
        if fast_window or allow_unvalidated:
            raise ValueError("Experimental model options require a saved model directory.")
    tokens = ["--transport", transport, "--participant-id", participant_id]
    if transport == "serial":
        tokens.extend(["--port", device])
    elif device:
        tokens.extend(["--ble-device", device])
    if model_dir:
        tokens.extend(["--model-dir", model_dir])
    else:
        tokens.extend(["--calibration-sbp", calibration_sbp, "--calibration-dbp", calibration_dbp])
    if fast_window:
        tokens.extend(["--experimental-fast-window", "30"])
    if allow_unvalidated:
        tokens.append("--allow-unvalidated")
    return parse_args(tokens)


def run_presentation_worker(
    args: argparse.Namespace,
    context: ViewerContext | None,
    stop: threading.Event,
    publish: Callable[[str, object], None],
    *,
    source_factory=create_line_source,
    context_loader=load_viewer_context,
    clock=time.monotonic,
    refresh_seconds: float = PRESENTATION_REFRESH_SECONDS,
) -> None:
    """Read and analyze off the UI/server thread and publish immutable snapshots."""
    try:
        if context is None:
            context = context_loader(args)
        if context.model_error:
            publish("model_error", context.model_error)
            return
        if stop.is_set():
            return
        engine = LiveBPPresentationEngine(context, clock())
        source = source_factory(args, timeout=0.05, reconnect=args.transport == "ble")
        with source:
            try:
                source.set_buffer_size(rx_size=SERIAL_RECEIVE_BUFFER_BYTES)
            except (AttributeError, NotImplementedError, OSError):
                pass
            if args.transport == "serial":
                if stop.wait(SERIAL_STARTUP_DELAY_SECONDS):
                    return
                source.reset_input_buffer()
            next_refresh = clock()
            while not stop.is_set():
                raw = source.readline()
                now = clock()
                engine.set_connected(source.connected)
                if raw:
                    engine.ingest(raw, now)
                if now >= next_refresh:
                    publish(
                        "snapshot",
                        engine.snapshot(
                            now,
                            source.display_name,
                            transport=args.transport,
                            baud=args.baud,
                        ),
                    )
                    next_refresh = now + refresh_seconds
    except (LineTransportError, OSError) as exc:
        publish("error", str(exc))
    except Exception as exc:
        publish("error", f"BP viewer stopped: {exc}")
    finally:
        publish("stopped", None)

