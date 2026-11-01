"""Label-independent acquisition readiness; never loads or predicts a BP model."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bp_core.config import load_config
from bp_core.datasets import LOCAL_IMU_COLUMNS, LOCAL_PPG_COLUMNS, Recording
from bp_core.feasibility import (
    FEASIBILITY_HEALTH_COUNTERS,
    load_feasibility_policy,
    process_gap_tolerant_frames,
)
from collect_ppg import (
    create_firmware_diagnostics,
    parse_firmware_status_line,
    parse_imu_row,
    parse_ppg_row,
    update_firmware_diagnostics,
)
from view_live_bp import (
    ANALYSIS_PERIOD_SECONDS,
    CONNECTION_STALE_SECONDS,
    MOTION_STALE_SECONDS,
    SERIAL_RECEIVE_BUFFER_BYTES,
)

REQUIRED_SECONDS = 30.0
BUFFER_SECONDS = 35.0
MAX_SAMPLES = 7000
# Firmware emits sensor statistics every five seconds. Allow two reporting
# periods so normal delivery jitter does not repeatedly erase a healthy buffer.
HEALTH_STALE_SECONDS = 10.0


class PrecheckError(ValueError):
    """Readiness failed before any trial recording began."""


@dataclass(frozen=True)
class PrecheckSettings:
    config: dict
    policy: dict
    provenance: dict


def load_settings(root: Path | None = None) -> PrecheckSettings:
    root = root or Path(__file__).resolve().parents[1]
    config_path = root / "config/bp_pipeline_v1.json"
    policy_path = root / "config/bp_feasibility_v1.json"
    config, _ = load_config(config_path)
    policy = load_feasibility_policy(policy_path)
    provenance = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "policy_path": str(policy_path),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "policy_name": policy["policy_name"],
        "minimum_accepted_windows": policy["minimum_accepted_windows_per_occasion"],
        "minimum_unique_clean_coverage_s": policy["minimum_unique_clean_coverage_seconds"],
    }
    return PrecheckSettings(config, policy, provenance)


@dataclass(frozen=True)
class Snapshot:
    generation: int
    created_at: float
    duration_s: float
    ppg: tuple
    imu: tuple
    metadata: dict


@dataclass(frozen=True)
class Assessment:
    snapshot: Snapshot
    usable: bool
    accepted: int
    total: int
    coverage_s: float
    reason: str


def assess(snapshot: Snapshot, settings: PrecheckSettings) -> Assessment:
    # Cuff labels are deliberately absent, even if collector arguments contain them.
    # Readiness requires a fully stationary slice, independently of window masks.
    if snapshot.ppg:
        start, end = snapshot.ppg[0][1], snapshot.ppg[-1][1]
        updates = sorted((u for u in snapshot.metadata.get("firmware_motion_updates", [])
                          if u["timestamp_ms"] <= end), key=lambda u: u["timestamp_ms"])
        prior = [u for u in updates if u["timestamp_ms"] <= start]
        applicable = (prior[-1:] if prior else []) + [u for u in updates if u["timestamp_ms"] > start]
        if any(u.get("status") != "still" for u in applicable):
            return Assessment(snapshot, False, 0, 0, 0.0, "motion_contaminated")
    recording = Recording(
        dataset_id="local_upper_arm", participant_id="precheck", session_id="precheck",
        recording_id="precheck", label_group_id="precheck", chronological_order="precheck",
        sensor_site="upper_arm", sample_rate_hz=100.0, ppg_path="", imu_path=None,
        sbp=math.nan, dbp=math.nan, reference_hr=None, label_source="none",
        label_timing=None, quality_status="", source_format="local_ppg_csv",
    )
    extraction = process_gap_tolerant_frames(
        recording, pd.DataFrame(snapshot.ppg, columns=LOCAL_PPG_COLUMNS),
        pd.DataFrame(snapshot.imu, columns=LOCAL_IMU_COLUMNS), snapshot.metadata,
        settings.config, settings.policy,
    )
    occasion = extraction.occasion
    reasons = str(occasion.get("occasion_rejection_reasons", ""))
    if not extraction.segments.empty:
        reasons += ";" + ";".join(extraction.segments["rejection_reason"].fillna("").astype(str))
    return Assessment(snapshot, bool(occasion.get("occasion_usable", False)),
                      int(occasion.get("accepted_segment_count", 0)),
                      int(occasion.get("total_segment_count", 0)),
                      float(occasion.get("unique_clean_coverage_s", 0)), reasons)


def adjustment(reason: str) -> str:
    if any(word in reason for word in ("contact", "clipping")):
        return "Contact or clipping problem; check alignment and strap fit."
    if any(word in reason for word in ("interval", "template", "beats", "pulse")):
        return "Inconsistent pulses; check contact and cable strain relief."
    if any(word in reason for word in ("timestamp", "sequence", "incomplete")):
        return "Sample timing failed; waiting for fresh continuous streams."
    if "motion" in reason:
        return "Movement detected; keep the supported arm still."
    return "Insufficient clean PPG; check contact and cable strain relief."


class PrecheckState:
    """Reader-owned bounded state. Workers receive only immutable sample snapshots."""

    def __init__(self, gap_threshold_ms: float = 40.0):
        self.gap_threshold_ms = gap_threshold_ms
        self.ppg = deque(maxlen=MAX_SAMPLES)
        self.imu = deque(maxlen=MAX_SAMPLES)
        self.previous = {}
        self.seen = {}
        self.health_seen = {}
        self.motion_seen = None
        self.motion_status = "unknown"
        self.diagnostics = create_firmware_diagnostics()
        self.motion_updates = deque(maxlen=200)
        self.generation = 0
        self.reset_count = 0
        self.result = None
        self.wait_reason = "Waiting for PPG and IMU."
        self.stale_latch = False

    def reset(self, reason: str):
        self.ppg.clear()
        self.imu.clear()
        self.previous.clear()
        self.result = None
        self.generation += 1
        self.reset_count += 1
        self.wait_reason = reason

    def ingest(self, line: str, now: float):
        parsed = parse_firmware_status_line(line)
        update_firmware_diagnostics(self.diagnostics, parsed)
        if parsed:
            kind, fields = parsed
            if kind in ("stats", "imu_stats"):
                required = ("i2c_errors", "ovf") if kind == "stats" else ("i2c_errors", "fifo_overflows")
                complete = True
                for name in required:
                    try:
                        value = float(fields[name])
                    except KeyError:
                        complete = False
                        continue
                    except (TypeError, ValueError):
                        raise PrecheckError("Sensor fault: invalid health telemetry; check wiring and restart the device.")
                    if not math.isfinite(value) or value != 0:
                        raise PrecheckError(f"Sensor fault: {kind} {name}={value}; check wiring and restart the device.")
                if complete:
                    self.health_seen[kind] = now
                else:
                    self.health_seen.pop(kind, None)
            for name in FEASIBILITY_HEALTH_COUNTERS:
                value = self.diagnostics["metadata_fields"].get(name)
                if value is not None:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        number = math.nan
                    if not math.isfinite(number) or number != 0:
                        raise PrecheckError(f"Sensor fault: {name}={value}; check wiring and restart the device.")
            if kind == "warning" and "i2c_errors" in fields:
                try:
                    clean = float(fields["i2c_errors"]) == 0
                except (TypeError, ValueError):
                    clean = False
                if not clean:
                    raise PrecheckError("Sensor fault: I2C warning; check wiring and restart the device.")
            if kind == "warning" and fields.get("event") in {
                "fifo_overflow", "imu_fifo_overflow", "fifo_overflow_recovery",
                "fifo_overflow_recovery_failed", "imu_init_failed", "imu_not_detected",
            }:
                raise PrecheckError(f"Sensor fault: {fields['event']}; check wiring and restart the device.")
            if kind == "motion":
                status = str(fields.get("status", "unknown")).lower()
                try:
                    valid_time = math.isfinite(float(fields["timestamp_ms"]))
                except (KeyError, ValueError, TypeError):
                    valid_time = False
                if not valid_time:
                    status = "unknown"
                self.motion_seen = now
                if status != "still" and (self.ppg or self.imu or self.motion_status != status):
                    self.reset("Movement detected; keep the supported arm still." if status == "moving" else "Waiting for fresh Still motion status.")
                self.motion_status = status
                if valid_time:
                    self.motion_updates.append(copy.deepcopy(fields))
        stream, row = "imu", parse_imu_row(line)
        if row is None:
            stream, row = "ppg", parse_ppg_row(line)
        if row is None:
            return
        self.seen[stream] = now
        previous = self.previous.get(stream)
        if previous is not None:
            dt = row[1] - previous[1]
            if row[0] != previous[0] + 1 or dt <= 0 or dt > self.gap_threshold_ms:
                self.reset("Sample continuity changed; collecting fresh stationary data.")
        self.previous[stream] = row
        if self.motion_status != "still" or self.motion_seen is None or now - self.motion_seen > MOTION_STALE_SECONDS:
            return
        samples = self.ppg if stream == "ppg" else self.imu
        samples.append(row)
        while samples and row[1] - samples[0][1] > BUFFER_SECONDS * 1000:
            samples.popleft()

    def guards(self, now: float) -> str | None:
        missing = [name for name in ("ppg", "imu") if name not in self.seen]
        if missing:
            return "Waiting for " + " and ".join(name.upper() for name in missing) + "."
        stale = any(now - self.seen[name] > CONNECTION_STALE_SECONDS for name in ("ppg", "imu"))
        motion_stale = self.motion_seen is None or now - self.motion_seen > MOTION_STALE_SECONDS
        health_stale = any(name not in self.health_seen or now - self.health_seen[name] > HEALTH_STALE_SECONDS
                           for name in ("stats", "imu_stats"))
        if stale or motion_stale or health_stale:
            reason = "Waiting for fresh " + ("PPG/IMU streams." if stale else "motion status." if motion_stale else "sensor health telemetry.")
            if not self.stale_latch and (self.ppg or self.imu):
                self.reset(reason)
            self.stale_latch = True
            return reason
        self.stale_latch = False
        if self.motion_status != "still":
            return "Movement detected; keep the supported arm still." if self.motion_status == "moving" else "Waiting for fresh Still motion status."
        return None

    def duration(self) -> float:
        if not self.ppg or not self.imu:
            return 0.0
        return max(0.0, (min(self.ppg[-1][1], self.imu[-1][1]) - max(self.ppg[0][1], self.imu[0][1])) / 1000)

    def snapshot(self, now: float) -> Snapshot | None:
        if self.guards(now) or self.duration() < 8:
            return None
        end = min(self.ppg[-1][1], self.imu[-1][1])
        start = max(self.ppg[0][1], self.imu[0][1], end - REQUIRED_SECONDS * 1000)
        metadata = copy.deepcopy(self.diagnostics["metadata_fields"])
        metadata.update(ppg_profile="upper_arm_experimental", imu_detected=True,
                        firmware_motion_updates=list(copy.deepcopy(self.motion_updates)))
        return Snapshot(self.generation, now, (end - start) / 1000,
                        tuple(row for row in self.ppg if start <= row[1] <= end),
                        tuple(row for row in self.imu if start <= row[1] <= end), metadata)

    def ready(self, now: float) -> bool:
        if self.guards(now) or self.result is None:
            return False
        result = self.result
        return (result.snapshot.generation == self.generation and result.usable
                and result.snapshot.duration_s >= REQUIRED_SECONDS
                and now - result.snapshot.created_at <= MOTION_STALE_SECONDS)

    def display(self, now: float, settings: PrecheckSettings) -> str:
        guard = self.guards(now)
        result = self.result
        if guard:
            status = guard
        elif result and not result.usable:
            status = adjustment(result.reason)
        else:
            status = "Collecting fresh stationary PPG; readiness is not BP validation."
        return ("PRE-CHECK — trial recording has not started\n"
                f"Stationary buffer: {min(self.duration(), REQUIRED_SECONDS):.1f} / {REQUIRED_SECONDS:g} s\n"
                f"Accepted windows: {result.accepted if result else 0} / {result.total if result else 0}\n"
                f"Unique clean coverage: {result.coverage_s if result else 0:.1f} / "
                f"{settings.policy['minimum_unique_clean_coverage_seconds']:g} s\n"
                f"Motion: {self.motion_status.title()}\n"
                f"Sensor health: {'OK' if all(k in self.health_seen and now-self.health_seen[k] <= HEALTH_STALE_SECONDS for k in ('stats','imu_stats')) else 'Waiting for fresh telemetry'}\n"
                f"Status: {status}")


@dataclass(frozen=True)
class PrecheckOutcome:
    metadata: dict
    diagnostics: dict


def run_precheck(source, settings: PrecheckSettings, timeout: float = 120.0, *,
                 clock=time.monotonic, output=print, evaluator=assess) -> PrecheckOutcome:
    state = PrecheckState(gap_threshold_ms=float(settings.policy["gap_detection_threshold_ms"]))
    started = clock()
    last_display = last_analysis = -math.inf
    future = None
    submitted_generation = None
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bp-precheck")
    try:
        try:
            source.set_buffer_size(rx_size=SERIAL_RECEIVE_BUFFER_BYTES)
        except (AttributeError, NotImplementedError, OSError):
            pass
        while True:
            now = clock()
            if now - started >= timeout:
                reason = state.guards(now) or (adjustment(state.result.reason) if state.result else state.wait_reason)
                raise PrecheckError(f"Readiness timed out after {timeout:g} s. {reason}")
            raw = source.readline()
            now = clock()
            if now - started >= timeout:
                raise PrecheckError(f"Readiness timed out after {timeout:g} s. " + (state.guards(now) or state.wait_reason))
            if raw:
                state.ingest(raw.decode("utf-8", errors="replace"), now)
            state.guards(now)
            if future is not None and future.done():
                if submitted_generation == state.generation:
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise PrecheckError(f"Signal assessment failed: {exc}") from exc
                    if result.snapshot.generation == state.generation:
                        state.result = result
                future = None
            if state.ready(now):
                result = state.result
                # Retain health/configuration, not pre-check signal or HR history.
                diagnostics = copy.deepcopy(state.diagnostics)
                diagnostics['hr_updates'] = []
                diagnostics['latest_hr'] = {}
                diagnostics['motion_updates'] = [diagnostics['latest_motion']] if diagnostics['latest_motion'] else []
                diagnostics['warning_events'] = []
                metadata = dict(settings.provenance, status="passed", elapsed_seconds=now-started,
                                stationary_buffer_seconds=result.snapshot.duration_s,
                                accepted_windows=result.accepted, total_windows=result.total,
                                unique_clean_coverage_s=result.coverage_s, buffer_reset_count=state.reset_count)
                output("READY — recording starts now", flush=True)
                return PrecheckOutcome(metadata, diagnostics)
            first_full_window = (state.duration() >= REQUIRED_SECONDS and
                                 (state.result is None or state.result.snapshot.duration_s < REQUIRED_SECONDS))
            if future is None and (now - last_analysis >= ANALYSIS_PERIOD_SECONDS or first_full_window):
                snapshot = state.snapshot(now)
                if snapshot:
                    submitted_generation = snapshot.generation
                    future = worker.submit(evaluator, snapshot, settings)
                    last_analysis = now
            if now - last_display >= 1:
                output(state.display(now, settings), flush=True)
                last_display = now
    finally:
        worker.shutdown(wait=False, cancel_futures=True)
