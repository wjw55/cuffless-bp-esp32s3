"""Gap-tolerant, label-independent BP feasibility feature extraction.

This module is deliberately separate from the strict BP path.  It may salvage
continuous windows around one brief synchronized PPG/IMU interruption, but it
never interpolates through the interruption or creates a window across it.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .datasets import LOCAL_IMU_COLUMNS, LOCAL_PPG_COLUMNS, Recording
from .features import (
    SENSOR_HEALTH_COUNTERS,
    SignalQualityError,
    aggregate_recording_features,
    process_signal,
)
from .inference import signal_from_ppg_frame

FEASIBILITY_HEALTH_COUNTERS = SENSOR_HEALTH_COUNTERS + (
    "firmware_fifo_overflow_recovery_count",
)


@dataclass(frozen=True)
class GapEvent:
    stream: str
    event_index: int
    before_timestamp_ms: float
    after_timestamp_ms: float
    span_ms: float
    missing_sequences: int

    @property
    def midpoint_ms(self) -> float:
        return (self.before_timestamp_ms + self.after_timestamp_ms) / 2.0


@dataclass
class FeasibilityExtraction:
    occasion: dict[str, Any]
    segments: pd.DataFrame
    gaps: pd.DataFrame
    admission_reasons: list[str]
    warnings: list[str]


def load_feasibility_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    required_positive = (
        "gap_detection_threshold_ms",
        "maximum_gap_span_ms",
        "gap_synchronization_tolerance_ms",
        "gap_guard_seconds",
        "minimum_accepted_windows_per_occasion",
        "minimum_unique_clean_coverage_seconds",
    )
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported feasibility policy schema_version")
    for key in required_positive:
        value = float(policy.get(key, 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if int(policy.get("maximum_gap_events", -1)) < 0:
        raise ValueError("maximum_gap_events must be non-negative")
    return policy


def _numeric_frame(frame: pd.DataFrame, columns: list[str], stream: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{stream}_missing_columns:{','.join(missing)}")
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[columns].isna().any().any():
        raise ValueError(f"{stream}_non_finite_values")
    return result


def find_gap_events(
    frame: pd.DataFrame,
    *,
    stream: str,
    sequence_column: str,
    threshold_ms: float,
) -> list[GapEvent]:
    """Find true discontinuities without treating ordinary 10/20 ms jitter as a gap."""
    timestamps = frame["timestamp_ms"].to_numpy(dtype=float)
    sequence = frame[sequence_column].to_numpy(dtype=float)
    if len(frame) < 2:
        return []
    timestamp_delta = np.diff(timestamps)
    sequence_delta = np.diff(sequence)
    if np.any(timestamp_delta <= 0):
        raise ValueError(f"non_monotonic_{stream}_timestamps")
    if np.any(sequence_delta <= 0):
        raise ValueError(f"non_monotonic_{stream}_sequences")
    indices = np.flatnonzero((timestamp_delta > float(threshold_ms)) | ~np.isclose(sequence_delta, 1.0))
    events: list[GapEvent] = []
    for event_index, index in enumerate(indices):
        missing_sequences = max(0, int(round(sequence_delta[index])) - 1)
        events.append(
            GapEvent(
                stream=stream,
                event_index=event_index,
                before_timestamp_ms=float(timestamps[index]),
                after_timestamp_ms=float(timestamps[index + 1]),
                span_ms=float(timestamp_delta[index]),
                missing_sequences=missing_sequences,
            )
        )
    return events


def _health_reasons(metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    for key in FEASIBILITY_HEALTH_COUNTERS:
        value = metadata.get(key)
        if value in (None, ""):
            warnings.append(f"missing_sensor_health_counter:{key}")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            reasons.append(f"invalid_sensor_health_counter:{key}")
            continue
        if not math.isfinite(number) or number < 0:
            reasons.append(f"invalid_sensor_health_counter:{key}")
        elif number > 0:
            reasons.append(f"sensor_health_error:{key}={number:g}")
    return reasons, warnings


def assess_gap_admission(
    ppg: pd.DataFrame,
    imu: pd.DataFrame,
    metadata: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[list[GapEvent], list[GapEvent], list[str], list[str]]:
    reasons, warnings = _health_reasons(metadata)
    try:
        ppg_gaps = find_gap_events(
            ppg,
            stream="ppg",
            sequence_column="sample_seq",
            threshold_ms=float(policy["gap_detection_threshold_ms"]),
        )
        imu_gaps = find_gap_events(
            imu,
            stream="imu",
            sequence_column="imu_seq",
            threshold_ms=float(policy["gap_detection_threshold_ms"]),
        )
    except ValueError as exc:
        return [], [], sorted(set(reasons + [str(exc)])), warnings

    maximum_events = int(policy["maximum_gap_events"])
    if len(ppg_gaps) > maximum_events:
        reasons.append(f"too_many_ppg_gaps:{len(ppg_gaps)}>{maximum_events}")
    if len(imu_gaps) > maximum_events:
        reasons.append(f"too_many_imu_gaps:{len(imu_gaps)}>{maximum_events}")
    if len(ppg_gaps) != len(imu_gaps):
        reasons.append(f"unsynchronized_gap_count:ppg={len(ppg_gaps)},imu={len(imu_gaps)}")

    maximum_span = float(policy["maximum_gap_span_ms"])
    for event in [*ppg_gaps, *imu_gaps]:
        if event.span_ms > maximum_span:
            reasons.append(f"gap_too_long:{event.stream}:{event.span_ms:.3f}>{maximum_span:.3f}ms")

    tolerance = float(policy["gap_synchronization_tolerance_ms"])
    for ppg_gap, imu_gap in zip(ppg_gaps, imu_gaps):
        offset = abs(ppg_gap.midpoint_ms - imu_gap.midpoint_ms)
        if offset > tolerance:
            reasons.append(f"unsynchronized_gap_time:{offset:.3f}>{tolerance:.3f}ms")
    return ppg_gaps, imu_gaps, sorted(set(reasons)), sorted(set(warnings))


def _continuous_intervals(
    ppg: pd.DataFrame,
    gaps: list[GapEvent],
    policy: dict[str, Any],
    recording: Recording,
) -> list[tuple[float, float, str]]:
    first = float(ppg["timestamp_ms"].iloc[0])
    last = float(ppg["timestamp_ms"].iloc[-1])
    guard_ms = float(policy["gap_guard_seconds"]) * 1000.0
    intervals: list[tuple[float, float, str]] = []
    start = first
    for gap in gaps:
        stop = gap.before_timestamp_ms - guard_ms
        if stop > start:
            intervals.append((start, stop, "continuous"))
        start = gap.after_timestamp_ms + guard_ms
    if last > start:
        intervals.append((start, last, "continuous"))

    protocol = policy.get("recovery_protocol", {})
    token = str(protocol.get("trial_name_contains", "")).lower()
    identity = f"{recording.session_id} {recording.recording_id}".lower()
    if token and token in identity:
        margin = float(protocol.get("movement_guard_seconds", 0.0)) * 1000.0
        movement_start = first + float(protocol["planned_movement_start_seconds"]) * 1000.0 - margin
        movement_end = first + float(protocol["planned_movement_end_seconds"]) * 1000.0 + margin
        allowed = [(first, movement_start, "still_baseline"), (movement_end, last, "still_recovery")]
        intervals = [
            (max(start, allow_start), min(end, allow_end), label)
            for start, end, _ in intervals
            for allow_start, allow_end, label in allowed
            if min(end, allow_end) > max(start, allow_start)
        ]
    return intervals


def _empty_occasion(recording: Recording, reasons: list[str]) -> dict[str, Any]:
    return {
        "dataset_id": recording.dataset_id,
        "participant_id": recording.participant_id,
        "session_id": recording.session_id,
        "recording_id": recording.recording_id,
        "label_group_id": recording.label_group_id,
        "sbp": recording.sbp,
        "dbp": recording.dbp,
        "occasion_usable": False,
        "accepted_segment_count": 0,
        "total_segment_count": 0,
        "unique_clean_coverage_s": 0.0,
        "occasion_rejection_reasons": ";".join(sorted(set(reasons))),
    }


def process_gap_tolerant_frames(
    recording: Recording,
    ppg_frame: pd.DataFrame,
    imu_frame: pd.DataFrame,
    metadata: dict[str, Any],
    base_config: dict[str, Any],
    policy: dict[str, Any],
) -> FeasibilityExtraction:
    """Extract morphology without crossing a brief synchronized stream gap."""
    try:
        ppg = _numeric_frame(ppg_frame, LOCAL_PPG_COLUMNS, "ppg")
        imu = _numeric_frame(imu_frame, LOCAL_IMU_COLUMNS, "imu")
    except ValueError as exc:
        reason = str(exc)
        return FeasibilityExtraction(_empty_occasion(recording, [reason]), pd.DataFrame(), pd.DataFrame(), [reason], [])
    if len(ppg) < 2 or len(imu) < 2:
        reason = "insufficient_ppg_or_imu_samples"
        return FeasibilityExtraction(
            _empty_occasion(recording, [reason]), pd.DataFrame(), pd.DataFrame(), [reason], []
        )

    ppg_gaps, imu_gaps, reasons, warnings = assess_gap_admission(ppg, imu, metadata, policy)
    gap_rows = [event.__dict__ for event in [*ppg_gaps, *imu_gaps]]
    gaps = pd.DataFrame(gap_rows)
    if reasons:
        return FeasibilityExtraction(_empty_occasion(recording, reasons), pd.DataFrame(), gaps, reasons, warnings)

    config = copy.deepcopy(base_config)
    config.setdefault("quality", {}).update(
        minimum_accepted_windows_per_occasion=int(policy["minimum_accepted_windows_per_occasion"]),
        minimum_unique_clean_coverage_seconds=float(policy["minimum_unique_clean_coverage_seconds"]),
        require_upper_arm_analyzer_acceptance=bool(policy["require_upper_arm_analyzer_acceptance"]),
    )
    context = replace(recording, quality_status="")
    full_start_ms = float(ppg["timestamp_ms"].iloc[0])
    rows: list[dict[str, Any]] = []
    intervals = _continuous_intervals(ppg, ppg_gaps, policy, recording)
    window_seconds = float(config["signal"]["window_seconds"])
    for interval_index, (start_ms, end_ms, phase) in enumerate(intervals):
        if end_ms - start_ms + 1e-9 < window_seconds * 1000.0:
            continue
        selected = ppg[(ppg["timestamp_ms"] >= start_ms) & (ppg["timestamp_ms"] <= end_ms)].copy()
        if len(selected) < 4:
            continue
        signal = signal_from_ppg_frame(selected, metadata)
        try:
            segment_rows = process_signal(context, signal, config)
        except SignalQualityError as exc:
            reasons.append(str(exc))
            continue
        offset_s = (float(selected["timestamp_ms"].iloc[0]) - full_start_ms) / 1000.0
        for row_index, row in enumerate(segment_rows):
            row["start_s"] = float(row["start_s"]) + offset_s
            row["end_s"] = float(row["end_s"]) + offset_s
            row["segment_id"] = f"{recording.recording_id}:feasibility:{interval_index:02d}:{row_index:04d}"
            row["feasibility_interval"] = interval_index
            row["protocol_phase"] = phase
            rows.append(row)

    segments = pd.DataFrame(rows)
    if reasons and segments.empty:
        occasion = _empty_occasion(recording, reasons)
    else:
        occasion = aggregate_recording_features(context, segments, config, recording_rejection_reasons=reasons)
    occasion.update(
        quality_policy=str(policy["policy_name"]),
        strict_policy_unchanged=True,
        gap_event_count=len(ppg_gaps),
        gap_tolerant=bool(ppg_gaps),
        feasibility_warnings=";".join(warnings),
    )
    return FeasibilityExtraction(occasion, segments, gaps, sorted(set(reasons)), warnings)


def process_gap_tolerant_recording(
    recording: Recording,
    base_config: dict[str, Any],
    policy: dict[str, Any],
) -> FeasibilityExtraction:
    if not recording.metadata_path or not recording.imu_path:
        reason = "missing_metadata_or_imu"
        return FeasibilityExtraction(_empty_occasion(recording, [reason]), pd.DataFrame(), pd.DataFrame(), [reason], [])
    metadata = json.loads(Path(recording.metadata_path).read_text(encoding="utf-8"))
    ppg = pd.read_csv(recording.ppg_path)
    imu = pd.read_csv(recording.imu_path)
    return process_gap_tolerant_frames(recording, ppg, imu, metadata, base_config, policy)
