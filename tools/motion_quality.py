"""Prepare synchronized PPG/IMU windows for manual motion-quality review."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from bp_core.features import extract_window_features


ADC_MAX = (1 << 18) - 1
REVIEW_LABELS = ("clean", "motion_corrupted", "contact_corrupted", "uncertain")
REVIEW_COLUMNS = ["window_id", "reviewed_label", "reviewer", "review_notes"]
HEALTH_COUNTERS = (
    "firmware_i2c_error_count",
    "firmware_fifo_overflow_count",
    "firmware_fifo_overflow_recovery_count",
    "imu_firmware_i2c_error_count",
    "imu_firmware_fifo_overflow_count",
)


@dataclass
class MotionTrial:
    participant_id: str
    session_id: str
    trial_id: str
    ppg_path: Path
    imu_path: Path
    metadata_path: Path
    annotations_path: Path
    metadata: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reason_counts(values: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values.fillna("").astype(str):
        for reason in filter(None, value.split(";")):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported motion-quality config schema")
    if float(config["window_seconds"]) <= 0 or float(config["window_step_seconds"]) <= 0:
        raise ValueError("Window and step durations must be positive")
    if not 0 < float(config["timing"]["minimum_completeness"]) <= 1:
        raise ValueError("timing.minimum_completeness must be in (0, 1]")
    return config


def _resolve_path(input_dir: Path, metadata_path: Path, value: Any, suffix: str) -> Path:
    if value:
        candidate = Path(str(value))
        candidates = [candidate, input_dir / candidate.name, metadata_path.parent / candidate.name]
        for path in candidates:
            if path.exists():
                return path.resolve()
    fallback = metadata_path.with_name(metadata_path.name.replace("_metadata.json", suffix))
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Missing input associated with {metadata_path.name}: {suffix}")


def discover_trials(input_dir: Path, session: str, protocol: str) -> list[MotionTrial]:
    trials: list[MotionTrial] = []
    for metadata_path in sorted(input_dir.glob("*_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("session_id", "")) != session:
            continue
        if str(metadata.get("motion_study_protocol", "")) != protocol:
            continue
        participant = str(metadata.get("subject_id", "")).strip()
        trial_id = str(metadata.get("trial_id", "")).strip()
        if not participant or not trial_id:
            raise ValueError(f"Missing participant/trial identity in {metadata_path}")
        trials.append(
            MotionTrial(
                participant_id=participant,
                session_id=session,
                trial_id=trial_id,
                ppg_path=_resolve_path(input_dir, metadata_path, metadata.get("output_csv_path"), "_ppg.csv"),
                imu_path=_resolve_path(input_dir, metadata_path, metadata.get("output_imu_csv_path"), "_imu.csv"),
                metadata_path=metadata_path.resolve(),
                annotations_path=_resolve_path(
                    input_dir,
                    metadata_path,
                    metadata.get("motion_study_annotation_path"),
                    "_activity_annotations.csv",
                ),
                metadata=metadata,
            )
        )
    if not trials:
        raise ValueError(f"No {protocol} trials found for session {session!r} in {input_dir}")
    identities = [(trial.participant_id, trial.session_id, trial.trial_id) for trial in trials]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate participant/session/trial identity")
    return trials


def _numeric_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame = frame[columns].apply(pd.to_numeric, errors="coerce")
    if frame.empty or not np.all(np.isfinite(frame.to_numpy(dtype=float))):
        raise ValueError(f"{path.name} contains missing or non-finite signal values")
    return frame


def _strict_stream_reasons(frame: pd.DataFrame, sequence: str, timestamp: str, prefix: str) -> list[str]:
    reasons: list[str] = []
    sequence_delta = np.diff(frame[sequence].to_numpy(dtype=float))
    timestamp_delta = np.diff(frame[timestamp].to_numpy(dtype=float))
    if len(sequence_delta) and np.any(sequence_delta != 1):
        reasons.append(f"{prefix}_missing_or_non_monotonic_sequence")
    if len(timestamp_delta) and np.any(timestamp_delta <= 0):
        reasons.append(f"{prefix}_non_monotonic_timestamp")
    return reasons


def _health_reasons(metadata: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in HEALTH_COUNTERS:
        value = metadata.get(key)
        try:
            if value not in (None, "") and float(value) != 0:
                reasons.append(f"sensor_health_error:{key}={value}")
        except (TypeError, ValueError):
            reasons.append(f"invalid_sensor_health_counter:{key}")
    return reasons


def _load_annotations(trial: MotionTrial, protocol: str) -> pd.DataFrame:
    annotations = pd.read_csv(trial.annotations_path)
    required = {
        "participant_id", "session_id", "trial_id", "protocol", "block_index", "block_name",
        "scheduled_start_s", "scheduled_end_s", "activity_label", "completion_status",
    }
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"{trial.annotations_path.name} is missing columns: {missing}")
    identity_checks = {
        "participant_id": trial.participant_id,
        "session_id": trial.session_id,
        "trial_id": trial.trial_id,
        "protocol": protocol,
    }
    for column, expected in identity_checks.items():
        if set(annotations[column].astype(str)) != {expected}:
            raise ValueError(f"Annotation {column} does not match metadata for {trial.trial_id}")
    if not (annotations["completion_status"].astype(str) == "complete").all():
        raise ValueError(f"Incomplete protocol annotations for {trial.trial_id}")
    annotations = annotations.sort_values("block_index").reset_index(drop=True)
    starts = pd.to_numeric(annotations["scheduled_start_s"], errors="raise").to_numpy(dtype=float)
    ends = pd.to_numeric(annotations["scheduled_end_s"], errors="raise").to_numpy(dtype=float)
    if starts[0] != 0 or np.any(ends <= starts) or np.any(np.abs(starts[1:] - ends[:-1]) > 1e-9):
        raise ValueError(f"Non-continuous protocol annotations for {trial.trial_id}")
    annotations["scheduled_start_s"] = starts
    annotations["scheduled_end_s"] = ends
    return annotations


def _causal_imu(imu: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    imu_config = config["imu"]
    axes = imu[["x_raw", "y_raw", "z_raw"]].to_numpy(dtype=float) * float(imu_config["scale_g_per_lsb"])
    gravity = np.empty_like(axes)
    gravity[0] = axes[0]
    alpha = float(imu_config["gravity_alpha"])
    for index in range(1, len(axes)):
        gravity[index] = gravity[index - 1] + alpha * (axes[index] - gravity[index - 1])
    dynamic_axes = axes - gravity
    dynamic = np.linalg.norm(dynamic_axes, axis=1)
    activity = (
        pd.Series(np.square(dynamic))
        .rolling(int(imu_config["activity_window_samples"]), min_periods=1)
        .mean()
        .pow(0.5)
        .to_numpy(dtype=float)
    )
    timestamps_s = imu["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    dt = np.diff(timestamps_s)
    jerk = np.zeros_like(dynamic)
    if len(jerk) > 1:
        jerk[1:] = np.linalg.norm(np.diff(axes, axis=0), axis=1) / dt
    return {
        "axes": axes,
        "gravity": gravity,
        "dynamic_axes": dynamic_axes,
        "dynamic": dynamic,
        "activity": activity,
        "jerk": jerk,
    }


def _filter(values: np.ndarray, sample_rate_hz: float, config: dict[str, Any]) -> np.ndarray:
    ppg = config["ppg"]
    high = min(float(ppg["bandpass_high_hz"]), 0.45 * sample_rate_hz)
    sos = butter(
        int(ppg["filter_order"]),
        [float(ppg["bandpass_low_hz"]), high],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return sosfiltfilt(sos, values)


def _spectral_features(values: np.ndarray, rate_hz: float, low: float, high: float) -> dict[str, float]:
    frequencies, power = welch(values, fs=rate_hz, nperseg=min(len(values), max(32, int(4 * rate_hz))))
    selected = (frequencies >= low) & (frequencies <= high)
    band_power = power[selected]
    if not len(band_power) or float(np.sum(band_power)) <= 0:
        return {"power": 0.0, "dominant_hz": math.nan, "entropy": 0.0, "prominence": 0.0}
    normalized = band_power / np.sum(band_power)
    entropy = -float(np.sum(normalized * np.log(normalized + np.finfo(float).eps))) / math.log(len(normalized)) if len(normalized) > 1 else 0.0
    return {
        "power": float(np.sum(band_power)),
        "dominant_hz": float(frequencies[selected][int(np.argmax(band_power))]),
        "entropy": entropy,
        "prominence": float(np.max(band_power) / (np.median(band_power) + np.finfo(float).eps)),
    }


def _max_lag_correlation(first: np.ndarray, second: np.ndarray, maximum_lag: int) -> float:
    if np.std(first) <= np.finfo(float).eps or np.std(second) <= np.finfo(float).eps:
        return 0.0
    first = (first - np.mean(first)) / (np.std(first) + np.finfo(float).eps)
    second = (second - np.mean(second)) / (np.std(second) + np.finfo(float).eps)
    best = 0.0
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left, right = first[-lag:], second[:lag]
        elif lag > 0:
            left, right = first[:-lag], second[lag:]
        else:
            left, right = first, second
        if len(left) >= 10:
            correlation = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(correlation):
                best = max(best, abs(correlation))
    return best


def _overlaps(start: float, end: float, interval_start: float, interval_end: float) -> bool:
    return start < interval_end and end > interval_start


def _protocol_context(start: float, end: float, annotations: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blocks = annotations[
        (annotations["scheduled_start_s"] <= start + 1e-9)
        & (annotations["scheduled_end_s"] >= end - 1e-9)
    ]
    context = {"block_name": "", "activity_label": ""}
    reasons: list[str] = []
    if len(blocks) != 1:
        reasons.append("crosses_activity_block")
    else:
        block = blocks.iloc[0]
        context = {"block_name": str(block["block_name"]), "activity_label": str(block["activity_label"])}
    duration = float(annotations["scheduled_end_s"].iloc[-1])
    edge = float(config["guards"]["recording_edge_seconds"])
    if _overlaps(start, end, 0.0, edge) or _overlaps(start, end, duration - edge, duration):
        reasons.append("recording_edge_guard")
    transition = float(config["guards"]["cue_transition_seconds"])
    for boundary in annotations["scheduled_start_s"].to_numpy(dtype=float)[1:]:
        if _overlaps(start, end, boundary - transition, boundary + transition):
            reasons.append("cue_transition_guard")
            break
    recovery = float(config["guards"]["early_recovery_seconds"])
    for _, block in annotations.iterrows():
        if str(block["block_name"]).startswith("still_recovery") and _overlaps(
            start, end, float(block["scheduled_start_s"]), float(block["scheduled_start_s"]) + recovery
        ):
            reasons.append("early_recovery_guard")
            break
    return context, sorted(set(reasons))


def _orientation_change(gravity: np.ndarray, count: int) -> float:
    count = max(1, min(count, len(gravity) // 2))
    first = np.median(gravity[:count], axis=0)
    last = np.median(gravity[-count:], axis=0)
    denominator = np.linalg.norm(first) * np.linalg.norm(last)
    if denominator <= 0:
        return math.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(first, last) / denominator, -1.0, 1.0))))


def _window_features(
    ppg: pd.DataFrame,
    imu: pd.DataFrame,
    imu_derived: dict[str, np.ndarray],
    ppg_indices: np.ndarray,
    imu_indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    ppg_time = ppg.loc[ppg_indices, "timestamp_ms"].to_numpy(dtype=float) / 1000.0
    ir = ppg.loc[ppg_indices, "ir"].to_numpy(dtype=float)
    red = ppg.loc[ppg_indices, "red"].to_numpy(dtype=float)
    imu_time = imu.loc[imu_indices, "timestamp_ms"].to_numpy(dtype=float) / 1000.0
    if ppg_time[0] < imu_time[0] or ppg_time[-1] > imu_time[-1]:
        raise ValueError("IMU interpolation would require extrapolation")
    ppg_rate = 1.0 / float(np.median(np.diff(ppg_time)))
    imu_rate = 1.0 / float(np.median(np.diff(imu_time)))
    relative = ppg_time - ppg_time[0]
    linear = np.polyval(np.polyfit(relative, ir, 1), relative)
    detrended = ir - linear
    filtered = _filter(ir, ppg_rate, config)
    filtered_red = _filter(red, ppg_rate, config)
    derivative = np.diff(ir) * ppg_rate
    ppg_spectrum = _spectral_features(
        filtered,
        ppg_rate,
        float(config["ppg"]["bandpass_low_hz"]),
        float(config["ppg"]["bandpass_high_hz"]),
    )
    pulse_spectrum = _spectral_features(
        filtered,
        ppg_rate,
        float(config["ppg"]["pulse_band_low_hz"]),
        float(config["ppg"]["pulse_band_high_hz"]),
    )
    signal_settings = {
        "filter_order": config["ppg"]["filter_order"],
        "bandpass_low_hz": config["ppg"]["bandpass_low_hz"],
        "bandpass_high_hz": config["ppg"]["bandpass_high_hz"],
        "minimum_hr_bpm": config["ppg"]["minimum_hr_bpm"],
        "maximum_hr_bpm": config["ppg"]["maximum_hr_bpm"],
        "template_samples": config["ppg"]["template_samples"],
    }
    quality = {
        "minimum_beats_per_window": config["ppg"]["minimum_beats_per_window"],
        "maximum_interval_cv": config["ppg"]["maximum_interval_cv"],
        "minimum_template_correlation": config["ppg"]["minimum_template_correlation"],
    }
    morphology, diagnostics = extract_window_features(
        relative, ir, red, ppg_rate, signal_settings, quality
    )
    axes = imu_derived["axes"][imu_indices]
    gravity = imu_derived["gravity"][imu_indices]
    dynamic_axes = imu_derived["dynamic_axes"][imu_indices]
    dynamic = imu_derived["dynamic"][imu_indices]
    activity = imu_derived["activity"][imu_indices]
    jerk = imu_derived["jerk"][imu_indices]
    imu_spectrum = _spectral_features(
        dynamic,
        imu_rate,
        float(config["cross_modal"]["frequency_low_hz"]),
        float(config["cross_modal"]["frequency_high_hz"]),
    )
    interpolated_dynamic = np.interp(ppg_time, imu_time, dynamic)
    maximum_lag = int(round(float(config["cross_modal"]["maximum_lag_seconds"]) * ppg_rate))
    cross_correlation = _max_lag_correlation(np.abs(np.gradient(filtered)), interpolated_dynamic, maximum_lag)
    ppg_freq, ppg_power = welch(filtered, fs=ppg_rate, nperseg=min(len(filtered), int(4 * ppg_rate)))
    imu_interp = interpolated_dynamic - np.mean(interpolated_dynamic)
    imu_freq, imu_power = welch(imu_interp, fs=ppg_rate, nperseg=min(len(imu_interp), int(4 * ppg_rate)))
    band = (ppg_freq >= float(config["cross_modal"]["frequency_low_hz"])) & (ppg_freq <= float(config["cross_modal"]["frequency_high_hz"]))
    imu_on_ppg = np.interp(ppg_freq[band], imu_freq, imu_power)
    p_norm = ppg_power[band] / (np.sum(ppg_power[band]) + np.finfo(float).eps)
    i_norm = imu_on_ppg / (np.sum(imu_on_ppg) + np.finfo(float).eps)
    features: dict[str, Any] = {
        "ppg_sample_rate_hz": ppg_rate,
        "ppg_dc_median": float(np.median(ir)),
        "ppg_dc_iqr": float(np.percentile(ir, 75) - np.percentile(ir, 25)),
        "ppg_dc_slope_counts_per_s": float(np.polyfit(relative, ir, 1)[0]),
        "ppg_ac_rms": float(np.sqrt(np.mean(np.square(detrended)))),
        "ppg_derivative_median_abs": float(np.median(np.abs(derivative))),
        "ppg_derivative_p95_abs": float(np.percentile(np.abs(derivative), 95)),
        "ppg_max_step_counts": float(np.max(np.abs(np.diff(ir)))),
        "ppg_clipping_fraction": float(np.mean((ir <= 16) | (ir >= ADC_MAX - 16))),
        "ppg_band_power": ppg_spectrum["power"],
        "ppg_pulse_band_power_fraction": pulse_spectrum["power"] / (ppg_spectrum["power"] + np.finfo(float).eps),
        "ppg_spectral_prominence": pulse_spectrum["prominence"],
        "ppg_spectral_entropy": ppg_spectrum["entropy"],
        "ppg_red_ir_correlation": (
            float(np.corrcoef(filtered_red, filtered)[0, 1])
            if np.std(filtered_red) > np.finfo(float).eps and np.std(filtered) > np.finfo(float).eps
            else 0.0
        ),
        "ppg_morphology_accepted": morphology is not None,
        "ppg_morphology_rejection_reason": diagnostics.get("rejection_reason", ""),
        "ppg_detected_peak_count": diagnostics.get("detected_peak_count"),
        "ppg_valid_beat_count": diagnostics.get("valid_beat_count"),
        "ppg_pulse_rate_bpm": diagnostics.get("pulse_rate_bpm"),
        "ppg_ibi_cv": diagnostics.get("interval_cv"),
        "ppg_template_correlation": diagnostics.get("template_correlation"),
        "imu_sample_rate_hz": imu_rate,
        "imu_accel_rms_g": float(np.sqrt(np.mean(np.sum(np.square(axes), axis=1)))),
        "imu_dynamic_rms_g": float(np.sqrt(np.mean(np.square(dynamic)))),
        "imu_dynamic_max_g": float(np.max(dynamic)),
        "imu_dynamic_p95_g": float(np.percentile(dynamic, 95)),
        "imu_activity_mean_g": float(np.mean(activity)),
        "imu_activity_max_g": float(np.max(activity)),
        "imu_activity_above_threshold_fraction": float(np.mean(activity > float(config["imu"]["firmware_motion_threshold_g"]))),
        "imu_axis_x_std_g": float(np.std(axes[:, 0])),
        "imu_axis_y_std_g": float(np.std(axes[:, 1])),
        "imu_axis_z_std_g": float(np.std(axes[:, 2])),
        "imu_signal_magnitude_area_g": float(np.mean(np.sum(np.abs(dynamic_axes), axis=1))),
        "imu_jerk_rms_g_per_s": float(np.sqrt(np.mean(np.square(jerk)))),
        "imu_jerk_p95_g_per_s": float(np.percentile(jerk, 95)),
        "imu_dominant_frequency_hz": imu_spectrum["dominant_hz"],
        "imu_spectral_entropy": imu_spectrum["entropy"],
        "imu_orientation_change_deg": _orientation_change(gravity, int(round(imu_rate))),
        "cross_modal_max_lag_correlation": cross_correlation,
        "cross_modal_spectral_overlap": float(np.sum(np.minimum(p_norm, i_norm))),
    }
    return features, filtered, interpolated_dynamic


def extract_synchronized_window_features(
    ppg: pd.DataFrame,
    imu: pd.DataFrame,
    config: dict[str, Any],
    start_timestamp_ms: float,
    end_timestamp_ms: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Extract the training-time features for one completed live/replay window.

    Gravity is estimated causally over all supplied IMU history, while the PPG,
    IMU and cross-modal features are calculated only for ``[start, end)``.  The
    function intentionally uses the same lower-level implementation as dataset
    preparation so a shadow prediction cannot drift onto a second feature path.
    """
    if end_timestamp_ms <= start_timestamp_ms:
        raise ValueError("Window end timestamp must be greater than its start")
    required_ppg = {"sample_seq", "timestamp_ms", "red", "ir"}
    required_imu = {"imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"}
    missing_ppg = sorted(required_ppg - set(ppg.columns))
    missing_imu = sorted(required_imu - set(imu.columns))
    if missing_ppg or missing_imu:
        raise ValueError(
            "Missing live feature columns: "
            + ", ".join([*(f"PPG {value}" for value in missing_ppg), *(f"IMU {value}" for value in missing_imu)])
        )

    ppg_columns = ["sample_seq", "timestamp_ms", "red", "ir"]
    imu_columns = ["imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"]
    ppg = ppg[ppg_columns].copy().reset_index(drop=True)
    imu = imu[imu_columns].copy().reset_index(drop=True)
    for column in ppg_columns:
        ppg[column] = pd.to_numeric(ppg[column], errors="raise")
    for column in imu_columns:
        imu[column] = pd.to_numeric(imu[column], errors="raise")

    reasons = sorted(set(
        _strict_stream_reasons(ppg, "sample_seq", "timestamp_ms", "ppg")
        + _strict_stream_reasons(imu, "imu_seq", "timestamp_ms", "imu")
    ))
    diagnostics: dict[str, Any] = {
        "start_timestamp_ms": float(start_timestamp_ms),
        "end_timestamp_ms": float(end_timestamp_ms),
        "window_seconds": float(end_timestamp_ms - start_timestamp_ms) / 1000.0,
        "rejection_reasons": reasons,
    }
    if len(ppg) < 4 or len(imu) < 4:
        diagnostics["rejection_reasons"] = sorted(set(reasons + ["insufficient_sensor_samples"]))
        return None, diagnostics

    ppg_times = ppg["timestamp_ms"].to_numpy(dtype=float)
    imu_times = imu["timestamp_ms"].to_numpy(dtype=float)
    ppg_rate = 1000.0 / float(np.median(np.diff(ppg_times)))
    imu_rate = 1000.0 / float(np.median(np.diff(imu_times)))
    ppg_indices = np.flatnonzero((ppg_times >= start_timestamp_ms) & (ppg_times < end_timestamp_ms))
    imu_indices = np.flatnonzero((imu_times >= start_timestamp_ms) & (imu_times < end_timestamp_ms))
    interpolation_left = max(0, int(np.searchsorted(imu_times, start_timestamp_ms, side="right")) - 1)
    interpolation_right = min(
        len(imu_times),
        int(np.searchsorted(imu_times, end_timestamp_ms, side="left")) + 1,
    )
    interpolation_indices = np.arange(interpolation_left, interpolation_right, dtype=int)
    window_seconds = diagnostics["window_seconds"]
    ppg_completeness = min(1.0, len(ppg_indices) / max(window_seconds * ppg_rate, 1.0))
    imu_completeness = min(1.0, len(imu_indices) / max(window_seconds * imu_rate, 1.0))
    diagnostics.update(
        {
            "ppg_sample_count": int(len(ppg_indices)),
            "imu_sample_count": int(len(imu_indices)),
            "ppg_completeness": float(ppg_completeness),
            "imu_completeness": float(imu_completeness),
        }
    )
    minimum = float(config["timing"]["minimum_completeness"])
    maximum_gap = float(config["timing"]["maximum_gap_ms"])
    if ppg_completeness < minimum:
        reasons.append("incomplete_ppg_window")
    if imu_completeness < minimum:
        reasons.append("incomplete_imu_window")
    if len(ppg_indices) < 2 or float(np.max(np.diff(ppg_times[ppg_indices]))) > maximum_gap:
        reasons.append("ppg_timestamp_gap")
    if len(imu_indices) < 2 or float(np.max(np.diff(imu_times[imu_indices]))) > maximum_gap:
        reasons.append("imu_timestamp_gap")
    if start_timestamp_ms < max(ppg_times[0], imu_times[0]) or end_timestamp_ms > min(
        ppg_times[-1] + 1000.0 / ppg_rate,
        imu_times[-1] + 1000.0 / imu_rate,
    ) + 1e-9:
        reasons.append("outside_common_sensor_coverage")
    if (
        len(ppg_indices)
        and (
            not len(interpolation_indices)
            or imu_times[interpolation_indices[0]] > ppg_times[ppg_indices[0]]
            or imu_times[interpolation_indices[-1]] < ppg_times[ppg_indices[-1]]
        )
    ):
        reasons.append("insufficient_imu_interpolation_coverage")
    diagnostics["rejection_reasons"] = sorted(set(reasons))
    if diagnostics["rejection_reasons"]:
        return None, diagnostics

    imu_derived = _causal_imu(imu, config)
    try:
        features, _, _ = _window_features(
            ppg,
            imu,
            imu_derived,
            ppg_indices,
            interpolation_indices,
            config,
        )
    except (ValueError, FloatingPointError) as exc:
        diagnostics["rejection_reasons"] = [f"feature_error:{exc}"]
        return None, diagnostics
    return features, diagnostics


def prepare_trial(trial: MotionTrial, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ppg = _numeric_frame(trial.ppg_path, ["sample_seq", "timestamp_ms", "red", "ir"])
    imu = _numeric_frame(trial.imu_path, ["imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"])
    annotations = _load_annotations(trial, str(config["protocol"]))
    recording_reasons = sorted(set(
        _strict_stream_reasons(ppg, "sample_seq", "timestamp_ms", "ppg")
        + _strict_stream_reasons(imu, "imu_seq", "timestamp_ms", "imu")
        + _health_reasons(trial.metadata)
    ))
    protocol_duration = float(annotations["scheduled_end_s"].iloc[-1])
    ppg_origin_ms = float(ppg["timestamp_ms"].iloc[0])
    common_start_ms = max(float(ppg["timestamp_ms"].iloc[0]), float(imu["timestamp_ms"].iloc[0]))
    common_end_ms = min(float(ppg["timestamp_ms"].iloc[-1]), float(imu["timestamp_ms"].iloc[-1]))
    imu_derived = _causal_imu(imu, config)
    ppg_rate = 1000.0 / float(np.median(np.diff(ppg["timestamp_ms"])))
    imu_rate = 1000.0 / float(np.median(np.diff(imu["timestamp_ms"])))
    imu_timestamps = imu["timestamp_ms"].to_numpy(dtype=float)
    window_s = float(config["window_seconds"])
    starts = np.arange(0.0, protocol_duration - window_s + 1e-9, float(config["window_step_seconds"]))
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {"ppg": ppg, "imu": imu, "imu_derived": imu_derived, "annotations": annotations, "windows": {}}
    for index, start_s in enumerate(starts):
        end_s = float(start_s + window_s)
        absolute_start = ppg_origin_ms + float(start_s) * 1000.0
        absolute_end = ppg_origin_ms + end_s * 1000.0
        ppg_indices = np.flatnonzero(
            (ppg["timestamp_ms"].to_numpy(dtype=float) >= absolute_start)
            & (ppg["timestamp_ms"].to_numpy(dtype=float) < absolute_end)
        )
        imu_indices = np.flatnonzero(
            (imu_timestamps >= absolute_start)
            & (imu_timestamps < absolute_end)
        )
        # Include one bracketing IMU sample on each side for interpolation only.
        # Completeness and gap checks continue to use samples inside the window.
        interpolation_left = max(0, int(np.searchsorted(imu_timestamps, absolute_start, side="right")) - 1)
        interpolation_right = min(
            len(imu_timestamps),
            int(np.searchsorted(imu_timestamps, absolute_end, side="left")) + 1,
        )
        interpolation_imu_indices = np.arange(interpolation_left, interpolation_right, dtype=int)
        context, exclusion_reasons = _protocol_context(float(start_s), end_s, annotations, config)
        ppg_completeness = min(1.0, len(ppg_indices) / max(window_s * ppg_rate, 1.0))
        imu_completeness = min(1.0, len(imu_indices) / max(window_s * imu_rate, 1.0))
        window_reasons = list(recording_reasons)
        minimum = float(config["timing"]["minimum_completeness"])
        if ppg_completeness < minimum:
            window_reasons.append("incomplete_ppg_window")
        if imu_completeness < minimum:
            window_reasons.append("incomplete_imu_window")
        maximum_gap = float(config["timing"]["maximum_gap_ms"])
        if len(ppg_indices) < 2 or np.max(np.diff(ppg.loc[ppg_indices, "timestamp_ms"])) > maximum_gap:
            window_reasons.append("ppg_timestamp_gap")
        if len(imu_indices) < 2 or np.max(np.diff(imu.loc[imu_indices, "timestamp_ms"])) > maximum_gap:
            window_reasons.append("imu_timestamp_gap")
        if absolute_start < common_start_ms or absolute_end > common_end_ms + 1e-9:
            window_reasons.append("outside_common_sensor_coverage")
        if (
            len(ppg_indices)
            and (
                not len(interpolation_imu_indices)
                or imu_timestamps[interpolation_imu_indices[0]] > float(ppg.loc[ppg_indices[0], "timestamp_ms"])
                or imu_timestamps[interpolation_imu_indices[-1]] < float(ppg.loc[ppg_indices[-1], "timestamp_ms"])
            )
        ):
            window_reasons.append("insufficient_imu_interpolation_coverage")
        window_id = f"{trial.participant_id}:{trial.session_id}:{trial.trial_id}:w{index:03d}"
        row: dict[str, Any] = {
            "window_id": window_id,
            "participant_id": trial.participant_id,
            "session_id": trial.session_id,
            "trial_id": trial.trial_id,
            "window_index": index,
            "start_s": float(start_s),
            "end_s": end_s,
            "ppg_start_timestamp_ms": absolute_start,
            "ppg_end_timestamp_ms": absolute_end,
            **context,
            "protocol_excluded": bool(exclusion_reasons),
            "protocol_exclusion_reasons": ";".join(exclusion_reasons),
            "recording_rejection_reasons": ";".join(recording_reasons),
            "window_rejection_reasons": ";".join(sorted(set(window_reasons))),
            "ppg_sample_count": len(ppg_indices),
            "imu_sample_count": len(imu_indices),
            "ppg_completeness": ppg_completeness,
            "imu_completeness": imu_completeness,
            "reviewable": False,
            "suggested_label": "",
            "suggestion_reasons": "",
        }
        if not window_reasons and len(ppg_indices) >= 4 and len(imu_indices) >= 4:
            try:
                features, filtered, interpolated_dynamic = _window_features(
                    ppg, imu, imu_derived, ppg_indices, interpolation_imu_indices, config
                )
                row.update(features)
                traces["windows"][window_id] = {
                    "ppg_indices": ppg_indices,
                    "filtered": filtered,
                    "interpolated_dynamic": interpolated_dynamic,
                }
                contact_like = bool(
                    features["ppg_clipping_fraction"] > 0
                    or features["ppg_max_step_counts"] > max(1000.0, 6.0 * features["ppg_dc_iqr"])
                )
                moving = features["imu_activity_above_threshold_fraction"] > 0.0
                suggestion_reasons: list[str] = []
                if contact_like:
                    suggestion = "contact_corrupted"
                    suggestion_reasons.append("large_ppg_step_or_clipping")
                elif moving:
                    suggestion = "motion_corrupted"
                    suggestion_reasons.append("imu_activity_above_reference_threshold")
                elif features["ppg_morphology_accepted"]:
                    suggestion = "clean"
                    suggestion_reasons.append("morphology_checks_passed")
                else:
                    suggestion = "uncertain"
                    suggestion_reasons.append(str(features["ppg_morphology_rejection_reason"] or "poor_waveform_quality"))
                row["suggested_label"] = suggestion
                row["suggestion_reasons"] = ";".join(suggestion_reasons)
            except (ValueError, FloatingPointError) as exc:
                row["window_rejection_reasons"] = f"feature_error:{exc}"
        row["reviewable"] = not exclusion_reasons and not row["window_rejection_reasons"]
        rows.append(row)
    report = {
        "participant_id": trial.participant_id,
        "session_id": trial.session_id,
        "trial_id": trial.trial_id,
        "recording_rejection_reasons": recording_reasons,
        "candidate_window_count": len(rows),
        "reviewable_window_count": sum(bool(row["reviewable"]) for row in rows),
        "protocol_excluded_window_count": sum(bool(row["protocol_excluded"]) for row in rows),
        "common_start_timestamp_ms": common_start_ms,
        "common_end_timestamp_ms": common_end_ms,
        "ppg_rate_hz": ppg_rate,
        "imu_rate_hz": imu_rate,
        "sensor_health": {key: trial.metadata.get(key) for key in HEALTH_COUNTERS},
        "input_sha256": {
            "ppg": file_sha256(trial.ppg_path),
            "imu": file_sha256(trial.imu_path),
            "metadata": file_sha256(trial.metadata_path),
            "annotations": file_sha256(trial.annotations_path),
        },
    }
    return pd.DataFrame(rows), report, traces


def plot_trial(trial_id: str, rows: pd.DataFrame, traces: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    ppg = traces["ppg"]
    imu = traces["imu"]
    origin = float(ppg["timestamp_ms"].iloc[0])
    ppg_time = (ppg["timestamp_ms"].to_numpy(dtype=float) - origin) / 1000.0
    imu_time = (imu["timestamp_ms"].to_numpy(dtype=float) - origin) / 1000.0
    ir = ppg["ir"].to_numpy(dtype=float)
    overview, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(ppg_time, ir - pd.Series(ir).rolling(100, center=True, min_periods=1).median(), linewidth=0.6)
    axes[0].set_ylabel("IR - local median")
    axes[1].plot(imu_time, traces["imu_derived"]["activity"], linewidth=0.6, color="tab:orange")
    axes[1].axhline(float(config["imu"]["firmware_motion_threshold_g"]), color="red", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Activity (g)")
    axes[1].set_xlabel("Seconds from first PPG sample")
    colors = {"still": "#d9f2d9", "gentle_arm_motion": "#ffe5b4", "object_handling": "#e6dcff", "whole_body_motion": "#ffd1d1", "sensor_contact_disturbance": "#f7c6de"}
    for _, block in traces["annotations"].iterrows():
        for axis in axes:
            axis.axvspan(float(block["scheduled_start_s"]), float(block["scheduled_end_s"]), color=colors.get(str(block["activity_label"]), "0.9"), alpha=0.2)
    guard_intervals: list[tuple[float, float]] = []
    protocol_end = float(traces["annotations"]["scheduled_end_s"].iloc[-1])
    edge = float(config["guards"]["recording_edge_seconds"])
    guard_intervals.extend([(0.0, edge), (protocol_end - edge, protocol_end)])
    transition = float(config["guards"]["cue_transition_seconds"])
    for boundary in traces["annotations"]["scheduled_start_s"].to_numpy(dtype=float)[1:]:
        guard_intervals.append((boundary - transition, boundary + transition))
    recovery = float(config["guards"]["early_recovery_seconds"])
    for _, block in traces["annotations"].iterrows():
        if str(block["block_name"]).startswith("still_recovery"):
            start = float(block["scheduled_start_s"])
            guard_intervals.append((start, start + recovery))
    for guard_start, guard_end in guard_intervals:
        for axis in axes:
            axis.axvspan(guard_start, guard_end, facecolor="0.4", alpha=0.12, hatch="///", edgecolor="0.25")
    for _, row in rows[rows["reviewable"] == True].iterrows():  # noqa: E712
        axes[0].text((float(row["start_s"]) + float(row["end_s"])) / 2, 0.97, f"w{int(row['window_index']):03d}", transform=axes[0].get_xaxis_transform(), fontsize=6, rotation=90, ha="center", va="top")
    overview.suptitle(f"{trial_id}: motion-quality overview")
    overview.tight_layout()
    overview_path = output_dir / f"{trial_id}_overview.png"
    overview.savefig(overview_path, dpi=140)
    plt.close(overview)

    paths = [overview_path]
    reviewable = rows[rows["reviewable"] == True].reset_index(drop=True)  # noqa: E712
    per_page = int(config["review"]["windows_per_page"])
    for page_start in range(0, len(reviewable), per_page):
        page_rows = reviewable.iloc[page_start : page_start + per_page]
        figure, page_axes = plt.subplots(4, 2, figsize=(14, 12), squeeze=False)
        for axis in page_axes.flat:
            axis.set_visible(False)
        for axis, (_, row) in zip(page_axes.flat, page_rows.iterrows()):
            axis.set_visible(True)
            window_id = str(row["window_id"])
            trace = traces["windows"][window_id]
            indices = trace["ppg_indices"]
            time = (ppg.loc[indices, "timestamp_ms"].to_numpy(dtype=float) - origin) / 1000.0
            filtered = np.asarray(trace["filtered"], dtype=float)
            filtered /= np.std(filtered) + np.finfo(float).eps
            raw_ir = ppg.loc[indices, "ir"].to_numpy(dtype=float)
            raw_relative = time - time[0]
            raw_detrended = raw_ir - np.polyval(np.polyfit(raw_relative, raw_ir, 1), raw_relative)
            raw_detrended /= np.std(raw_detrended) + np.finfo(float).eps
            red = ppg.loc[indices, "red"].to_numpy(dtype=float)
            red_filtered = _filter(red, float(row["ppg_sample_rate_hz"]), config)
            red_filtered /= np.std(red_filtered) + np.finfo(float).eps
            axis.plot(time, raw_detrended, linewidth=0.5, alpha=0.4, color="0.35", label="IR detrended")
            axis.plot(time, filtered, linewidth=0.8, label="IR filtered")
            axis.plot(time, red_filtered, linewidth=0.6, alpha=0.65, color="tab:red", label="Red filtered")
            axis.set_ylabel("normalized PPG", fontsize=7)
            axis.legend(fontsize=6, loc="upper left", ncol=3)
            secondary = axis.twinx()
            secondary.plot(time, trace["interpolated_dynamic"], linewidth=0.6, color="tab:green", alpha=0.7)
            secondary.axhline(float(config["imu"]["firmware_motion_threshold_g"]), color="black", linestyle="--", linewidth=0.5)
            secondary.set_ylabel("dynamic g", fontsize=7)
            axis.set_title(f"w{int(row['window_index']):03d} | {row['activity_label']} | {row['start_s']:.0f}-{row['end_s']:.0f}s", fontsize=9)
            axis.tick_params(labelsize=7)
        page_number = page_start // per_page + 1
        figure.suptitle(f"{trial_id}: manual review page {page_number}")
        figure.tight_layout()
        page_path = output_dir / f"{trial_id}_review_{page_number:02d}.png"
        figure.savefig(page_path, dpi=140)
        plt.close(figure)
        paths.append(page_path)
    return paths


def prepare_dataset(config: dict[str, Any], input_dir: Path, session: str, output_dir: Path) -> dict[str, Any]:
    trials = discover_trials(input_dir, session, str(config["protocol"]))
    all_rows: list[pd.DataFrame] = []
    trial_reports: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    plots_dir = output_dir / "review_plots"
    for trial in trials:
        rows, report, traces = prepare_trial(trial, config)
        all_rows.append(rows)
        trial_reports.append(report)
        plot_paths.extend(str(path) for path in plot_trial(trial.trial_id, rows, traces, plots_dir, config))
    features = pd.concat(all_rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "window_features.csv"
    features.to_csv(feature_path, index=False)
    reviewable = features[features["reviewable"] == True]  # noqa: E712
    review = pd.DataFrame(
        {
            "window_id": reviewable["window_id"],
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        },
        columns=REVIEW_COLUMNS,
    )
    review.to_csv(output_dir / "window_review.csv", index=False)
    model_feature_start = features.columns.get_loc("ppg_sample_rate_hz")
    model_feature_columns = [
        column
        for column in features.columns[model_feature_start:]
        if pd.api.types.is_numeric_dtype(features[column]) or pd.api.types.is_bool_dtype(features[column])
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "session_id": session,
        "single_subject_development": True,
        "population_validated": False,
        "bp_labels_used": False,
        "activity_labels_are_model_features": False,
        "model_feature_columns": model_feature_columns,
        "config": config,
        "trial_count": len(trials),
        "candidate_window_count": len(features),
        "reviewable_window_count": len(review),
        "protocol_excluded_window_count": int(features["protocol_excluded"].sum()),
        "quality_rejected_window_count": int((~features["reviewable"] & ~features["protocol_excluded"]).sum()),
        "candidate_activity_context_counts": {
            str(key): int(value) for key, value in features["activity_label"].value_counts().sort_index().items()
        },
        "reviewable_activity_context_counts": {
            str(key): int(value)
            for key, value in reviewable["activity_label"].value_counts().sort_index().items()
        },
        "reviewable_suggested_label_counts": {
            str(key): int(value)
            for key, value in reviewable["suggested_label"].value_counts().sort_index().items()
        },
        "protocol_exclusion_reason_counts": _reason_counts(features["protocol_exclusion_reasons"]),
        "window_rejection_reason_counts": _reason_counts(features["window_rejection_reasons"]),
        "trials": trial_reports,
        "plot_files": plot_paths,
        "window_features_sha256": file_sha256(feature_path),
    }
    (output_dir / "study_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def finalize_dataset(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    feature_path = run_dir / "window_features.csv"
    review_path = run_dir / "window_review.csv"
    report_path = run_dir / "study_report.json"
    if not feature_path.exists() or not review_path.exists() or not report_path.exists():
        raise FileNotFoundError("Run directory must contain window_features.csv, window_review.csv and study_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if file_sha256(feature_path) != report.get("window_features_sha256"):
        raise ValueError("window_features.csv changed after preparation; rerun prepare")
    features = pd.read_csv(feature_path)
    review = pd.read_csv(review_path, keep_default_na=False)
    if list(review.columns) != REVIEW_COLUMNS:
        raise ValueError(f"window_review.csv columns must be exactly: {REVIEW_COLUMNS}")
    if review["window_id"].duplicated().any():
        raise ValueError("window_review.csv contains duplicate window IDs")
    expected = set(features.loc[features["reviewable"] == True, "window_id"].astype(str))  # noqa: E712
    actual = set(review["window_id"].astype(str))
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"Review window IDs do not match; missing={missing}, unknown={unknown}")
    invalid = sorted(set(review["reviewed_label"].astype(str)) - set(REVIEW_LABELS))
    if invalid:
        raise ValueError(f"Invalid or blank reviewed labels: {invalid}; allowed={list(REVIEW_LABELS)}")
    reviewed = features.merge(review, on="window_id", how="inner", validate="one_to_one")
    reviewed["usable"] = reviewed["reviewed_label"].map(
        {"clean": 1, "motion_corrupted": 0, "contact_corrupted": 0}
    )
    reviewed["supervised_training_eligible"] = reviewed["reviewed_label"] != "uncertain"
    reviewed["artifact_subtype"] = reviewed["reviewed_label"].map(
        {"motion_corrupted": "motion", "contact_corrupted": "contact"}
    ).fillna("")
    reviewed.to_csv(run_dir / "reviewed_windows.csv", index=False)
    counts = {label: int((reviewed["reviewed_label"] == label).sum()) for label in REVIEW_LABELS}
    minimums = {
        "clean": int(config["review"]["minimum_clean"]),
        "motion_corrupted": int(config["review"]["minimum_motion_corrupted"]),
        "contact_corrupted": int(config["review"]["minimum_contact_corrupted"]),
    }
    warnings = [f"insufficient_{label}:{counts[label]}<{minimum}" for label, minimum in minimums.items() if counts[label] < minimum]
    final_report = {
        "schema_version": 1,
        "single_subject_development": True,
        "population_validated": False,
        "bp_labels_used": False,
        "activity_labels_are_model_features": False,
        "model_feature_columns": report.get("model_feature_columns", []),
        "reviewed_window_count": len(reviewed),
        "supervised_training_window_count": int(reviewed["supervised_training_eligible"].sum()),
        "label_counts": counts,
        "training_readiness_warnings": warnings,
        "reviewed_windows_sha256": file_sha256(run_dir / "reviewed_windows.csv"),
    }
    (run_dir / "finalization_report.json").write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
    return final_report
