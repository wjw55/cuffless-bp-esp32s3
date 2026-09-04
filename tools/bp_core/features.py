from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from .datasets import Recording, SignalData, load_recording


MODEL_FEATURES = [
    "pulse_rate_bpm",
    "median_ibi_s",
    "ibi_cv",
    "rise_time_s",
    "fall_time_s",
    "peak_phase",
    "width_25_phase",
    "width_50_phase",
    "width_75_phase",
    "systolic_area_norm",
    "diastolic_area_norm",
    "area_ratio",
    "max_d1_phase",
    "min_d1_phase",
    "max_d1_norm",
    "min_d1_norm",
    "template_correlation",
]


def scaled_mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0
    center = float(np.median(finite))
    return 1.4826 * float(np.median(np.abs(finite - center)))


def estimate_sample_rate(time_s: np.ndarray) -> float | None:
    differences = np.diff(np.asarray(time_s, dtype=float))
    valid = differences[np.isfinite(differences) & (differences > 0)]
    return float(1.0 / np.median(valid)) if len(valid) else None


def _expand_mask(mask: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0 or not np.any(mask):
        return mask.copy()
    return np.convolve(mask.astype(np.int8), np.ones(2 * samples + 1, dtype=np.int8), mode="same") > 0


def _motion_mask(signal: SignalData, sample_rate_hz: float, margin_s: float) -> np.ndarray:
    mask = np.zeros(len(signal.time_s), dtype=bool)
    updates = signal.metadata.get("firmware_motion_updates")
    first_timestamp_ms = signal.metadata.get("bp_pipeline_first_timestamp_ms")
    if isinstance(updates, list) and first_timestamp_ms is not None:
        valid: list[tuple[float, str]] = []
        for update in updates:
            try:
                elapsed_s = (float(update["timestamp_ms"]) - float(first_timestamp_ms)) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            valid.append((elapsed_s, str(update.get("status", ""))))
        valid.sort()
        for index, (start_s, status) in enumerate(valid):
            if status != "moving":
                continue
            end_s = valid[index + 1][0] if index + 1 < len(valid) else float(signal.time_s[-1])
            mask |= (signal.time_s >= start_s) & (signal.time_s <= end_s)
        return _expand_mask(mask, int(round(margin_s * sample_rate_hz)))

    acceleration = signal.acceleration_m_s2
    if acceleration is None or len(acceleration) != len(mask):
        return mask
    axes = np.asarray(acceleration, dtype=float) / 9.80665
    window = max(3, int(round(sample_rate_hz)))
    gravity = pd.DataFrame(axes).rolling(window, center=True, min_periods=1).median().to_numpy()
    dynamic = np.linalg.norm(axes - gravity, axis=1)
    threshold = max(0.03, float(np.nanmedian(dynamic)) + 6.0 * scaled_mad(dynamic))
    mask = np.isfinite(dynamic) & (dynamic > threshold)
    return _expand_mask(mask, int(round(margin_s * sample_rate_hz)))


def _contact_masks(
    signal: SignalData,
    recording: Recording,
    sample_rate_hz: float,
    quality: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ir = np.asarray(signal.ir, dtype=float)
    contact_step = np.zeros(len(ir), dtype=bool)
    poor_contact = np.zeros(len(ir), dtype=bool)
    clipping = np.zeros(len(ir), dtype=bool)
    if not len(ir):
        return contact_step, poor_contact, clipping

    if recording.dataset_id in {"local_upper_arm", "one_month_wrist"}:
        clipping = (ir <= 16.0) | (ir >= float((1 << 18) - 17))
    else:
        finite = ir[np.isfinite(ir)]
        if len(finite):
            minimum, maximum = float(np.min(finite)), float(np.max(finite))
            clipping = (ir == minimum) | (ir == maximum)
            if float(np.mean(clipping)) <= quality["maximum_clipped_fraction"]:
                clipping[:] = False

    if recording.dataset_id == "local_upper_arm":
        threshold = float(quality["local_contact_threshold_counts"])
        poor_contact = ir < threshold

    finite_time = np.isfinite(signal.time_s)
    second_index = np.floor(signal.time_s[finite_time]).astype(int)
    medians = pd.Series(ir[finite_time]).groupby(second_index).median().sort_index()
    if len(medians) >= 3:
        differences = np.abs(np.diff(medians.to_numpy(dtype=float)))
        step_threshold = max(float(np.median(differences)) + 6.0 * scaled_mad(differences), np.finfo(float).eps)
        labels = medians.index.to_numpy(dtype=float)
        for index in np.flatnonzero(differences > step_threshold):
            boundary = labels[index + 1]
            contact_step |= np.abs(signal.time_s - boundary) <= float(quality["contact_margin_seconds"])
    return contact_step, poor_contact, clipping


def _filter(signal: np.ndarray, sample_rate_hz: float, settings: dict[str, Any]) -> np.ndarray:
    nyquist = sample_rate_hz / 2.0
    high = min(float(settings["bandpass_high_hz"]), nyquist * 0.90)
    low = float(settings["bandpass_low_hz"])
    if high <= low:
        raise ValueError(f"Sampling rate {sample_rate_hz:.3f} Hz is too low for configured band-pass")
    sos = butter(int(settings["filter_order"]), [low, high], btype="bandpass", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))


@dataclass
class BeatCandidate:
    sign: int
    peaks: np.ndarray
    templates: np.ndarray
    beat_features: list[dict[str, float]]
    pulse_rate_bpm: float | None
    ibi_cv: float | None
    template_correlation: float | None

    @property
    def score(self) -> tuple[int, float, float]:
        return (
            len(self.beat_features),
            self.template_correlation if self.template_correlation is not None else -1.0,
            -(self.ibi_cv if self.ibi_cv is not None else 1.0),
        )


def _crossing_width(template: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(template >= level)
    return float((indices[-1] - indices[0]) / max(len(template) - 1, 1)) if len(indices) >= 2 else math.nan


def _candidate(signal: np.ndarray, time_s: np.ndarray, sample_rate_hz: float, sign: int, settings: dict[str, Any]) -> BeatCandidate:
    oriented = sign * np.asarray(signal, dtype=float)
    frequencies, power = welch(
        oriented,
        fs=sample_rate_hz,
        nperseg=len(oriented),
        nfft=max(2048, 1 << int(np.ceil(np.log2(max(len(oriented), 2))))),
    )
    pulse_band = (
        (frequencies >= float(settings["minimum_hr_bpm"]) / 60.0)
        & (frequencies <= float(settings["maximum_hr_bpm"]) / 60.0)
    )
    spectral_bpm = None
    if np.any(pulse_band):
        spectral_bpm = float(frequencies[pulse_band][np.argmax(power[pulse_band])] * 60.0)
    refractory_s = 60.0 / float(settings["maximum_hr_bpm"])
    if spectral_bpm is not None and spectral_bpm > 0:
        refractory_s = max(refractory_s, 0.65 * 60.0 / spectral_bpm)
    distance = max(1, int(math.floor(sample_rate_hz * refractory_s)))
    prominence = max(0.25 * scaled_mad(oriented), np.finfo(float).eps)
    peaks, _ = find_peaks(oriented, distance=distance, prominence=prominence)
    if len(peaks) >= 2:
        intervals = np.diff(time_s[peaks])
        valid = intervals[
            (intervals >= 60.0 / float(settings["maximum_hr_bpm"]))
            & (intervals <= 60.0 / float(settings["minimum_hr_bpm"]))
        ]
    else:
        valid = np.array([], dtype=float)
    pulse_rate = 60.0 / float(np.median(valid)) if len(valid) else None
    ibi_cv = float(np.std(valid) / np.mean(valid)) if len(valid) >= 2 and np.mean(valid) > 0 else (0.0 if len(valid) == 1 else None)

    template_samples = int(settings["template_samples"])
    phase = np.linspace(0.0, 1.0, template_samples)
    templates: list[np.ndarray] = []
    features: list[dict[str, float]] = []
    for index, peak in enumerate(peaks):
        left_boundary = 0 if index == 0 else int((peaks[index - 1] + peak) // 2)
        right_boundary = len(oriented) - 1 if index + 1 == len(peaks) else int((peak + peaks[index + 1]) // 2)
        if peak <= left_boundary or right_boundary <= peak:
            continue
        left = left_boundary + int(np.argmin(oriented[left_boundary : peak + 1]))
        right = peak + int(np.argmin(oriented[peak : right_boundary + 1]))
        duration = float(time_s[right] - time_s[left])
        if duration < 60.0 / float(settings["maximum_hr_bpm"]) or duration > 60.0 / float(settings["minimum_hr_bpm"]):
            continue
        baseline = np.linspace(oriented[left], oriented[right], right - left + 1)
        beat = oriented[left : right + 1] - baseline
        amplitude = float(beat[peak - left])
        if not math.isfinite(amplitude) or amplitude <= np.finfo(float).eps:
            continue
        beat /= amplitude
        beat_phase = (time_s[left : right + 1] - time_s[left]) / duration
        normalized = np.interp(phase, beat_phase, beat)
        peak_phase = float((time_s[peak] - time_s[left]) / duration)
        peak_index = int(round(peak_phase * (template_samples - 1)))
        derivative = np.gradient(normalized, phase)
        systolic_area = float(np.trapezoid(normalized[: peak_index + 1], phase[: peak_index + 1])) if peak_index >= 1 else math.nan
        diastolic_area = float(np.trapezoid(normalized[peak_index:], phase[peak_index:])) if peak_index < template_samples - 1 else math.nan
        templates.append(normalized)
        features.append(
            {
                "rise_time_s": float(time_s[peak] - time_s[left]),
                "fall_time_s": float(time_s[right] - time_s[peak]),
                "peak_phase": peak_phase,
                "width_25_phase": _crossing_width(normalized, 0.25),
                "width_50_phase": _crossing_width(normalized, 0.50),
                "width_75_phase": _crossing_width(normalized, 0.75),
                "systolic_area_norm": systolic_area,
                "diastolic_area_norm": diastolic_area,
                "area_ratio": systolic_area / diastolic_area if diastolic_area > 0 else math.nan,
                "max_d1_phase": float(np.argmax(derivative) / (template_samples - 1)),
                "min_d1_phase": float(np.argmin(derivative) / (template_samples - 1)),
                "max_d1_norm": float(np.max(derivative)),
                "min_d1_norm": float(np.min(derivative)),
            }
        )

    matrix = np.asarray(templates, dtype=float) if templates else np.empty((0, template_samples))
    template_correlation = None
    if len(matrix) == 1:
        template_correlation = 1.0
    elif len(matrix) >= 2:
        median_template = np.median(matrix, axis=0)
        correlations = [
            float(np.corrcoef(beat, median_template)[0, 1])
            for beat in matrix
            if np.std(beat) > 0 and np.std(median_template) > 0
        ]
        if correlations:
            template_correlation = float(np.median(correlations))
    return BeatCandidate(sign, peaks, matrix, features, pulse_rate, ibi_cv, template_correlation)


def extract_window_features(
    time_s: np.ndarray,
    ir: np.ndarray,
    red: np.ndarray | None,
    sample_rate_hz: float,
    signal_settings: dict[str, Any],
    quality: dict[str, Any],
    short_signal: bool = False,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    filtered = _filter(ir, sample_rate_hz, signal_settings)
    candidates = [
        _candidate(filtered, time_s, sample_rate_hz, sign, signal_settings) for sign in (1, -1)
    ]
    selected = max(candidates, key=lambda value: value.score)
    minimum_beats = 2 if short_signal else int(quality["minimum_beats_per_window"])
    reasons: list[str] = []
    if len(selected.peaks) < minimum_beats or len(selected.beat_features) < max(1, minimum_beats - 1):
        reasons.append("insufficient_beats")
    if selected.pulse_rate_bpm is None:
        reasons.append("implausible_or_unavailable_pulse_rate")
    if selected.ibi_cv is not None and selected.ibi_cv > float(quality["maximum_interval_cv"]):
        reasons.append("interval_cv_too_high")
    if not short_signal and (
        selected.template_correlation is None
        or selected.template_correlation < float(quality["minimum_template_correlation"])
    ):
        reasons.append("template_correlation_too_low")
    diagnostics = {
        "polarity": selected.sign,
        "detected_peak_count": len(selected.peaks),
        "valid_beat_count": len(selected.beat_features),
        "pulse_rate_bpm": selected.pulse_rate_bpm,
        "interval_cv": selected.ibi_cv,
        "template_correlation": selected.template_correlation,
    }
    if red is not None and len(red) == len(ir) and np.std(red) > 0:
        try:
            filtered_red = _filter(red, sample_rate_hz, signal_settings)
            diagnostics["red_ir_correlation"] = float(np.corrcoef(filtered_red, filtered)[0, 1])
        except ValueError:
            diagnostics["red_ir_correlation"] = None
    if reasons:
        diagnostics["rejection_reason"] = ";".join(reasons)
        return None, diagnostics

    frame = pd.DataFrame(selected.beat_features)
    features = {column: float(frame[column].median()) for column in frame.columns}
    features.update(
        {
            "pulse_rate_bpm": float(selected.pulse_rate_bpm),
            "median_ibi_s": 60.0 / float(selected.pulse_rate_bpm),
            "ibi_cv": float(selected.ibi_cv or 0.0),
            "template_correlation": float(selected.template_correlation or 1.0),
        }
    )
    return features, diagnostics


def _window_bounds(signal: SignalData, settings: dict[str, Any]) -> list[tuple[float, float, bool]]:
    finite_time = signal.time_s[np.isfinite(signal.time_s)]
    if not len(finite_time):
        return []
    start_time = float(finite_time[0])
    end_time = float(finite_time[-1])
    duration = end_time - start_time
    window = float(settings["window_seconds"])
    if duration < window:
        return [(start_time, end_time, True)]
    starts = np.arange(start_time, end_time - window + 1e-9, float(settings["window_step_seconds"]))
    return [(float(start), float(start + window), False) for start in starts]


def process_signal(
    recording: Recording,
    signal: SignalData,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract segment features from an in-memory signal using the training path."""
    settings = config["signal"]
    quality = config["quality"]
    sample_rate = estimate_sample_rate(signal.time_s) or recording.sample_rate_hz
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("sample_rate_unavailable")
    finite_time = signal.time_s[np.isfinite(signal.time_s)]
    if len(finite_time) < 2 or np.any(np.diff(finite_time) <= 0):
        raise ValueError("non_monotonic_timestamps")
    if signal.sample_sequence is not None and len(signal.sample_sequence) > 1:
        differences = np.diff(signal.sample_sequence)
        if np.any(differences <= 0) or np.any(differences > 1):
            raise ValueError("missing_or_non_monotonic_sample_sequence")

    motion = _motion_mask(signal, sample_rate, float(quality["motion_margin_seconds"]))
    contact, poor_contact, clipping = _contact_masks(signal, recording, sample_rate, quality)
    rows: list[dict[str, Any]] = []
    for segment_index, (start_s, end_s, short_signal) in enumerate(_window_bounds(signal, settings)):
        inclusive_end = short_signal
        selected = (signal.time_s >= start_s) & ((signal.time_s <= end_s) if inclusive_end else (signal.time_s < end_s))
        indices = np.flatnonzero(selected)
        base = {
            "dataset_id": recording.dataset_id,
            "participant_id": recording.participant_id,
            "session_id": recording.session_id,
            "recording_id": recording.recording_id,
            "label_group_id": recording.label_group_id,
            "segment_id": f"{recording.recording_id}:{segment_index:04d}",
            "start_s": start_s,
            "end_s": end_s,
            "sample_rate_hz": sample_rate,
            "sbp": recording.sbp,
            "dbp": recording.dbp,
            "reference_hr": recording.reference_hr,
            "calibration_occasion": recording.calibration_occasion,
            "accepted": False,
            "rejection_reason": "",
        }
        if len(indices) < 4:
            base["rejection_reason"] = "insufficient_samples"
            rows.append(base)
            continue
        expected = len(indices) if short_signal else float(settings["window_seconds"]) * sample_rate
        completeness = min(1.0, len(indices) / max(expected, 1.0))
        base["sample_completeness"] = completeness
        rejection_masks = [
            (motion, "motion"),
            (contact, "contact_step"),
            (poor_contact, "poor_contact"),
            (clipping, "clipping"),
        ]
        reasons = [reason for mask, reason in rejection_masks if np.any(mask[indices])]
        if completeness < float(quality["minimum_sample_completeness"]):
            reasons.append("incomplete_window")
        if not np.all(np.isfinite(signal.ir[indices])):
            reasons.append("non_finite_ppg")
        if float(np.mean(clipping[indices])) > float(quality["maximum_clipped_fraction"]):
            reasons.append("clipping_fraction_too_high")
        if reasons:
            base["rejection_reason"] = ";".join(sorted(set(reasons)))
            rows.append(base)
            continue
        try:
            features, diagnostics = extract_window_features(
                signal.time_s[indices],
                signal.ir[indices],
                signal.red[indices] if signal.red is not None else None,
                sample_rate,
                settings,
                quality,
                short_signal=short_signal,
            )
        except (ValueError, FloatingPointError) as exc:
            base["rejection_reason"] = f"filter_or_feature_error:{exc}"
            rows.append(base)
            continue
        base.update({f"quality__{key}": value for key, value in diagnostics.items() if key != "rejection_reason"})
        if features is None:
            base["rejection_reason"] = str(diagnostics.get("rejection_reason", "poor_waveform_quality"))
        else:
            base["accepted"] = True
            base.update({f"feature__{key}": value for key, value in features.items()})
        rows.append(base)
    return rows


def process_recording(recording: Recording, config: dict[str, Any]) -> list[dict[str, Any]]:
    return process_signal(recording, load_recording(recording), config)


def aggregate_recording_features(
    recording: Recording,
    segment_rows: list[dict[str, Any]] | pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate one recording exactly as the training pipeline aggregates an occasion."""
    group = segment_rows.copy() if isinstance(segment_rows, pd.DataFrame) else pd.DataFrame(segment_rows)
    minimum = int(config["quality"]["minimum_accepted_windows_per_occasion"])
    if group.empty:
        return {
            "dataset_id": recording.dataset_id,
            "participant_id": recording.participant_id,
            "session_id": recording.session_id,
            "label_group_id": recording.label_group_id,
            "chronological_order": recording.chronological_order,
            "sensor_site": recording.sensor_site,
            "sbp": recording.sbp,
            "dbp": recording.dbp,
            "reference_hr": recording.reference_hr,
            "calibration_occasion": recording.calibration_occasion,
            "total_segment_count": 0,
            "accepted_segment_count": 0,
            "accepted_segment_fraction": 0.0,
            "occasion_usable": False,
            "occasion_status": "insufficient_clean_data",
        }
    accepted = group[group["accepted"] == True]  # noqa: E712
    total = len(group)
    row: dict[str, Any] = {
        "dataset_id": recording.dataset_id,
        "participant_id": recording.participant_id,
        "session_id": recording.session_id,
        "label_group_id": recording.label_group_id,
        "chronological_order": recording.chronological_order,
        "sensor_site": recording.sensor_site,
        "sbp": recording.sbp,
        "dbp": recording.dbp,
        "reference_hr": recording.reference_hr,
        "calibration_occasion": recording.calibration_occasion,
        "total_segment_count": total,
        "accepted_segment_count": len(accepted),
        "accepted_segment_fraction": len(accepted) / total if total else 0.0,
        "occasion_usable": len(accepted) >= minimum,
        "occasion_status": "usable" if len(accepted) >= minimum else "insufficient_clean_data",
    }
    for column in [name for name in group.columns if name.startswith("feature__")]:
        values = pd.to_numeric(accepted[column], errors="coerce")
        row[f"median__{column}"] = float(values.median()) if values.notna().any() else math.nan
        row[f"iqr__{column}"] = (
            float(values.quantile(0.75) - values.quantile(0.25)) if values.notna().any() else math.nan
        )
    return row


def build_occasion_features(
    recordings: list[Recording], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    segment_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    recording_by_group: dict[str, list[Recording]] = {}
    for recording in recordings:
        recording_by_group.setdefault(recording.label_group_id, []).append(recording)
        try:
            segment_rows.extend(process_recording(recording, config))
        except Exception as exc:
            errors.append(
                {
                    "dataset_id": recording.dataset_id,
                    "participant_id": recording.participant_id,
                    "recording_id": recording.recording_id,
                    "error": str(exc),
                }
            )
    segments = pd.DataFrame(segment_rows)
    occasions: list[dict[str, Any]] = []
    if segments.empty:
        return segments, pd.DataFrame(), errors
    for label_group_id, group in segments.groupby("label_group_id", sort=True):
        source_rows = recording_by_group[label_group_id]
        sbp_values = {row.sbp for row in source_rows}
        dbp_values = {row.dbp for row in source_rows}
        if len(sbp_values) != 1 or len(dbp_values) != 1:
            raise ValueError(f"Conflicting BP labels inside {label_group_id}")
        representative = replace(
            source_rows[0],
            chronological_order=min(item.chronological_order for item in source_rows),
            calibration_occasion=any(item.calibration_occasion for item in source_rows),
        )
        row = aggregate_recording_features(representative, group, config)
        occasions.append(row)
    return segments, pd.DataFrame(occasions), errors
