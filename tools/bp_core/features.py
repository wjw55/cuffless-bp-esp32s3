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

SENSOR_HEALTH_COUNTERS = (
    "firmware_i2c_error_count",
    "firmware_fifo_overflow_count",
    "imu_firmware_i2c_error_count",
    "imu_firmware_fifo_overflow_count",
)


class SignalQualityError(ValueError):
    """A recording-level fault that makes every window unusable."""


def _counter_rejection_reasons(metadata: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in SENSOR_HEALTH_COUNTERS:
        if key not in metadata or metadata[key] in (None, ""):
            continue
        try:
            value = float(metadata[key])
        except (TypeError, ValueError):
            reasons.append(f"invalid_sensor_health_counter:{key}")
            continue
        if not math.isfinite(value) or value < 0:
            reasons.append(f"invalid_sensor_health_counter:{key}")
        elif value > 0:
            reasons.append(f"sensor_health_error:{key}={value:g}")
    return reasons


def recording_quality_reasons(recording: Recording, signal: SignalData) -> list[str]:
    """Return label-independent recording faults shared by every BP path."""
    reasons = _counter_rejection_reasons(signal.metadata)
    time_s = np.asarray(signal.time_s, dtype=float)
    if len(time_s) < 2 or not np.all(np.isfinite(time_s)):
        reasons.append("missing_ppg_timestamps")
    else:
        timestamp_differences = np.diff(time_s)
        if np.any(timestamp_differences <= 0):
            reasons.append("non_monotonic_ppg_timestamps")
        else:
            typical_interval = float(np.median(timestamp_differences))
            if typical_interval > 0 and np.any(timestamp_differences > 1.5 * typical_interval):
                reasons.append("missing_ppg_timestamps")

    sequence = signal.sample_sequence
    if sequence is None or len(sequence) != len(time_s):
        reasons.append("missing_ppg_sequences")
    else:
        sequence = np.asarray(sequence, dtype=float)
        if not np.all(np.isfinite(sequence)):
            reasons.append("missing_ppg_sequences")
        elif len(sequence) > 1:
            differences = np.diff(sequence)
            if np.any(differences <= 0):
                reasons.append("non_monotonic_ppg_sequences")
            if np.any((differences > 0) & ~np.isclose(differences, 1.0, rtol=0.0, atol=1e-9)):
                reasons.append("missing_ppg_sequences")

    rejected_statuses = {"reject", "rejected", "unusable", "poor", "pending_manual_review", "uncertain"}
    status = str(recording.quality_status or "").strip().lower()
    if recording.dataset_id == "local_upper_arm" and status in rejected_statuses:
        reasons.append(f"unresolved_quality_status:{status}")
    return sorted(set(reasons))


def _upper_arm_analyzer_reasons(
    recording: Recording,
    signal: SignalData,
    quality: dict[str, Any],
) -> list[str]:
    if (
        recording.dataset_id != "local_upper_arm"
        or not bool(quality.get("require_upper_arm_analyzer_acceptance", False))
    ):
        return []
    # Import lazily so external-dataset-only uses of bp_core do not depend on
    # the upper-arm presentation module being imported at startup.
    from upper_arm_hr import analyze_upper_arm_ppg

    first_timestamp_ms = float(signal.metadata.get("bp_pipeline_first_timestamp_ms", 0.0))
    sequence = np.asarray(signal.sample_sequence, dtype=float)
    frame = pd.DataFrame(
        {
            "sample_seq": sequence,
            "timestamp_ms": first_timestamp_ms + np.asarray(signal.time_s, dtype=float) * 1000.0,
            "red": np.asarray(signal.red if signal.red is not None else signal.ir, dtype=float),
            "ir": np.asarray(signal.ir, dtype=float),
        }
    )
    try:
        result = analyze_upper_arm_ppg(frame, signal.metadata)
    except Exception as exc:
        return [f"upper_arm_quality_error:{exc}"]
    if result.status == "usable":
        return []
    status_map = {
        "motion_contaminated": "unresolved_motion_artifact",
        "contact_artifact": "unresolved_contact_artifact",
        "poor_contact": "unresolved_contact_artifact",
        "clipping": "unresolved_contact_artifact",
        "invalid_timing": "invalid_upper_arm_timing",
        "ambiguous_hr": "poor_waveform_quality",
        "poor_waveform_quality": "poor_waveform_quality",
        "insufficient_clean_data": "poor_waveform_quality",
    }
    category = status_map.get(result.status, "poor_waveform_quality")
    return [f"{category}:upper_arm_analyzer={result.status}"]


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
            if status == "still":
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
    recording_reasons = recording_quality_reasons(recording, signal)
    if not recording_reasons:
        recording_reasons.extend(_upper_arm_analyzer_reasons(recording, signal, quality))
    if recording_reasons:
        raise SignalQualityError(";".join(recording_reasons))
    sample_rate = estimate_sample_rate(signal.time_s) or recording.sample_rate_hz
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("sample_rate_unavailable")
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
            "quality_status": recording.quality_status,
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
    recording_rejection_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate one recording exactly as the training pipeline aggregates an occasion."""
    group = segment_rows.copy() if isinstance(segment_rows, pd.DataFrame) else pd.DataFrame(segment_rows)
    quality = config["quality"]
    minimum = int(quality.get("minimum_accepted_windows_per_occasion", 3))
    minimum_coverage = float(quality.get("minimum_unique_clean_coverage_seconds", 60.0))
    recording_rejection_reasons = sorted(set(recording_rejection_reasons or []))

    def clean_coverage_seconds(accepted_rows: pd.DataFrame) -> float:
        if accepted_rows.empty:
            return 0.0
        duration = 0.0
        grouping = "recording_id" if "recording_id" in accepted_rows.columns else None
        groups = accepted_rows.groupby(grouping, sort=False) if grouping else [("recording", accepted_rows)]
        for _, rows in groups:
            intervals = sorted(
                (float(start), float(end))
                for start, end in zip(rows["start_s"], rows["end_s"])
                if math.isfinite(float(start)) and math.isfinite(float(end)) and float(end) > float(start)
            )
            merged: list[list[float]] = []
            for start, end in intervals:
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            duration += sum(end - start for start, end in merged)
        return float(duration)

    def rejection_reasons(accepted_rows: pd.DataFrame, all_rows: pd.DataFrame) -> list[str]:
        reasons = list(recording_rejection_reasons)
        coverage = clean_coverage_seconds(accepted_rows)
        if len(accepted_rows) < minimum:
            reasons.append(f"insufficient_accepted_windows:{len(accepted_rows)}<{minimum}")
        if coverage + 1e-9 < minimum_coverage:
            reasons.append(f"insufficient_unique_clean_coverage:{coverage:.3f}<{minimum_coverage:.3f}")
        rejected_text = ";".join(all_rows.get("rejection_reason", pd.Series(dtype=str)).fillna("").astype(str))
        if coverage + 1e-9 < minimum_coverage and "motion" in rejected_text:
            reasons.append("unresolved_motion_artifact")
        if coverage + 1e-9 < minimum_coverage and any(
            value in rejected_text for value in ("contact_step", "poor_contact", "clipping")
        ):
            reasons.append("unresolved_contact_artifact")
        return sorted(set(reasons))
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
            "unique_clean_coverage_s": 0.0,
            "occasion_status": "rejected",
            "occasion_rejection_reasons": ";".join(
                recording_rejection_reasons
                or [
                    f"insufficient_accepted_windows:0<{minimum}",
                    f"insufficient_unique_clean_coverage:0.000<{minimum_coverage:.3f}",
                ]
            ),
        }
    accepted = group[group["accepted"] == True]  # noqa: E712
    total = len(group)
    clean_coverage = clean_coverage_seconds(accepted)
    occasion_reasons = rejection_reasons(accepted, group)
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
        "quality_status": recording.quality_status,
        "calibration_occasion": recording.calibration_occasion,
        "total_segment_count": total,
        "accepted_segment_count": len(accepted),
        "accepted_segment_fraction": len(accepted) / total if total else 0.0,
        "unique_clean_coverage_s": clean_coverage,
        "occasion_usable": not occasion_reasons,
        "occasion_status": "usable" if not occasion_reasons else "rejected",
        "occasion_rejection_reasons": ";".join(occasion_reasons),
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
    failures_by_group: dict[str, list[str]] = {}
    for recording in recordings:
        recording_by_group.setdefault(recording.label_group_id, []).append(recording)
        try:
            segment_rows.extend(process_recording(recording, config))
        except SignalQualityError as exc:
            failures_by_group.setdefault(recording.label_group_id, []).extend(str(exc).split(";"))
        except Exception as exc:
            failures_by_group.setdefault(recording.label_group_id, []).extend(str(exc).split(";"))
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
    segment_groups = {key: value for key, value in segments.groupby("label_group_id", sort=True)} if not segments.empty else {}
    for label_group_id in sorted(recording_by_group):
        group = segment_groups.get(label_group_id, pd.DataFrame())
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
        row = aggregate_recording_features(
            representative,
            group,
            config,
            recording_rejection_reasons=failures_by_group.get(label_group_id),
        )
        occasions.append(row)
    return segments, pd.DataFrame(occasions), errors
