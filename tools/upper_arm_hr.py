"""Conservative offline heart-rate estimation for upper-arm MAX30102 PPG.

This module is intentionally offline-only.  It rejects ambiguous recordings
instead of forcing a heart-rate estimate and never changes the raw input data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


LOW_HR_BPM = 40.0
HIGH_HR_BPM = 180.0
CONTACT_THRESHOLD = 50_000.0
ADC_MAX = float((1 << 18) - 1)
EDGE_GUARD_S = 5.0
MOTION_MARGIN_S = 1.0
CONTACT_MARGIN_S = 2.0
WINDOW_S = 20.0
WINDOW_STEP_S = 5.0
MIN_REFRACTORY_S = 0.4
METHOD_AGREEMENT_BPM = 6.0
WINDOW_CLUSTER_BPM = 5.0
MIN_ACCEPTED_WINDOWS = 3
MIN_CLEAN_COVERAGE_S = 30.0
MAX_TRIAL_TREND_BPM_PER_S = 0.15


def profile_parameters() -> dict[str, Any]:
    return {
        "bandpass_hz": [0.7, 3.0],
        "filter_order": 4,
        "filter_phase": "zero_phase_offline",
        "edge_guard_s": EDGE_GUARD_S,
        "motion_margin_s": MOTION_MARGIN_S,
        "contact_margin_s": CONTACT_MARGIN_S,
        "contact_threshold_counts": CONTACT_THRESHOLD,
        "window_s": WINDOW_S,
        "window_step_s": WINDOW_STEP_S,
        "refractory_s": MIN_REFRACTORY_S,
        "method_agreement_bpm": METHOD_AGREEMENT_BPM,
        "minimum_beats_per_window": 8,
        "maximum_interval_cv": 0.20,
        "minimum_template_correlation": 0.60,
        "cluster_tolerance_bpm": WINDOW_CLUSTER_BPM,
        "minimum_accepted_windows": MIN_ACCEPTED_WINDOWS,
        "minimum_clean_coverage_s": MIN_CLEAN_COVERAGE_S,
        "minimum_winning_cluster_fraction": 0.60,
        "minimum_winner_to_runner_up_ratio": 2.0,
        "maximum_consensus_trend_bpm_per_s": MAX_TRIAL_TREND_BPM_PER_S,
    }


@dataclass
class WindowResult:
    start_s: float
    end_s: float
    status: str
    reason: str
    bpm: float | None = None
    spectral_bpm: float | None = None
    autocorrelation_bpm: float | None = None
    peak_bpm: float | None = None
    beat_count: int = 0
    interval_cv: float | None = None
    valid_interval_fraction: float | None = None
    template_correlation: float | None = None
    spectral_prominence: float | None = None
    red_ir_correlation: float | None = None
    peak_indices: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "status": self.status,
            "reason": self.reason,
            "bpm": self.bpm,
            "spectral_bpm": self.spectral_bpm,
            "autocorrelation_bpm": self.autocorrelation_bpm,
            "peak_bpm": self.peak_bpm,
            "beat_count": self.beat_count,
            "interval_cv": self.interval_cv,
            "valid_interval_fraction": self.valid_interval_fraction,
            "template_correlation": self.template_correlation,
            "spectral_prominence": self.spectral_prominence,
            "red_ir_correlation": self.red_ir_correlation,
        }


@dataclass
class UpperArmResult:
    bpm: float | None
    status: str
    status_reason: str
    processed_ir: np.ndarray
    time_s: np.ndarray
    motion_mask: np.ndarray
    contact_step_mask: np.ndarray
    poor_contact_mask: np.ndarray
    clipping_mask: np.ndarray
    edge_mask: np.ndarray
    usable_mask: np.ndarray
    windows: list[WindowResult]
    accepted_window_count: int
    clean_coverage_s: float
    motion_fraction: float
    contact_step_fraction: float
    contact_step_threshold_counts: float | None
    poor_contact_fraction: float
    clipping_fraction: float
    median_interval_cv: float | None
    median_template_correlation: float | None
    median_spectral_prominence: float | None

    @property
    def detected_peak_count(self) -> int:
        accepted = [window for window in self.windows if window.status == "accepted"]
        return max((window.beat_count for window in accepted), default=0)


def scaled_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _mask_interval(time_s: np.ndarray, mask: np.ndarray, start_s: float, end_s: float) -> None:
    mask |= (time_s >= start_s) & (time_s <= end_s)


def build_motion_mask(
    timestamps_ms: np.ndarray,
    metadata: dict[str, Any],
    margin_s: float = MOTION_MARGIN_S,
) -> np.ndarray:
    """Map firmware `moving` reports to PPG samples and add guard margins."""
    mask = np.zeros(len(timestamps_ms), dtype=bool)
    updates = metadata.get("firmware_motion_updates")
    if not isinstance(updates, list) or not len(timestamps_ms):
        return mask

    valid_updates: list[tuple[float, str]] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        try:
            update_ms = float(update["timestamp_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        valid_updates.append((update_ms, str(update.get("status", ""))))
    valid_updates.sort()

    recording_end_ms = float(timestamps_ms[-1])
    for index, (start_ms, status) in enumerate(valid_updates):
        if status != "moving":
            continue
        next_ms = valid_updates[index + 1][0] if index + 1 < len(valid_updates) else recording_end_ms
        mask |= (timestamps_ms >= start_ms - margin_s * 1000.0) & (
            timestamps_ms <= next_ms + margin_s * 1000.0
        )
    return mask


def build_contact_masks(
    time_s: np.ndarray,
    ir: np.ndarray,
    contact_threshold: float = CONTACT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Return pressure-step, low-contact, and clipping masks."""
    contact_step_mask = np.zeros(len(ir), dtype=bool)
    poor_contact_mask = ir < contact_threshold
    clipping_mask = (ir <= 16.0) | (ir >= ADC_MAX - 16.0)
    if not len(ir):
        return contact_step_mask, poor_contact_mask, clipping_mask, None

    second_index = np.floor(time_s).astype(int)
    second_medians = pd.Series(ir).groupby(second_index).median().sort_index()
    if len(second_medians) < 3:
        return contact_step_mask, poor_contact_mask, clipping_mask, None

    differences = np.abs(np.diff(second_medians.to_numpy(dtype=float)))
    difference_median = float(np.median(differences))
    threshold = difference_median + 6.0 * scaled_mad(differences)
    # A perfectly constant synthetic baseline has zero MAD.  Requiring a
    # strictly positive change still detects a real step without a raw-count
    # magic number.
    threshold = max(threshold, np.finfo(float).eps)

    second_labels = second_medians.index.to_numpy(dtype=float)
    for index in np.flatnonzero(differences > threshold):
        boundary_s = second_labels[index + 1]
        _mask_interval(
            time_s,
            contact_step_mask,
            boundary_s - CONTACT_MARGIN_S,
            boundary_s + CONTACT_MARGIN_S,
        )
    return contact_step_mask, poor_contact_mask, clipping_mask, threshold


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def bandpass_clean_segments(ir: np.ndarray, usable_mask: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    processed = np.full(len(ir), np.nan, dtype=float)
    if len(ir) == 0 or sample_rate_hz <= 0:
        return processed
    sos = butter(4, [0.7, 3.0], btype="bandpass", fs=sample_rate_hz, output="sos")
    minimum_samples = max(int(round(3.0 * sample_rate_hz)), 28)
    for start, end in _contiguous_true_runs(usable_mask):
        if end - start < minimum_samples:
            continue
        segment = ir[start:end].astype(float)
        try:
            processed[start:end] = sosfiltfilt(sos, segment)
        except ValueError:
            continue
    return processed


def _spectral_estimate(signal: np.ndarray, sample_rate_hz: float) -> tuple[float | None, float | None]:
    nfft = max(4096, 1 << int(np.ceil(np.log2(max(len(signal), 2)))))
    frequencies, power = welch(
        signal,
        fs=sample_rate_hz,
        window="hann",
        nperseg=len(signal),
        noverlap=0,
        nfft=nfft,
        detrend="constant",
    )
    band = (frequencies >= LOW_HR_BPM / 60.0) & (frequencies <= HIGH_HR_BPM / 60.0)
    if not np.any(band):
        return None, None
    band_power = power[band]
    band_frequencies = frequencies[band]
    peak_index = int(np.argmax(band_power))
    bpm = float(band_frequencies[peak_index] * 60.0)
    noise_floor = float(np.median(band_power))
    prominence = float(band_power[peak_index] / max(noise_floor, np.finfo(float).eps))
    return bpm, prominence


def _autocorrelation_estimate(signal: np.ndarray, sample_rate_hz: float) -> float | None:
    centered = signal - float(np.mean(signal))
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if not len(correlation) or correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]
    min_lag = max(1, int(np.floor(sample_rate_hz * 60.0 / HIGH_HR_BPM)))
    max_lag = min(len(correlation) - 1, int(np.ceil(sample_rate_hz * 60.0 / LOW_HR_BPM)))
    if max_lag <= min_lag:
        return None
    peaks, _ = find_peaks(correlation[min_lag : max_lag + 1])
    if not len(peaks):
        return None
    candidate_lags = peaks + min_lag
    lag = int(candidate_lags[np.argmax(correlation[candidate_lags])])
    return float(60.0 * sample_rate_hz / lag)


def _template_correlation(signal: np.ndarray, peaks: np.ndarray, sample_rate_hz: float) -> float | None:
    before = int(round(0.25 * sample_rate_hz))
    after = int(round(0.40 * sample_rate_hz))
    beats = [signal[index - before : index + after + 1] for index in peaks if index >= before and index + after < len(signal)]
    if len(beats) < 3:
        return None
    beat_matrix = np.asarray(beats)
    template = np.median(beat_matrix, axis=0)
    correlations: list[float] = []
    for beat in beat_matrix:
        if np.std(beat) == 0 or np.std(template) == 0:
            continue
        correlations.append(float(np.corrcoef(beat, template)[0, 1]))
    return float(np.median(correlations)) if correlations else None


def _agreeing_estimates(estimates: list[float], tolerance_bpm: float) -> list[float]:
    best: list[float] = []
    for center in estimates:
        group = [value for value in estimates if abs(value - center) <= tolerance_bpm]
        if len(group) > len(best):
            best = group
    return best


def estimate_window(
    time_s: np.ndarray,
    ir_signal: np.ndarray,
    red: np.ndarray,
    sample_rate_hz: float,
    global_indices: np.ndarray,
) -> WindowResult:
    start_s = float(time_s[0])
    end_s = float(time_s[-1])
    spectral_bpm, spectral_prominence = _spectral_estimate(ir_signal, sample_rate_hz)
    autocorrelation_bpm = _autocorrelation_estimate(ir_signal, sample_rate_hz)

    robust_sigma = scaled_mad(ir_signal)
    expected_period_s = 60.0 / spectral_bpm if spectral_bpm is not None and spectral_bpm > 0 else 0.0
    peak_distance_s = max(MIN_REFRACTORY_S, 0.65 * expected_period_s)
    peaks, _ = find_peaks(
        ir_signal,
        distance=max(1, int(round(peak_distance_s * sample_rate_hz))),
        prominence=max(robust_sigma * 0.35, np.finfo(float).eps),
    )
    peak_bpm = None
    interval_cv = None
    valid_interval_fraction = None
    if len(peaks) >= 2:
        intervals_s = np.diff(time_s[peaks])
        plausible = intervals_s[
            (intervals_s >= 60.0 / HIGH_HR_BPM) & (intervals_s <= 60.0 / LOW_HR_BPM)
        ]
        if len(plausible) >= 2:
            initial_median = float(np.median(plausible))
            interval_mad = scaled_mad(plausible)
            tolerance = max(3.0 * interval_mad, 0.10 * initial_median)
            valid_intervals = plausible[np.abs(plausible - initial_median) <= tolerance]
            valid_interval_fraction = float(len(valid_intervals) / len(intervals_s))
            if len(valid_intervals) >= 2:
                median_interval = float(np.median(valid_intervals))
                peak_bpm = 60.0 / median_interval
                interval_cv = (
                    float(np.std(valid_intervals) / np.mean(valid_intervals))
                    if np.mean(valid_intervals) > 0
                    else None
                )

    template_correlation = _template_correlation(ir_signal, peaks, sample_rate_hz)
    red_ir_correlation = None
    if np.std(red) > 0 and np.std(ir_signal) > 0:
        # This is diagnostic only; upper-arm red/IR correlation is not a gate.
        red_sos = butter(4, [0.7, 3.0], btype="bandpass", fs=sample_rate_hz, output="sos")
        filtered_red = sosfiltfilt(red_sos, red.astype(float))
        red_ir_correlation = float(np.corrcoef(filtered_red, ir_signal)[0, 1])

    estimates = [value for value in (spectral_bpm, autocorrelation_bpm, peak_bpm) if value is not None]
    agreeing = _agreeing_estimates(estimates, METHOD_AGREEMENT_BPM)
    reasons: list[str] = []
    if len(peaks) < 8:
        reasons.append(f"beats={len(peaks)}<8")
    if interval_cv is None or interval_cv > 0.20:
        reasons.append("interval_cv_unavailable_or_gt_0.20")
    if valid_interval_fraction is None or valid_interval_fraction < 0.70:
        reasons.append("valid_interval_fraction_unavailable_or_lt_0.70")
    if template_correlation is None or template_correlation < 0.60:
        reasons.append("template_correlation_unavailable_or_lt_0.60")
    if len(agreeing) < 2:
        reasons.append("fewer_than_two_methods_agree_within_6_bpm")

    bpm = round(float(np.median(agreeing)), 2) if len(agreeing) >= 2 and not reasons else None
    return WindowResult(
        start_s=start_s,
        end_s=end_s,
        status="accepted" if bpm is not None else "poor_waveform_quality",
        reason="accepted" if bpm is not None else "; ".join(reasons),
        bpm=bpm,
        spectral_bpm=round(spectral_bpm, 2) if spectral_bpm is not None else None,
        autocorrelation_bpm=round(autocorrelation_bpm, 2) if autocorrelation_bpm is not None else None,
        peak_bpm=round(peak_bpm, 2) if peak_bpm is not None else None,
        beat_count=len(peaks),
        interval_cv=round(interval_cv, 4) if interval_cv is not None else None,
        valid_interval_fraction=(
            round(valid_interval_fraction, 4) if valid_interval_fraction is not None else None
        ),
        template_correlation=round(template_correlation, 4) if template_correlation is not None else None,
        spectral_prominence=round(spectral_prominence, 2) if spectral_prominence is not None else None,
        red_ir_correlation=round(red_ir_correlation, 4) if red_ir_correlation is not None else None,
        peak_indices=global_indices[peaks].astype(int).tolist(),
    )


def _winning_cluster(bpms: list[float]) -> tuple[list[float], int]:
    best: list[float] = []
    for center in bpms:
        group = [value for value in bpms if abs(value - center) <= WINDOW_CLUSTER_BPM]
        if len(group) > len(best):
            best = group
        elif len(group) == len(best) and group and np.std(group) < np.std(best):
            best = group
    remaining = list(bpms)
    for value in best:
        remaining.remove(value)
    runner_up = 0
    for center in remaining:
        runner_up = max(runner_up, sum(abs(value - center) <= WINDOW_CLUSTER_BPM for value in remaining))
    return best, runner_up


def analyze_upper_arm_ppg(df: pd.DataFrame, metadata: dict[str, Any]) -> UpperArmResult:
    timestamps_ms = df["timestamp_ms"].to_numpy(dtype=float)
    ir = df["ir"].to_numpy(dtype=float)
    red = df["red"].to_numpy(dtype=float)
    if len(df):
        time_s = (timestamps_ms - timestamps_ms[0]) / 1000.0
    else:
        time_s = np.array([], dtype=float)

    dt_ms = np.diff(timestamps_ms)
    sample_rate_hz = 1000.0 / float(np.median(dt_ms)) if len(dt_ms) and np.median(dt_ms) > 0 else 100.0
    edge_mask = np.zeros(len(df), dtype=bool)
    if len(time_s):
        edge_mask = (time_s < EDGE_GUARD_S) | (time_s > time_s[-1] - EDGE_GUARD_S)
    motion_mask = build_motion_mask(timestamps_ms, metadata)
    contact_step_mask, poor_contact_mask, clipping_mask, contact_step_threshold = build_contact_masks(time_s, ir)
    usable_mask = ~(edge_mask | motion_mask | contact_step_mask | poor_contact_mask | clipping_mask)
    processed_ir = bandpass_clean_segments(ir, usable_mask, sample_rate_hz)

    windows: list[WindowResult] = []
    covered = np.zeros(len(df), dtype=bool)
    if len(time_s) and time_s[-1] >= 2 * EDGE_GUARD_S + WINDOW_S:
        last_start = time_s[-1] - EDGE_GUARD_S - WINDOW_S
        candidate_starts = list(np.arange(EDGE_GUARD_S, last_start + 0.001, WINDOW_STEP_S))
        # Add windows aligned to the exact boundaries of every sufficiently
        # long clean run.  A fixed five-second grid can otherwise discard a
        # valid 30-second still interval merely because it starts at 54.2 s.
        for run_start, run_end in _contiguous_true_runs(usable_mask & np.isfinite(processed_ir)):
            run_start_s = float(time_s[run_start])
            run_end_s = float(time_s[run_end - 1])
            if run_end_s - run_start_s < WINDOW_S:
                continue
            candidate_starts.extend(np.arange(run_start_s, run_end_s - WINDOW_S + 0.001, WINDOW_STEP_S))
            candidate_starts.append(run_end_s - WINDOW_S)
        unique_starts: list[float] = []
        for start_s in sorted(candidate_starts):
            if not unique_starts or abs(start_s - unique_starts[-1]) > 0.10:
                unique_starts.append(float(start_s))

        for start_s in unique_starts:
            end_s = start_s + WINDOW_S
            indices = np.flatnonzero((time_s >= start_s) & (time_s < end_s))
            if not len(indices):
                continue
            masks = [
                (motion_mask, "motion_contaminated"),
                (contact_step_mask, "contact_artifact"),
                (poor_contact_mask, "poor_contact"),
                (clipping_mask, "clipping"),
                (edge_mask, "edge_guard"),
            ]
            rejection = next((reason for mask, reason in masks if np.any(mask[indices])), None)
            if rejection is not None or np.any(~np.isfinite(processed_ir[indices])):
                reason = rejection or "filtered_segment_unavailable"
                windows.append(WindowResult(start_s, end_s, reason, reason))
                continue
            window = estimate_window(
                time_s[indices],
                processed_ir[indices],
                red[indices],
                sample_rate_hz,
                indices,
            )
            windows.append(window)
            if window.status == "accepted":
                covered[indices] = True

    accepted = [window for window in windows if window.status == "accepted" and window.bpm is not None]
    clean_coverage_s = float(np.count_nonzero(covered) / sample_rate_hz) if sample_rate_hz > 0 else 0.0
    bpm: float | None = None
    status = "insufficient_clean_data"
    status_reason = f"accepted_windows={len(accepted)}, clean_coverage_s={clean_coverage_s:.1f}"
    if len(accepted) >= MIN_ACCEPTED_WINDOWS and clean_coverage_s >= MIN_CLEAN_COVERAGE_S:
        winning, runner_up_count = _winning_cluster([float(window.bpm) for window in accepted])
        required_count = int(np.ceil(0.60 * len(accepted)))
        if len(winning) >= required_count and len(winning) >= 2 * runner_up_count:
            winning_windows = [window for window in accepted if any(abs(float(window.bpm) - value) < 1e-9 for value in winning)]
            centers = np.asarray([(window.start_s + window.end_s) / 2.0 for window in winning_windows])
            values = np.asarray([float(window.bpm) for window in winning_windows])
            trend = float(np.polyfit(centers, values, 1)[0]) if len(values) >= 2 else 0.0
            if abs(trend) <= MAX_TRIAL_TREND_BPM_PER_S:
                bpm = round(float(np.median(winning)), 2)
                status = "usable"
                status_reason = (
                    f"consensus_windows={len(winning)}/{len(accepted)}, "
                    f"runner_up_windows={runner_up_count}, trend_bpm_per_s={trend:.3f}"
                )
            else:
                status = "ambiguous_hr"
                status_reason = f"consensus HR changes too quickly: trend_bpm_per_s={trend:.3f}"
        else:
            status = "ambiguous_hr"
            status_reason = (
                f"largest_cluster={len(winning)}/{len(accepted)}, "
                f"runner_up_windows={runner_up_count}, required={required_count}"
            )
    elif windows:
        rejection_counts = {
            reason: sum(window.status == reason for window in windows)
            for reason in ("motion_contaminated", "contact_artifact", "poor_contact", "clipping")
        }
        most_common = max(rejection_counts, key=rejection_counts.get)
        if rejection_counts[most_common] > 0:
            status = "contact_artifact" if most_common in {"contact_artifact", "poor_contact", "clipping"} else most_common
            status_reason += f", dominant_rejection={most_common}"
        elif any(window.status == "poor_waveform_quality" for window in windows):
            status = "poor_waveform_quality"

    def fraction(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if len(mask) else 0.0

    def accepted_metric(name: str) -> float | None:
        values = [getattr(window, name) for window in accepted if getattr(window, name) is not None]
        return round(float(np.median(values)), 4) if values else None

    invalid_timing_reasons: list[str] = []
    sequence_delta = np.diff(df["sample_seq"].to_numpy(dtype=float)) if len(df) > 1 else np.array([])
    if np.any(sequence_delta > 1) or np.any(sequence_delta <= 0):
        invalid_timing_reasons.append("missing_or_non_increasing_sample_sequence")
    if np.any(dt_ms <= 0) or np.count_nonzero(dt_ms > 20.0) > 5 or (len(dt_ms) and np.max(dt_ms) > 40.0):
        invalid_timing_reasons.append("invalid_ppg_timestamps")
    if metadata.get("timing_quality") == "reject":
        invalid_timing_reasons.append("metadata_timing_quality_reject")
    for field in (
        "firmware_fifo_overflow_count",
        "firmware_i2c_error_count",
        "imu_firmware_fifo_overflow_count",
        "imu_firmware_i2c_error_count",
    ):
        value = metadata.get(field)
        try:
            if value not in (None, "") and int(value) > 0:
                invalid_timing_reasons.append(f"{field}={value}")
        except (TypeError, ValueError):
            invalid_timing_reasons.append(f"{field}=invalid")
    if invalid_timing_reasons:
        bpm = None
        status = "invalid_timing"
        status_reason = "; ".join(invalid_timing_reasons)

    return UpperArmResult(
        bpm=bpm,
        status=status,
        status_reason=status_reason,
        processed_ir=processed_ir,
        time_s=time_s,
        motion_mask=motion_mask,
        contact_step_mask=contact_step_mask,
        poor_contact_mask=poor_contact_mask,
        clipping_mask=clipping_mask,
        edge_mask=edge_mask,
        usable_mask=usable_mask,
        windows=windows,
        accepted_window_count=len(accepted),
        clean_coverage_s=round(clean_coverage_s, 2),
        motion_fraction=round(fraction(motion_mask), 4),
        contact_step_fraction=round(fraction(contact_step_mask), 4),
        contact_step_threshold_counts=(
            round(float(contact_step_threshold), 3) if contact_step_threshold is not None else None
        ),
        poor_contact_fraction=round(fraction(poor_contact_mask), 4),
        clipping_fraction=round(fraction(clipping_mask), 4),
        median_interval_cv=accepted_metric("interval_cv"),
        median_template_correlation=accepted_metric("template_correlation"),
        median_spectral_prominence=accepted_metric("spectral_prominence"),
    )


def mask_to_intervals(time_s: np.ndarray, mask: np.ndarray, category: str, source: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for start, end in _contiguous_true_runs(mask):
        if end <= start:
            continue
        sample_step = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
        intervals.append(
            {
                "start_s": round(float(time_s[start]), 3),
                "end_s": round(float(time_s[end - 1] + sample_step), 3),
                "category": category,
                "source": source,
                "review_status": "pending_manual_review" if category in {"contact_step", "poor_contact"} else "automatic",
            }
        )
    return intervals
