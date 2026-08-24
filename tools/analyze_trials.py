"""Summarize Omron-labeled PPG pilot trials.

This script validates collection quality and estimates a simple PPG heart rate.
It does not train a model or predict blood pressure.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_COLUMNS = ["sample_seq", "timestamp_ms", "red", "ir"]
MIN_PEAKS_FOR_HR = 3
MIN_IR_SPAN = 1000
MIN_HR_BPM = 40.0
MAX_HR_BPM = 180.0

SUMMARY_FIELDS = [
    "subject_id",
    "session_id",
    "trial_id",
    "csv_file",
    "metadata_file",
    "systolic_mmHg",
    "diastolic_mmHg",
    "cuff_hr_bpm",
    "cuff_start_time_s",
    "cuff_reading_time_s",
    "sample_count",
    "data_duration_seconds",
    "approximate_sampling_rate_hz",
    "missing_sample_sequences",
    "median_sample_interval_ms",
    "mean_sample_interval_ms",
    "max_sample_interval_ms",
    "p95_sample_interval_ms",
    "p99_sample_interval_ms",
    "timestamp_gaps_gt_15ms",
    "timestamp_gaps_gt_20ms",
    "non_increasing_timestamp_count",
    "timing_quality",
    "timing_quality_reason",
    "firmware_captured_samples",
    "firmware_interval_rate_hz",
    "firmware_effective_rate_hz",
    "firmware_latest_fifo_available",
    "firmware_fifo_overflow_count",
    "firmware_i2c_error_count",
    "firmware_timestamp_resync_count",
    "firmware_timestamp_correction_count",
    "firmware_timestamp_lag_warning_count",
    "imu_csv_file",
    "imu_location",
    "imu_orientation",
    "imu_sample_count",
    "imu_approximate_sampling_rate_hz",
    "imu_missing_sample_sequences",
    "imu_timing_quality",
    "imu_firmware_fifo_overflow_count",
    "imu_firmware_i2c_error_count",
    "imu_firmware_effective_rate_hz",
    "imu_firmware_clock_adjustment_count",
    "imu_firmware_clock_adjustment_total_us",
    "imu_motion_threshold_g",
    "imu_motion_candidate_fraction",
    "imu_warnings",
    "warnings",
    "ignored_legacy_warnings",
    "red_min",
    "red_max",
    "red_span",
    "ir_min",
    "ir_max",
    "ir_span",
    "estimated_ppg_hr_bpm",
    "num_detected_peaks",
    "hr_error_vs_cuff_bpm",
    "live_hr_update_count",
    "live_hr_stable_update_count",
    "live_hr_mean_bpm",
    "live_hr_median_bpm",
    "live_hr_mean_absolute_error_vs_offline_bpm",
    "live_hr_max_absolute_error_vs_offline_bpm",
    "live_hr_mean_absolute_error_vs_cuff_bpm",
    "live_hr_max_absolute_error_vs_cuff_bpm",
    "analysis_quality",
    "analysis_quality_reason",
]


@dataclass
class TrialPair:
    csv_path: Path | None
    metadata_path: Path | None
    metadata: dict[str, Any]
    problem: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze collected PPG pilot trials.")
    parser.add_argument("--input-dir", default="data/raw", help="Folder containing raw PPG CSV and metadata JSON files")
    parser.add_argument("--session", required=True, help="Session ID to analyze, for example omron_pilot_001")
    parser.add_argument("--subject", help="Optional subject ID filter")
    parser.add_argument("--output-dir", default="data/processed", help="Base output folder")
    parser.add_argument("--make-plots", action="store_true", help="Save per-trial IR peak diagnostic plots")
    parser.add_argument(
        "--include-borderline",
        action="store_true",
        help="Include borderline trials in the printed clean pilot candidate list",
    )
    parser.add_argument("--verbose", action="store_true", help="Print missing-file and per-trial diagnostic details")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_matches(metadata: dict[str, Any], session_id: str, subject_id: str | None) -> bool:
    if metadata.get("session_id") != session_id:
        return False
    if subject_id is not None and metadata.get("subject_id") != subject_id:
        return False
    return True


def resolve_csv_path(input_dir: Path, metadata_path: Path, metadata: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []

    output_csv_path = metadata.get("output_csv_path")
    if output_csv_path:
        raw_path = Path(str(output_csv_path))
        candidates.append(raw_path)
        candidates.append(input_dir / raw_path.name)

    output_csv_filename = metadata.get("output_csv_filename")
    if output_csv_filename:
        candidates.append(input_dir / str(output_csv_filename))

    if metadata_path.name.endswith("_metadata.json"):
        candidates.append(input_dir / metadata_path.name.replace("_metadata.json", "_ppg.csv"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1] if candidates else None


def infer_identity_from_csv(csv_path: Path, session_id: str, subject_id: str | None) -> dict[str, str | None]:
    base_name = csv_path.stem.removesuffix("_ppg")
    marker = f"_{session_id}_"

    if marker in base_name:
        inferred_subject, inferred_trial = base_name.split(marker, 1)
        return {
            "subject_id": subject_id or inferred_subject,
            "session_id": session_id,
            "trial_id": inferred_trial,
        }

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "trial_id": base_name,
    }


def discover_trial_pairs(
    input_dir: Path,
    session_id: str,
    subject_id: str | None = None,
) -> tuple[list[TrialPair], list[str]]:
    pairs: list[TrialPair] = []
    problems: list[str] = []
    seen_csv_paths: set[Path] = set()

    for metadata_path in sorted(input_dir.glob("*_metadata.json")):
        try:
            metadata = load_json(metadata_path)
        except json.JSONDecodeError as exc:
            problems.append(f"Could not parse metadata {metadata_path}: {exc}")
            continue

        if not metadata_matches(metadata, session_id, subject_id):
            continue

        csv_path = resolve_csv_path(input_dir, metadata_path, metadata)
        if csv_path is None or not csv_path.exists():
            problem = f"Missing CSV for metadata {metadata_path}"
            problems.append(problem)
            pairs.append(TrialPair(None, metadata_path, metadata, problem))
            continue

        seen_csv_paths.add(csv_path.resolve())
        pairs.append(TrialPair(csv_path, metadata_path, metadata))

    for csv_path in sorted(input_dir.glob("*_ppg.csv")):
        if csv_path.resolve() in seen_csv_paths:
            continue
        if session_id not in csv_path.stem:
            continue
        if subject_id is not None and not csv_path.stem.startswith(f"{subject_id}_"):
            continue

        metadata_path = input_dir / csv_path.name.replace("_ppg.csv", "_metadata.json")
        if metadata_path.exists():
            continue

        metadata = infer_identity_from_csv(csv_path, session_id, subject_id)
        problem = f"Missing metadata for CSV {csv_path}"
        problems.append(problem)
        pairs.append(TrialPair(csv_path, None, metadata, problem))

    return pairs, problems


def get_number(metadata: dict[str, Any], key: str, fallback=None):
    value = metadata.get(key, fallback)
    if value == "":
        return fallback
    return value


def load_ppg_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_columns = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing raw column(s): {', '.join(missing_columns)}")
    return df[RAW_COLUMNS].copy()


def compute_csv_sampling_stats(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "sample_count": 0,
            "data_duration_seconds": None,
            "missing_sample_sequences": None,
            "median_sample_interval_ms": None,
            "mean_sample_interval_ms": None,
            "max_sample_interval_ms": None,
            "p95_sample_interval_ms": None,
            "p99_sample_interval_ms": None,
            "timestamp_gaps_gt_15ms": None,
            "timestamp_gaps_gt_20ms": None,
            "non_increasing_timestamp_count": None,
            "approximate_sampling_rate_hz": None,
        }

    sample_seq_delta = df["sample_seq"].diff().dropna()
    forward_seq_gaps = sample_seq_delta[sample_seq_delta > 1]
    missing_sample_sequences = int((forward_seq_gaps - 1).sum()) if len(forward_seq_gaps) else 0

    if len(df) < 2:
        return {
            "sample_count": int(len(df)),
            "data_duration_seconds": 0.0,
            "missing_sample_sequences": missing_sample_sequences,
            "median_sample_interval_ms": None,
            "mean_sample_interval_ms": None,
            "max_sample_interval_ms": None,
            "p95_sample_interval_ms": None,
            "p99_sample_interval_ms": None,
            "timestamp_gaps_gt_15ms": 0,
            "timestamp_gaps_gt_20ms": 0,
            "non_increasing_timestamp_count": 0,
            "approximate_sampling_rate_hz": None,
        }

    dt_ms = df["timestamp_ms"].diff().dropna()
    median_dt = float(dt_ms.median())
    return {
        "sample_count": int(len(df)),
        "data_duration_seconds": float((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0),
        "missing_sample_sequences": missing_sample_sequences,
        "median_sample_interval_ms": median_dt,
        "mean_sample_interval_ms": float(dt_ms.mean()),
        "min_sample_interval_ms": float(dt_ms.min()),
        "max_sample_interval_ms": float(dt_ms.max()),
        "p95_sample_interval_ms": float(dt_ms.quantile(0.95)),
        "p99_sample_interval_ms": float(dt_ms.quantile(0.99)),
        "timestamp_gaps_gt_15ms": int((dt_ms > 15).sum()),
        "timestamp_gaps_gt_20ms": int((dt_ms > 20).sum()),
        "non_increasing_timestamp_count": int((dt_ms <= 0).sum()),
        "approximate_sampling_rate_hz": (1000.0 / median_dt) if median_dt > 0 else None,
    }


def infer_timing_quality(stats: dict[str, Any]) -> tuple[str | None, str | None]:
    missing_sequences = stats.get("missing_sample_sequences")
    non_increasing = stats.get("non_increasing_timestamp_count")
    gaps_gt_15ms = stats.get("timestamp_gaps_gt_15ms")
    gaps_gt_20ms = stats.get("timestamp_gaps_gt_20ms")
    max_dt = stats.get("max_sample_interval_ms")

    required_values = [missing_sequences, non_increasing, gaps_gt_20ms, max_dt]
    if any(value is None for value in required_values):
        return None, "timing stats unavailable"

    missing_sequences = int(missing_sequences)
    non_increasing = int(non_increasing)
    gaps_gt_20ms = int(gaps_gt_20ms)
    max_dt = float(max_dt)
    gaps_gt_15ms = int(gaps_gt_15ms) if gaps_gt_15ms is not None else None

    reason = (
        f"missing_seq={missing_sequences}, non_increasing={non_increasing}, "
        f"gaps_gt_15ms={gaps_gt_15ms if gaps_gt_15ms is not None else 'n/a'}, "
        f"gaps_gt_20ms={gaps_gt_20ms}, max_dt={max_dt:.1f} ms"
    )

    if missing_sequences > 0 or non_increasing > 0 or gaps_gt_20ms > 5 or max_dt > 40:
        return "reject", reason
    if gaps_gt_15ms == 0 and max_dt <= 15:
        return "good", reason
    if gaps_gt_20ms == 0 and max_dt <= 20:
        return "usable", reason
    if gaps_gt_20ms <= 5 and max_dt <= 40:
        return "borderline", reason
    return "reject", reason


def baseline_remove_ir(df: pd.DataFrame) -> np.ndarray:
    ir = df["ir"].astype(float).to_numpy()
    if len(ir) == 0:
        return ir

    dt_ms = df["timestamp_ms"].diff().dropna()
    median_dt_ms = float(dt_ms.median()) if len(dt_ms) else 10.0
    sample_rate_hz = 1000.0 / median_dt_ms if median_dt_ms > 0 else 100.0
    baseline_window = max(5, int(round(sample_rate_hz * 0.75)))
    smooth_window = max(3, int(round(sample_rate_hz * 0.05)))

    series = pd.Series(ir)
    baseline = series.rolling(baseline_window, center=True, min_periods=1).mean()
    detrended = series - baseline
    smoothed = detrended.rolling(smooth_window, center=True, min_periods=1).mean()
    return smoothed.to_numpy()


def detect_peaks(time_s: np.ndarray, signal: np.ndarray) -> list[int]:
    if len(signal) < 3:
        return []

    min_distance_s = 60.0 / MAX_HR_BPM
    threshold = max(0.0, float(np.std(signal)) * 0.25)
    peaks: list[int] = []

    for index in range(1, len(signal) - 1):
        is_peak = signal[index] > threshold and signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]
        if not is_peak:
            continue

        if not peaks or (time_s[index] - time_s[peaks[-1]]) >= min_distance_s:
            peaks.append(index)
        elif signal[index] > signal[peaks[-1]]:
            peaks[-1] = index

    return peaks


def estimate_ppg_hr(df: pd.DataFrame) -> tuple[float | None, int, list[int], np.ndarray]:
    if len(df) < 3:
        return None, 0, [], np.array([])

    time_s = (df["timestamp_ms"].astype(float).to_numpy() - float(df["timestamp_ms"].iloc[0])) / 1000.0
    processed_ir = baseline_remove_ir(df)
    peak_indices = detect_peaks(time_s, processed_ir)

    if len(peak_indices) < MIN_PEAKS_FOR_HR:
        return None, len(peak_indices), peak_indices, processed_ir

    peak_times_s = time_s[peak_indices]
    intervals_s = np.diff(peak_times_s)
    plausible_intervals_s = intervals_s[
        (intervals_s >= (60.0 / MAX_HR_BPM)) & (intervals_s <= (60.0 / MIN_HR_BPM))
    ]
    if len(plausible_intervals_s) < 2:
        return None, len(peak_indices), peak_indices, processed_ir

    estimated_hr_bpm = 60.0 / float(np.median(plausible_intervals_s))
    if not (MIN_HR_BPM <= estimated_hr_bpm <= MAX_HR_BPM):
        return None, len(peak_indices), peak_indices, processed_ir

    return round(estimated_hr_bpm, 2), len(peak_indices), peak_indices, processed_ir


def classify_analysis_quality(
    timing_quality: str | None,
    missing_sample_sequences: int | None,
    num_detected_peaks: int,
    estimated_ppg_hr_bpm: float | None,
    hr_error_vs_cuff_bpm: float | None,
    ir_span: float | None,
    metadata_warnings: list[str],
) -> tuple[str, str]:
    # Keep this intentionally transparent: reject data that cannot support
    # analysis, flag questionable trials as borderline, and leave modeling for
    # a later stage after manual review.
    if missing_sample_sequences is not None and missing_sample_sequences > 0:
        return "reject", f"missing_sample_sequences={missing_sample_sequences}"
    if timing_quality == "reject":
        return "reject", "timing_quality=reject"
    if estimated_ppg_hr_bpm is None or not (MIN_HR_BPM <= estimated_ppg_hr_bpm <= MAX_HR_BPM):
        return "reject", "PPG HR unavailable or outside plausible range"
    if num_detected_peaks < MIN_PEAKS_FOR_HR:
        return "reject", f"too few PPG peaks detected: {num_detected_peaks}"

    borderline_reasons: list[str] = []
    if timing_quality == "borderline":
        borderline_reasons.append("borderline timing")
    if metadata_warnings:
        borderline_reasons.append("metadata warnings present")
    if hr_error_vs_cuff_bpm is not None and abs(hr_error_vs_cuff_bpm) > 10:
        borderline_reasons.append(f"PPG HR differs from cuff HR by {abs(hr_error_vs_cuff_bpm):.1f} bpm")
    if ir_span is not None and ir_span < MIN_IR_SPAN:
        borderline_reasons.append(f"small IR span: {ir_span}")

    if borderline_reasons:
        return "borderline", "; ".join(borderline_reasons)

    return "usable", f"{timing_quality or 'unknown'} timing and plausible PPG HR"


def is_legacy_timestamp_warning(warning: str) -> bool:
    return "timestamp" in warning.lower()


def split_effective_warnings(
    timing_quality: str | None,
    metadata_warnings: list[str],
) -> tuple[list[str], list[str]]:
    if timing_quality not in {"good", "usable"}:
        return metadata_warnings, []

    effective_warnings: list[str] = []
    ignored_legacy_warnings: list[str] = []
    for warning in metadata_warnings:
        if is_legacy_timestamp_warning(warning):
            ignored_legacy_warnings.append(warning)
        else:
            effective_warnings.append(warning)

    return effective_warnings, ignored_legacy_warnings


def metadata_warnings_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if value == "":
        return []
    return [str(value)]


def summarize_live_hr(metadata: dict[str, Any], offline_hr_bpm: float | None, cuff_hr_bpm: float | None) -> dict[str, Any]:
    updates = metadata.get("firmware_hr_updates")
    if not isinstance(updates, list):
        updates = []

    stable_bpms: list[float] = []
    for update in updates:
        if not isinstance(update, dict) or update.get("status") != "stable":
            continue
        bpm = update.get("bpm")
        if isinstance(bpm, (int, float)) and MIN_HR_BPM <= float(bpm) <= MAX_HR_BPM:
            stable_bpms.append(float(bpm))

    def errors(reference_bpm: float | None) -> tuple[float | None, float | None]:
        if reference_bpm is None or not stable_bpms:
            return None, None
        absolute_errors = [abs(bpm - float(reference_bpm)) for bpm in stable_bpms]
        return round(float(np.mean(absolute_errors)), 2), round(max(absolute_errors), 2)

    offline_mae, offline_max_error = errors(offline_hr_bpm)
    cuff_mae, cuff_max_error = errors(cuff_hr_bpm)
    return {
        "live_hr_update_count": len(updates),
        "live_hr_stable_update_count": len(stable_bpms),
        "live_hr_mean_bpm": round(float(np.mean(stable_bpms)), 2) if stable_bpms else None,
        "live_hr_median_bpm": round(float(np.median(stable_bpms)), 2) if stable_bpms else None,
        "live_hr_mean_absolute_error_vs_offline_bpm": offline_mae,
        "live_hr_max_absolute_error_vs_offline_bpm": offline_max_error,
        "live_hr_mean_absolute_error_vs_cuff_bpm": cuff_mae,
        "live_hr_max_absolute_error_vs_cuff_bpm": cuff_max_error,
    }


def empty_summary_row(pair: TrialPair, quality_reason: str) -> dict[str, Any]:
    metadata = pair.metadata
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(
        {
            "subject_id": metadata.get("subject_id"),
            "session_id": metadata.get("session_id"),
            "trial_id": metadata.get("trial_id"),
            "csv_file": str(pair.csv_path) if pair.csv_path is not None else None,
            "metadata_file": str(pair.metadata_path) if pair.metadata_path is not None else None,
            "analysis_quality": "reject",
            "analysis_quality_reason": quality_reason,
        }
    )
    return row


def build_summary_row(pair: TrialPair) -> dict[str, Any]:
    if pair.metadata_path is None:
        return empty_summary_row(pair, "missing metadata")
    if pair.csv_path is None:
        return empty_summary_row(pair, "missing CSV")

    metadata = pair.metadata
    df = load_ppg_csv(pair.csv_path)
    csv_stats = compute_csv_sampling_stats(df)
    estimated_hr, num_peaks, _peak_indices, _processed_ir = estimate_ppg_hr(df)

    red_min = int(df["red"].min()) if len(df) else None
    red_max = int(df["red"].max()) if len(df) else None
    ir_min = int(df["ir"].min()) if len(df) else None
    ir_max = int(df["ir"].max()) if len(df) else None
    red_span = (red_max - red_min) if red_min is not None and red_max is not None else None
    ir_span = (ir_max - ir_min) if ir_min is not None and ir_max is not None else None

    cuff_hr_bpm = metadata.get("cuff_hr_bpm")
    hr_error = None
    if cuff_hr_bpm is not None and estimated_hr is not None:
        hr_error = round(float(estimated_hr) - float(cuff_hr_bpm), 2)
    live_hr_summary = summarize_live_hr(metadata, estimated_hr, cuff_hr_bpm)

    missing_sample_sequences = get_number(metadata, "missing_sample_sequences", csv_stats["missing_sample_sequences"])
    timing_stats = {
        "missing_sample_sequences": missing_sample_sequences,
        "non_increasing_timestamp_count": get_number(
            metadata,
            "non_increasing_timestamp_count",
            csv_stats["non_increasing_timestamp_count"],
        ),
        "timestamp_gaps_gt_15ms": get_number(
            metadata,
            "timestamp_gaps_gt_15ms",
            csv_stats["timestamp_gaps_gt_15ms"],
        ),
        "timestamp_gaps_gt_20ms": get_number(
            metadata,
            "timestamp_gaps_gt_20ms",
            csv_stats["timestamp_gaps_gt_20ms"],
        ),
        "max_sample_interval_ms": get_number(metadata, "max_sample_interval_ms", csv_stats["max_sample_interval_ms"]),
    }
    inferred_timing_quality, inferred_timing_reason = infer_timing_quality(timing_stats)
    timing_quality = metadata.get("timing_quality") or inferred_timing_quality
    timing_quality_reason = metadata.get("timing_quality_reason") or inferred_timing_reason
    metadata_warnings = metadata_warnings_to_list(metadata.get("warnings"))
    effective_warnings, ignored_legacy_warnings = split_effective_warnings(timing_quality, metadata_warnings)

    analysis_quality, analysis_reason = classify_analysis_quality(
        timing_quality=timing_quality,
        missing_sample_sequences=missing_sample_sequences,
        num_detected_peaks=num_peaks,
        estimated_ppg_hr_bpm=estimated_hr,
        hr_error_vs_cuff_bpm=hr_error,
        ir_span=ir_span,
        metadata_warnings=effective_warnings,
    )

    row = {
        "subject_id": metadata.get("subject_id"),
        "session_id": metadata.get("session_id"),
        "trial_id": metadata.get("trial_id"),
        "csv_file": str(pair.csv_path),
        "metadata_file": str(pair.metadata_path),
        "systolic_mmHg": metadata.get("systolic_mmHg"),
        "diastolic_mmHg": metadata.get("diastolic_mmHg"),
        "cuff_hr_bpm": cuff_hr_bpm,
        "cuff_start_time_s": metadata.get("cuff_start_time_s"),
        "cuff_reading_time_s": metadata.get("cuff_reading_time_s"),
        "sample_count": get_number(metadata, "sample_count", csv_stats["sample_count"]),
        "data_duration_seconds": get_number(metadata, "data_duration_seconds", csv_stats["data_duration_seconds"]),
        "approximate_sampling_rate_hz": get_number(
            metadata,
            "approximate_sampling_rate_hz",
            csv_stats["approximate_sampling_rate_hz"],
        ),
        "missing_sample_sequences": missing_sample_sequences,
        "median_sample_interval_ms": get_number(
            metadata,
            "median_sample_interval_ms",
            csv_stats["median_sample_interval_ms"],
        ),
        "mean_sample_interval_ms": get_number(metadata, "mean_sample_interval_ms", csv_stats["mean_sample_interval_ms"]),
        "max_sample_interval_ms": get_number(metadata, "max_sample_interval_ms", csv_stats["max_sample_interval_ms"]),
        "p95_sample_interval_ms": get_number(metadata, "p95_sample_interval_ms", csv_stats["p95_sample_interval_ms"]),
        "p99_sample_interval_ms": get_number(metadata, "p99_sample_interval_ms", csv_stats["p99_sample_interval_ms"]),
        "timestamp_gaps_gt_15ms": get_number(
            metadata,
            "timestamp_gaps_gt_15ms",
            csv_stats["timestamp_gaps_gt_15ms"],
        ),
        "timestamp_gaps_gt_20ms": get_number(
            metadata,
            "timestamp_gaps_gt_20ms",
            csv_stats["timestamp_gaps_gt_20ms"],
        ),
        "non_increasing_timestamp_count": get_number(
            metadata,
            "non_increasing_timestamp_count",
            csv_stats["non_increasing_timestamp_count"],
        ),
        "timing_quality": timing_quality,
        "timing_quality_reason": timing_quality_reason,
        "firmware_captured_samples": metadata.get("firmware_captured_samples"),
        "firmware_interval_rate_hz": metadata.get("firmware_interval_rate_hz"),
        "firmware_effective_rate_hz": metadata.get("firmware_effective_rate_hz"),
        "firmware_latest_fifo_available": metadata.get("firmware_latest_fifo_available"),
        "firmware_fifo_overflow_count": metadata.get("firmware_fifo_overflow_count"),
        "firmware_i2c_error_count": metadata.get("firmware_i2c_error_count"),
        "firmware_timestamp_resync_count": metadata.get("firmware_timestamp_resync_count"),
        "firmware_timestamp_correction_count": metadata.get("firmware_timestamp_correction_count"),
        "firmware_timestamp_lag_warning_count": metadata.get("firmware_timestamp_lag_warning_count"),
        "imu_csv_file": metadata.get("output_imu_csv_path"),
        "imu_location": metadata.get("imu_location"),
        "imu_orientation": metadata.get("imu_orientation"),
        "imu_sample_count": metadata.get("imu_sample_count"),
        "imu_approximate_sampling_rate_hz": metadata.get("imu_approximate_sampling_rate_hz"),
        "imu_missing_sample_sequences": metadata.get("imu_missing_sample_sequences"),
        "imu_timing_quality": metadata.get("imu_timing_quality"),
        "imu_firmware_fifo_overflow_count": metadata.get("imu_firmware_fifo_overflow_count"),
        "imu_firmware_i2c_error_count": metadata.get("imu_firmware_i2c_error_count"),
        "imu_firmware_effective_rate_hz": metadata.get("imu_firmware_effective_rate_hz"),
        "imu_firmware_clock_adjustment_count": metadata.get("imu_firmware_clock_adjustment_count"),
        "imu_firmware_clock_adjustment_total_us": metadata.get("imu_firmware_clock_adjustment_total_us"),
        "imu_motion_threshold_g": metadata.get("imu_motion_threshold_g"),
        "imu_motion_candidate_fraction": metadata.get("imu_motion_candidate_fraction"),
        "imu_warnings": "; ".join(metadata_warnings_to_list(metadata.get("imu_warnings"))),
        "warnings": "; ".join(metadata_warnings),
        "ignored_legacy_warnings": "; ".join(ignored_legacy_warnings),
        "red_min": red_min,
        "red_max": red_max,
        "red_span": red_span,
        "ir_min": ir_min,
        "ir_max": ir_max,
        "ir_span": ir_span,
        "estimated_ppg_hr_bpm": estimated_hr,
        "num_detected_peaks": num_peaks,
        "hr_error_vs_cuff_bpm": hr_error,
        "analysis_quality": analysis_quality,
        "analysis_quality_reason": analysis_reason,
    }
    row.update(live_hr_summary)
    return row


def save_peak_plot(pair: TrialPair, row: dict[str, Any], plot_path: Path) -> None:
    if pair.csv_path is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_ppg_csv(pair.csv_path)
    hr_bpm, _num_peaks, peak_indices, processed_ir = estimate_ppg_hr(df)
    time_s = (df["timestamp_ms"].astype(float).to_numpy() - float(df["timestamp_ms"].iloc[0])) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, processed_ir, linewidth=0.8, label="IR baseline removed")
    if peak_indices:
        ax.scatter(time_s[peak_indices], processed_ir[peak_indices], color="tab:red", s=18, label="Detected peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IR - baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    cuff_hr = row.get("cuff_hr_bpm")
    ax.set_title(
        f"{row.get('trial_id')} | PPG HR={hr_bpm if hr_bpm is not None else 'n/a'} bpm | "
        f"Cuff HR={cuff_hr if cuff_hr is not None else 'n/a'} bpm | {row.get('analysis_quality')}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def rows_to_jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                cleaned_row[key] = None
            elif isinstance(value, np.integer):
                cleaned_row[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_row[key] = float(value)
            else:
                cleaned_row[key] = value
        cleaned_rows.append(cleaned_row)
    return cleaned_rows


def summarize_live_hr_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    offline_maes = [
        float(row["live_hr_mean_absolute_error_vs_offline_bpm"])
        for row in rows
        if row.get("live_hr_mean_absolute_error_vs_offline_bpm") is not None
    ]
    offline_max_errors = [
        float(row["live_hr_max_absolute_error_vs_offline_bpm"])
        for row in rows
        if row.get("live_hr_max_absolute_error_vs_offline_bpm") is not None
    ]
    cuff_maes = [
        float(row["live_hr_mean_absolute_error_vs_cuff_bpm"])
        for row in rows
        if row.get("live_hr_mean_absolute_error_vs_cuff_bpm") is not None
    ]
    cuff_max_errors = [
        float(row["live_hr_max_absolute_error_vs_cuff_bpm"])
        for row in rows
        if row.get("live_hr_max_absolute_error_vs_cuff_bpm") is not None
    ]

    return {
        "trials_with_offline_comparison": len(offline_maes),
        "mean_trial_mae_vs_offline_bpm": round(float(np.mean(offline_maes)), 2) if offline_maes else None,
        "max_update_error_vs_offline_bpm": round(max(offline_max_errors), 2) if offline_max_errors else None,
        "trials_with_cuff_comparison": len(cuff_maes),
        "mean_trial_mae_vs_cuff_bpm": round(float(np.mean(cuff_maes)), 2) if cuff_maes else None,
        "max_update_error_vs_cuff_bpm": round(max(cuff_max_errors), 2) if cuff_max_errors else None,
    }


def print_console_summary(
    session_id: str,
    rows: list[dict[str, Any]],
    problems: list[str],
    summary_csv_path: Path,
    summary_json_path: Path,
    include_borderline: bool,
    verbose: bool,
    live_hr_validation: dict[str, Any],
) -> None:
    usable_count = sum(row.get("analysis_quality") == "usable" for row in rows)
    borderline_count = sum(row.get("analysis_quality") == "borderline" for row in rows)
    reject_count = sum(row.get("analysis_quality") == "reject" for row in rows)
    candidate_qualities = {"usable", "borderline"} if include_borderline else {"usable"}
    clean_candidates = [
        str(row.get("trial_id"))
        for row in rows
        if row.get("analysis_quality") in candidate_qualities and row.get("trial_id")
    ]

    print(f"Analyzed session: {session_id}")
    print(f"Trials found: {len(rows)}")
    print(f"Usable: {usable_count}")
    print(f"Borderline: {borderline_count}")
    print(f"Reject: {reject_count}")
    if problems:
        print(f"File problems: {len(problems)}")
        if verbose:
            for problem in problems:
                print(f"- {problem}")
    print()
    print("Clean pilot candidates:")
    if clean_candidates:
        for trial_id in clean_candidates:
            print(f"- {trial_id}")
    else:
        print("- none")
    print()
    print("Live HR validation:")
    print(f"- trials compared with offline HR: {live_hr_validation['trials_with_offline_comparison']}")
    print(f"- mean trial MAE vs offline HR: {live_hr_validation['mean_trial_mae_vs_offline_bpm']}")
    print(f"- max live update error vs offline HR: {live_hr_validation['max_update_error_vs_offline_bpm']}")
    print(f"- trials compared with cuff HR: {live_hr_validation['trials_with_cuff_comparison']}")
    print(f"- mean trial MAE vs cuff HR: {live_hr_validation['mean_trial_mae_vs_cuff_bpm']}")
    print(f"- max live update error vs cuff HR: {live_hr_validation['max_update_error_vs_cuff_bpm']}")
    print()
    print("Saved:")
    print(f"- {summary_csv_path}")
    print(f"- {summary_json_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_session_dir = Path(args.output_dir) / args.session
    output_session_dir.mkdir(parents=True, exist_ok=True)

    pairs, problems = discover_trial_pairs(input_dir, args.session, args.subject)
    rows = [build_summary_row(pair) for pair in pairs]
    live_hr_validation = summarize_live_hr_validation(rows)

    summary_csv_path = output_session_dir / "session_summary.csv"
    summary_json_path = output_session_dir / "session_summary.json"
    pd.DataFrame(rows, columns=SUMMARY_FIELDS).to_csv(summary_csv_path, index=False)

    if args.make_plots:
        plots_dir = output_session_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for pair, row in zip(pairs, rows):
            if pair.csv_path is None or row.get("trial_id") is None:
                continue
            save_peak_plot(pair, row, plots_dir / f"{row['trial_id']}_ir_peaks.png")

    payload = {
        "session_id": args.session,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "trials": rows_to_jsonable(rows),
        "problems": problems,
        "counts": {
            "usable": sum(row.get("analysis_quality") == "usable" for row in rows),
            "borderline": sum(row.get("analysis_quality") == "borderline" for row in rows),
            "reject": sum(row.get("analysis_quality") == "reject" for row in rows),
        },
        "live_hr_validation": live_hr_validation,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_console_summary(
        args.session,
        rows,
        problems,
        summary_csv_path,
        summary_json_path,
        args.include_borderline,
        args.verbose,
        live_hr_validation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
