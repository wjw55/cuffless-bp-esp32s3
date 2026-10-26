"""Offline-only harmonic/peak-spacing experiment. Never imports a BP model.

The alternative uses waveform-derived periodicity, not reference HR or BP.
It is deliberately not selectable by a collector or live viewer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import upper_arm_hr as hr
from analyze_trials import load_session_labels

ROOT = Path(__file__).resolve().parents[2]
POLICY = {
    "name": "autocorrelation_guided_peaks_v1_offline",
    "minimum_autocorrelation": 0.30,
    "minimum_fundamental_relative_power": 0.10,
    "period_fraction_peak_spacing": 0.70,
    "detection_lowpass_frequency_factor": 1.50,
    "shared_masking_and_numeric_thresholds": hr.profile_parameters(),
    "live_enabled": False,
    "bp_estimated": False,
}


def periodicity(signal: np.ndarray, fs: float) -> tuple[float | None, float, float]:
    """Return autocorrelation BPM, strength and relative fundamental power.

    Relative power is a veto, not an independent second vote for the same BPM.
    It prevents choosing an unsupported subharmonic merely to slow the detector.
    """
    bpm = hr._autocorrelation_estimate(signal, fs)
    if bpm is None:
        return None, 0.0, 0.0
    centered = signal - np.mean(signal)
    ac = np.correlate(centered, centered, mode="full")[len(signal) - 1:]
    lag = int(round(fs * 60.0 / bpm))
    strength = float(ac[lag] / ac[0]) if ac[0] > 0 and lag < len(ac) else 0.0
    f, p = welch(signal, fs=fs, nperseg=len(signal), noverlap=0,
                 nfft=max(4096, 1 << int(np.ceil(np.log2(len(signal))))))
    band = (f >= hr.LOW_HR_BPM / 60) & (f <= hr.HIGH_HR_BPM / 60)
    near = abs(f - bpm / 60) <= max(1 / (len(signal) / fs), fs / 4096)
    relative = float(np.max(p[near]) / max(np.max(p[band]), np.finfo(float).eps))
    return bpm, strength, relative


def estimate_guided(time_s, ir_signal, red, sample_rate_hz, global_indices):
    """Guide peak spacing by autocorrelation; retain original-signal morphology.

    Detection-only smoothing is not passed to template correlation or BP features.
    The numerical beat count, interval, template and agreement thresholds stay
    unchanged. This remains susceptible to sufficiently periodic artifacts.
    """
    time_s, ir_signal = np.asarray(time_s), np.asarray(ir_signal)
    fs = float(sample_rate_hz)
    if (len(ir_signal) < 28 or fs <= 6 or not np.isfinite(ir_signal).all()
            or len(time_s) != len(ir_signal) or np.any(np.diff(time_s) <= 0)):
        return hr.WindowResult(float(time_s[0]) if len(time_s) else 0.,
                               float(time_s[-1]) if len(time_s) else 0.,
                               "poor_waveform_quality", "invalid_detector_input")
    spectral, prominence = hr._spectral_estimate(ir_signal, fs)
    auto, strength, relative = periodicity(ir_signal, fs)
    if auto is None:
        return hr.WindowResult(float(time_s[0]), float(time_s[-1]),
                               "poor_waveform_quality", "autocorrelation_unavailable")
    cutoff = min(3.0, POLICY["detection_lowpass_frequency_factor"] * auto / 60)
    detection = sosfiltfilt(butter(2, cutoff, btype="lowpass", fs=fs, output="sos"), ir_signal)
    distance = max(hr.MIN_REFRACTORY_S, POLICY["period_fraction_peak_spacing"] * 60 / auto)
    peaks, _ = find_peaks(detection, distance=max(1, int(round(distance * fs))),
                         prominence=max(hr.scaled_mad(detection) * .35, np.finfo(float).eps))
    intervals = np.diff(time_s[peaks])
    plausible = intervals[(intervals >= 60 / hr.HIGH_HR_BPM) & (intervals <= 60 / hr.LOW_HR_BPM)]
    valid = np.array([])
    if len(plausible) >= 2:
        median = float(np.median(plausible))
        valid = plausible[abs(plausible - median) <= max(3 * hr.scaled_mad(plausible), .10 * median)]
    peak_bpm = float(60 / np.median(valid)) if len(valid) >= 2 else None
    cv = float(np.std(valid) / np.mean(valid)) if len(valid) >= 2 else None
    fraction = float(len(valid) / len(intervals)) if len(intervals) else None
    template = hr._template_correlation(ir_signal, peaks, fs)
    reasons = []
    if strength < POLICY["minimum_autocorrelation"]:
        reasons.append("weak_autocorrelation")
    if relative < POLICY["minimum_fundamental_relative_power"]:
        reasons.append("unsupported_fundamental")
    if len(peaks) < 8:
        reasons.append(f"beats={len(peaks)}<8")
    if cv is None or cv > .20:
        reasons.append("interval_cv_unavailable_or_gt_0.20")
    if fraction is None or fraction < .70:
        reasons.append("valid_interval_fraction_unavailable_or_lt_0.70")
    if template is None or template < .60:
        reasons.append("template_correlation_unavailable_or_lt_0.60")
    if peak_bpm is None or abs(peak_bpm - auto) > hr.METHOD_AGREEMENT_BPM:
        reasons.append("autocorrelation_and_peaks_disagree")
    bpm = None if reasons else round(float(np.median([auto, peak_bpm])), 2)
    return hr.WindowResult(
        float(time_s[0]), float(time_s[-1]), "accepted" if bpm is not None else "poor_waveform_quality",
        "; ".join(reasons) if reasons else "accepted", bpm=bpm,
        spectral_bpm=spectral, autocorrelation_bpm=auto, peak_bpm=peak_bpm,
        beat_count=len(peaks), interval_cv=cv, valid_interval_fraction=fraction,
        template_correlation=template, spectral_prominence=prominence,
        peak_indices=np.asarray(global_indices)[peaks].astype(int).tolist(),
    )


def reference_hr(metadata, labels_dir):
    """Evaluation only: called after both detectors finish; conflicts are fatal."""
    labels = load_session_labels(labels_dir, str(metadata["session_id"]))
    key = tuple(str(metadata[field]) for field in ("subject_id", "session_id", "trial_id"))
    value = labels.get(key, {}).get("omron_hr")
    saved = metadata.get("cuff_hr_bpm")
    if value not in (None, "") and saved not in (None, "") and float(value) != float(saved):
        raise ValueError(f"Conflicting reference HR for {key}")
    result = value if value not in (None, "") else saved
    return float(result) if result not in (None, "") else None


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_recording(ppg_path, metadata):
    frame = pd.read_csv(ppg_path)
    required = ["sample_seq", "timestamp_ms", "red", "ir"]
    if not set(required) <= set(frame) or len(frame) < 2:
        raise ValueError("Recording needs at least two rows and the original PPG columns")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite raw samples or timing fields")
    # Only acquisition/health/motion metadata enters the detectors. Strip all
    # remaining fields, including BP/HR labels, notes and scheduled activities.
    allowed = {"firmware_motion_updates", "timing_quality", "firmware_fifo_overflow_count",
               "firmware_i2c_error_count", "imu_firmware_fifo_overflow_count", "imu_firmware_i2c_error_count"}
    health = {key: value for key, value in metadata.items() if key in allowed}
    baseline = hr.analyze_upper_arm_ppg(frame, health)
    candidate = hr.analyze_upper_arm_ppg(frame, health, window_estimator=estimate_guided)
    rows = []
    for method, result in (("baseline", baseline), ("candidate", candidate)):
        for window in result.windows:
            rows.append({"recording": ppg_path.stem.removesuffix("_ppg"), "method": method,
                         **window.as_dict()})
    summary = {"recording": ppg_path.stem.removesuffix("_ppg")}
    for method, result in (("baseline", baseline), ("candidate", candidate)):
        summary.update({f"{method}_status": result.status, f"{method}_reason": result.status_reason,
                        f"{method}_bpm": result.bpm, f"{method}_accepted_windows": result.accepted_window_count,
                        f"{method}_unique_clean_seconds": result.clean_coverage_s})
    return summary, rows, frame, baseline, candidate


def review_overlaps(frame, result, reviews, method, recording):
    """Compare exact timestamp overlaps; never turn 8-s labels into 20-s truth."""
    first = float(frame.timestamp_ms.iloc[0])
    rows = []
    for review in reviews:
        start, end = float(review["start_timestamp_ms"]), float(review["end_timestamp_ms"])
        if not start < end:
            raise ValueError("Review interval must have positive duration")
        accepted = []
        for window in result.windows:
            if window.status == "accepted":
                left, right = max(start, first + window.start_s * 1000), min(end, first + window.end_s * 1000)
                if right > left:
                    accepted.append((left, right))
        covered, cursor = 0., start
        for left, right in sorted(accepted):
            covered += max(0., right - max(left, cursor))
            cursor = max(cursor, right)
        rows.append({"recording": recording, "method": method, "window_id": review["window_id"],
                     "reviewed_label": review["reviewed_label"], "review_duration_s": (end-start)/1000,
                     "accepted_overlap_s": covered / 1000,
                     "overlap_warning": review["reviewed_label"] in {"motion_corrupted", "contact_corrupted"}
                                        and covered > 0})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "data/labels")
    parser.add_argument("--session", action="append", required=True, help="Exact session; repeat to add replay sources")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--review-recording", help="Exact recording prefix belonging to --review-csv")
    args = parser.parse_args(argv)
    if bool(args.review_csv) != bool(args.review_recording):
        parser.error("--review-csv and --review-recording must be supplied together")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    review_rows = []
    if args.review_csv:
        with args.review_csv.open(newline="", encoding="utf-8") as handle:
            review_rows = list(csv.DictReader(handle))
        ids = [row["window_id"] for row in review_rows]
        if len(ids) != len(set(ids)):
            parser.error("Duplicate review IDs")
        for row in review_rows:
            if not row["window_id"].startswith(args.review_recording + ":"):
                parser.error("Review recording does not match review window IDs")
            if row["reviewed_label"] not in {"clean", "motion_corrupted", "contact_corrupted", "uncertain"}:
                parser.error("Unknown reviewed label")
    summaries, windows, overlaps, provenance, errors = [], [], [], [], []
    for metadata_path in sorted(args.input_dir.glob("*_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("session_id") not in args.session:
            continue
        ppg_path = metadata_path.with_name(metadata_path.name.replace("_metadata.json", "_ppg.csv"))
        try:
            summary, rows, frame, baseline, candidate = compare_recording(ppg_path, metadata)
            # Labels are evaluated only now, after all waveform decisions.
            ref = reference_hr(metadata, args.labels_dir)
            summary["reference_hr_bpm"] = ref
            for method in ("baseline", "candidate"):
                estimate = summary[f"{method}_bpm"]
                summary[f"{method}_absolute_hr_error"] = abs(estimate-ref) if estimate is not None and ref is not None else None
            summaries.append(summary)
            windows.extend(rows)
            if args.review_csv and summary["recording"] == args.review_recording:
                for method, result in (("baseline", baseline), ("candidate", candidate)):
                    overlaps.extend(review_overlaps(frame, result, review_rows, method, summary["recording"]))
            label_path = args.labels_dir / f"{metadata['session_id']}_labels.csv"
            sources = [metadata_path, ppg_path] + ([label_path] if label_path.exists() else [])
            for source in sources:
                provenance.append({"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            print(f"{summary['recording']}: {baseline.status} ({baseline.accepted_window_count}) -> "
                  f"{candidate.status} ({candidate.accepted_window_count})")
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"recording": str(metadata_path), "error": str(exc)})
    if not summaries:
        parser.error(f"No recordings compared: {errors}")
    if args.review_csv and not overlaps:
        parser.error("Review recording not found among selected recordings")
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "recording_comparison.csv", summaries)
    write_csv(args.output_dir / "window_comparison.csv", windows)
    write_csv(args.output_dir / "review_overlap.csv", overlaps)
    report = {"research_only": True, "policy": POLICY, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "sessions": sorted(set(args.session)),
              "recordings_compared": len(summaries), "errors": errors, "raw_sources": provenance,
              "corrupted_review_overlap_warnings": {
                  method: sum(row["overlap_warning"] for row in overlaps if row["method"] == method)
                  for method in ("baseline", "candidate")},
              "limitations": ["Development replay, not independent validation or BP accuracy.",
                  "8-second review overlaps are warnings, not exact 20-second classification error rates.",
                  "Uncertain labels are not clean ground truth. No independent artifact adjudication.",
                  "Autocorrelation-guided peaks are not independent estimators; periodic artifacts can pass.",
                  "Reference HR is measured after acquisition, not synchronized beat-to-beat truth."]}
    report["source_code_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in (Path(__file__).resolve(), ROOT / "tools/upper_arm_hr.py")}
    if args.review_csv:
        report["review_source"] = {"path": str(args.review_csv.resolve()),
                                   "sha256": hashlib.sha256(args.review_csv.read_bytes()).hexdigest()}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved offline comparison: {args.output_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
