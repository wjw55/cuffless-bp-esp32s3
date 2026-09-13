"""Offline signal feasibility only. Never fits a model or produces BP estimates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from motion_bp import extract_recording, load_config, quality_decision
from upper_arm_hr import build_contact_masks

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"clean", "motion_corrupted", "contact_corrupted", "uncertain"}


def union_ms(intervals, start, end):
    total, cursor = 0.0, start
    for left, right in sorted(intervals):
        left, right = max(start, left), min(end, right)
        if right > max(left, cursor):
            total += right - max(left, cursor)
        cursor = max(cursor, right)
    return total


def contained(windows, start, end):
    """Exclude whole windows crossing a guarded activity boundary."""
    return windows[(windows.start_timestamp_ms >= start) & (windows.end_timestamp_ms <= end)].copy()


def review_errors(frame, prediction):
    """Window counts, not independent observations; uncertain labels are excluded."""
    result = {}
    for name, labels, error_value in (
        ("false_acceptance", {"motion_corrupted", "contact_corrupted"}, True),
        ("false_rejection", {"clean"}, False),
    ):
        subset = frame[frame.reviewed_label.isin(labels)]
        known = subset[prediction].notna()
        errors = int((subset.loc[known, prediction] == error_value).sum())
        result[name] = {"reviewed_windows": len(subset), "scored_windows": int(known.sum()),
                        "unscored_windows": int((~known).sum()), "errors": errors,
                        "rate": errors / int(known.sum()) if known.any() else None}
    return result


def attach_reviews(windows, reviews):
    required = {"window_id", "reviewed_label", "reviewer"}
    if not required <= set(reviews) or reviews.window_id.duplicated().any():
        raise ValueError("Reviews require unique window_id, reviewed_label and reviewer")
    if not set(reviews.window_id) <= set(windows.window_id):
        raise ValueError("Review contains windows from another recording")
    reviews = reviews.fillna("")
    labelled = reviews.reviewed_label != ""
    if not set(reviews.loc[labelled, "reviewed_label"]) <= LABELS:
        raise ValueError("Unknown review label")
    if reviews.loc[labelled, "reviewer"].str.strip().eq("").any():
        raise ValueError("Labelled windows need a reviewer")
    return windows.merge(reviews[list(required)], on="window_id", how="left", validate="one_to_one")


def summarize(windows, start, end, ppg=None, masks=None):
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("Invalid reporting interval")
    windows = contained(windows, start, end).sort_values("start_timestamp_ms")
    duration = end - start
    seconds = dict.fromkeys(("stationary", "mild", "moderate", "severe", "unknown"), 0.0)
    centers = (windows.start_timestamp_ms.to_numpy() + windows.end_timestamp_ms.to_numpy()) / 2
    # Each instant belongs to at most one window: split overlaps at center midpoints.
    for i, row in enumerate(windows.itertuples()):
        left = max(row.start_timestamp_ms, (centers[i-1] + centers[i]) / 2 if i else start)
        right = min(row.end_timestamp_ms, (centers[i] + centers[i+1]) / 2 if i+1 < len(centers) else end)
        seconds[row.imu_severity] += max(0, right-left) / 1000
    seconds["unknown"] += max(0, duration / 1000 - sum(seconds.values()))
    result = {"duration_s": duration / 1000, "windows": len(windows),
              "severity_percent": {k: 100*v/(duration/1000) for k, v in seconds.items()}}
    result["still_percent"] = result["severity_percent"]["stationary"]
    result["moving_percent"] = sum(result["severity_percent"][k] for k in ("mild", "moderate", "severe"))
    result["unknown_percent"] = result["severity_percent"]["unknown"]
    for column in ("ppg_candidate", "signal_eligible", "classifier_accept"):
        selected = windows[windows[column].eq(True)]
        scored = windows[windows[column].notna()]
        result[column + "_coverage_percent"] = (100 * union_ms(
            zip(selected.start_timestamp_ms, selected.end_timestamp_ms), start, end) / duration
            if len(scored) else None)
        result[column + "_scored_coverage_percent"] = 100 * union_ms(
            zip(scored.start_timestamp_ms, scored.end_timestamp_ms), start, end) / duration
        result[column + "_review_errors"] = review_errors(windows, column)
    result["median_ppg_template_correlation"] = None
    if "ppg_template_correlation" in windows and windows.ppg_template_correlation.notna().any():
        result["median_ppg_template_correlation"] = float(windows.ppg_template_correlation.median())
    if ppg is not None:
        times = ppg.timestamp_ms.to_numpy(dtype=float)
        weights = np.maximum(0, np.minimum(times[1:], end) - np.maximum(times[:-1], start))
        for name, mask in masks.items():
            result[name + "_time_percent"] = float(100 * weights[mask[:-1]].sum() / duration)
    return result


def score_classifier(windows, model_path, metadata):
    windows["classifier_accept"] = pd.Series(pd.NA, index=windows.index, dtype="boolean")
    if model_path is None:
        return {"available": False}
    import joblib
    package = joblib.load(model_path)  # Only load trusted local project packages.
    columns = package["feature_columns"]
    features = windows.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(features.to_numpy()).all(axis=1)
    if valid.any():
        model = package["model"]
        positive = list(model.classes_).index(1)
        probability = np.asarray(model.predict_proba(features.loc[valid]))[:, positive]
        if not np.isfinite(probability).all():
            raise ValueError("Non-finite classifier probabilities")
        windows.loc[valid, "classifier_accept"] = probability < package["decision_threshold"]
    return {"available": True, "scored_windows": int(valid.sum()), "shadow_only": True,
            "training_trial_overlap": str(metadata.get("trial_id")) in set(map(str, package["trial_ids"])),
            "training_participant_overlap": str(metadata.get("subject_id")) in set(map(str, package["participant_ids"])),
            "independent_validation": False}


def run(metadata_path, output_dir, activities=None, reviews=None, model_path=None):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prefix = metadata_path.name.removesuffix("_metadata.json")
    ppg = pd.read_csv(metadata_path.with_name(prefix + "_ppg.csv"))
    imu = pd.read_csv(metadata_path.with_name(prefix + "_imu.csv"))
    config = load_config(ROOT / "config/motion_bp_v1.json")
    windows = extract_recording(ppg, imu, metadata, config)
    report = {"recording": prefix, "bp_estimates": "suppressed_all_windows",
              "severity_thresholds_provisional_g": config["dynamic_rms_boundaries_g"],
              "alignment": "timestamp-based; software checks do not prove physical synchronization",
              "review_limitation": "Cue labels describe instructed activity, not waveform truth. Overlapping windows are correlated."}
    if "start_timestamp_ms" not in windows:
        report.update(timing_eligible=False, rejection_reasons=windows.iloc[0].rejection_reasons)
    else:
        report["timing_eligible"] = True
        windows["window_id"] = [f"{prefix}:w{i:04d}" for i in range(len(windows))]
        severity = ("stationary", "mild", "moderate", "severe")
        windows["imu_severity"] = [severity[int(np.searchsorted(config["dynamic_rms_boundaries_g"], x, side="right"))]
            if np.isfinite(x) else "unknown" for x in
            pd.to_numeric(windows.reindex(columns=["imu_dynamic_rms_g"]).iloc[:, 0], errors="coerce")]
        # Reuse the existing PPG rules with the motion veto disabled only in this
        # separate diagnostic column. It never enables BP or changes the baseline.
        windows["ppg_candidate"] = [quality_decision(dict(row, imu_dynamic_rms_g=0.0),
            ["signal_extraction_failed"] if row.get("rejection_reasons") else [], config)["signal_eligible"]
            for row in windows.to_dict("records")]
        report["classifier"] = score_classifier(windows, model_path, metadata)
        if reviews is not None:
            windows = attach_reviews(windows, pd.read_csv(reviews))
        else:
            windows["reviewed_label"] = ""
            windows["reviewer"] = ""
        start = float(max(ppg.timestamp_ms.iloc[0], imu.timestamp_ms.iloc[0]))
        end = float(min(ppg.timestamp_ms.iloc[-1], imu.timestamp_ms.iloc[-1]))
        step, poor, clip, threshold = build_contact_masks(
            (ppg.timestamp_ms.to_numpy()-start)/1000, ppg.ir.to_numpy())
        masks = {"contact_step_flag": step, "poor_contact_flag": poor, "clipping_flag": clip}
        report["contact_step_threshold_counts"] = threshold
        report["whole_recording"] = summarize(windows, start, end, ppg, masks)
        report["activities"] = {}
        if activities is not None:
            blocks = pd.read_csv(activities).sort_values("start_timestamp_ms")
            previous = start
            for block in blocks.itertuples():
                left, right = float(block.start_timestamp_ms), float(block.end_timestamp_ms)
                if not np.isfinite([left, right]).all() or not previous <= left < right <= end:
                    raise ValueError("Guarded activities must be ordered, disjoint and within the recording")
                if str(block.block) in report["activities"]:
                    raise ValueError("Duplicate activity name")
                report["activities"][str(block.block)] = summarize(windows, left, right, ppg, masks)
                previous = right
        report["by_imu_severity"] = {}
        for name in severity:
            subset = windows[windows.imu_severity == name]
            report["by_imu_severity"][name] = {"windows": len(subset),
                "ppg_candidate_windows": int(subset.ppg_candidate.sum()),
                "signal_eligible_windows": int(subset.signal_eligible.sum()),
                "classifier_review_errors": review_errors(subset, "classifier_accept")}
    output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_dir / "window_features.csv", index=False)
    if "window_id" in windows:
        windows[["window_id", "start_timestamp_ms", "end_timestamp_ms", "reviewed_label", "reviewer"]].to_csv(
            output_dir / "window_review.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activities", type=Path, help="Existing guarded intervals in device timestamps")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--model-package", type=Path, help="Trusted local frozen quality classifier; shadow only")
    args = parser.parse_args()
    run(args.metadata, args.output_dir, args.activities, args.reviews, args.model_package)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
