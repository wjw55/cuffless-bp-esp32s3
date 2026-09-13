#!/usr/bin/env python3
"""Source-domain PPG/continuous-BP short-window feasibility experiment.

Only the Graphene dataset's fingertip PPG and Finapres BP channels are used.
ECG, Bio-Z and pulse-transit-time signals are deliberately out of scope.
Nothing produced here is eligible for the upper-arm live viewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import find_peaks, resample_poly
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bp_core.features import MODEL_FEATURES, extract_window_features, estimate_sample_rate


TRIAL_PATTERN = re.compile(r"data_trial(?P<trial>\d+)_ppg\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class GrapheneTrial:
    participant_id: str
    day_id: str
    setup_id: str
    trial_number: int
    recording_id: str
    chronological_order: str
    ppg_path: str
    finapres_path: str | None
    discovery_status: str


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported Graphene experiment schema")
    durations = [float(value) for value in config["observation"]["durations_seconds"]]
    if not durations or any(value <= 0 for value in durations) or durations != sorted(set(durations)):
        raise ValueError("Observation durations must be unique, positive and sorted")
    if bool(config.get("deployment_eligible", True)):
        raise ValueError("This source-domain experiment must remain deployment_eligible=false")
    return config


def _identity(relative: Path, trial_number: int) -> tuple[str, str, str, str]:
    parts = relative.parts
    participant_day = parts[-3] if len(parts) >= 3 else "unknown_subject_unknown_day"
    match = re.match(r"(?P<subject>subject\d+)_(?P<day>day\d+)$", participant_day, re.IGNORECASE)
    participant = match.group("subject").lower() if match else participant_day.lower()
    day = match.group("day").lower() if match else "unknown_day"
    setup = parts[-2].lower() if len(parts) >= 2 else "unknown_setup"
    chronological = f"{day}:{setup}:trial{trial_number:03d}"
    return participant, day, setup, chronological


def discover_trials(root: Path) -> list[GrapheneTrial]:
    trials: list[GrapheneTrial] = []
    if not root.exists():
        return trials
    for ppg_path in sorted(root.rglob("data_trial*_ppg.csv")):
        match = TRIAL_PATTERN.match(ppg_path.name)
        if not match:
            continue
        trial_number = int(match.group("trial"))
        bp_path = ppg_path.with_name(ppg_path.name.replace("_ppg.csv", "_finapresBP.csv"))
        relative = ppg_path.relative_to(root)
        participant, day, setup, chronological = _identity(relative, trial_number)
        recording_id = relative.as_posix().removesuffix("_ppg.csv")
        trials.append(
            GrapheneTrial(
                participant_id=participant,
                day_id=day,
                setup_id=setup,
                trial_number=trial_number,
                recording_id=recording_id,
                chronological_order=chronological,
                ppg_path=str(ppg_path),
                finapres_path=str(bp_path) if bp_path.exists() else None,
                discovery_status="paired" if bp_path.exists() else "missing_finapres_pair",
            )
        )
    return trials


def _read_signal(path: Path, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["time", value_column]:
        raise ValueError(f"unexpected_schema:{path.name}:{','.join(map(str, frame.columns))}")
    time_s = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    if len(time_s) < 4 or len(values) != len(time_s):
        raise ValueError(f"insufficient_samples:{path.name}")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(values)):
        raise ValueError(f"non_finite_samples:{path.name}")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"non_monotonic_timestamps:{path.name}")
    return time_s, values


def _resample_ppg(time_s: np.ndarray, values: np.ndarray, target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    native_hz = estimate_sample_rate(time_s)
    if native_hz is None or native_hz <= 0:
        raise ValueError("ppg_sample_rate_unavailable")
    if native_hz <= target_hz * 1.05:
        return time_s, values
    ratio = Fraction(target_hz / native_hz).limit_denominator(1000)
    resampled = resample_poly(values, ratio.numerator, ratio.denominator)
    resampled_time = time_s[0] + np.arange(len(resampled), dtype=float) / target_hz
    valid = resampled_time <= time_s[-1] + 1e-9
    return resampled_time[valid], resampled[valid]


def _merge_coverage(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return float(sum(end - start for start, end in merged))


def extract_finapres_reference(
    time_s: np.ndarray,
    pressure: np.ndarray,
    settings: dict[str, Any],
) -> tuple[dict[str, float] | None, str]:
    fs = estimate_sample_rate(time_s)
    if fs is None or fs <= 0 or len(time_s) < 4:
        return None, "reference_timing_invalid"
    minimum_distance = max(1, int(round(fs * 60.0 / 180.0)))
    peaks, _ = find_peaks(
        pressure,
        distance=minimum_distance,
        prominence=float(settings["peak_prominence_mmHg"]),
    )
    if len(peaks) < int(settings["minimum_beats"]):
        return None, "insufficient_reference_beats"
    ibi = np.diff(time_s[peaks])
    reference_hr = float(60.0 / np.median(ibi))
    interval_cv = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else math.inf
    if not 40.0 <= reference_hr <= 180.0 or interval_cv > float(settings["maximum_interval_cv"]):
        return None, "inconsistent_reference_beats"
    systolic: list[float] = []
    diastolic: list[float] = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        peak_value = float(pressure[left])
        trough_value = float(np.min(pressure[left : right + 1]))
        if (
            float(settings["minimum_sbp_mmHg"]) <= peak_value <= float(settings["maximum_sbp_mmHg"])
            and float(settings["minimum_dbp_mmHg"]) <= trough_value <= float(settings["maximum_dbp_mmHg"])
            and peak_value - trough_value >= float(settings["minimum_pulse_pressure_mmHg"])
        ):
            systolic.append(peak_value)
            diastolic.append(trough_value)
    if len(systolic) < int(settings["minimum_beats"]) - 1:
        return None, "implausible_reference_pulses"
    result = {
        "sbp": float(np.median(systolic)),
        "dbp": float(np.median(diastolic)),
        "reference_hr": reference_hr,
        "reference_beat_count": float(len(systolic)),
    }
    return result, ""


def _precompute_ppg_segments(
    time_s: np.ndarray,
    ppg: np.ndarray,
    common_start: float,
    common_end: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    signal = config["signal"]
    quality = config["quality"]
    window = float(signal["window_seconds"])
    step = float(signal["window_step_seconds"])
    fs = estimate_sample_rate(time_s)
    if fs is None:
        raise ValueError("ppg_sample_rate_unavailable")
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(np.arange(common_start, common_end - window + 1e-9, step)):
        end = float(start + window)
        selected = (time_s >= start) & (time_s < end)
        base: dict[str, Any] = {
            "segment_index": index,
            "start_s": float(start),
            "end_s": end,
            "accepted": False,
            "rejection_reason": "",
        }
        expected = window * fs
        if selected.sum() < 0.995 * expected:
            base["rejection_reason"] = "incomplete_ppg_segment"
            rows.append(base)
            continue
        features, diagnostics = extract_window_features(
            time_s[selected], ppg[selected], None, fs, signal, quality
        )
        if features is None:
            base["rejection_reason"] = str(diagnostics.get("rejection_reason", "poor_waveform_quality"))
        else:
            base["accepted"] = True
            base.update({f"feature__{key}": value for key, value in features.items()})
        rows.append(base)
    return pd.DataFrame(rows)


def extract_trial_observations(trial: GrapheneTrial, config: dict[str, Any]) -> list[dict[str, Any]]:
    if trial.finapres_path is None:
        return []
    ppg_time, ppg = _read_signal(Path(trial.ppg_path), "PPG")
    bp_time, bp = _read_signal(Path(trial.finapres_path), "FinapresBP")
    ppg_time, ppg = _resample_ppg(ppg_time, ppg, float(config["signal"]["analysis_sample_rate_hz"]))
    common_start = max(float(ppg_time[0]), float(bp_time[0]))
    common_end = min(float(ppg_time[-1]), float(bp_time[-1]))
    if common_end <= common_start:
        raise ValueError("signals_do_not_overlap")
    segments = _precompute_ppg_segments(ppg_time, ppg, common_start, common_end, config)
    observation = config["observation"]
    step = float(observation["update_step_seconds"])
    minimum_clean = float(observation["minimum_clean_fraction"])
    minimum_segments = int(observation["minimum_accepted_segments"])
    maximum_shortfall = float(observation.get("maximum_duration_shortfall_seconds", 0.0))
    rows: list[dict[str, Any]] = []
    available = common_end - common_start
    for requested_duration in map(float, observation["durations_seconds"]):
        actual_duration = requested_duration
        if available < requested_duration:
            if requested_duration - available <= maximum_shortfall:
                actual_duration = available
            else:
                continue
        update_count = int(math.floor(max(0.0, available - actual_duration) / step)) + 1
        ends = np.sort(common_end - np.arange(update_count, dtype=float) * step)
        for ordinal, end in enumerate(ends):
            start = float(end - actual_duration)
            inside = segments[(segments["start_s"] >= start - 1e-9) & (segments["end_s"] <= end + 1e-9)]
            accepted = inside[inside["accepted"] == True]  # noqa: E712
            coverage = _merge_coverage(zip(accepted.get("start_s", []), accepted.get("end_s", [])))
            ppg_reasons: list[str] = []
            if len(accepted) < minimum_segments:
                ppg_reasons.append("insufficient_accepted_ppg_segments")
            if coverage / actual_duration < minimum_clean:
                ppg_reasons.append("insufficient_clean_ppg_coverage")
            base: dict[str, Any] = {
                "participant_id": trial.participant_id,
                "day_id": trial.day_id,
                "setup_id": trial.setup_id,
                "recording_id": trial.recording_id,
                "chronological_order": trial.chronological_order,
                "observation_id": f"{trial.recording_id}:d{requested_duration:g}:u{ordinal:04d}",
                "requested_duration_s": requested_duration,
                "actual_duration_s": actual_duration,
                "start_s": float(start),
                "end_s": end,
                "candidate_segment_count": len(inside),
                "accepted_segment_count": len(accepted),
                "unique_clean_coverage_s": coverage,
                "clean_fraction": coverage / actual_duration,
                "signal_usable": not ppg_reasons,
                "signal_rejection_reason": ";".join(ppg_reasons),
                "model_evaluation_row": ordinal % max(1, int(math.ceil(requested_duration / step))) == 0,
                "matched_endpoint_row": bool(np.isclose(end, common_end, rtol=0.0, atol=1e-8)),
            }
            for feature in MODEL_FEATURES:
                column = f"feature__{feature}"
                values = pd.to_numeric(accepted.get(column, pd.Series(dtype=float)), errors="coerce")
                base[f"median__{column}"] = float(values.median()) if values.notna().any() else math.nan
                base[f"iqr__{column}"] = (
                    float(values.quantile(0.75) - values.quantile(0.25)) if values.notna().any() else math.nan
                )
            label_start = max(start, end - float(config["reference"]["label_window_seconds"]))
            bp_selected = (bp_time >= label_start) & (bp_time < end)
            reference, reference_reason = extract_finapres_reference(
                bp_time[bp_selected], bp[bp_selected], config["reference"]
            )
            base["reference_usable"] = reference is not None
            base["reference_rejection_reason"] = reference_reason
            if reference:
                base.update(reference)
            else:
                base.update({"sbp": math.nan, "dbp": math.nan, "reference_hr": math.nan, "reference_beat_count": 0})
            base["accepted"] = bool(base["signal_usable"] and base["reference_usable"])
            rows.append(base)
    return rows


def build_personalized_examples(observations: pd.DataFrame, duration: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    evaluation_column = "matched_endpoint_row" if "matched_endpoint_row" in observations else "model_evaluation_row"
    eligible = observations[
        (observations["accepted"] == True)  # noqa: E712
        & (observations[evaluation_column] == True)  # noqa: E712
        & np.isclose(observations["requested_duration_s"].astype(float), duration)
    ].copy()
    feature_columns = [column for column in eligible.columns if column.startswith(("median__feature__", "iqr__feature__"))]
    examples: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for participant, group in eligible.groupby("participant_id", sort=True):
        group = group.sort_values(["chronological_order", "start_s", "observation_id"])
        recording_order = list(dict.fromkeys(group["recording_id"].tolist()))
        if len(recording_order) < 2:
            continue
        calibration_recording = recording_order[0]
        calibration_group = group[group["recording_id"] == calibration_recording]
        calibration_features = calibration_group[feature_columns].median(numeric_only=True)
        calibration_sbp = float(calibration_group["sbp"].median())
        calibration_dbp = float(calibration_group["dbp"].median())
        calibration_rows.append(
            {
                "participant_id": participant,
                "duration_s": duration,
                "calibration_recording_id": calibration_recording,
                "calibration_sbp": calibration_sbp,
                "calibration_dbp": calibration_dbp,
            }
        )
        for _, current in group[group["recording_id"] != calibration_recording].iterrows():
            row: dict[str, Any] = {
                "participant_id": participant,
                "recording_id": current["recording_id"],
                "observation_id": current["observation_id"],
                "duration_s": duration,
                "sbp": float(current["sbp"]),
                "dbp": float(current["dbp"]),
                "calibration_sbp": calibration_sbp,
                "calibration_dbp": calibration_dbp,
                "delta_sbp": float(current["sbp"]) - calibration_sbp,
                "delta_dbp": float(current["dbp"]) - calibration_dbp,
            }
            for column in feature_columns:
                value = float(current[column]) if pd.notna(current[column]) else math.nan
                calibration_value = float(calibration_features.get(column, math.nan))
                row[f"current__{column}"] = value
                row[f"change__{column}"] = value - calibration_value
            examples.append(row)
    return pd.DataFrame(examples), calibration_rows


def evaluate_duration(
    observations: pd.DataFrame,
    duration: float,
    config: dict[str, Any],
    matched_recordings: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluation_observations = observations
    if matched_recordings is not None:
        evaluation_observations = observations[observations["recording_id"].isin(matched_recordings)]
    examples, calibrations = build_personalized_examples(evaluation_observations, duration)
    participants = sorted(examples.get("participant_id", pd.Series(dtype=str)).unique())
    minimum_participants = int(config["model"]["minimum_participants"])
    minimum_examples = int(config["model"]["minimum_training_examples"])
    candidate_count = int(np.isclose(observations["requested_duration_s"].astype(float), duration).sum())
    accepted_count = int(
        (
            np.isclose(observations["requested_duration_s"].astype(float), duration)
            & (observations["accepted"] == True)  # noqa: E712
        ).sum()
    )
    summary: dict[str, Any] = {
        "duration_s": duration,
        "candidate_observations": candidate_count,
        "accepted_observations": accepted_count,
        "accepted_observation_fraction": accepted_count / candidate_count if candidate_count else 0.0,
        "participant_count": len(participants),
        "personalized_example_count": len(examples),
        "matched_recording_count": len(matched_recordings) if matched_recordings is not None else None,
        "time_to_first_estimate_s": duration,
        "overlapping_updates_are_not_independent": True,
        "evaluation_status": "available",
    }
    duration_rows = observations[np.isclose(observations["requested_duration_s"].astype(float), duration)]
    if not duration_rows.empty:
        summary["minimum_actual_duration_s"] = float(duration_rows["actual_duration_s"].min())
        summary["maximum_actual_duration_s"] = float(duration_rows["actual_duration_s"].max())
    if len(participants) < minimum_participants or len(examples) < minimum_examples:
        summary["evaluation_status"] = "insufficient_participants_or_examples"
        return [], calibrations, summary

    feature_columns = [column for column in examples.columns if column.startswith(("current__", "change__"))]
    predictions: list[dict[str, Any]] = []
    for held_out in participants:
        train = examples[examples["participant_id"] != held_out]
        test = examples[examples["participant_id"] == held_out]
        if len(train) < minimum_examples or test.empty:
            continue
        for target in ("sbp", "dbp"):
            delta_target = f"delta_{target}"
            calibration_target = f"calibration_{target}"
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=float(config["model"]["ridge_alpha"]))),
                ]
            )
            model.fit(train[feature_columns], train[delta_target])
            predicted_delta = model.predict(test[feature_columns])
            predicted = test[calibration_target].to_numpy(dtype=float) + predicted_delta
            for (_, source), estimate, delta in zip(test.iterrows(), predicted, predicted_delta):
                predictions.append(
                    {
                        "duration_s": duration,
                        "held_out_participant": held_out,
                        "recording_id": source["recording_id"],
                        "observation_id": source["observation_id"],
                        "target": target.upper(),
                        "true_mmHg": float(source[target]),
                        "predicted_mmHg": float(estimate),
                        "predicted_change_mmHg": float(delta),
                        "zero_change_mmHg": float(source[calibration_target]),
                    }
                )
    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        summary["evaluation_status"] = "insufficient_cross_participant_training_data"
        return predictions, calibrations, summary
    for target in ("SBP", "DBP"):
        subset = prediction_frame[prediction_frame["target"] == target]
        truth = subset["true_mmHg"].to_numpy(dtype=float)
        estimate = subset["predicted_mmHg"].to_numpy(dtype=float)
        baseline = subset["zero_change_mmHg"].to_numpy(dtype=float)
        error = estimate - truth
        summary[target.lower()] = {
            "model_mae_mmHg": float(mean_absolute_error(truth, estimate)),
            "model_rmse_mmHg": float(mean_squared_error(truth, estimate) ** 0.5),
            "model_bias_mmHg": float(np.mean(error)),
            "zero_change_mae_mmHg": float(mean_absolute_error(truth, baseline)),
            "beats_zero_change": bool(mean_absolute_error(truth, estimate) < mean_absolute_error(truth, baseline)),
            "prediction_count": len(subset),
        }
    return predictions, calibrations, summary


def _resolve_dataset_root(config_path: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    project_root = config_path.resolve().parent.parent
    return project_root / path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(config_path: Path, output_dir: Path, max_recordings: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config_path, str(config["dataset"]["root"]))
    trials = discover_trials(root)
    if max_recordings is not None:
        trials = trials[:max_recordings]
    pd.DataFrame([asdict(trial) for trial in trials]).to_csv(output_dir / "dataset_manifest.csv", index=False)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for trial in trials:
        if trial.discovery_status != "paired":
            continue
        try:
            rows.extend(extract_trial_observations(trial, config))
        except Exception as exc:
            errors.append({"recording_id": trial.recording_id, "error": str(exc)})
    observations = pd.DataFrame(rows)
    observations.to_csv(output_dir / "observation_features.csv", index=False)
    pd.DataFrame(errors, columns=["recording_id", "error"]).to_csv(output_dir / "recording_errors.csv", index=False)

    all_predictions: list[dict[str, Any]] = []
    all_calibrations: list[dict[str, Any]] = []
    duration_metrics: list[dict[str, Any]] = []
    if not observations.empty:
        endpoint_rows = observations[observations["matched_endpoint_row"] == True]  # noqa: E712
        accepted_recordings_by_duration = []
        for duration in map(float, config["observation"]["durations_seconds"]):
            accepted_recordings_by_duration.append(
                set(
                    endpoint_rows[
                        np.isclose(endpoint_rows["requested_duration_s"].astype(float), duration)
                        & (endpoint_rows["accepted"] == True)  # noqa: E712
                    ]["recording_id"]
                )
            )
        matched_recordings = (
            set.intersection(*accepted_recordings_by_duration) if accepted_recordings_by_duration else set()
        )
        for duration in map(float, config["observation"]["durations_seconds"]):
            predictions, calibrations, metrics = evaluate_duration(
                observations, duration, config, matched_recordings=matched_recordings
            )
            all_predictions.extend(predictions)
            all_calibrations.extend(calibrations)
            duration_metrics.append(metrics)
        longest = max(duration_metrics, key=lambda item: float(item["duration_s"]))
        longest_coverage = float(longest["accepted_observation_fraction"])
        for metrics in duration_metrics:
            metrics["coverage_ratio_vs_longest"] = (
                float(metrics["accepted_observation_fraction"]) / longest_coverage
                if longest_coverage > 0
                else None
            )
            for target in ("sbp", "dbp"):
                if target in metrics and target in longest:
                    metrics[target]["model_mae_difference_vs_longest_mmHg"] = (
                        float(metrics[target]["model_mae_mmHg"])
                        - float(longest[target]["model_mae_mmHg"])
                    )
    pd.DataFrame(all_predictions).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(all_calibrations).to_csv(output_dir / "calibrations.csv", index=False)
    pd.DataFrame(duration_metrics).to_csv(output_dir / "duration_metrics.csv", index=False)

    report = {
        "experiment_id": config["experiment_id"],
        "research_only": True,
        "deployment_eligible": False,
        "source_domain": "fingertip_ppg_with_finapres",
        "target_domain": "upper_arm_ppg",
        "trial_count": len(trials),
        "paired_trial_count": sum(trial.discovery_status == "paired" for trial in trials),
        "observation_count": len(observations),
        "recording_error_count": len(errors),
        "matched_endpoint_recording_count": (
            duration_metrics[0].get("matched_recording_count") if duration_metrics else 0
        ),
        "durations": duration_metrics,
        "interpretation": (
            "This experiment measures source-domain short-window feasibility only. "
            "It cannot validate upper-arm, live, recovery, or motion-compensated BP."
        ),
    }
    _write_json(output_dir / "report.json", report)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": _config_hash(config_path),
        "dataset_root": str(root.resolve()),
        "dataset_source_url": config["dataset"]["source_url"],
        "dataset_version": config["dataset"]["version"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "model_exported": False,
        "viewer_modified": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return report


def audit_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = _resolve_dataset_root(config_path, str(config["dataset"]["root"]))
    trials = discover_trials(root)
    pd.DataFrame([asdict(trial) for trial in trials]).to_csv(output_dir / "dataset_manifest.csv", index=False)
    result = {
        "dataset_root": str(root.resolve()),
        "trial_count": len(trials),
        "paired_trial_count": sum(trial.discovery_status == "paired" for trial in trials),
        "missing_pair_count": sum(trial.discovery_status != "paired" for trial in trials),
        "participant_count": len({trial.participant_id for trial in trials}),
        "research_only": True,
        "deployment_eligible": False,
    }
    _write_json(output_dir / "audit.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "run":
            child.add_argument("--max-recordings", type=int)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit_dataset(args.config, args.output_dir)
            print(f"Graphene audit: {args.output_dir}")
            print(
                f"Trials={result['trial_count']}, paired={result['paired_trial_count']}, "
                f"participants={result['participant_count']}"
            )
        else:
            result = run_experiment(args.config, args.output_dir, args.max_recordings)
            print(f"Graphene PPG/BP experiment: {args.output_dir}")
            print(
                f"Trials={result['trial_count']}, observations={result['observation_count']}, "
                f"errors={result['recording_error_count']}"
            )
            for item in result["durations"]:
                print(
                    f"- {item['duration_s']:g}s: coverage={item['accepted_observation_fraction']:.1%}, "
                    f"evaluation={item['evaluation_status']}"
                )
            print("Research-only fingertip source-domain result; no viewer model was created.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
