"""Offline-only motion/BP primitives. Never imported by the live viewer.

Quality functions accept sensor information only. Reference admission and BP
evaluation are separate operations. All defaults are research, not deployment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from bp_core.features import MODEL_FEATURES, SENSOR_HEALTH_COUNTERS, extract_window_features
from motion_quality import (_causal_imu, classify_motion_intensity,
                            extract_synchronized_window_features,
                            load_config as load_motion_quality_config,
                            motion_intensity_settings)

ROOT = Path(__file__).resolve().parents[1]
PPG_COLUMNS = ["sample_seq", "timestamp_ms", "red", "ir"]
IMU_COLUMNS = ["imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"]
STATES = ("Stable estimate", "Motion-compensated estimate", "Low confidence",
          "Motion too severe", "Poor contact", "Insufficient data")
IMU_FEATURES = ["imu_accel_rms_g", "imu_dynamic_rms_g", "imu_dynamic_max_g", "imu_dynamic_p95_g",
    "imu_activity_mean_g", "imu_activity_max_g", "imu_activity_above_threshold_fraction",
    "imu_axis_x_std_g", "imu_axis_y_std_g", "imu_axis_z_std_g", "imu_signal_magnitude_area_g",
    "imu_jerk_rms_g_per_s", "imu_jerk_p95_g_per_s", "imu_dominant_frequency_hz",
    "imu_spectral_entropy", "imu_orientation_change_deg", "imu_movement_duration_s",
    "imu_longest_motion_bout_s"]
CROSS_FEATURES = ["cross_modal_max_lag_correlation", "cross_modal_spectral_overlap"]
MOTION_BAND_FEATURES = ["imu_band_mild", "imu_band_moderate", "imu_band_severe"]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["signal_config"] = load_motion_quality_config(path.parent / config["quality_config"])
    intensity = motion_intensity_settings(config["signal_config"])
    if intensity is None or intensity["labels"] != ["stationary", "mild", "moderate", "severe"]:
        raise ValueError("Motion BP requires the shared stationary/mild/moderate/severe bands")
    if not np.isclose(float(config["movement_threshold_g"]), intensity["boundaries_g"][0]):
        raise ValueError("movement_threshold_g must match the shared mild-motion boundary")
    if config.get("deployment_eligible") is not False:
        raise ValueError("This offline pipeline cannot enable deployment")
    for key in ("maximum_gap_ms", "maximum_timestamp_lag_ms", "ridge_alpha", "minimum_uncertainty_groups"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid positive setting: " + key)
    if not 0 < config["interval_coverage"] < 1:
        raise ValueError("Interval coverage must be between zero and one")
    return config


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stream_reasons(frame: pd.DataFrame, columns: list[str], config: dict) -> list[str]:
    if not set(columns) <= set(frame.columns) or len(frame) < 2:
        return ["missing_or_empty_stream"]
    try:
        values = frame[columns].to_numpy(dtype=float)
    except (ValueError, TypeError):
        return ["nonnumeric_stream"]
    if not np.isfinite(values).all():
        return ["nonfinite_stream"]
    reasons = []
    if np.any(np.diff(values[:, 0]) != 1):
        reasons.append("sequence_discontinuity")
    delta = np.diff(values[:, 1])
    if np.any(delta <= 0):
        reasons.append("nonmonotonic_time")
    if np.any(delta > config["maximum_gap_ms"]):
        reasons.append("timestamp_gap")
    return reasons


def timing_reasons(metadata: dict, config: dict) -> list[str]:
    reasons = []
    if metadata.get('firmware_clock_rejected_observation_count', 0):
        reasons.append('ppg_clock_observation_rejected')
    for key in SENSOR_HEALTH_COUNTERS:
        try:
            value = float(metadata[key])
            if not np.isfinite(value) or value != 0:
                reasons.append("sensor_health:" + key)
        except (KeyError, ValueError, TypeError):
            reasons.append("unknown_sensor_health:" + key)
    warnings = metadata.get("firmware_warning_events", [])
    if not isinstance(warnings, list):
        reasons.append('invalid_warning_evidence')
        warnings = []
    for warning in warnings:
        if not isinstance(warning, dict):
            reasons.append('invalid_warning_evidence')
            continue
        if warning.get('event') == 'ppg_clock_observation_rejected':
            reasons.append('ppg_clock_observation_rejected')
        if warning.get("event") == "timestamp_lag":
            try:
                lag = float(warning["lag_us"]) / 1000
                if not np.isfinite(lag) or abs(lag) > config["maximum_timestamp_lag_ms"]:
                    reasons.append("timestamp_lag_uncertain")
            except (KeyError, ValueError, TypeError):
                reasons.append("timestamp_lag_unknown")
    return sorted(set(reasons))


def audit(raw_dir: Path, config: dict, identities: dict | None = None) -> pd.DataFrame:
    """Record file evidence, never infer a participant alias or a BP label."""
    rows = []
    identities = identities or {}
    for path in sorted(raw_dir.glob("*_metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        prefix = path.name.removesuffix("_metadata.json")
        row = {"recording_id": prefix, "metadata_path": str(path.resolve()),
               "metadata_sha256": sha256(path), "subject_id": metadata.get("subject_id", ""),
               "participant_id": identities.get(metadata.get("subject_id"), ""),
               "session_id": metadata.get("session_id", ""),
               "recording_start_time": metadata.get("recording_start_time", ""),
               "firmware_git_commit": metadata.get("firmware_git_commit", ""),
               "firmware_note": metadata.get("notes", "")}
        reasons = timing_reasons(metadata, config)
        for stream, columns in (("ppg", PPG_COLUMNS), ("imu", IMU_COLUMNS)):
            source = raw_dir / (prefix + "_" + stream + ".csv")
            row[stream + "_path"] = str(source.resolve())
            row[stream + "_sha256"] = sha256(source) if source.exists() else ""
            try:
                frame = pd.read_csv(source)
                row[stream + "_rows"] = len(frame)
                issues = stream_reasons(frame, columns, config)
            except (OSError, ValueError, pd.errors.ParserError):
                row[stream + "_rows"] = 0
                issues = ["unreadable_stream"]
            reasons.extend(stream + ":" + issue for issue in issues)
        paired = row["ppg_rows"] > 1 and row["imu_rows"] > 1
        row["available_modalities"] = "paired" if paired else "imu_only" if row["imu_rows"] > 1 else "ppg_only" if row["ppg_rows"] > 1 else "none"
        row["timing_reasons"] = ";".join(sorted(set(reasons)))
        row["paired_feature_eligible"] = paired and not reasons
        row["identity_resolved"] = bool(row["participant_id"])
        row["motion_bp_label_status"] = "no_admitted_continuous_reference"
        rows.append(row)
    return pd.DataFrame(rows)


def movement_duration(time_ms: np.ndarray, dynamic_g: np.ndarray, start: float,
                      end: float, threshold: float) -> dict:
    """Integrate sample-supported intervals, clipped to the window; no row counts."""
    total = longest = run = 0.0
    for i in range(len(time_ms) - 1):
        duration = max(0.0, min(end, time_ms[i + 1]) - max(start, time_ms[i])) / 1000
        if duration and dynamic_g[i] > threshold:
            total += duration
            run += duration
            longest = max(longest, run)
        else:
            run = 0.0
    return {"imu_movement_duration_s": total, "imu_longest_motion_bout_s": longest}


def quality_decision(features: dict | None, reasons: list[str], config: dict) -> dict:
    """Signal information only: never accept labels, residuals or predictions."""
    if reasons or features is None:
        return {"severity": "unknown", "recoverability": "uncertain", "status": STATES[5], "signal_eligible": False}
    def number(key):
        try:
            value = float(features.get(key, np.nan))
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError):
            return np.nan
    intensity = motion_intensity_settings(config["signal_config"])
    activity = number(intensity["source_feature"])
    if not np.isfinite(activity):
        return quality_decision(None, ["missing_motion"], config)
    severity = classify_motion_intensity(activity, config["signal_config"])
    result = {"severity": severity, "recoverability": "uncertain", "status": STATES[2], "signal_eligible": False}
    if number("ppg_dc_median") < config["minimum_contact_counts"] or number("ppg_clipping_fraction") > config["maximum_clipping_fraction"]:
        return dict(result, recoverability="unrecoverable", status=STATES[4])
    if severity == "severe":
        return dict(result, recoverability="unrecoverable", status=STATES[3])
    checks = [number("ppg_template_correlation") >= config["minimum_template_correlation"],
              number("ppg_ibi_cv") <= config["maximum_ibi_cv"],
              number("ppg_valid_beat_count") >= config["minimum_valid_beats"],
              np.isfinite(number('ppg_dc_median')), np.isfinite(number('ppg_clipping_fraction')),
              number('ppg_morphology_accepted') == 1.0]
    if all(checks):
        # Eligibility is a research signal decision, not permission to display BP.
        return dict(result, recoverability="preserved" if severity == "stationary" else "potentially_recoverable", signal_eligible=True)
    return result


def extract_recording(ppg: pd.DataFrame, imu: pd.DataFrame, metadata: dict, config: dict) -> pd.DataFrame:
    reasons = timing_reasons(metadata, config)
    reasons += ["ppg:" + x for x in stream_reasons(ppg, PPG_COLUMNS, config)]
    reasons += ["imu:" + x for x in stream_reasons(imu, IMU_COLUMNS, config)]
    if reasons:
        return pd.DataFrame([{**quality_decision(None, reasons, config), "rejection_reasons": ";".join(reasons)}])
    settings = config["signal_config"]
    start = max(ppg.timestamp_ms.iloc[0], imu.timestamp_ms.iloc[0])
    end = min(ppg.timestamp_ms.iloc[-1], imu.timestamp_ms.iloc[-1])
    width = settings["window_seconds"] * 1000
    step = settings["window_step_seconds"] * 1000
    derived = _causal_imu(imu, settings)
    rows = []
    for left in np.arange(start, end - width + 1e-8, step):
        right = left + width
        features, diagnostics = extract_synchronized_window_features(ppg, imu, settings, left, right)
        rejection = diagnostics["rejection_reasons"]
        row = {"start_timestamp_ms": left, "end_timestamp_ms": right, "rejection_reasons": ";".join(rejection)}
        if features is not None:
            features.update(movement_duration(imu.timestamp_ms.to_numpy(), derived["dynamic"], left, right, config["movement_threshold_g"]))
            window = ppg[(ppg.timestamp_ms >= left) & (ppg.timestamp_ms < right)]
            times = window.timestamp_ms.to_numpy() / 1000
            morphology, _ = extract_window_features(times - times[0], window.ir.to_numpy(), window.red.to_numpy(),
                1 / np.median(np.diff(times)), settings["ppg"], settings["ppg"])
            if morphology:
                features.update({"morph__" + key: value for key, value in morphology.items() if key in MODEL_FEATURES})
            row.update(features)
        row.update(quality_decision(features, rejection, config))
        row["bp_status"] = "not_evaluable_without_admitted_reference"
        rows.append(row)
    if not rows:
        rows.append({**quality_decision(None, ["short_recording"], config), "rejection_reasons": "short_recording"})
    return pd.DataFrame(rows)


def validate_reference(frame: pd.DataFrame) -> None:
    """Motion training accepts only independently verified continuous references.

    Cuff values must never be propagated through movement sequences. The table is
    a derived, reviewed reference artifact, not a modification of raw CSVs.
    """
    required = {"reference_kind", "reference_valid", "reference_start_ms", "reference_end_ms",
                "start_timestamp_ms", "end_timestamp_ms", "true_sbp", "true_dbp",
                "reference_source_sha256", "reference_alignment_verified"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("Missing continuous-reference evidence")
    for _, row in frame.iterrows():
        numbers = np.array([row[x] for x in ["reference_start_ms", "reference_end_ms", "start_timestamp_ms", "end_timestamp_ms", "true_sbp", "true_dbp"]], dtype=float)
        if (row.reference_kind != "continuous" or row.reference_valid != True
                or row.reference_alignment_verified != True or not np.isfinite(numbers).all()
                or not row.reference_start_ms <= row.start_timestamp_ms < row.end_timestamp_ms <= row.reference_end_ms
                or not 0 < row.true_dbp < row.true_sbp
                or len(str(row.reference_source_sha256)) != 64):
            raise ValueError("Unusable or non-continuous BP reference")


def assert_split(train: pd.DataFrame, test: pd.DataFrame, *, held_out_participant: bool = False) -> None:
    if train.empty or test.empty:
        raise ValueError("Empty split")
    for column in ("recording_id", "recording_sha256", "cuff_occasion_id"):
        if column not in train or column not in test or train[column].isna().any() or test[column].isna().any() or (train[column] == "").any() or (test[column] == "").any():
            raise ValueError("Missing split provenance: " + column)
        if set(train[column]) & set(test[column]):
            raise ValueError("Split leakage: " + column)
    if held_out_participant and set(train.participant_id) & set(test.participant_id):
        raise ValueError("Participant leakage")
    for frame in (train, test):
        start = pd.to_datetime(frame.support_start_utc, utc=True)
        end = pd.to_datetime(frame.support_end_utc, utc=True)
        if start.isna().any() or end.isna().any() or not (start < end).all():
            raise ValueError("Invalid signal support")
    # Absolute support intervals include filtering and IMU history, not just windows.
    for participant in set(train.participant_id) & set(test.participant_id):
        a = train[train.participant_id == participant]
        b = test[test.participant_id == participant]
        if pd.to_datetime(a.support_end_utc, utc=True).max() >= pd.to_datetime(b.support_start_utc, utc=True).min():
            raise ValueError("Nonchronological or overlapping signal support")


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    # Explicit allowlist of namespaces; identity, cue and reference columns cannot enter models.
    morphology = ["morph__" + x for x in MODEL_FEATURES]
    base = [x for x in morphology if x in frame]
    base += ["calibration__" + x for x in MODEL_FEATURES if "calibration__" + x in frame]
    base += ["baseline_sbp", "baseline_dbp"]
    if not any(x.startswith("morph__") for x in base) or not any(x.startswith("calibration__") for x in base):
        raise ValueError("Current and personal calibration morphology are required")
    intensity_names = ["imu_activity_mean_g", "imu_dynamic_rms_g", "imu_movement_duration_s"] + MOTION_BAND_FEATURES
    intensity = [x for x in intensity_names if x in frame]
    full = [x for x in IMU_FEATURES + CROSS_FEATURES + MOTION_BAND_FEATURES if x in frame]
    if len(intensity) != len(intensity_names) or not full:
        raise ValueError("Missing IMU features for ablation")
    return {"ppg_only": base, "ppg_intensity": base + intensity, "ppg_full_imu": base + full}


def attach_motion_band_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Derive model inputs from the shared signal-only Stage 1 motion bands."""
    output = frame.copy()
    settings = motion_intensity_settings(config["signal_config"])
    source = settings["source_feature"]
    if source not in output:
        raise ValueError("Missing shared motion-intensity source feature: " + source)
    output["severity"] = output[source].map(
        lambda value: classify_motion_intensity(value, config["signal_config"])
    )
    for band in ("mild", "moderate", "severe"):
        output["imu_band_" + band] = (output["severity"] == band).astype(float)
    return output


def chronological_split(frame: pd.DataFrame, minimum_groups: int = 10) -> dict[str, pd.DataFrame]:
    """Lock complete occasions in time order: 60% fit, 20% uncertainty, 20% test.

    This is deliberately separate from the unchanged stationary 70/30 split.
    Reject shared recordings crossing an occasion boundary rather than splitting
    their overlapping windows. Split membership is never based on BP values.
    """
    if frame.participant_id.nunique() != 1:
        raise ValueError("Chronological development requires one resolved participant")
    ordered = frame.assign(_time=pd.to_datetime(frame.support_start_utc, utc=True)).groupby('cuff_occasion_id')._time.min().sort_values()
    count = len(ordered)
    held = max(minimum_groups, int(np.ceil(count * .2)))
    if count - 2 * held < minimum_groups:
        raise ValueError("Need at least three disjoint sets of independent occasions")
    groups = [ordered.index[:-2*held], ordered.index[-2*held:-held], ordered.index[-held:]]
    result = {name: frame[frame.cuff_occasion_id.isin(ids)].copy() for name, ids in zip(('train', 'uncertainty', 'test'), groups)}
    assert_split(result['train'], result['uncertainty'])
    assert_split(result['train'], result['test'])
    assert_split(result['uncertainty'], result['test'])
    return result


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if not len(actual):
        return {"count": 0, "mae": None, "rmse": None, "bias": None, "maximum_absolute_error": None}
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite evaluation values")
    error = predicted - actual
    return {"count": len(error), "mae": float(np.mean(abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)), "maximum_absolute_error": float(np.max(abs(error)))}


def interval_union(intervals) -> float:
    total, last = 0.0, -np.inf
    for start, end in sorted(intervals):
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError("Invalid coverage interval")
        total += max(0, end - max(start, last))
        last = max(last, end)
    return total


def coverage(frame: pd.DataFrame, accepted: np.ndarray) -> float:
    total = usable = 0.0
    for _, group in frame.assign(_accepted=np.asarray(accepted, bool)).groupby("recording_id"):
        pairs = list(zip(group.start_timestamp_ms, group.end_timestamp_ms))
        total += interval_union(pairs)
        usable += interval_union([pair for pair, yes in zip(pairs, group._accepted) if yes])
    return usable / total if total else 0.0


def accepted_unique_seconds(frame: pd.DataFrame, accepted: np.ndarray) -> float:
    """Count accepted signal time once even when adjacent windows overlap."""
    total_ms = 0.0
    for _, group in frame.assign(_accepted=np.asarray(accepted, bool)).groupby("recording_id"):
        intervals = list(zip(group.start_timestamp_ms, group.end_timestamp_ms))
        total_ms += interval_union([
            interval for interval, yes in zip(intervals, group._accepted) if yes
        ]) if len(group) else 0.0
    return total_ms / 1000.0


def ablation_comparison(outputs: list[pd.DataFrame]) -> dict:
    """Compare IMU models with PPG-only on identical accepted test windows."""
    selected = {
        frame.model.iloc[0]: frame.reset_index(drop=True)
        for frame in outputs
        if frame.model.iloc[0] in {"ppg_only", "ppg_intensity", "ppg_full_imu"}
    }
    if set(selected) != {"ppg_only", "ppg_intensity", "ppg_full_imu"}:
        raise ValueError("Missing model output for motion ablation comparison")
    base = selected["ppg_only"]
    keys = ["participant_id", "recording_id", "cuff_occasion_id", "start_timestamp_ms", "end_timestamp_ms"]
    report = {}
    for candidate_name in ("ppg_intensity", "ppg_full_imu"):
        candidate = selected[candidate_name]
        if not base[keys].equals(candidate[keys]):
            raise ValueError("Model outputs are not aligned to identical test windows")
        bands = {}
        for band in ("all", "stationary", "motion", "mild", "moderate", "severe"):
            if band == "all":
                band_mask = np.ones(len(base), dtype=bool)
            elif band == "motion":
                band_mask = base.severity.isin(["mild", "moderate", "severe"]).to_numpy()
            else:
                band_mask = (base.severity == band).to_numpy()
            common = band_mask & base.accepted.to_numpy(bool) & candidate.accepted.to_numpy(bool)
            base_band = base.loc[band_mask]
            candidate_band = candidate.loc[band_mask]
            common_base = base.loc[common]
            common_candidate = candidate.loc[common]
            targets = {}
            improvements = []
            for target in ("sbp", "dbp"):
                base_metrics = metrics(common_base["true_" + target], common_base["predicted_" + target])
                candidate_metrics = metrics(common_candidate["true_" + target], common_candidate["predicted_" + target])
                delta = (candidate_metrics["mae"] - base_metrics["mae"]
                         if candidate_metrics["mae"] is not None and base_metrics["mae"] is not None else None)
                targets[target] = {
                    "ppg_only": base_metrics,
                    "candidate": candidate_metrics,
                    "mae_delta_candidate_minus_ppg_only": delta,
                }
                improvements.append(delta is not None and delta < 0)
            base_coverage = coverage(base_band, base_band.accepted.to_numpy()) if len(base_band) else 0.0
            candidate_coverage = coverage(candidate_band, candidate_band.accepted.to_numpy()) if len(candidate_band) else 0.0
            bands[band] = {
                "test_window_count": int(band_mask.sum()),
                "common_accepted_window_count": int(common.sum()),
                "common_accepted_unique_seconds": accepted_unique_seconds(base, common),
                "ppg_only_accepted_coverage": base_coverage,
                "candidate_accepted_coverage": candidate_coverage,
                "coverage_delta_candidate_minus_ppg_only": candidate_coverage - base_coverage,
                "targets": targets,
                "improves_both_targets_on_common_windows": bool(common.any() and all(improvements)),
            }
        report[candidate_name + "_vs_ppg_only"] = bands
    return report


def prediction_report(output: pd.DataFrame, maximum_width: float) -> dict:
    report = {}
    for severity in ["all", "stationary", "motion", "mild", "moderate", "severe"]:
        subset = output if severity == "all" else output[output.severity.isin(['mild', 'moderate', 'severe'])] if severity == 'motion' else output[output.severity == severity]
        accepted = subset[subset.accepted]
        report[severity] = {"coverage": coverage(subset, subset.accepted.to_numpy()),
            "window_count": len(subset), "participant_count": subset.participant_id.nunique(),
            "occasion_count": subset.cuff_occasion_id.nunique(),
            **{target: metrics(accepted["true_" + target], accepted["predicted_" + target]) for target in ("sbp", "dbp")}}
    for label in ('severely_corrupted', 'contact_corrupted'):
        reviewed = output[output.reviewed_quality == label] if 'reviewed_quality' in output else output.iloc[:0]
        report['false_acceptance_' + label] = {"reviewed_windows": len(reviewed),
            "accepted_windows": int(reviewed.accepted.sum()),
            "rate": float(reviewed.accepted.mean()) if len(reviewed) else None}
    # Severe intensity and severe corruption are different denominators.
    report['severe_motion_acceptances'] = int(output.loc[output.severity == 'severe', 'accepted'].sum())
    report['error_coverage_curve'] = []
    for limit in sorted(set([0., 2., 5., maximum_width, 15., 20.])):
        selected = output.signal_eligible & (output.severity != 'severe') & (output.half_width_sbp <= limit) & (output.half_width_dbp <= limit) & (output.predicted_dbp > 0) & (output.predicted_sbp > output.predicted_dbp)
        row = {"maximum_half_width_mmhg": limit, "coverage": coverage(output, selected.to_numpy())}
        for target in ('sbp', 'dbp'):
            row[target] = metrics(output.loc[selected, 'true_' + target], output.loc[selected, 'predicted_' + target])
        report['error_coverage_curve'].append(row)
    report['interval_empirical_coverage'] = {}
    for target in ('sbp', 'dbp'):
        accepted = output[output.accepted]
        report['interval_empirical_coverage'][target] = float((abs(accepted['true_' + target] - accepted['predicted_' + target]) <= accepted['half_width_' + target]).mean()) if len(accepted) else None
    return report


def evaluate_ablation(train: pd.DataFrame, uncertainty: pd.DataFrame, test: pd.DataFrame,
                      config: dict, *, held_out_participant: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fixed hyperparameters; disjoint uncertainty set; never select on test errors.

    The caller supplies all test windows, including rejected windows. Group-max
    residual intervals reduce pseudoreplication from overlapping windows.
    """
    train, uncertainty, test = (
        attach_motion_band_features(frame, config) for frame in (train, uncertainty, test)
    )
    for frame in (train, uncertainty, test):
        validate_reference(frame)
        if frame.participant_id.isna().any() or (frame.participant_id == "").any():
            raise ValueError("Unresolved participant identity")
        if not frame.signal_eligible.isin([True, False]).all() or not frame.severity.isin(['stationary', 'mild', 'moderate', 'severe', 'unknown']).all():
            raise ValueError("Invalid signal decisions")
        for participant, group in frame.groupby("participant_id"):
            if group.calibration_occasion_id.nunique() != 1:
                raise ValueError("More than one personal calibration")
            if set(group.calibration_occasion_id) & set(group.cuff_occasion_id):
                raise ValueError("Calibration occasion included in outcomes")
            if not (pd.to_datetime(group.calibration_end_utc, utc=True) < pd.to_datetime(group.support_start_utc, utc=True)).all():
                raise ValueError("Future calibration")
    assert_split(train, uncertainty)
    assert_split(train, test, held_out_participant=held_out_participant)
    assert_split(uncertainty, test, held_out_participant=held_out_participant)
    all_rows = pd.concat([train, uncertainty, test], ignore_index=True)
    for _, group in all_rows.groupby("participant_id"):
        for column in ["calibration_occasion_id", "baseline_sbp", "baseline_dbp"] + [x for x in all_rows if x.startswith('calibration__')]:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError("Inconsistent personal calibration")
    sets = feature_sets(train)
    for frame in (train, uncertainty, test):
        values = frame[sets['ppg_full_imu']].to_numpy(dtype=float)
        if np.isinf(values).any() or frame[['baseline_sbp', 'baseline_dbp']].isna().any().any():
            raise ValueError("Invalid feature/calibration values")
        if not ((frame.baseline_sbp > frame.baseline_dbp) & (frame.baseline_dbp > 0)).all():
            raise ValueError("Invalid calibration BP")
    predictions = []
    eligible_train = train[train.signal_eligible == True]
    eligible_uncertainty = uncertainty[uncertainty.signal_eligible == True]
    if eligible_train.empty or eligible_uncertainty.cuff_occasion_id.nunique() < config["minimum_uncertainty_groups"]:
        raise ValueError("Insufficient independent training/uncertainty groups")
    for name, columns in sets.items():
        output = test.copy()
        output["model"] = name
        for target in ("sbp", "dbp"):
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=config["ridge_alpha"]))
            model.fit(eligible_train[columns], eligible_train["true_" + target] - eligible_train["baseline_" + target])
            residual = abs(eligible_uncertainty["true_" + target] - (model.predict(eligible_uncertainty[columns]) + eligible_uncertainty["baseline_" + target]))
            widths = {}
            for severity, group in eligible_uncertainty.groupby('severity'):
                group_error = residual.loc[group.index].groupby(group.cuff_occasion_id).max().to_numpy()
                rank = int(np.ceil((len(group_error) + 1) * config["interval_coverage"]))
                widths[severity] = float(np.sort(group_error)[rank - 1]) if len(group_error) >= config['minimum_uncertainty_groups'] and rank <= len(group_error) else np.inf
            output["predicted_" + target] = model.predict(test[columns]) + test["baseline_" + target]
            output["half_width_" + target] = test.severity.map(widths).fillna(np.inf)
        output["accepted"] = (test.signal_eligible == True) & (test.severity != "severe") & (output.half_width_sbp <= config["maximum_interval_half_width_mmhg"]) & (output.half_width_dbp <= config["maximum_interval_half_width_mmhg"]) & (output.predicted_dbp > 0) & (output.predicted_sbp > output.predicted_dbp)
        predictions.append(output)
    rejection_only = predictions[0].copy()
    rejection_only['model'] = 'ppg_imu_rejection_only'
    predictions.append(rejection_only)
    baseline = predictions[0].copy()
    baseline["model"] = "zero_change"
    baseline["predicted_sbp"], baseline["predicted_dbp"] = test.baseline_sbp, test.baseline_dbp
    predictions.append(baseline)
    report = {"research_only": True, "deployment_eligible": False, "hyperparameters_selected_on_test": False,
              "coverage_denominator": "union of supplied reference-labelled test windows; not full operational time",
              "models": {}}
    report['ppg_only_gate'] = 'shared PPG+IMU signal gate; rejection-only control has identical predictions'
    report['uncertainty_note'] = 'Group-max development residual intervals; temporal distribution shift can invalidate nominal coverage'
    common = np.logical_and.reduce([x.accepted.to_numpy() for x in predictions[:3]])
    report['common_accepted_comparison'] = {}
    for output in predictions:
        report["models"][output.model.iloc[0]] = prediction_report(output, config['maximum_interval_half_width_mmhg'])
        subset = output.loc[common]
        report['common_accepted_comparison'][output.model.iloc[0]] = {
            target: metrics(subset['true_' + target], subset['predicted_' + target]) for target in ('sbp', 'dbp')}
    report["imu_ablation_comparison"] = ablation_comparison(predictions[:3])
    return pd.concat(predictions, ignore_index=True), report
