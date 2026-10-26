"""Setup-aware, single-person Ridge development; deliberately not a live model format.

Every setup episode has its own explicit, earlier calibration. Features are
extracted identically for training and offline inference through the existing
feasibility implementation. Prospective IDs are never used for fitting.
"""
from __future__ import annotations

import json
import math
import platform
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from .config import load_config
from .datasets import Recording, _discover_local, sha256_file
from .feasibility import load_feasibility_policy, process_gap_tolerant_recording
from .models import _model_feature_columns, _scaled_pipeline, build_personalized_examples
from .reporting import write_json
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_KIND = "setup_aware_single_subject_research_v1"
PROCESSING_FILES = ("bp_core/combined.py", "bp_core/config.py", "bp_core/datasets.py",
                    "bp_core/feasibility.py", "bp_core/features.py", "bp_core/models.py",
                    "bp_core/inference.py", "upper_arm_hr.py")


def identity(occasion_id: str, participant: str) -> tuple[str, str, str]:
    parts = str(occasion_id).split(":")
    if (len(parts) != 4 or parts[0] != "local_upper_arm" or parts[1] != participant
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts[1:])):
        raise ValueError(f"Invalid or wrong-participant occasion ID: {occasion_id}")
    return tuple(parts[1:])


def validate_study(study: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str]]:
    """Build an explicit ID->(episode, role) map; no inferred setup assignments."""
    if not isinstance(study, dict) or study.get("schema_version") != 1 or not study.get("participant_id"):
        raise ValueError("Study needs schema_version=1 and participant_id")
    if study.get("training_variant") != "combined":
        raise ValueError("The selected variant must be fixed to combined before testing")
    for field in ("minimum_development_occasions", "minimum_test_occasions"):
        if type(study.get(field)) is not int or study[field] < 5:
            raise ValueError(f"{field} must be an integer >= 5")
    episodes = study.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 2 or any(not isinstance(item, dict) for item in episodes):
        raise ValueError("Declare at least two calibration episodes")
    episode_ids = [episode.get("episode_id") for episode in episodes]
    if (any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in episode_ids)
            or len(set(episode_ids)) != len(episode_ids)):
        raise ValueError("Episode IDs must be unique safe identifiers")
    target = study.get("target_episode_id")
    if target not in episode_ids:
        raise ValueError("target_episode_id is not declared")
    mapped = {}
    for episode in episodes:
        if not episode.get("setup_id") or not episode.get("sensor_location"):
            raise ValueError("Every episode needs an explicit setup_id and sensor_location")
        for role, values in (
            ("calibration", [episode.get("calibration_occasion_id")]),
            ("development", episode.get("development_occasion_ids")),
            ("test", episode.get("test_occasion_ids")),
        ):
            if not isinstance(values, list) or (role == "development" and not values):
                raise ValueError("Every episode needs explicit development/test ID lists")
            if role == "test" and values and episode["episode_id"] != target:
                raise ValueError("Prospective tests must belong to the target episode")
            for occasion_id in values:
                identity(occasion_id, study["participant_id"])
                if occasion_id in mapped:
                    raise ValueError(f"Calibration/development/test overlap or duplicate: {occasion_id}")
                mapped[occasion_id] = (episode, role)
    if sum(role == "test" for _, role in mapped.values()) < study["minimum_test_occasions"]:
        raise ValueError("Reserve at least minimum_test_occasions explicit prospective IDs")
    return mapped


def extract_selected(study, config, policy, roles):
    """Only selected roles enter canonical discovery, including label merging."""
    mapped = validate_study(study)
    selected = {key for key, (_, role) in mapped.items() if role in roles}
    selected_identities = {identity(key, study["participant_id"]) for key in selected}
    recordings = _discover_local(config["datasets"]["local_upper_arm"], selected_identities=selected_identities)
    found = {recording.label_group_id for recording in recordings}
    if found != selected or len(recordings) != len(found):
        raise ValueError(f"Missing/unlabelled/duplicate selected recordings: {sorted(selected-found)}")
    occasions, segments, sources = [], [], []
    seen_ppg = set()
    for recording in recordings:
        episode, role = mapped[recording.label_group_id]
        ppg_hash = sha256_file(Path(recording.ppg_path))
        if ppg_hash in seen_ppg:
            raise ValueError("Duplicate raw PPG content across cuff occasions")
        seen_ppg.add(ppg_hash)
        if recording.sensor_site != episode["sensor_location"]:
            raise ValueError(f"Sensor location does not match declared episode: {recording.label_group_id}")
        if not (math.isfinite(recording.sbp) and math.isfinite(recording.dbp) and recording.sbp > recording.dbp):
            raise ValueError(f"Invalid reference BP: {recording.label_group_id}")
        result = process_gap_tolerant_recording(recording, config, policy)
        row = dict(result.occasion)
        row.update(episode_id=episode["episode_id"], setup_id=episode["setup_id"], role=role,
                   calibration_occasion=role == "calibration", chronological_order=recording.chronological_order,
                   recording_id=recording.recording_id)
        occasions.append(row)
        for segment in result.segments.to_dict("records"):
            segment.update(episode_id=episode["episode_id"], setup_id=episode["setup_id"], role=role)
            segments.append(segment)
        for kind, value in (("ppg", recording.ppg_path), ("imu", recording.imu_path), ("metadata", recording.metadata_path)):
            if value and Path(value).exists():
                sources.append({"occasion_id": recording.label_group_id, "kind": kind, "path": str(Path(value).resolve()),
                                "sha256": sha256_file(Path(value))})
    frame = pd.DataFrame(occasions)
    frame["_time"] = pd.to_datetime(frame.chronological_order, errors="coerce", utc=True)
    if frame._time.isna().any() or frame._time.duplicated().any():
        raise ValueError("Selected occasions need valid, distinct chronological timestamps")
    return frame.sort_values(["_time", "label_group_id"]).reset_index(drop=True), pd.DataFrame(segments), sources


def episode_examples(occasions, study):
    """Calibrate within episodes; never use a later or another setup's waveform."""
    output, calibrations = [], {}
    for episode in study["episodes"]:
        group = occasions[occasions.episode_id == episode["episode_id"]].copy()
        calibration = group[group.label_group_id == episode["calibration_occasion_id"]]
        if len(calibration) != 1 or not bool(calibration.iloc[0].occasion_usable):
            raise ValueError(f"Explicit calibration failed quality/missing: {episode['episode_id']}; no fallback")
        cal = calibration.iloc[0]
        if any(group[group.role != "calibration"]._time <= cal._time):
            raise ValueError(f"Occasion does not follow its episode calibration: {episode['episode_id']}")
        calibrations[episode["episode_id"]] = cal.to_dict()
        usable = group[group.occasion_usable == True].copy()  # noqa: E712
        examples = build_personalized_examples(usable)
        if not examples.empty:
            examples["episode_id"] = episode["episode_id"]
            examples["setup_id"] = episode["setup_id"]
            output.append(examples)
    if not output:
        raise ValueError("No usable post-calibration development occasions")
    examples = pd.concat(output, ignore_index=True)
    examples["_time"] = pd.to_datetime(examples.chronological_order, utc=True)
    if examples.label_group_id.duplicated().any():
        raise ValueError("Duplicate cuff occasion in personalized examples")
    return examples.sort_values(["_time", "label_group_id"]).reset_index(drop=True), calibrations


def forward_folds(examples, target_episode, variant, minimum, max_splits):
    """Score only future target-setup occasions; auxiliary history stays earlier."""
    target_indices = np.flatnonzero(examples.episode_id.eq(target_episode).to_numpy())
    if len(examples) < minimum or (variant == "new_housing_only" and len(target_indices) < minimum):
        raise ValueError(f"Insufficient development occasions for {variant}; need {minimum}")
    if len(target_indices) < 3:
        raise ValueError("At least three target-episode follow-ups are required for forward validation")
    splits = []
    for _, validation in TimeSeriesSplit(n_splits=min(max_splits, len(target_indices)-1)).split(target_indices):
        validation_index = target_indices[validation]
        boundary = examples.iloc[validation_index]._time.min()
        training = examples._time.lt(boundary)
        if variant == "new_housing_only":
            training &= examples.episode_id.eq(target_episode)
        train_index = np.flatnonzero(training.to_numpy())
        if len(train_index) >= 2:
            splits.append((train_index, validation_index))
    if len(splits) < 2:
        raise ValueError("Fewer than two safe forward development folds")
    return splits


def fit_variant(examples, config, study, variant):
    splits = forward_folds(examples, study["target_episode_id"], variant,
                           study["minimum_development_occasions"], int(config["models"]["inner_max_splits"]))
    final_rows = examples if variant == "combined" else examples[examples.episode_id == study["target_episode_id"]]
    packages, scored, predictions = {}, [], []
    fold_ids = [{"train_ids": examples.iloc[train].label_group_id.tolist(),
                 "validation_ids": examples.iloc[val].label_group_id.tolist()} for train, val in splits]
    for target in ("sbp", "dbp"):
        column = f"delta_{target}"
        features = _model_feature_columns(examples)
        options = []
        for alpha in sorted(set(float(value) for value in config["models"]["ridge_alphas"])):
            if not math.isfinite(alpha) or alpha < 0:
                raise ValueError("Ridge alphas must be finite and nonnegative")
            estimator = _scaled_pipeline(Ridge(alpha=alpha))
            rows = []
            try:
                for fold, (train, val) in enumerate(splits):
                    fitted = clone(estimator).fit(examples.iloc[train][features], examples.iloc[train][column])
                    values = fitted.predict(examples.iloc[val][features])
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite forward predictions")
                    for index, value in zip(val, values):
                        rows.append(prediction_row(examples.iloc[index], target, "ridge", value, fold))
            except ValueError as exc:
                scored.append({"variant": variant, "target": target, "model": "ridge", "alpha": alpha,
                               "status": "failed", "reason": str(exc), "mae": None})
                continue
            mae = float(np.mean([abs(row["error"]) for row in rows]))
            scored.append({"variant": variant, "target": target, "model": "ridge", "alpha": alpha,
                           "status": "ok", "reason": "", "mae": mae})
            options.append((mae, alpha, estimator, rows))
        if not options:
            raise ValueError(f"No Ridge candidate fitted for {variant}/{target}")
        _, alpha, estimator, rows = min(options, key=lambda item: (item[0], item[1]))
        estimator.fit(final_rows[features], final_rows[column])
        packages[target] = {"estimator": estimator, "feature_columns": features, "alpha": alpha, "target": target}
        predictions.extend(rows)
        # Baselines use exactly the same forward target validation occasions.
        hr_features = _model_feature_columns(examples, hr_only=True)
        for fold, (train, val) in enumerate(splits):
            mean = float(examples.iloc[train][column].mean())
            hr_model = _scaled_pipeline(LinearRegression()).fit(examples.iloc[train][hr_features], examples.iloc[train][column])
            hr_values = hr_model.predict(examples.iloc[val][hr_features])
            for index, hr_value in zip(val, hr_values):
                for name, value in (("zero_change", 0.), ("mean_change", mean), ("hr_only", float(hr_value))):
                    predictions.append(prediction_row(examples.iloc[index], target, name, value, fold))
    return packages, scored, predictions, fold_ids


def prediction_row(row, target, model, delta, fold=None):
    estimate = float(row[f"baseline_{target}"]) + float(delta)
    return {"label_group_id": row["label_group_id"], "episode_id": row["episode_id"],
            "target": target, "model": model, "fold": fold,
            "true_bp": float(row[f"true_{target}"]), "predicted_bp": estimate,
            "calibration_occasion_id": row["calibration_label_group_id"],
            "error": estimate - float(row[f"true_{target}"])}


def metrics(rows):
    frame = pd.DataFrame(rows)
    result = {}
    for (target, model), group in frame.groupby(["target", "model"]):
        error = group.error.to_numpy(float)
        result[f"{target}:{model}"] = {"count": len(group), "mae": float(np.mean(abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(np.mean(error)),
            "max_absolute_error": float(np.max(abs(error)))}
    return result


def _code_hashes():
    return {name: sha256_file(ROOT / "tools" / name) for name in PROCESSING_FILES}


def train_combined(config_path, policy_path, study_path, output_dir, prepare_only=False):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Run directory exists; choose a new run ID (no overwrite)")
    config, _ = load_config(config_path)
    policy = load_feasibility_policy(policy_path)
    study = json.loads(Path(study_path).read_text(encoding="utf-8"))
    mapped = validate_study(study)
    occasions, segments, sources = extract_selected(study, config, policy, {"calibration", "development"})
    examples, calibrations = episode_examples(occasions, study)
    target_cal = calibrations[study["target_episode_id"]]
    output.mkdir(parents=True)
    occasions.drop(columns="_time").to_csv(output / "occasion_features.csv", index=False)
    segments.to_csv(output / "segment_features.csv", index=False)
    examples.drop(columns="_time").to_csv(output / "development_examples.csv", index=False)
    for name, payload in (("config_snapshot.json", config), ("policy_snapshot.json", policy), ("study_snapshot.json", study)):
        write_json(output / name, payload)
    split = {"method": "explicit_episode_calibrations_and_prospective_ids",
             "requested_development_ids": sorted(key for key, (_, role) in mapped.items() if role == "development"),
             "usable_development_ids": examples.label_group_id.tolist(),
             "rejected_development_ids": occasions.loc[(occasions.role == "development") & ~occasions.occasion_usable.astype(bool), "label_group_id"].tolist(),
             "calibration_ids": {key: value["label_group_id"] for key, value in calibrations.items()},
             "reserved_test_ids": sorted(key for key, (_, role) in mapped.items() if role == "test"),
             "development_cutoff": occasions._time.max().isoformat()}
    write_json(output / "split.json", split)
    report = {"research_only": True, "single_subject_development": True, "population_validated": False,
              "viewer_eligible": False, "prospective_validation_pending": True,
              "participant_id": study["participant_id"], "selected_variant": "combined",
              "selected_model_family": "ridge", "variants": {}, "raw_sources": sources,
              "counts": {"calibration_occasions": len(calibrations), "usable_development": len(examples),
                         "rejected_development": len(split["rejected_development_ids"]),
                         "reserved_tests": len(split["reserved_test_ids"]),
                         "usable_development_by_episode": examples.episode_id.value_counts().to_dict()},
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    selected = None
    for variant in (() if prepare_only else ("combined", "new_housing_only")):
        try:
            packages, scores, rows, folds = fit_variant(examples, config, study, variant)
        except ValueError as exc:
            report["variants"][variant] = {"status": "insufficient_or_failed", "reason": str(exc)}
            continue
        pd.DataFrame(scores).to_csv(output / f"{variant}_cv_scores.csv", index=False)
        pd.DataFrame(rows).to_csv(output / f"{variant}_cv_predictions.csv", index=False)
        write_json(output / f"{variant}_forward_folds.json", folds)
        report["variants"][variant] = {"status": "development_only", "metrics": metrics(rows),
                                       "selected_alphas": {key: value["alpha"] for key, value in packages.items()}}
        if variant == "combined":
            selected = packages
    report["status"] = "prepared_only" if prepare_only else ("development_model_fitted" if selected else "insufficient_development")
    write_json(output / "development_report.json", report)
    if selected:
        calibration_snapshot = {"label_group_id": target_cal["label_group_id"],
                                "chronological_order": target_cal["chronological_order"],
                                "sbp": float(target_cal["sbp"]), "dbp": float(target_cal["dbp"]),
                                "features": {key: float(value) for key, value in target_cal.items()
                                             if key.startswith(("median__feature__", "iqr__feature__")) and pd.notna(value)}}
        write_json(output / "calibration_snapshot.json", calibration_snapshot)
        manifest = {"schema_version": 1, "package_kind": PACKAGE_KIND, "research_only": True,
                    "single_subject_development": True, "population_validated": False, "viewer_eligible": False,
                    "prospective_validation_pending": True, "participant_id": study["participant_id"],
                    "target_episode_id": study["target_episode_id"], "setup_id": target_cal["setup_id"],
                    "sensor_location": target_cal["sensor_site"],
                    "calibration": calibration_snapshot,
                    "snapshot_sha256": {name: sha256_file(output / name) for name in
                                        ("config_snapshot.json", "policy_snapshot.json", "study_snapshot.json", "split.json", "calibration_snapshot.json", "development_report.json", "occasion_features.csv", "development_examples.csv")},
                    "processing_code_sha256": _code_hashes(), "models": {}}
        for target, package in selected.items():
            package.update(package_kind=PACKAGE_KIND, participant_id=study["participant_id"],
                           calibration_id=target_cal["label_group_id"])
            path = output / f"research_{target}_ridge.joblib"
            joblib.dump(package, path)
            manifest["models"][target] = {"file": path.name, "sha256": sha256_file(path),
                                           "feature_columns": package["feature_columns"], "alpha": package["alpha"]}
        # Intentionally not model_manifest.json: legacy/live loaders fail closed.
        write_json(output / "combined_model_manifest.json", manifest)
    return report


def load_combined(run_dir):
    try:
        return _load_combined(run_dir)
    except (KeyError, TypeError, AttributeError, EOFError) as exc:
        raise ValueError(f"Malformed combined research package: {exc}") from exc


def _load_combined(run_dir):
    root = Path(run_dir)
    manifest = json.loads((root / "combined_model_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("package_kind") != PACKAGE_KIND
            or manifest.get("viewer_eligible") is not False):
        raise ValueError("Invalid combined research manifest; live eligibility is never enabled")
    for name in ("config_snapshot.json", "policy_snapshot.json", "study_snapshot.json", "split.json", "calibration_snapshot.json", "development_report.json", "occasion_features.csv", "development_examples.csv"):
        if manifest.get("snapshot_sha256", {}).get(name) != sha256_file(root / name):
            raise ValueError(f"Frozen snapshot checksum mismatch: {name}")
    if manifest.get("processing_code_sha256") != _code_hashes():
        raise ValueError("Processing implementation changed since fitting; do not silently reuse features")
    config, _ = load_config(root / "config_snapshot.json")
    policy = load_feasibility_policy(root / "policy_snapshot.json")
    study = json.loads((root / "study_snapshot.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    validate_study(study)
    if manifest["participant_id"] != study["participant_id"] or manifest["target_episode_id"] != study["target_episode_id"]:
        raise ValueError("Manifest participant/episode mismatch")
    target_episode = next(item for item in study["episodes"] if item["episode_id"] == study["target_episode_id"])
    calibration = json.loads((root / "calibration_snapshot.json").read_text(encoding="utf-8"))
    if (manifest["calibration"] != calibration or calibration["label_group_id"] != target_episode["calibration_occasion_id"]
            or manifest["setup_id"] != target_episode["setup_id"]
            or manifest["sensor_location"] != target_episode["sensor_location"]):
        raise ValueError("Manifest calibration/setup does not match frozen study")
    if (not calibration["features"] or not all(math.isfinite(value) for value in calibration["features"].values())
            or not math.isfinite(calibration["sbp"]) or not math.isfinite(calibration["dbp"])
            or calibration["sbp"] <= calibration["dbp"]):
        raise ValueError("Invalid calibration BP/features")
    packages = {}
    for target in ("sbp", "dbp"):
        entry = manifest["models"][target]
        path = root / entry["file"]
        if path.parent.resolve() != root.resolve() or sha256_file(path) != entry["sha256"]:
            raise ValueError("Model path/checksum mismatch")
        try:
            package = joblib.load(path)  # Only locally produced, trusted packages.
        except Exception as exc:
            raise ValueError(f"Cannot load trusted local research model: {exc}") from exc
        if (package.get("package_kind") != PACKAGE_KIND or package.get("target") != target
                or package.get("participant_id") != manifest["participant_id"]
                or package.get("calibration_id") != manifest["calibration"]["label_group_id"]
                or package.get("feature_columns") != entry["feature_columns"]):
            raise ValueError("Model participant/calibration/feature schema mismatch")
        packages[target] = package
    return manifest, config, policy, study, split, packages


def predict_occasion(manifest, packages, occasion):
    """The same calibration-relative construction as episode-level training."""
    if not bool(occasion.get("occasion_usable", False)):
        return {"status": "signal_rejected", "reason": occasion.get("occasion_rejection_reasons", "quality_failed"),
                "sbp": None, "dbp": None}
    cal = manifest["calibration"]
    calibration = {**cal["features"], "dataset_id": "local_upper_arm", "participant_id": manifest["participant_id"],
                   "label_group_id": cal["label_group_id"], "chronological_order": "0", "sbp": cal["sbp"],
                   "dbp": cal["dbp"], "occasion_usable": True, "calibration_occasion": True}
    current = {**occasion, "dataset_id": "local_upper_arm", "participant_id": manifest["participant_id"],
               "label_group_id": "current", "chronological_order": "1", "sbp": np.nan, "dbp": np.nan,
               "occasion_usable": True, "calibration_occasion": False}
    example = build_personalized_examples(pd.DataFrame([calibration, current]))
    estimates, deltas = {}, {}
    for target, package in packages.items():
        columns = package["feature_columns"]
        if not set(columns) <= set(example):
            raise ValueError("Current/calibration feature schema does not match trained model")
        delta = float(package["estimator"].predict(example[columns])[0])
        deltas[target] = delta
        estimates[target] = float(cal[target]) + delta
    if not all(math.isfinite(value) for value in estimates.values()) or estimates["sbp"] <= estimates["dbp"]:
        return {"status": "invalid_model_output", "reason": "Non-finite output or SBP <= DBP", "sbp": None, "dbp": None}
    return {"status": "unvalidated_development_estimate", **estimates,
            "delta_sbp": deltas["sbp"], "delta_dbp": deltas["dbp"]}


def predict_combined(run_dir, ppg_path, imu_path, metadata_path, setup_id, *, allow_unvalidated=False):
    from .feasibility import process_gap_tolerant_frames
    manifest, config, policy, study, split, packages = load_combined(run_dir)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if (metadata.get("subject_id") != manifest["participant_id"] or setup_id != manifest["setup_id"]
            or metadata.get("sensor_location") != manifest["sensor_location"]
            or metadata.get("ppg_profile") != config["datasets"]["local_upper_arm"]["required_profile"]):
        raise ValueError("Recording participant/setup/location/profile does not match research package")
    occasion_id = f"local_upper_arm:{metadata['subject_id']}:{metadata['session_id']}:{metadata['trial_id']}"
    mapped = validate_study(study)
    if occasion_id in mapped and mapped[occasion_id][0]["setup_id"] != setup_id:
        raise ValueError("Known recording belongs to a different calibration/setup episode")
    if occasion_id in split["reserved_test_ids"]:
        raise ValueError("Reserved test recording: use combined-evaluate, not ad hoc prediction")
    recording = Recording("local_upper_arm", manifest["participant_id"], metadata["session_id"], metadata["trial_id"],
                          occasion_id, str(metadata.get("recording_start_time", "")), manifest["sensor_location"],
                          100., str(ppg_path), str(imu_path), np.nan, np.nan, None, "none", None, "")
    extraction = process_gap_tolerant_frames(recording, pd.read_csv(ppg_path), pd.read_csv(imu_path), metadata, config, policy)
    if not allow_unvalidated and bool(extraction.occasion.get("occasion_usable", False)):
        result = {"status": "model_validation_pending", "reason": "Use --allow-unvalidated only for an explicitly labelled offline development estimate",
                  "sbp": None, "dbp": None}
    elif occasion_id == manifest["calibration"]["label_group_id"]:
        result = {"status": "calibration_only", "reason": "Calibration is not a prediction example", "sbp": None, "dbp": None}
    else:
        result = predict_occasion(manifest, packages, extraction.occasion)
    return {**result, "research_only": True, "population_validated": False, "viewer_eligible": False,
            "allow_unvalidated": bool(allow_unvalidated),
            "development_recording": occasion_id in split["usable_development_ids"],
            "accepted_windows": extraction.occasion.get("accepted_segment_count", 0),
            "unique_clean_coverage_s": extraction.occasion.get("unique_clean_coverage_s", 0),
            "calibration": {key: manifest["calibration"][key] for key in ("label_group_id", "sbp", "dbp")},
            "quality_policy": policy["policy_name"]}


def evaluate_combined(run_dir):
    root = Path(run_dir)
    destination = root / "prospective_evaluation"
    if destination.exists():
        raise FileExistsError("Prospective results already exist; do not reuse this test batch for tuning")
    manifest, config, policy, study, split, packages = load_combined(root)
    occasions, segments, sources = extract_selected(study, config, policy, {"test"})
    development_report = json.loads((root / "development_report.json").read_text(encoding="utf-8"))
    development_ppg = {row["sha256"] for row in development_report["raw_sources"] if row["kind"] == "ppg"}
    if any(row["kind"] == "ppg" and row["sha256"] in development_ppg for row in sources):
        raise ValueError("Duplicate raw PPG content across development and test occasions")
    if any(occasions._time <= pd.Timestamp(split["development_cutoff"])):
        raise ValueError("Test occasions must be strictly later than every development/calibration occasion")
    predictions, decisions = [], []
    for _, occasion in occasions.iterrows():
        prediction = predict_occasion(manifest, packages, occasion.to_dict())
        decisions.append({"label_group_id": occasion.label_group_id, **prediction})
        if prediction["sbp"] is not None:
            for target in ("sbp", "dbp"):
                row = {"label_group_id": occasion.label_group_id, "episode_id": occasion.episode_id,
                       f"baseline_{target}": manifest["calibration"][target],
                       "calibration_label_group_id": manifest["calibration"]["label_group_id"],
                       f"true_{target}": occasion[target]}
                predictions.append(prediction_row(row, target, "ridge", prediction[f"delta_{target}"]))
                predictions.append(prediction_row(row, target, "zero_change", 0.))
    summary = metrics(predictions) if predictions else {}
    accepted = sum(row["sbp"] is not None for row in decisions)
    gate = accepted >= study["minimum_test_occasions"] and all(
        summary.get(f"{target}:ridge", {}).get("mae", math.inf)
        < summary.get(f"{target}:zero_change", {}).get("mae", -math.inf) for target in ("sbp", "dbp"))
    result = {"research_only": True, "population_validated": False, "viewer_eligible": False,
              "selected_variant": "combined", "test_occasions": len(occasions), "accepted_occasions": accepted,
              "coverage": accepted / len(occasions), "beats_zero_change_both_targets": bool(gate),
              "metrics": summary, "model_refitted": False, "raw_sources": sources}
    destination.mkdir()
    occasions.drop(columns="_time").to_csv(destination / "occasion_features.csv", index=False)
    segments.to_csv(destination / "segment_features.csv", index=False)
    pd.DataFrame(predictions).to_csv(destination / "predictions.csv", index=False)
    pd.DataFrame(decisions).to_csv(destination / "quality_decisions.csv", index=False)
    write_json(destination / "metrics.json", result)
    return result
