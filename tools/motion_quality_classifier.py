"""Leakage-safe training and frozen evaluation for the motion-quality classifier."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from motion_quality import (
    MOTION_INTENSITY_COLUMN,
    attach_motion_intensity,
    motion_intensity_settings,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FORBIDDEN_FEATURES = {
    "activity_label",
    "block_name",
    "end_s",
    "participant_id",
    "ppg_end_timestamp_ms",
    "ppg_start_timestamp_ms",
    "protocol_excluded",
    "review_notes",
    "reviewed_label",
    "reviewer",
    "session_id",
    "start_s",
    "suggested_label",
    "suggestion_reasons",
    "trial_id",
    "usable",
    "window_id",
    "window_index",
}
MODEL_NAMES = ("logistic_l2", "random_forest")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError(f"{name} contains invalid Boolean values")
    return normalized.isin({"true", "1"})


def load_training_data(run_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    reviewed_path = run_dir / "reviewed_windows.csv"
    report_path = run_dir / "finalization_report.json"
    if not reviewed_path.exists() or not report_path.exists():
        raise FileNotFoundError("Run directory must contain reviewed_windows.csv and finalization_report.json")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hash = report.get("reviewed_windows_sha256")
    if expected_hash and file_sha256(reviewed_path) != expected_hash:
        raise ValueError("reviewed_windows.csv changed after finalization; finalize the review again")
    if report.get("bp_labels_used") is not False:
        raise ValueError("Finalization report does not prove that BP labels were excluded")
    if report.get("activity_labels_are_model_features") is not False:
        raise ValueError("Finalization report does not prove that activity context was excluded")

    feature_columns = list(report.get("model_feature_columns", []))
    if not feature_columns:
        raise ValueError("Finalization report contains no model feature columns")
    forbidden = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError("Forbidden context or identity features requested: " + ", ".join(forbidden))

    frame = pd.read_csv(reviewed_path)
    required = {
        "window_id",
        "participant_id",
        "session_id",
        "trial_id",
        "reviewed_label",
        "usable",
        "supervised_training_eligible",
        *feature_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Reviewed dataset is missing columns: " + ", ".join(missing))
    if frame["window_id"].duplicated().any():
        raise ValueError("Reviewed dataset contains duplicate window IDs")

    eligible = _as_bool(frame["supervised_training_eligible"], "supervised_training_eligible")
    frame = frame.loc[eligible].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No supervised training windows are available")
    if frame["usable"].isna().any():
        raise ValueError("An eligible training row has no usable label")
    if not set(frame["usable"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("usable must contain only 0 or 1 for eligible rows")
    if (frame["reviewed_label"] == "uncertain").any():
        raise ValueError("Uncertain windows must not enter supervised training")

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {column} contains infinite values")
        if frame[column].notna().sum() == 0:
            raise ValueError(f"Feature {column} has no finite training values")
    return frame, feature_columns, report


def grouped_trial_splits(frame: pd.DataFrame, minimum_trials: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["trial_id"].astype(str).to_numpy()
    trials = sorted(np.unique(groups).tolist())
    if len(trials) < minimum_trials:
        raise ValueError(f"At least {minimum_trials} trials are required; found {len(trials)}")
    target = (1 - frame["usable"].astype(int)).to_numpy()
    splits = list(LeaveOneGroupOut().split(frame, target, groups))
    for train_index, test_index in splits:
        if set(groups[train_index]) & set(groups[test_index]):
            raise RuntimeError("Trial leakage detected")
        if len(np.unique(target[train_index])) < 2 or len(np.unique(target[test_index])) < 2:
            held_out = sorted(set(groups[test_index]))
            raise ValueError(f"Every trial fold must contain usable and unusable windows: {held_out}")
    return splits


def build_candidate(name: str, settings: dict[str, Any]) -> Pipeline:
    seed = int(settings["random_seed"])
    if name == "logistic_l2":
        model_settings = settings[name]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(model_settings["c"]),
                        class_weight="balanced",
                        max_iter=int(model_settings["maximum_iterations"]),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        model_settings = settings[name]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(model_settings["trees"]),
                        max_depth=int(model_settings["maximum_depth"]),
                        min_samples_leaf=int(model_settings["minimum_leaf_samples"]),
                        max_features=model_settings["maximum_features"],
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classifier: {name}")


def metric_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    true_usable, false_unusable = int(matrix[0, 0]), int(matrix[0, 1])
    false_usable, true_unusable = int(matrix[1, 0]), int(matrix[1, 1])
    usable_count = true_usable + false_unusable
    unusable_count = true_unusable + false_usable
    auc: float | None = None
    if probability is not None and len(np.unique(truth)) == 2:
        auc = float(roc_auc_score(truth, probability))
    return {
        "window_count": int(len(truth)),
        "usable_count": usable_count,
        "unusable_count": unusable_count,
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "usable_recall": true_usable / usable_count if usable_count else None,
        "unusable_recall": true_unusable / unusable_count if unusable_count else None,
        "unusable_precision": float(precision_score(truth, prediction, pos_label=1, zero_division=0)),
        "false_usable_count": false_usable,
        "false_unusable_count": false_unusable,
        "roc_auc": auc,
        "confusion_matrix": [[true_usable, false_unusable], [false_usable, true_unusable]],
    }


def _passes_acceptance(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    return bool(
        metrics["balanced_accuracy"] >= float(gates["minimum_balanced_accuracy"])
        and metrics["usable_recall"] >= float(gates["minimum_usable_recall"])
        and metrics["unusable_recall"] >= float(gates["minimum_unusable_recall"])
    )


def evaluate_candidates(
    frame: pd.DataFrame,
    feature_columns: list[str],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], str, bool]:
    splits = grouped_trial_splits(frame, int(settings["minimum_trials"]))
    features = frame[feature_columns]
    truth = (1 - frame["usable"].astype(int)).to_numpy()
    threshold = float(settings["decision_threshold"])
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    baseline_predictions = {
        "always_unusable_baseline": np.ones(len(frame), dtype=int),
        "firmware_threshold_baseline": (
            frame["imu_activity_above_threshold_fraction"].fillna(0).to_numpy(dtype=float) > 0
        ).astype(int),
    }
    for name, prediction in baseline_predictions.items():
        metrics = metric_summary(truth, prediction)
        metrics.update({"model": name, "model_type": "baseline", "passes_development_gates": False})
        summaries.append(metrics)
        for index, predicted in enumerate(prediction):
            rows.append(_prediction_row(frame.iloc[index], name, truth[index], predicted, None))

    for name in MODEL_NAMES:
        prediction = np.zeros(len(frame), dtype=int)
        probability = np.full(len(frame), np.nan, dtype=float)
        for train_index, test_index in splits:
            model = build_candidate(name, settings)
            model.fit(features.iloc[train_index], truth[train_index])
            fold_probability = model.predict_proba(features.iloc[test_index])[:, 1]
            probability[test_index] = fold_probability
            prediction[test_index] = (fold_probability >= threshold).astype(int)

        metrics = metric_summary(truth, prediction, probability)
        metrics.update(
            {
                "model": name,
                "model_type": "learned",
                "passes_development_gates": _passes_acceptance(metrics, settings["acceptance"]),
            }
        )
        summaries.append(metrics)
        for index, predicted in enumerate(prediction):
            rows.append(_prediction_row(frame.iloc[index], name, truth[index], predicted, probability[index]))

    learned = [item for item in summaries if item["model_type"] == "learned"]
    passing = [item for item in learned if item["passes_development_gates"]]
    pool = passing or learned
    selected = max(
        pool,
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["unusable_recall"]),
            float(item["usable_recall"]),
            item["model"] == "logistic_l2",
        ),
    )
    predictions = pd.DataFrame(rows)
    return predictions, summaries, str(selected["model"]), bool(passing)


def _prediction_row(
    source: pd.Series,
    model: str,
    true_unusable: int,
    predicted_unusable: int,
    probability: float | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "window_id": source["window_id"],
        "participant_id": source["participant_id"],
        "session_id": source["session_id"],
        "trial_id": source["trial_id"],
        "start_s": source.get("start_s", np.nan),
        "end_s": source.get("end_s", np.nan),
        MOTION_INTENSITY_COLUMN: source.get(MOTION_INTENSITY_COLUMN, "unknown"),
        "reviewed_label": source["reviewed_label"],
        "true_unusable": int(true_unusable),
        "predicted_unusable": int(predicted_unusable),
        "unusable_probability": probability,
        "correct": bool(true_unusable == predicted_unusable),
    }


def per_trial_metrics(predictions: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    selected = predictions[predictions["model"] == model_name]
    results = []
    for trial, group in selected.groupby("trial_id", sort=True):
        metrics = metric_summary(
            group["true_unusable"].to_numpy(dtype=int),
            group["predicted_unusable"].to_numpy(dtype=int),
            group["unusable_probability"].to_numpy(dtype=float),
        )
        metrics["trial_id"] = trial
        results.append(metrics)
    return results


def subtype_metrics(predictions: pd.DataFrame, model_name: str) -> dict[str, dict[str, int | float]]:
    selected = predictions[predictions["model"] == model_name]
    result: dict[str, dict[str, int | float]] = {}
    for label, group in selected.groupby("reviewed_label", sort=True):
        expected = 0 if label == "clean" else 1
        correct = int((group["predicted_unusable"] == expected).sum())
        result[str(label)] = {
            "window_count": int(len(group)),
            "correct_count": correct,
            "recall": correct / len(group),
        }
    return result


def _empty_stratified_metrics() -> dict[str, Any]:
    return {
        "window_count": 0,
        "usable_count": 0,
        "unusable_count": 0,
        "balanced_accuracy": None,
        "usable_recall": None,
        "unusable_recall": None,
        "unusable_precision": None,
        "false_usable_count": 0,
        "false_unusable_count": 0,
        "roc_auc": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "supported_unique_seconds": None,
        "true_usable_unique_seconds": None,
        "predicted_usable_unique_seconds": None,
        "true_usable_time_coverage": None,
        "predicted_usable_time_coverage": None,
    }


def _interval_union_seconds(group: pd.DataFrame, selection: np.ndarray | None = None) -> float | None:
    required = {"participant_id", "session_id", "trial_id", "start_s", "end_s"}
    if not required.issubset(group.columns):
        return None
    selected = group if selection is None else group.loc[np.asarray(selection, dtype=bool)]
    if selected.empty:
        return 0.0
    values = selected[["start_s", "end_s"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 1] <= values[:, 0]):
        return None
    total = 0.0
    for _, trial in selected.groupby(["participant_id", "session_id", "trial_id"], sort=False):
        cursor: float | None = None
        for start, end in sorted(trial[["start_s", "end_s"]].itertuples(index=False, name=None)):
            if cursor is None or start > cursor:
                total += end - start
                cursor = end
            elif end > cursor:
                total += end - cursor
                cursor = end
    return float(total)


def _stratified_metric_summary(group: pd.DataFrame) -> dict[str, Any]:
    """Return honest per-band metrics; two-class metrics need both labels."""
    if group.empty:
        return _empty_stratified_metrics()
    truth = group["true_unusable"].to_numpy(dtype=int)
    prediction = group["predicted_unusable"].to_numpy(dtype=int)
    probability = group["unusable_probability"].to_numpy(dtype=float)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    true_usable, false_unusable = int(matrix[0, 0]), int(matrix[0, 1])
    false_usable, true_unusable = int(matrix[1, 0]), int(matrix[1, 1])
    usable_count = true_usable + false_unusable
    unusable_count = true_unusable + false_usable
    both_classes = usable_count > 0 and unusable_count > 0
    supported_seconds = _interval_union_seconds(group)
    true_usable_seconds = _interval_union_seconds(group, truth == 0)
    predicted_usable_seconds = _interval_union_seconds(group, prediction == 0)
    return {
        "window_count": int(len(group)),
        "usable_count": usable_count,
        "unusable_count": unusable_count,
        "balanced_accuracy": (
            (true_usable / usable_count + true_unusable / unusable_count) / 2.0
            if both_classes
            else None
        ),
        "usable_recall": true_usable / usable_count if usable_count else None,
        "unusable_recall": true_unusable / unusable_count if unusable_count else None,
        "unusable_precision": (
            true_unusable / (true_unusable + false_unusable)
            if true_unusable + false_unusable
            else None
        ),
        "false_usable_count": false_usable,
        "false_unusable_count": false_unusable,
        "roc_auc": (
            float(roc_auc_score(truth, probability))
            if both_classes and np.isfinite(probability).all()
            else None
        ),
        "confusion_matrix": [[true_usable, false_unusable], [false_usable, true_unusable]],
        "supported_unique_seconds": supported_seconds,
        "true_usable_unique_seconds": true_usable_seconds,
        "predicted_usable_unique_seconds": predicted_usable_seconds,
        "true_usable_time_coverage": (
            true_usable_seconds / supported_seconds
            if supported_seconds and true_usable_seconds is not None
            else None
        ),
        "predicted_usable_time_coverage": (
            predicted_usable_seconds / supported_seconds
            if supported_seconds and predicted_usable_seconds is not None
            else None
        ),
    }


def intensity_band_metrics(
    predictions: pd.DataFrame,
    model_names: list[str],
    configured_bands: list[str],
) -> list[dict[str, Any]]:
    """Calculate diagnostic metrics for every model and configured intensity band."""
    observed = [
        str(value)
        for value in predictions[MOTION_INTENSITY_COLUMN].dropna().astype(str).unique()
        if str(value) not in configured_bands
    ]
    bands = [*configured_bands, *sorted(observed)]
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        selected = predictions[predictions["model"] == model_name]
        for band in bands:
            metrics = _stratified_metric_summary(
                selected[selected[MOTION_INTENSITY_COLUMN].astype(str) == band]
            )
            rows.append(
                {
                    "model": model_name,
                    MOTION_INTENSITY_COLUMN: band,
                    "diagnostic_only": True,
                    **metrics,
                }
            )
    return rows


def feature_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_, dtype=float)
        signed = np.full(len(importance), np.nan)
    else:
        signed = np.asarray(classifier.coef_[0], dtype=float)
        importance = np.abs(signed)
    return pd.DataFrame(
        {"feature": feature_columns, "importance": importance, "signed_coefficient": signed}
    ).sort_values(["importance", "feature"], ascending=[False, True], ignore_index=True)


def _plot_confusion(
    metrics: dict[str, Any],
    model_name: str,
    output: Path,
    title: str | None = None,
) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], ["Usable", "Unusable"])
    axis.set_yticks([0, 1], ["Usable", "Unusable"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Reviewed label")
    axis.set_title(title or f"Trial-held-out confusion matrix: {model_name}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_probabilities(
    predictions: pd.DataFrame,
    model_name: str,
    output: Path,
    title: str | None = None,
) -> None:
    selected = predictions[predictions["model"] == model_name].copy()
    labels = ["clean", "motion_corrupted", "contact_corrupted"]
    values = [selected.loc[selected["reviewed_label"] == label, "unusable_probability"] for label in labels]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.boxplot(values, tick_labels=["Clean", "Motion", "Contact"], showfliers=True)
    axis.axhline(0.5, color="#C44E52", linestyle="--", linewidth=1.2, label="Decision threshold")
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Predicted unusable probability")
    axis.set_title(title or "Trial-held-out development predictions")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train_and_evaluate(
    config: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    settings = config.get("classifier")
    if not isinstance(settings, dict):
        raise ValueError("Configuration is missing classifier settings")
    frame, feature_columns, finalization = load_training_data(run_dir)
    frame = attach_motion_intensity(frame, config)
    predictions, summaries, selected_name, gates_passed = evaluate_candidates(frame, feature_columns, settings)
    selected_summary = next(item for item in summaries if item["model"] == selected_name)
    intensity = motion_intensity_settings(config)
    configured_bands = list(intensity["labels"]) if intensity else ["unknown"]
    per_band = intensity_band_metrics(
        predictions,
        [str(item["model"]) for item in summaries],
        configured_bands,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "fold_predictions.csv", index=False)
    pd.DataFrame(summaries).drop(columns=["confusion_matrix"]).to_csv(
        output_dir / "model_metrics.csv", index=False
    )
    pd.DataFrame(per_band).drop(columns=["confusion_matrix"]).to_csv(
        output_dir / "model_intensity_band_metrics.csv", index=False
    )

    truth = (1 - frame["usable"].astype(int)).to_numpy()
    fitted = build_candidate(selected_name, settings)
    fitted.fit(frame[feature_columns], truth)
    importance = feature_importance(fitted, feature_columns)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    package = {
        "schema_version": int(settings["schema_version"]),
        "model": fitted,
        "model_name": selected_name,
        "feature_columns": feature_columns,
        "decision_threshold": float(settings["decision_threshold"]),
        "label_definition": {"0": "usable", "1": "unusable"},
        "participant_ids": sorted(frame["participant_id"].astype(str).unique().tolist()),
        "trial_ids": sorted(frame["trial_id"].astype(str).unique().tolist()),
        "training_window_count": int(len(frame)),
        "config_sha256": file_sha256(config_path),
        "reviewed_windows_sha256": finalization["reviewed_windows_sha256"],
        "motion_intensity": intensity,
        "single_subject_development": True,
        "independent_validation": False,
        "deployment_eligible": False,
    }
    joblib.dump(package, output_dir / "motion_quality_classifier.joblib")

    report = {
        "schema_version": int(settings["schema_version"]),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_git_commit": _git_commit(config_path.resolve().parents[1]),
        "participant_ids": package["participant_ids"],
        "session_ids": sorted(frame["session_id"].astype(str).unique().tolist()),
        "trial_ids": package["trial_ids"],
        "training_window_count": int(len(frame)),
        "usable_window_count": int((truth == 0).sum()),
        "unusable_window_count": int((truth == 1).sum()),
        "uncertain_windows_excluded": int(finalization["reviewed_window_count"] - len(frame)),
        "split_method": "leave_one_trial_out",
        "overlapping_windows_cross_trial_boundary": False,
        "preprocessing_fitted_inside_each_fold": True,
        "activity_context_used_as_feature": False,
        "bp_or_hr_labels_used": False,
        "candidate_metrics": summaries,
        "selected_model": selected_name,
        "selected_model_metrics": selected_summary,
        "selected_model_per_trial_metrics": per_trial_metrics(predictions, selected_name),
        "selected_model_subtype_metrics": subtype_metrics(predictions, selected_name),
        "candidate_intensity_band_metrics": per_band,
        "selected_model_per_intensity_band_metrics": [
            row for row in per_band if row["model"] == selected_name
        ],
        "intensity_band_metrics_are_diagnostic_only": True,
        "intensity_bands_used_as_classifier_features": False,
        "development_acceptance_gates": settings["acceptance"],
        "development_acceptance_passed": bool(gates_passed),
        "single_subject_development": True,
        "independent_validation": False,
        "population_validated": False,
        "deployment_eligible": False,
        "model_use": "development_only",
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "config_sha256": package["config_sha256"],
        "reviewed_windows_sha256": package["reviewed_windows_sha256"],
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _plot_confusion(selected_summary, selected_name, output_dir / "confusion_matrix.png")
    _plot_probabilities(predictions, selected_name, output_dir / "prediction_probabilities.png")
    return report


def evaluate_frozen_model(
    config: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    model_package_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one previously fitted package without fitting on validation data."""
    settings = config.get("classifier")
    if not isinstance(settings, dict):
        raise ValueError("Configuration is missing classifier settings")
    if not model_package_path.exists():
        raise FileNotFoundError(f"Model package does not exist: {model_package_path}")

    model_hash_before = file_sha256(model_package_path)
    package = joblib.load(model_package_path)
    if not isinstance(package, dict):
        raise ValueError("Model package must be a dictionary")

    required_package_fields = {
        "schema_version",
        "model",
        "model_name",
        "feature_columns",
        "decision_threshold",
        "participant_ids",
        "trial_ids",
        "config_sha256",
        "reviewed_windows_sha256",
    }
    missing_package_fields = sorted(required_package_fields - set(package))
    if missing_package_fields:
        raise ValueError("Model package is missing fields: " + ", ".join(missing_package_fields))

    expected_config_hash = file_sha256(config_path)
    if package["config_sha256"] != expected_config_hash:
        raise ValueError("Model package configuration does not match the validation configuration")
    if int(package["schema_version"]) != int(settings["schema_version"]):
        raise ValueError("Model package schema version does not match the validation configuration")

    frame, feature_columns, finalization = load_training_data(run_dir)
    frame = attach_motion_intensity(frame, config)
    package_features = list(package["feature_columns"])
    if package_features != feature_columns:
        raise ValueError("Model package feature schema does not match the finalized validation data")

    training_trials = {str(value) for value in package["trial_ids"]}
    validation_trials = set(frame["trial_id"].astype(str))
    overlap = sorted(training_trials & validation_trials)
    if overlap:
        raise ValueError("Training and validation trial IDs overlap: " + ", ".join(overlap))

    model = package["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Frozen model does not support probability prediction")
    model_classes = np.asarray(getattr(model, "classes_", []))
    if not np.array_equal(model_classes, np.asarray([0, 1])):
        raise ValueError("Frozen model classes must be ordered as usable=0 and unusable=1")
    threshold = float(package["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Model decision threshold must be between zero and one")

    features = frame[feature_columns]
    probability_matrix = np.asarray(model.predict_proba(features), dtype=float)
    if probability_matrix.ndim != 2 or probability_matrix.shape != (len(frame), 2):
        raise ValueError("Frozen model returned an unexpected probability shape")
    probability = probability_matrix[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Frozen model returned non-finite probabilities")
    prediction = (probability >= threshold).astype(int)
    truth = (1 - frame["usable"].astype(int)).to_numpy()

    model_name = str(package["model_name"])
    prediction_rows = [
        _prediction_row(frame.iloc[index], model_name, truth[index], prediction[index], probability[index])
        for index in range(len(frame))
    ]
    predictions = pd.DataFrame(prediction_rows)
    intensity = motion_intensity_settings(config)
    configured_bands = list(intensity["labels"]) if intensity else ["unknown"]
    per_band = intensity_band_metrics(predictions, [model_name], configured_bands)
    metrics = metric_summary(truth, prediction, probability)
    metrics.update(
        {
            "model": model_name,
            "model_type": "frozen_learned",
            "passes_validation_gates": _passes_acceptance(metrics, settings["acceptance"]),
        }
    )

    model_hash_after = file_sha256(model_package_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("Frozen model package changed during validation")

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(per_band).drop(columns=["confusion_matrix"]).to_csv(
        output_dir / "validation_intensity_band_metrics.csv", index=False
    )
    report = {
        "schema_version": int(settings["schema_version"]),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_git_commit": _git_commit(config_path.resolve().parents[1]),
        "model_name": model_name,
        "model_package_path": str(model_package_path.resolve()),
        "model_package_sha256_before": model_hash_before,
        "model_package_sha256_after": model_hash_after,
        "model_frozen": True,
        "model_refit_on_validation": False,
        "validation_data_used_for_training": False,
        "training_participant_ids": sorted(str(value) for value in package["participant_ids"]),
        "validation_participant_ids": sorted(frame["participant_id"].astype(str).unique().tolist()),
        "training_trial_ids": sorted(training_trials),
        "validation_trial_ids": sorted(validation_trials),
        "training_validation_trial_overlap": False,
        "validation_window_count": int(len(frame)),
        "usable_window_count": int((truth == 0).sum()),
        "unusable_window_count": int((truth == 1).sum()),
        "uncertain_windows_excluded": int(finalization["reviewed_window_count"] - len(frame)),
        "feature_columns": feature_columns,
        "decision_threshold": threshold,
        "validation_metrics": metrics,
        "per_trial_metrics": per_trial_metrics(predictions, model_name),
        "subtype_metrics": subtype_metrics(predictions, model_name),
        "motion_intensity": intensity,
        "per_intensity_band_metrics": per_band,
        "intensity_band_metrics_are_diagnostic_only": True,
        "intensity_bands_used_as_classifier_features": False,
        "validation_acceptance_gates": settings["acceptance"],
        "validation_acceptance_passed": bool(metrics["passes_validation_gates"]),
        "independent_trial_validation": True,
        "participant_independent_validation": False,
        "population_validated": False,
        "deployment_eligible": False,
        "model_use": "research_validation_only",
        "bp_or_hr_labels_used": False,
        "activity_context_used_as_feature": False,
        "config_sha256": expected_config_hash,
        "validation_reviewed_windows_sha256": finalization["reviewed_windows_sha256"],
        "training_reviewed_windows_sha256": package["reviewed_windows_sha256"],
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _plot_confusion(
        metrics,
        model_name,
        output_dir / "validation_confusion_matrix.png",
        title=f"Frozen-model validation: {model_name}",
    )
    _plot_probabilities(
        predictions,
        model_name,
        output_dir / "validation_probabilities.png",
        title="Independent validation predictions",
    )
    return report
