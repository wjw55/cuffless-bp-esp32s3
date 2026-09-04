from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IDENTITY_COLUMNS = {
    "dataset_id",
    "participant_id",
    "session_id",
    "label_group_id",
    "chronological_order",
    "sensor_site",
    "sbp",
    "dbp",
    "reference_hr",
    "calibration_occasion",
    "total_segment_count",
    "accepted_segment_count",
    "accepted_segment_fraction",
    "occasion_usable",
    "occasion_status",
}


def _numeric_feature_columns(occasions: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in occasions.columns
        if (column.startswith("median__feature__") or column.startswith("iqr__feature__"))
    )


def build_personalized_examples(occasions: pd.DataFrame) -> pd.DataFrame:
    """Create one-time-calibrated examples without treating windows as labels."""
    if occasions.empty:
        return pd.DataFrame()
    usable = occasions[occasions["occasion_usable"] == True].copy()  # noqa: E712
    feature_columns = _numeric_feature_columns(usable)
    examples: list[dict[str, Any]] = []
    for (dataset_id, participant_id), group in usable.groupby(["dataset_id", "participant_id"], sort=True):
        ordered = group.sort_values(["chronological_order", "label_group_id"]).reset_index(drop=True)
        if len(ordered) < 2:
            continue
        marked = ordered.index[ordered["calibration_occasion"] == True].tolist()  # noqa: E712
        calibration_index = marked[0] if marked else 0
        calibration = ordered.iloc[calibration_index]
        for row_index, current in ordered.iterrows():
            if row_index == calibration_index:
                continue
            row: dict[str, Any] = {
                "dataset_id": dataset_id,
                "participant_id": participant_id,
                "participant_group": f"{dataset_id}:{participant_id}",
                "label_group_id": current["label_group_id"],
                "calibration_label_group_id": calibration["label_group_id"],
                "chronological_order": current["chronological_order"],
                "baseline_sbp": float(calibration["sbp"]),
                "baseline_dbp": float(calibration["dbp"]),
                "true_sbp": float(current["sbp"]),
                "true_dbp": float(current["dbp"]),
                "delta_sbp": float(current["sbp"] - calibration["sbp"]),
                "delta_dbp": float(current["dbp"] - calibration["dbp"]),
            }
            for column in feature_columns:
                current_value = pd.to_numeric(pd.Series([current.get(column)]), errors="coerce").iloc[0]
                baseline_value = pd.to_numeric(pd.Series([calibration.get(column)]), errors="coerce").iloc[0]
                short_name = column.replace("median__feature__", "median__").replace("iqr__feature__", "iqr__")
                row[f"current__{short_name}"] = current_value
                row[f"calibration__{short_name}"] = baseline_value
                row[f"change__{short_name}"] = current_value - baseline_value
            examples.append(row)
    return pd.DataFrame(examples)


def _is_true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def prepare_single_subject_split(
    occasions: pd.DataFrame,
    participant_id: str,
    *,
    test_fraction: float = 0.30,
    minimum_test_occasions: int = 5,
    minimum_development_occasions: int = 5,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create a deterministic calibration/development/test split for one participant.

    Only occasions after the selected calibration are eligible. The latest test
    occasions are locked away from all fitting and model selection.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if minimum_test_occasions < 1 or minimum_development_occasions < 2:
        raise ValueError("minimum occasion counts are invalid")
    required = {
        "dataset_id",
        "participant_id",
        "label_group_id",
        "chronological_order",
        "sbp",
        "dbp",
        "calibration_occasion",
        "occasion_usable",
    }
    missing = sorted(required.difference(occasions.columns))
    if missing:
        raise ValueError(f"Occasion table is missing required columns: {', '.join(missing)}")

    local = occasions[
        (occasions["dataset_id"].astype(str) == "local_upper_arm")
        & (occasions["participant_id"].astype(str) == str(participant_id))
        & occasions["occasion_usable"].map(_is_true)
    ].copy()
    if local.empty:
        raise ValueError(f"No usable local_upper_arm occasions found for participant {participant_id}")
    if local["label_group_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate cuff-occasion IDs found for the selected participant")
    local["_chronological_timestamp"] = pd.to_datetime(
        local["chronological_order"], errors="coerce", utc=True
    )
    if local["_chronological_timestamp"].isna().any():
        invalid = local.loc[local["_chronological_timestamp"].isna(), "label_group_id"].astype(str).tolist()
        raise ValueError(f"Invalid chronological timestamps: {', '.join(invalid)}")
    local = local.sort_values(["_chronological_timestamp", "label_group_id"]).reset_index(drop=True)

    marked = local.index[local["calibration_occasion"].map(_is_true)].tolist()
    calibration_index = marked[0] if marked else 0
    calibration = local.iloc[calibration_index].copy()
    eligible = local.iloc[calibration_index:].drop(columns=["_chronological_timestamp"]).copy()
    if not marked:
        eligible.loc[eligible.index[0], "calibration_occasion"] = True
    examples = build_personalized_examples(eligible)
    examples = examples.sort_values(["chronological_order", "label_group_id"]).reset_index(drop=True)

    test_count = max(minimum_test_occasions, int(math.ceil(len(examples) * test_fraction)))
    development_count = len(examples) - test_count
    if development_count < minimum_development_occasions:
        required_followups = minimum_development_occasions + minimum_test_occasions
        raise ValueError(
            f"Insufficient post-calibration occasions for participant {participant_id}: "
            f"found {len(examples)}, need at least {required_followups} "
            f"({minimum_development_occasions} development and {minimum_test_occasions} test)"
        )
    development = examples.iloc[:development_count].copy().reset_index(drop=True)
    test = examples.iloc[development_count:].copy().reset_index(drop=True)
    development_ids = development["label_group_id"].astype(str).tolist()
    test_ids = test["label_group_id"].astype(str).tolist()
    if set(development_ids).intersection(test_ids):
        raise ValueError("Cuff-occasion leakage detected between development and test splits")

    split = {
        "method": "single_subject_chronological_70_30",
        "participant_id": str(participant_id),
        "test_fraction_requested": float(test_fraction),
        "calibration_selection": "first_explicit" if marked else "first_chronological_usable",
        "calibration_occasion_id": str(calibration["label_group_id"]),
        "calibration_chronological_order": str(calibration["chronological_order"]),
        "calibration_sbp": float(calibration["sbp"]),
        "calibration_dbp": float(calibration["dbp"]),
        "excluded_pre_calibration_occasion_ids": local.iloc[:calibration_index]["label_group_id"].astype(str).tolist(),
        "development_occasion_ids": development_ids,
        "test_occasion_ids": test_ids,
        "development_count": len(development_ids),
        "test_count": len(test_ids),
    }
    return calibration.drop(labels=["_chronological_timestamp"]), development, test, split


def _model_feature_columns(examples: pd.DataFrame, hr_only: bool = False) -> list[str]:
    columns = [
        column
        for column in examples.columns
        if column.startswith(("current__", "calibration__", "change__"))
    ]
    if hr_only:
        columns = [column for column in columns if "pulse_rate_bpm" in column or "median_ibi_s" in column]
    return ["baseline_sbp", "baseline_dbp"] + sorted(columns)


def _scaled_pipeline(regressor: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def _candidate(name: str, settings: dict[str, Any], seed: int) -> tuple[Any, dict[str, list[Any]]]:
    if name == "hr_only":
        return _scaled_pipeline(LinearRegression()), {}
    if name == "ridge":
        return _scaled_pipeline(Ridge()), {"regressor__alpha": list(settings["ridge_alphas"])}
    if name == "elastic_net":
        return _scaled_pipeline(
            ElasticNet(max_iter=10000, tol=1e-2, selection="cyclic", random_state=seed)
        ), {
            "regressor__alpha": list(settings["elastic_net_alphas"]),
            "regressor__l1_ratio": list(settings["elastic_net_l1_ratios"]),
        }
    if name == "hist_gradient_boosting":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    HistGradientBoostingRegressor(max_iter=200, random_state=seed, l2_regularization=1.0),
                ),
            ]
        )
        return pipeline, {
            "regressor__learning_rate": list(settings["hist_gradient_boosting_learning_rates"]),
            "regressor__max_leaf_nodes": list(settings["hist_gradient_boosting_max_leaf_nodes"]),
            "regressor__min_samples_leaf": list(settings["hist_gradient_boosting_min_samples_leaf"]),
        }
    raise ValueError(f"Unknown model: {name}")


def _forward_chronological_splits(frame: pd.DataFrame, config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    maximum = int(config["models"]["inner_max_splits"])
    split_count = min(maximum, len(frame) - 1)
    if split_count < 2:
        raise ValueError("At least three development occasions are required for forward validation")
    return list(TimeSeriesSplit(n_splits=split_count).split(frame))


def _fit_single_subject_candidate(
    name: str,
    development: pd.DataFrame,
    target_column: str,
    config: dict[str, Any],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Any, list[str], dict[str, Any], list[dict[str, Any]]]:
    """Select hyperparameters using forward-only development folds, then fit development data."""
    feature_columns = _model_feature_columns(development, hr_only=name == "hr_only")
    base_estimator, grid = _candidate(name, config["models"], int(config["random_seed"]))
    candidates = list(ParameterGrid(grid)) if grid else [{}]
    scored: list[dict[str, Any]] = []
    for parameters in candidates:
        absolute_errors: list[float] = []
        validation_count = 0
        failed = False
        for train_index, validation_index in splits:
            estimator = clone(base_estimator).set_params(**parameters)
            train = development.iloc[train_index]
            validation = development.iloc[validation_index]
            try:
                estimator.fit(train[feature_columns], train[target_column])
                predicted = estimator.predict(validation[feature_columns])
            except (ValueError, FloatingPointError):
                failed = True
                break
            absolute_errors.extend(np.abs(predicted - validation[target_column].to_numpy(float)).tolist())
            validation_count += len(validation)
        if not failed and absolute_errors:
            scored.append(
                {
                    "target": target_column.removeprefix("delta_"),
                    "model": name,
                    "parameters": dict(parameters),
                    "cv_mae": float(np.mean(absolute_errors)),
                    "validation_count": int(validation_count),
                    "fold_count": len(splits),
                }
            )
    if not scored:
        raise ValueError(f"{name} could not be fitted in the forward chronological folds")
    scored.sort(key=lambda item: (item["cv_mae"], repr(sorted(item["parameters"].items()))))
    selected_parameters = scored[0]["parameters"]
    final_estimator = clone(base_estimator).set_params(**selected_parameters)
    final_estimator.fit(development[feature_columns], development[target_column])
    return final_estimator, feature_columns, selected_parameters, scored


def _single_subject_prediction_row(
    row: pd.Series,
    target: str,
    model: str,
    predicted_delta: float,
    selected_model: bool,
) -> dict[str, Any]:
    baseline = float(row[f"baseline_{target}"])
    true = float(row[f"true_{target}"])
    predicted = baseline + float(predicted_delta)
    return {
        "evaluation": "single_subject_development",
        "dataset_id": str(row["dataset_id"]),
        "participant_id": str(row["participant_id"]),
        "label_group_id": str(row["label_group_id"]),
        "calibration_label_group_id": str(row["calibration_label_group_id"]),
        "chronological_order": str(row["chronological_order"]),
        "target": target,
        "model": model,
        "selected_model": bool(selected_model),
        "baseline_bp": baseline,
        "true_delta": float(row[f"delta_{target}"]),
        "predicted_delta": float(predicted_delta),
        "true_bp": true,
        "predicted_bp": predicted,
        "error": predicted - true,
        "absolute_error": abs(predicted - true),
    }


def summarize_single_subject_predictions(
    predictions: pd.DataFrame,
    selected_model_names: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "single_subject_development": True,
        "population_validated": False,
        "targets": {},
    }
    target_passes: list[bool] = []
    for target in ("sbp", "dbp"):
        target_rows = predictions[predictions["target"] == target]
        models: dict[str, Any] = {}
        for model, group in target_rows.groupby("model", sort=True):
            values = _metrics(group["true_bp"].to_numpy(float), group["predicted_bp"].to_numpy(float))
            values["maximum_absolute_error"] = float(group["absolute_error"].max())
            models[str(model)] = values
        selected_name = selected_model_names[target]
        selected_mae = models[selected_name]["mae"]
        baseline_mae = models["zero_change"]["mae"]
        passed = bool(selected_mae < baseline_mae)
        target_passes.append(passed)
        result["targets"][target] = {
            "selected_model": selected_name,
            "selected_model_metrics": models[selected_name],
            "zero_change_metrics": models["zero_change"],
            "beats_zero_change": passed,
            "all_model_metrics": models,
        }
    result["passes_both_targets"] = bool(all(target_passes))
    return result


def evaluate_single_subject_models(
    development: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[str, Any, list[str], dict[str, Any]]],
    dict[str, Any],
]:
    """Select on forward development folds and evaluate once on the locked test set."""
    if development.empty or test.empty:
        raise ValueError("Development and test occasions must both be non-empty")
    development_ids = set(development["label_group_id"].astype(str))
    test_ids = set(test["label_group_id"].astype(str))
    if development_ids.intersection(test_ids):
        raise ValueError("Cuff-occasion leakage detected between development and test data")
    development = development.sort_values(["chronological_order", "label_group_id"]).reset_index(drop=True)
    test = test.sort_values(["chronological_order", "label_group_id"]).reset_index(drop=True)
    splits = _forward_chronological_splits(development, config)
    learned_names = ("hr_only", "ridge", "elastic_net")
    prediction_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    selected_models: dict[str, tuple[str, Any, list[str], dict[str, Any]]] = {}
    selected_names: dict[str, str] = {}

    for target in ("sbp", "dbp"):
        target_column = f"delta_{target}"
        zero_cv_errors: list[float] = []
        mean_cv_errors: list[float] = []
        validation_count = 0
        for train_index, validation_index in splits:
            train_delta = development.iloc[train_index][target_column].to_numpy(float)
            validation_delta = development.iloc[validation_index][target_column].to_numpy(float)
            zero_cv_errors.extend(np.abs(validation_delta).tolist())
            mean_cv_errors.extend(np.abs(validation_delta - float(np.mean(train_delta))).tolist())
            validation_count += len(validation_delta)
        cv_rows.extend(
            [
                {
                    "target": target,
                    "model": "zero_change",
                    "parameters": {},
                    "cv_mae": float(np.mean(zero_cv_errors)),
                    "validation_count": validation_count,
                    "fold_count": len(splits),
                },
                {
                    "target": target,
                    "model": "mean_change",
                    "parameters": {},
                    "cv_mae": float(np.mean(mean_cv_errors)),
                    "validation_count": validation_count,
                    "fold_count": len(splits),
                },
            ]
        )

        fitted: dict[str, tuple[Any, list[str], dict[str, Any], float]] = {}
        for name in learned_names:
            estimator, columns, parameters, scored = _fit_single_subject_candidate(
                name, development, target_column, config, splits
            )
            cv_rows.extend(scored)
            best_cv_mae = min(item["cv_mae"] for item in scored)
            fitted[name] = (estimator, columns, parameters, best_cv_mae)
        selected_name = min(learned_names, key=lambda name: (fitted[name][3], name))
        selected_names[target] = selected_name
        selected_estimator, selected_columns, selected_parameters, _ = fitted[selected_name]
        selected_models[target] = (
            selected_name,
            selected_estimator,
            selected_columns,
            selected_parameters,
        )

        baseline_delta = np.zeros(len(test), dtype=float)
        mean_delta = np.full(len(test), float(development[target_column].mean()))
        for row_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                _single_subject_prediction_row(row, target, "zero_change", baseline_delta[row_index], False)
            )
            prediction_rows.append(
                _single_subject_prediction_row(row, target, "mean_change", mean_delta[row_index], False)
            )
        for name in learned_names:
            estimator, columns, _, _ = fitted[name]
            predicted_delta = estimator.predict(test[columns])
            for row_index, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    _single_subject_prediction_row(
                        row,
                        target,
                        name,
                        float(predicted_delta[row_index]),
                        name == selected_name,
                    )
                )

    cv_results = pd.DataFrame(cv_rows)
    if not cv_results.empty:
        cv_results["parameters"] = cv_results["parameters"].map(
            lambda value: repr(dict(sorted(value.items())))
        )
        cv_results["selected_model"] = [
            row.model == selected_names[str(row.target)]
            and math.isclose(
                float(row.cv_mae),
                float(
                    min(
                        item["cv_mae"]
                        for item in cv_rows
                        if item["target"] == row.target and item["model"] == row.model
                    )
                ),
            )
            for row in cv_results.itertuples()
        ]
    predictions = pd.DataFrame(prediction_rows)
    metrics = summarize_single_subject_predictions(predictions, selected_names)
    return predictions, cv_results, selected_models, metrics


def _fit_candidate(
    name: str,
    frame: pd.DataFrame,
    target_column: str,
    config: dict[str, Any],
) -> tuple[Any, list[str], dict[str, Any]]:
    feature_columns = _model_feature_columns(frame, hr_only=name == "hr_only")
    estimator, grid = _candidate(name, config["models"], int(config["random_seed"]))
    groups = frame["participant_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two training participants are required")
    splits = min(int(config["models"]["inner_max_splits"]), len(unique_groups))
    combination_count = int(np.prod([len(values) for values in grid.values()])) if grid else 0
    if grid and combination_count == 1:
        fixed_parameters = {name: values[0] for name, values in grid.items()}
        estimator.set_params(**fixed_parameters)
        estimator.fit(frame[feature_columns], frame[target_column])
        return estimator, feature_columns, fixed_parameters
    if grid and splits >= 2:
        search = GridSearchCV(
            estimator,
            grid,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=splits),
            n_jobs=1,
            error_score="raise",
        )
        search.fit(frame[feature_columns], frame[target_column], groups=groups)
        return search.best_estimator_, feature_columns, search.best_params_
    estimator.fit(frame[feature_columns], frame[target_column])
    return estimator, feature_columns, {}


def _metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    errors = predicted - true
    correlation = None
    if len(true) >= 2 and np.std(true) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(true, predicted)[0, 1])
    return {
        "count": int(len(true)),
        "mae": float(mean_absolute_error(true, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(true, predicted))),
        "bias": float(np.mean(errors)),
        "error_std": float(np.std(errors, ddof=1)) if len(errors) >= 2 else 0.0,
        "correlation": correlation,
    }


def _bootstrap_mae_ci(frame: pd.DataFrame, iterations: int, seed: int) -> list[float] | None:
    working = frame.assign(_absolute_error=np.abs(frame["predicted_bp"] - frame["true_bp"]))
    participant_stats = working.groupby("participant_id")["_absolute_error"].agg(["sum", "count"])
    if len(participant_stats) < 2:
        return None
    generator = np.random.default_rng(seed)
    participant_count = len(participant_stats)
    sampled_indices = generator.integers(
        0, participant_count, size=(iterations, participant_count)
    )
    sums = participant_stats["sum"].to_numpy(dtype=float)[sampled_indices].sum(axis=1)
    counts = participant_stats["count"].to_numpy(dtype=float)[sampled_indices].sum(axis=1)
    scores = sums / counts
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def summarize_predictions(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"evaluations": {}, "scientific_feasibility": {}}
    if predictions.empty:
        return summary
    for (evaluation, target, model), group in predictions.groupby(["evaluation", "target", "model"], sort=True):
        values = _metrics(group["true_bp"].to_numpy(float), group["predicted_bp"].to_numpy(float))
        values["participant_bootstrap_mae_95_ci"] = _bootstrap_mae_ci(
            group,
            int(config["models"]["bootstrap_iterations"]),
            int(config["random_seed"]),
        )
        summary["evaluations"].setdefault(evaluation, {}).setdefault(target, {})[model] = values
    for evaluation, targets in summary["evaluations"].items():
        feasibility: dict[str, Any] = {}
        for target, models in targets.items():
            baseline = models.get("zero_change", {}).get("mae")
            learned = {name: value for name, value in models.items() if name not in {"zero_change", "mean_change"}}
            if learned:
                best_name = min(learned, key=lambda name: learned[name]["mae"])
                best_mae = learned[best_name]["mae"]
            else:
                best_name, best_mae = None, None
            feasibility[target] = {
                "best_learned_model": best_name,
                "best_learned_mae": best_mae,
                "zero_change_mae": baseline,
                "beats_zero_change": bool(best_mae is not None and baseline is not None and best_mae < baseline),
            }
        feasibility["passes_both_targets"] = bool(
            feasibility.get("sbp", {}).get("beats_zero_change")
            and feasibility.get("dbp", {}).get("beats_zero_change")
        )
        summary["scientific_feasibility"][evaluation] = feasibility
    return summary


def evaluate_personalized_models(
    examples: pd.DataFrame,
    config: dict[str, Any],
    evaluation_dataset: str,
    auxiliary_datasets: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, tuple[Any, list[str]]]]:
    evaluation = examples[examples["dataset_id"] == evaluation_dataset].copy()
    participants = sorted(evaluation["participant_id"].unique()) if not evaluation.empty else []
    if len(participants) < 3:
        return pd.DataFrame(), [], {}
    auxiliary = examples[examples["dataset_id"].isin(auxiliary_datasets)].copy()
    prediction_rows: list[dict[str, Any]] = []
    fold_parameters: list[dict[str, Any]] = []
    learned_models = ["hr_only", "ridge", "elastic_net", "hist_gradient_boosting"]
    for held_out in participants:
        test = evaluation[evaluation["participant_id"] == held_out].copy()
        train = pd.concat(
            [evaluation[evaluation["participant_id"] != held_out], auxiliary], ignore_index=True
        )
        if train["participant_group"].nunique() < 2:
            continue
        for target in ("sbp", "dbp"):
            target_column = f"delta_{target}"
            baseline_column = f"baseline_{target}"
            true_column = f"true_{target}"
            simple_predictions = {
                "zero_change": np.zeros(len(test), dtype=float),
                "mean_change": np.full(len(test), float(train[target_column].mean())),
            }
            for model_name, predicted_delta in simple_predictions.items():
                for row_index, (_, row) in enumerate(test.iterrows()):
                    prediction_rows.append(
                        _prediction_row(evaluation_dataset, held_out, row, target, model_name, predicted_delta[row_index])
                    )
            for model_name in learned_models:
                try:
                    estimator, feature_columns, parameters = _fit_candidate(model_name, train, target_column, config)
                    predicted = estimator.predict(test[feature_columns])
                    fold_parameters.append(
                        {
                            "evaluation": evaluation_dataset,
                            "held_out_participant": held_out,
                            "target": target,
                            "model": model_name,
                            "parameters": parameters,
                            "feature_count": len(feature_columns),
                        }
                    )
                    for row_index, (_, row) in enumerate(test.iterrows()):
                        prediction_rows.append(
                            _prediction_row(evaluation_dataset, held_out, row, target, model_name, float(predicted[row_index]))
                        )
                except (ValueError, FloatingPointError) as exc:
                    fold_parameters.append(
                        {
                            "evaluation": evaluation_dataset,
                            "held_out_participant": held_out,
                            "target": target,
                            "model": model_name,
                            "error": str(exc),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    final_models: dict[str, tuple[Any, list[str]]] = {}
    metrics = summarize_predictions(predictions, config)
    evaluation_metrics = metrics.get("evaluations", {}).get(evaluation_dataset, {})
    full_training = pd.concat([evaluation, auxiliary], ignore_index=True)
    for target in ("sbp", "dbp"):
        available = evaluation_metrics.get(target, {})
        learned = {name: value for name, value in available.items() if name in learned_models}
        if not learned:
            continue
        best_name = min(learned, key=lambda name: learned[name]["mae"])
        try:
            estimator, columns, parameters = _fit_candidate(best_name, full_training, f"delta_{target}", config)
        except ValueError:
            continue
        final_models[target] = (estimator, columns)
        fold_parameters.append(
            {
                "evaluation": evaluation_dataset,
                "held_out_participant": None,
                "target": target,
                "model": best_name,
                "parameters": parameters,
                "feature_count": len(columns),
                "final_fit": True,
            }
        )
    return predictions, fold_parameters, final_models


def _prediction_row(
    evaluation: str,
    participant: str,
    row: pd.Series,
    target: str,
    model: str,
    predicted_delta: float,
) -> dict[str, Any]:
    baseline = float(row[f"baseline_{target}"])
    true = float(row[f"true_{target}"])
    predicted = baseline + float(predicted_delta)
    return {
        "evaluation": evaluation,
        "dataset_id": row["dataset_id"],
        "participant_id": participant,
        "label_group_id": row["label_group_id"],
        "calibration_label_group_id": row["calibration_label_group_id"],
        "target": target,
        "model": model,
        "baseline_bp": baseline,
        "true_delta": float(row[f"delta_{target}"]),
        "predicted_delta": float(predicted_delta),
        "true_bp": true,
        "predicted_bp": predicted,
        "error": predicted - true,
        "absolute_error": abs(predicted - true),
    }


def evaluate_absolute_ppg_bp(
    occasions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Evaluate a fingertip source-domain comparator without personalization."""
    frame = occasions[
        (occasions["dataset_id"] == "ppg_bp") & (occasions["occasion_usable"] == True)  # noqa: E712
    ].copy()
    if frame["participant_id"].nunique() < 5:
        return pd.DataFrame()
    feature_columns = _numeric_feature_columns(frame)
    groups = frame["participant_id"].to_numpy(str)
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rows: list[dict[str, Any]] = []
    for fold_index, (train_index, test_index) in enumerate(folds.split(frame, groups=groups)):
        train, test = frame.iloc[train_index], frame.iloc[test_index]
        for target in ("sbp", "dbp"):
            mean_value = float(train[target].mean())
            ridge = _scaled_pipeline(Ridge(alpha=10.0))
            ridge.fit(train[feature_columns], train[target])
            predictions = {
                "population_mean": np.full(len(test), mean_value),
                "ridge_absolute": ridge.predict(test[feature_columns]),
            }
            for model, values in predictions.items():
                for position, (_, item) in enumerate(test.iterrows()):
                    predicted = float(values[position])
                    rows.append(
                        {
                            "evaluation": "ppg_bp_source_domain",
                            "dataset_id": "ppg_bp",
                            "participant_id": item["participant_id"],
                            "label_group_id": item["label_group_id"],
                            "calibration_label_group_id": None,
                            "target": target,
                            "model": model,
                            "baseline_bp": math.nan,
                            "true_delta": math.nan,
                            "predicted_delta": math.nan,
                            "true_bp": float(item[target]),
                            "predicted_bp": predicted,
                            "error": predicted - float(item[target]),
                            "absolute_error": abs(predicted - float(item[target])),
                            "fold": fold_index,
                        }
                    )
    return pd.DataFrame(rows)


def participant_learning_curve(
    examples: pd.DataFrame, target: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Deterministic Ridge learning curve using whole participants as units."""
    participants = sorted(examples["participant_id"].unique()) if not examples.empty else []
    if len(participants) < 4:
        return pd.DataFrame()
    feature_columns = _model_feature_columns(examples)
    rows: list[dict[str, Any]] = []
    target_column = f"delta_{target}"
    baseline_column = f"baseline_{target}"
    true_column = f"true_{target}"
    for count in sorted(set([2, max(2, len(participants) // 2), len(participants) - 1])):
        scores: list[float] = []
        for rotation in range(len(participants)):
            rotated = participants[rotation:] + participants[:rotation]
            training_ids = set(rotated[:count])
            testing_ids = set(rotated[count:])
            if not testing_ids:
                continue
            train = examples[examples["participant_id"].isin(training_ids)]
            test = examples[examples["participant_id"].isin(testing_ids)]
            if train.empty or test.empty:
                continue
            estimator = _scaled_pipeline(Ridge(alpha=10.0))
            try:
                estimator.fit(train[feature_columns], train[target_column])
                delta = estimator.predict(test[feature_columns])
            except ValueError:
                continue
            absolute = test[baseline_column].to_numpy(float) + delta
            scores.append(float(mean_absolute_error(test[true_column], absolute)))
        if scores:
            rows.append(
                {
                    "target": target,
                    "training_participant_count": count,
                    "mean_mae": float(np.mean(scores)),
                    "std_mae": float(np.std(scores)),
                    "repeat_count": len(scores),
                }
            )
    return pd.DataFrame(rows)
