"""Evaluation-only personalized BP comparison for feasibility occasion features.

The command performs participant-held-out evaluation and never saves a fitted
deployment model. Each held-out participant contributes only their first usable
calibration occasion; every later occasion remains unseen by the training fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bp_core.config import load_config
from bp_core.models import (
    build_personalized_examples,
    evaluate_personalized_models,
    summarize_predictions,
)


REQUIRED_COLUMNS = {
    "dataset_id",
    "participant_id",
    "session_id",
    "label_group_id",
    "chronological_order",
    "sbp",
    "dbp",
    "calibration_occasion",
    "occasion_usable",
    "quality_policy",
}
LEARNED_MODELS = {"hr_only", "ridge", "elastic_net", "hist_gradient_boosting"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/bp_pipeline_v1.json")
    parser.add_argument("--occasion-features", help="Run one evaluation from one feature table")
    parser.add_argument("--ah-occasion-features", help="AH table for the three-way comparison suite")
    parser.add_argument("--p001-occasion-features", help="P001 table for the three-way comparison suite")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-participants", type=int, default=3)
    return parser.parse_args(argv)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _recording_role(label_group_id: str) -> str:
    trial = str(label_group_id).rsplit(":", 1)[-1].lower()
    if "recovery" in trial:
        return "recovery"
    if "stationary" in trial:
        return "stationary"
    if "calibration" in trial:
        return "calibration"
    return "other"


def prepare_evaluation(
    occasions: pd.DataFrame,
    *,
    minimum_participants: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing = sorted(REQUIRED_COLUMNS.difference(occasions.columns))
    if missing:
        raise ValueError(f"Occasion feature table is missing required columns: {', '.join(missing)}")
    if occasions["label_group_id"].astype(str).duplicated().any():
        duplicates = occasions.loc[
            occasions["label_group_id"].astype(str).duplicated(keep=False), "label_group_id"
        ].astype(str).unique().tolist()
        raise ValueError(f"Duplicate cuff occasions would cause leakage: {', '.join(sorted(duplicates))}")
    policies = sorted(occasions["quality_policy"].dropna().astype(str).unique())
    if len(policies) != 1:
        raise ValueError(f"Expected exactly one frozen feasibility policy, found: {policies}")

    working = occasions.copy()
    working["occasion_usable"] = working["occasion_usable"].map(_as_bool)
    working["calibration_occasion"] = working["calibration_occasion"].map(_as_bool)
    chronological = pd.to_datetime(
        working["chronological_order"], errors="coerce", utc=True
    )
    invalid_usable = working["occasion_usable"] & chronological.isna()
    if invalid_usable.any():
        invalid = working.loc[invalid_usable, "label_group_id"].astype(str).tolist()
        raise ValueError(f"Invalid chronological timestamps: {', '.join(invalid)}")
    working["_chronological_timestamp"] = chronological

    usable = working[working["occasion_usable"]].copy()
    usable["chronological_order"] = usable["_chronological_timestamp"].map(
        lambda value: value.isoformat()
    )
    usable = usable.drop(columns=["_chronological_timestamp"])
    participants = sorted(usable["participant_id"].astype(str).unique())
    if len(participants) < minimum_participants:
        raise ValueError(
            f"Participant-held-out evaluation requires at least {minimum_participants} usable participants; "
            f"found {len(participants)}"
        )

    calibration_rows: list[dict[str, Any]] = []
    normalized_groups: list[pd.DataFrame] = []
    for participant, group in usable.groupby("participant_id", sort=True):
        ordered = group.sort_values(["chronological_order", "label_group_id"]).copy()
        marked = ordered[ordered["calibration_occasion"]]
        calibration = marked.iloc[0] if not marked.empty else ordered.iloc[0]
        later = ordered[ordered["chronological_order"] > calibration["chronological_order"]]
        if later.empty:
            raise ValueError(f"Participant {participant} has no usable post-calibration occasion")
        ordered["calibration_occasion"] = False
        ordered.loc[calibration.name, "calibration_occasion"] = True
        normalized_groups.append(ordered)
        calibration_rows.append(
            {
                "participant_id": str(participant),
                "calibration_label_group_id": str(calibration["label_group_id"]),
                "calibration_chronological_order": str(calibration["chronological_order"]),
                "calibration_sbp": float(calibration["sbp"]),
                "calibration_dbp": float(calibration["dbp"]),
                "selection": "first_explicit_usable" if not marked.empty else "first_chronological_usable",
                "post_calibration_occasion_count": int(len(later)),
            }
        )
    normalized = pd.concat(normalized_groups, ignore_index=True)
    examples = build_personalized_examples(normalized)
    if examples.empty:
        raise ValueError("No personalized post-calibration examples were created")
    calibration_ids = {row["calibration_label_group_id"] for row in calibration_rows}
    if calibration_ids.intersection(set(examples["label_group_id"].astype(str))):
        raise ValueError("Calibration occasion leakage detected in prediction examples")
    if examples["label_group_id"].astype(str).duplicated().any():
        raise ValueError("A cuff occasion appears more than once in personalized examples")

    lookup = normalized.set_index("label_group_id")
    examples["recording_role"] = examples["label_group_id"].map(
        lambda value: _recording_role(str(value))
    )
    examples["gap_tolerant"] = examples["label_group_id"].map(
        lambda value: _as_bool(lookup.loc[value].get("gap_tolerant", False))
    )
    manifest = {
        "quality_policy": policies[0],
        "participant_count": len(participants),
        "participants": participants,
        "usable_occasion_count": int(len(usable)),
        "personalized_example_count": int(len(examples)),
        "calibrations": calibration_rows,
    }
    return examples, pd.DataFrame(calibration_rows), manifest


def _metrics(frame: pd.DataFrame, *, participant_balanced: bool = False) -> dict[str, Any]:
    errors = frame["predicted_bp"].to_numpy(float) - frame["true_bp"].to_numpy(float)
    if participant_balanced:
        counts = frame.groupby("participant_id")["participant_id"].transform("size").to_numpy(float)
        participant_count = int(frame["participant_id"].nunique())
        weights = 1.0 / (counts * participant_count)
    else:
        participant_count = int(frame["participant_id"].nunique())
        weights = np.full(len(frame), 1.0 / len(frame))
    correlation = None
    if len(frame) >= 2 and frame["true_bp"].std(ddof=0) > 0 and frame["predicted_bp"].std(ddof=0) > 0:
        correlation = float(frame[["true_bp", "predicted_bp"]].corr().iloc[0, 1])
    bias = float(np.sum(weights * errors))
    return {
        "count": int(len(frame)),
        "participant_count": participant_count,
        "mae": float(np.sum(weights * np.abs(errors))),
        "rmse": float(np.sqrt(np.sum(weights * errors ** 2))),
        "bias": bias,
        "error_std": float(np.sqrt(np.sum(weights * (errors - bias) ** 2))),
        "maximum_absolute_error": float(np.max(np.abs(errors))),
        "correlation": correlation,
    }


def detailed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groupings = [
        ("overall", ["target", "model"]),
        ("recording_role", ["target", "model", "recording_role"]),
    ]
    for aggregation, balanced in (("occasion_weighted", False), ("participant_balanced", True)):
        for scope, columns in groupings:
            for keys, group in predictions.groupby(columns, sort=True):
                keys = keys if isinstance(keys, tuple) else (keys,)
                row = {column: value for column, value in zip(columns, keys)}
                row.update(scope=scope, aggregation=aggregation, **_metrics(group, participant_balanced=balanced))
                rows.append(row)
    return pd.DataFrame(rows)


def metric_decision(metrics: pd.DataFrame, aggregation: str) -> dict[str, Any]:
    result: dict[str, Any] = {"aggregation": aggregation, "targets": {}}
    passes: list[bool] = []
    overall = metrics[(metrics["scope"] == "overall") & (metrics["aggregation"] == aggregation)]
    for target in ("sbp", "dbp"):
        target_rows = overall[overall["target"] == target].set_index("model")
        if "zero_change" not in target_rows.index:
            raise ValueError(f"Missing zero-change predictions for {target}")
        learned = target_rows.loc[target_rows.index.intersection(sorted(LEARNED_MODELS))]
        if learned.empty:
            raise ValueError(f"No learned-model predictions for {target}")
        selected = str(learned["mae"].astype(float).idxmin())
        selected_mae = float(learned.loc[selected, "mae"])
        zero_mae = float(target_rows.loc["zero_change", "mae"])
        passed = selected_mae < zero_mae
        passes.append(passed)
        result["targets"][target] = {
            "best_learned_model": selected,
            "best_learned_mae": selected_mae,
            "zero_change_mae": zero_mae,
            "beats_zero_change": passed,
        }
    result["passes_both_targets"] = bool(all(passes))
    return result


def balanced_p001_combination(
    ah_occasions: pd.DataFrame,
    p001_occasions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cap P001 to a protocol-matched follow-up count based on AH participants."""
    ah_examples, _, ah_manifest = prepare_evaluation(ah_occasions)
    p001_examples, p001_calibrations, _ = prepare_evaluation(p001_occasions, minimum_participants=1)
    ah_counts = ah_examples.groupby("participant_id").size()
    cap = max(1, int(round(float(ah_counts.median()))))
    ordered = p001_examples.sort_values(["chronological_order", "label_group_id"]).copy()
    selected: list[str] = []
    # Prefer the same two protocol roles used by the AH study before adding any
    # other chronological P001 occasion.
    for role in ("stationary", "recovery", "other"):
        candidates = ordered[ordered["recording_role"] == role]
        if not candidates.empty and len(selected) < cap:
            selected.append(str(candidates.iloc[0]["label_group_id"]))
    for label_group_id in ordered["label_group_id"].astype(str):
        if len(selected) >= cap:
            break
        if label_group_id not in selected:
            selected.append(label_group_id)
    calibration_id = str(p001_calibrations.iloc[0]["calibration_label_group_id"])
    selected_ids = {calibration_id, *selected}
    balanced_p001 = p001_occasions[
        p001_occasions["label_group_id"].astype(str).isin(selected_ids)
    ].copy()
    combined = pd.concat([ah_occasions, balanced_p001], ignore_index=True, sort=False)
    return combined, {
        "method": "cap_P001_to_median_AH_post_calibration_count",
        "ah_post_calibration_counts": {str(key): int(value) for key, value in ah_counts.items()},
        "p001_followup_cap": cap,
        "p001_calibration_id": calibration_id,
        "selected_p001_followup_ids": selected,
        "participant_count": ah_manifest["participant_count"] + 1,
    }


def run_evaluation(
    occasions: pd.DataFrame,
    *,
    config: dict[str, Any],
    config_path: Path,
    output: Path,
    experiment_name: str,
    input_sources: list[Path],
    minimum_participants: int,
    balancing_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    examples, calibrations, manifest = prepare_evaluation(
        occasions, minimum_participants=minimum_participants
    )
    predictions, fold_parameters, final_models = evaluate_personalized_models(
        examples,
        config,
        "local_upper_arm",
        fit_final_models=False,
    )
    if final_models:
        raise RuntimeError("Evaluation-only command unexpectedly fitted final models")
    if predictions.empty:
        raise RuntimeError("Participant-held-out evaluation produced no predictions")

    context = examples[["participant_id", "label_group_id", "chronological_order", "recording_role", "gap_tolerant"]]
    predictions = predictions.merge(
        context, on=["participant_id", "label_group_id"], how="left", validate="many_to_one"
    )
    metrics = detailed_metrics(predictions)
    legacy_summary = summarize_predictions(predictions, config)
    occasion_decision = metric_decision(metrics, "occasion_weighted")
    participant_decision = metric_decision(metrics, "participant_balanced")
    summary = {
        "experiment": experiment_name,
        "legacy_occasion_weighted_summary": legacy_summary,
        "occasion_weighted_decision": occasion_decision,
        "participant_balanced_decision": participant_decision,
    }
    summary.update(
        {
            "evaluation_only": True,
            "development_pilot": True,
            "population_validated": False,
            "medical_device_validation": False,
            "saved_deployment_model": False,
            "quality_policy": manifest["quality_policy"],
            "input_occasion_count": int(len(occasions)),
            "usable_occasion_count": manifest["usable_occasion_count"],
            "participant_count": manifest["participant_count"],
            "personalized_example_count": manifest["personalized_example_count"],
            "passes_both_targets_against_zero_change": bool(
                participant_decision["passes_both_targets"]
            ),
            "primary_decision_metric": "participant_balanced_mae",
            "balancing_manifest": balancing_manifest,
            "warning": (
                "These previously examined feasibility recordings are development evidence only. "
                "Freeze any selected model before testing on new untouched multi-day occasions."
            ),
        }
    )

    split_manifest = {
        "method": "leave_one_participant_out",
        "quality_policy": manifest["quality_policy"],
        "calibrations": manifest["calibrations"],
        "folds": [
            {
                "held_out_participant": participant,
                "training_participants": [
                    item for item in manifest["participants"] if item != participant
                ],
                "test_occasion_ids": examples.loc[
                    examples["participant_id"].astype(str) == participant, "label_group_id"
                ].astype(str).tolist(),
            }
            for participant in manifest["participants"]
        ],
        "calibration_rows_excluded_from_metrics": True,
        "overlapping_windows_used_as_independent_labels": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output / "personalized_examples.csv", index=False)
    calibrations.to_csv(output / "calibration_manifest.csv", index=False)
    predictions.to_csv(output / "participant_held_out_predictions.csv", index=False)
    pd.DataFrame(fold_parameters).to_csv(output / "fold_parameters.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "experiment": experiment_name,
        "inputs": [str(path.resolve()) for path in input_sources],
        "config": str(Path(config_path).resolve()),
        "quality_policy": manifest["quality_policy"],
        "evaluation_only": True,
        "final_model_fitted": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"[{experiment_name}] Participants={manifest['participant_count']}, "
        f"usable_occasions={manifest['usable_occasion_count']}, "
        f"post_calibration_examples={manifest['personalized_example_count']}"
    )
    for target in ("sbp", "dbp"):
        result = participant_decision["targets"][target]
        print(
            f"- {target.upper()} participant-balanced: best={result.get('best_learned_model')}, "
            f"MAE={result.get('best_learned_mae')}, "
            f"zero_change_MAE={result.get('zero_change_mae')}, "
            f"beats_zero_change={result.get('beats_zero_change')}"
        )
    print(f"Passes both targets: {summary['passes_both_targets_against_zero_change']}")
    print(f"Results: {output.resolve()}")
    return summary


def _comparison_rows(summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment, summary in summaries.items():
        for aggregation_key in ("occasion_weighted_decision", "participant_balanced_decision"):
            decision = summary[aggregation_key]
            for target, result in decision["targets"].items():
                rows.append(
                    {
                        "experiment": experiment,
                        "aggregation": decision["aggregation"],
                        "target": target,
                        "participant_count": summary["participant_count"],
                        "usable_occasion_count": summary["usable_occasion_count"],
                        "personalized_example_count": summary["personalized_example_count"],
                        **result,
                        "passes_both_targets": decision["passes_both_targets"],
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_path = load_config(args.config)
    output = Path(args.output_dir)
    comparison_mode = bool(args.ah_occasion_features or args.p001_occasion_features)
    if comparison_mode:
        if args.occasion_features or not (args.ah_occasion_features and args.p001_occasion_features):
            raise ValueError(
                "Use either --occasion-features, or both --ah-occasion-features and "
                "--p001-occasion-features"
            )
        ah_path = Path(args.ah_occasion_features)
        p001_path = Path(args.p001_occasion_features)
        ah = pd.read_csv(ah_path)
        p001 = pd.read_csv(p001_path)
        combined_all = pd.concat([ah, p001], ignore_index=True, sort=False)
        combined_balanced, balancing = balanced_p001_combination(ah, p001)
        experiments = {
            "ah_only": (ah, None),
            "combined_all": (combined_all, {"method": "all_compatible_P001_occasions"}),
            "combined_balanced": (combined_balanced, balancing),
        }
        summaries: dict[str, dict[str, Any]] = {}
        for name, (table, balance_manifest) in experiments.items():
            summaries[name] = run_evaluation(
                table,
                config=config,
                config_path=config_path,
                output=output / name,
                experiment_name=name,
                input_sources=[ah_path] if name == "ah_only" else [ah_path, p001_path],
                minimum_participants=args.minimum_participants,
                balancing_manifest=balance_manifest,
            )
        comparison = _comparison_rows(summaries)
        output.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output / "comparison_summary.csv", index=False)
        (output / "comparison_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Comparison summary: {(output / 'comparison_summary.csv').resolve()}")
        return 0

    if not args.occasion_features:
        raise ValueError(
            "Provide --occasion-features for one evaluation, or both comparison-suite inputs"
        )
    input_path = Path(args.occasion_features)
    occasions = pd.read_csv(input_path)
    run_evaluation(
        occasions,
        config=config,
        config_path=config_path,
        output=output,
        experiment_name="single_table",
        input_sources=[input_path],
        minimum_participants=args.minimum_participants,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
