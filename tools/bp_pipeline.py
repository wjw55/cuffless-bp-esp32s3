"""Reproducible offline PPG-to-BP research pipeline.

This command never opens a serial port and never changes firmware or raw data.
It audits source datasets, extracts device-resistant morphology features, and
evaluates one-time-calibrated SBP/DBP baselines with participant-level splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
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

from bp_core.config import load_config, resolve_path
from bp_core.datasets import audit_datasets, discover_recordings
from bp_core.features import build_occasion_features
from bp_core.inference import ModelCompatibilityError, load_model_bundle, predict_frame
from bp_core.models import (
    build_personalized_examples,
    evaluate_absolute_ppg_bp,
    evaluate_personalized_models,
    evaluate_single_subject_models,
    participant_learning_curve,
    prepare_single_subject_split,
    summarize_predictions,
)
from bp_core.reporting import (
    plot_bp_distribution,
    plot_learning_curve,
    plot_prediction_diagnostics,
    plot_single_subject_diagnostics,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PPG datasets and run leakage-safe personalized BP baselines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit configured datasets and write a canonical manifest")
    audit.add_argument("--config", default="config/bp_pipeline_v1.json")
    audit.add_argument("--output-dir", help="Override the default data/processed/bp/audit directory")

    run = subparsers.add_parser("run", help="Audit, extract features, train, and evaluate")
    run.add_argument("--config", default="config/bp_pipeline_v1.json")
    run.add_argument("--run-id", help="Stable output folder name; defaults to a timestamp")
    run.add_argument("--overwrite", action="store_true", help="Replace an existing run folder")
    run.add_argument("--resume", action="store_true", help="Reuse saved features in an interrupted run folder")
    run.add_argument("--no-plots", action="store_true", help="Skip diagnostic plot generation")

    evaluate = subparsers.add_parser("evaluate", help="Rebuild metrics and plots from saved predictions")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--no-plots", action="store_true")

    single = subparsers.add_parser(
        "single-subject",
        help="Run a retrospective chronological development evaluation for one local participant",
    )
    single.add_argument("--config", default="config/bp_pipeline_v1.json")
    single.add_argument("--participant-id", required=True)
    single.add_argument("--run-dir", required=True, help="Existing BP run containing occasion_features.csv")
    single.add_argument("--test-fraction", type=float, default=0.30)
    single.add_argument("--minimum-test-occasions", type=int, default=5)
    single.add_argument("--minimum-development-occasions", type=int, default=5)
    single.add_argument("--no-plots", action="store_true")

    predict = subparsers.add_parser("predict", help="Predict experimental BP from one saved PPG recording")
    predict.add_argument("--config", default="config/bp_pipeline_v1.json")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--ppg-csv", required=True)
    predict.add_argument("--metadata-json", required=True)
    predict.add_argument("--allow-unvalidated", action="store_true")
    predict.add_argument("--output", help="Optional JSON result path")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _save_audit(
    output_dir: Path,
    config: dict[str, Any],
    recordings: list[Any],
    statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_datasets(config, recordings, statuses)
    write_json(output_dir / "dataset_audit.json", audit)
    pd.DataFrame(statuses).to_csv(output_dir / "dataset_status.csv", index=False)
    manifest = pd.DataFrame([recording.as_manifest_row() for recording in recordings])
    manifest.to_csv(output_dir / "canonical_manifest.csv", index=False)
    return audit


def command_audit(args: argparse.Namespace) -> int:
    config, _ = load_config(args.config)
    recordings, statuses = discover_recordings(config)
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_root"]) / "audit"
    audit = _save_audit(output_dir, config, recordings, statuses)
    print(f"Dataset audit: {output_dir / 'dataset_audit.json'}")
    for item in audit["datasets"]:
        print(
            f"- {item['dataset_id']}: {item['status']}, "
            f"participants={item.get('participant_count', 0)}, recordings={item.get('recording_count', 0)}"
        )
    return 1 if any(item["status"] in {"missing_required", "error"} for item in audit["datasets"]) else 0


def _prepare_run_dir(root: Path, run_id: str, overwrite: bool, resume: bool = False) -> Path:
    run_dir = root / run_id
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if run_dir.exists():
        if resume:
            return run_dir
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite to replace it.")
        resolved_root = root.resolve()
        resolved_run = run_dir.resolve()
        if resolved_run.parent != resolved_root:
            raise ValueError(f"Refusing to replace unsafe run path: {resolved_run}")
        shutil.rmtree(resolved_run)
    run_dir.mkdir(parents=True)
    return run_dir


def _split_manifest(examples: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"method": "leave_one_participant_out", "evaluations": {}}
    if examples.empty:
        return result
    for dataset, group in examples.groupby("dataset_id"):
        result["evaluations"][dataset] = {
            "participants": sorted(group["participant_id"].unique().tolist()),
            "folds": [
                {
                    "held_out_participant": participant,
                    "test_label_groups": sorted(
                        group[group["participant_id"] == participant]["label_group_id"].tolist()
                    ),
                }
                for participant in sorted(group["participant_id"].unique())
            ],
        }
    return result


def _run_manifest(config_path: Path, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_git_commit": _project_commit(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "random_seed": None,
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "research_only": True,
        "medical_device_validation": False,
        "firmware_modified": False,
    }


def command_run(args: argparse.Namespace) -> int:
    config, config_path = load_config(args.config)
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    output_root = resolve_path(config["output_root"])
    run_dir = _prepare_run_dir(output_root, run_id, args.overwrite, args.resume)
    shutil.copy2(config_path, run_dir / "config_snapshot.json")

    recordings, statuses = discover_recordings(config)
    audit = _save_audit(run_dir, config, recordings, statuses)
    if not recordings:
        write_json(run_dir / "run_manifest.json", _run_manifest(config_path, run_id))
        print(f"No labelled recordings were discovered. Audit saved to {run_dir}", file=sys.stderr)
        return 2

    segment_path = run_dir / "segment_features.csv"
    occasion_path = run_dir / "occasion_features.csv"
    error_path = run_dir / "extraction_errors.csv"
    if args.resume and segment_path.exists() and occasion_path.exists():
        print("Reusing saved segment and occasion features...", flush=True)
        segments = pd.read_csv(segment_path)
        occasions = pd.read_csv(occasion_path)
        extraction_errors = pd.read_csv(error_path).to_dict("records") if error_path.exists() else []
    else:
        print(f"Discovered {len(recordings)} labelled recordings; extracting morphology features...", flush=True)
        segments, occasions, extraction_errors = build_occasion_features(recordings, config)
        segments.to_csv(segment_path, index=False)
        occasions.to_csv(occasion_path, index=False)
        pd.DataFrame(extraction_errors, columns=["dataset_id", "participant_id", "recording_id", "error"]).to_csv(
            error_path, index=False
        )

    examples = build_personalized_examples(occasions)
    examples.to_csv(run_dir / "personalized_examples.csv", index=False)
    write_json(run_dir / "splits.json", _split_manifest(examples))

    prediction_frames: list[pd.DataFrame] = []
    all_parameters: list[dict[str, Any]] = []
    model_manifest: dict[str, Any] = {}
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Built {len(occasions)} cuff occasions and {len(examples)} personalized examples; evaluating models...", flush=True)
    one_month_predictions, parameters, fitted = evaluate_personalized_models(
        examples, config, "one_month_wrist"
    )
    if not one_month_predictions.empty:
        prediction_frames.append(one_month_predictions)
    all_parameters.extend(parameters)
    _save_models(models_dir, "one_month_wrist", fitted, model_manifest)

    local_predictions, parameters, fitted = evaluate_personalized_models(
        examples,
        config,
        "local_upper_arm",
        auxiliary_datasets=("one_month_wrist",),
    )
    if not local_predictions.empty:
        prediction_frames.append(local_predictions)
    all_parameters.extend(parameters)
    _save_models(models_dir, "local_upper_arm", fitted, model_manifest)

    ppg_bp_predictions = evaluate_absolute_ppg_bp(occasions, config)
    if not ppg_bp_predictions.empty:
        prediction_frames.append(ppg_bp_predictions)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "model_parameters.json", all_parameters)
    write_json(run_dir / "model_manifest.json", model_manifest)

    metrics = summarize_predictions(predictions, config)
    metrics["audit_status"] = {item["dataset_id"]: item["status"] for item in audit["datasets"]}
    metrics["feature_extraction"] = {
        "recording_count": len(recordings),
        "segment_count": len(segments),
        "accepted_segment_count": int(segments["accepted"].sum()) if not segments.empty else 0,
        "occasion_count": len(occasions),
        "usable_occasion_count": int(occasions["occasion_usable"].sum()) if not occasions.empty else 0,
        "error_count": len(extraction_errors),
    }
    write_json(run_dir / "metrics.json", metrics)

    learning_frames: list[pd.DataFrame] = []
    for dataset in ("one_month_wrist", "local_upper_arm"):
        subset = examples[examples["dataset_id"] == dataset] if not examples.empty else examples
        for target in ("sbp", "dbp"):
            curve = participant_learning_curve(subset, target, config)
            if not curve.empty:
                curve.insert(0, "evaluation", dataset)
                learning_frames.append(curve)
    learning = pd.concat(learning_frames, ignore_index=True) if learning_frames else pd.DataFrame()
    learning.to_csv(run_dir / "learning_curve.csv", index=False)

    manifest = _run_manifest(config_path, run_id)
    manifest["random_seed"] = int(config["random_seed"])
    manifest["counts"] = metrics["feature_extraction"]
    write_json(run_dir / "run_manifest.json", manifest)

    if not args.no_plots:
        print("Generating diagnostic plots...", flush=True)
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_bp_distribution(occasions, plots_dir)
        plot_prediction_diagnostics(predictions, metrics, plots_dir)
        for evaluation, curve in learning.groupby("evaluation") if not learning.empty else []:
            plot_learning_curve(curve, plots_dir, str(evaluation))

    print(f"BP pipeline run: {run_dir}")
    print(
        f"Recordings={len(recordings)}, accepted_segments={metrics['feature_extraction']['accepted_segment_count']}, "
        f"usable_occasions={metrics['feature_extraction']['usable_occasion_count']}"
    )
    for evaluation, result in metrics.get("scientific_feasibility", {}).items():
        print(f"- {evaluation}: beats zero-change for SBP and DBP = {result['passes_both_targets']}")
    if extraction_errors:
        print(f"Warning: {len(extraction_errors)} recordings failed extraction; see extraction_errors.csv")
    return 0


def _save_models(
    models_dir: Path,
    evaluation: str,
    fitted: dict[str, tuple[Any, list[str]]],
    manifest: dict[str, Any],
) -> None:
    for target, (estimator, columns) in fitted.items():
        filename = f"{evaluation}_{target}.joblib"
        joblib.dump(
            {
                "estimator": estimator,
                "feature_columns": columns,
                "target": target,
                "evaluation_domain": evaluation,
                "research_only": True,
            },
            models_dir / filename,
        )
        manifest[f"{evaluation}:{target}"] = {
            "file": f"models/{filename}",
            "feature_count": len(columns),
            "research_only": True,
        }


def command_evaluate(args: argparse.Namespace) -> int:
    run_dir = resolve_path(args.run_dir)
    config_path = run_dir / "config_snapshot.json"
    predictions_path = run_dir / "predictions.csv"
    if not config_path.exists() or not predictions_path.exists():
        print("ERROR: run directory must contain config_snapshot.json and predictions.csv", file=sys.stderr)
        return 2
    config, _ = load_config(config_path)
    predictions = pd.read_csv(predictions_path)
    metrics = summarize_predictions(predictions, config)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("audit_status", "feature_extraction"):
            if key in previous:
                metrics[key] = previous[key]
    write_json(metrics_path, metrics)
    if not args.no_plots:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_prediction_diagnostics(predictions, metrics, plots_dir)
        occasions_path = run_dir / "occasion_features.csv"
        if occasions_path.exists():
            plot_bp_distribution(pd.read_csv(occasions_path), plots_dir)
    print(f"Rebuilt metrics: {run_dir / 'metrics.json'}")
    return 0


def _safe_participant_folder(participant_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not participant_id or participant_id in {".", ".."} or any(char not in allowed for char in participant_id):
        raise ValueError("participant ID may contain only letters, numbers, period, underscore, and hyphen")
    return participant_id


def command_single_subject(args: argparse.Namespace) -> int:
    config, config_path = load_config(args.config)
    run_dir = resolve_path(args.run_dir)
    occasions_path = run_dir / "occasion_features.csv"
    if not occasions_path.exists():
        raise FileNotFoundError(f"Existing run does not contain occasion_features.csv: {run_dir}")
    participant_folder = _safe_participant_folder(str(args.participant_id))
    occasions = pd.read_csv(occasions_path)
    calibration, development, test, split = prepare_single_subject_split(
        occasions,
        str(args.participant_id),
        test_fraction=float(args.test_fraction),
        minimum_test_occasions=int(args.minimum_test_occasions),
        minimum_development_occasions=int(args.minimum_development_occasions),
    )

    predictions, cv_results, selected_models, metrics = evaluate_single_subject_models(
        development, test, config
    )
    output_dir = run_dir / "single_subject" / participant_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(output_dir / "development_examples.csv", index=False)
    test.to_csv(output_dir / "locked_test_examples.csv", index=False)
    predictions.to_csv(output_dir / "locked_test_predictions.csv", index=False)
    cv_results.to_csv(output_dir / "development_cv_results.csv", index=False)
    write_json(output_dir / "split.json", split)

    selected_names = {target: values[0] for target, values in selected_models.items()}
    metrics.update(
        {
            "participant_id": str(args.participant_id),
            "dataset_id": "local_upper_arm",
            "calibration": {
                "label_group_id": str(calibration["label_group_id"]),
                "chronological_order": str(calibration["chronological_order"]),
                "sbp": float(calibration["sbp"]),
                "dbp": float(calibration["dbp"]),
            },
            "split": split,
            "model_selection_data": "development_forward_chaining_only",
            "locked_test_used_for_model_selection": False,
            "research_only": True,
            "medical_device_validation": False,
        }
    )
    write_json(output_dir / "metrics.json", metrics)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    calibration_features = {
        str(column): float(calibration[column])
        for column in calibration.index
        if str(column).startswith(("median__feature__", "iqr__feature__"))
        and pd.notna(calibration[column])
    }
    # Write the snapshot before hashing it into the deployment manifest.
    shutil.copy2(config_path, output_dir / "config_snapshot.json")
    model_manifest: dict[str, Any] = {
        "schema_version": 1,
        "single_subject_development": True,
        "population_validated": False,
        "participant_id": str(args.participant_id),
        "trained_on_locked_test": False,
        "viewer_eligible": bool(metrics["passes_both_targets"]),
        "config_sha256": _sha256(output_dir / "config_snapshot.json"),
        "calibration": {
            "label_group_id": str(calibration["label_group_id"]),
            "chronological_order": str(calibration["chronological_order"]),
            "sbp": float(calibration["sbp"]),
            "dbp": float(calibration["dbp"]),
            "features": calibration_features,
        },
        "passes_both_targets": bool(metrics["passes_both_targets"]),
        "models": {},
    }
    for target, (model_name, estimator, columns, parameters) in selected_models.items():
        filename = f"single_subject_{target}.joblib"
        target_passed = bool(metrics["targets"][target]["beats_zero_change"])
        joblib.dump(
            {
                "estimator": estimator,
                "feature_columns": columns,
                "target": target,
                "model": model_name,
                "parameters": parameters,
                "participant_id": str(args.participant_id),
                "calibration_label_group_id": str(calibration["label_group_id"]),
                "calibration_bp": float(calibration[target]),
                "calibration_features": calibration_features,
                "development_occasion_ids": split["development_occasion_ids"],
                "locked_test_occasion_ids": split["test_occasion_ids"],
                "trained_on_locked_test": False,
                "single_subject_development": True,
                "population_validated": False,
                "beats_zero_change_on_locked_test": target_passed,
                "research_only": True,
                "model_manifest_schema_version": 1,
                "config_sha256": model_manifest["config_sha256"],
            },
            models_dir / filename,
        )
        model_manifest["models"][target] = {
            "file": f"models/{filename}",
            "model": model_name,
            "parameters": parameters,
            "feature_count": len(columns),
            "feature_columns": list(columns),
            "beats_zero_change_on_locked_test": target_passed,
        }
    write_json(output_dir / "model_manifest.json", model_manifest)

    if not args.no_plots:
        plot_single_subject_diagnostics(predictions, selected_names, output_dir / "plots")

    print(f"Single-subject development evaluation: {output_dir}")
    print(
        f"Participant={args.participant_id}, calibration=1, "
        f"development={len(development)}, locked_test={len(test)}"
    )
    for target in ("sbp", "dbp"):
        target_metrics = metrics["targets"][target]
        selected = target_metrics["selected_model"]
        selected_mae = target_metrics["selected_model_metrics"]["mae"]
        baseline_mae = target_metrics["zero_change_metrics"]["mae"]
        print(
            f"- {target.upper()}: selected={selected}, MAE={selected_mae:.2f}, "
            f"zero_change_MAE={baseline_mae:.2f}, "
            f"beats_zero_change={target_metrics['beats_zero_change']}"
        )
    print(f"Passes both targets: {metrics['passes_both_targets']}")
    print("Research-only retrospective result; population_validated=False")
    return 0


def command_predict(args: argparse.Namespace) -> int:
    config, config_path = load_config(args.config)
    bundle = load_model_bundle(
        resolve_path(args.model_dir),
        allow_unvalidated=bool(args.allow_unvalidated),
    )
    if _sha256(config_path) != _sha256(bundle.model_dir / "config_snapshot.json"):
        raise ModelCompatibilityError("--config does not match the model's frozen configuration")
    ppg_path = resolve_path(args.ppg_csv)
    metadata_path = resolve_path(args.metadata_json)
    if not ppg_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("PPG CSV and metadata JSON must both exist")
    frame = pd.read_csv(ppg_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recording_participant = str(metadata.get("subject_id", ""))
    if recording_participant and recording_participant != bundle.participant_id:
        raise ModelCompatibilityError(
            f"recording participant {recording_participant!r} does not match model participant {bundle.participant_id!r}"
        )
    result = predict_frame(bundle, frame, metadata)
    payload = {
        "status": result.status,
        "reason": result.reason,
        "participant_id": bundle.participant_id,
        "model_eligible": bundle.viewer_eligible,
        "allow_unvalidated": bool(args.allow_unvalidated),
        "calibration": {
            "label_group_id": bundle.calibration_id,
            "sbp": bundle.calibration_sbp,
            "dbp": bundle.calibration_dbp,
        },
        "accepted_windows": result.accepted_windows,
        "total_windows": result.total_windows,
        "pulse_rate_bpm": result.pulse_rate_bpm,
        "delta_sbp": result.delta_sbp,
        "delta_dbp": result.delta_dbp,
        "estimated_sbp": result.sbp,
        "estimated_dbp": result.dbp,
        "research_only": True,
        "population_validated": False,
    }
    if args.output:
        write_json(resolve_path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.numeric_available or result.status == "model_validation_failed" else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "audit":
            return command_audit(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "evaluate":
            return command_evaluate(args)
        if args.command == "single-subject":
            return command_single_subject(args)
        if args.command == "predict":
            return command_predict(args)
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
