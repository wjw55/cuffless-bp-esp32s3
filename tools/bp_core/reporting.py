from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True), encoding="utf-8")


def plot_bp_distribution(occasions: pd.DataFrame, output_dir: Path) -> None:
    usable = occasions[occasions.get("occasion_usable", False) == True] if not occasions.empty else occasions  # noqa: E712
    if usable.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, target, color in zip(axes, ("sbp", "dbp"), ("tab:red", "tab:blue")):
        for dataset, group in usable.groupby("dataset_id"):
            axis.hist(group[target].dropna(), bins=12, alpha=0.5, label=dataset, color=None)
        axis.set_xlabel(f"{target.upper()} (mmHg)")
        axis.set_ylabel("Cuff occasions")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("BP label distributions by dataset")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "bp_distribution.png", dpi=160)
    plt.close(fig)


def plot_prediction_diagnostics(predictions: pd.DataFrame, metrics: dict[str, Any], output_dir: Path) -> None:
    if predictions.empty:
        return
    for evaluation in sorted(predictions["evaluation"].unique()):
        for target in ("sbp", "dbp"):
            subset = predictions[(predictions["evaluation"] == evaluation) & (predictions["target"] == target)]
            if subset.empty:
                continue
            feasibility = metrics.get("scientific_feasibility", {}).get(evaluation, {}).get(target, {})
            best_model = feasibility.get("best_learned_model")
            if not best_model:
                model_metrics = metrics.get("evaluations", {}).get(evaluation, {}).get(target, {})
                if not model_metrics:
                    continue
                best_model = min(model_metrics, key=lambda name: model_metrics[name]["mae"])
            selected = subset[subset["model"] == best_model]
            if selected.empty:
                continue
            true = selected["true_bp"].to_numpy(float)
            predicted = selected["predicted_bp"].to_numpy(float)
            low = float(min(np.min(true), np.min(predicted)))
            high = float(max(np.max(true), np.max(predicted)))
            fig, axis = plt.subplots(figsize=(5, 5))
            axis.scatter(true, predicted, alpha=0.75)
            axis.plot([low, high], [low, high], "k--", linewidth=1)
            axis.set_xlabel(f"Reference {target.upper()} (mmHg)")
            axis.set_ylabel(f"Predicted {target.upper()} (mmHg)")
            axis.set_title(f"{evaluation}: {best_model}")
            axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(output_dir / f"prediction_{evaluation}_{target}.png", dpi=160)
            plt.close(fig)

            means = (true + predicted) / 2.0
            differences = predicted - true
            bias = float(np.mean(differences))
            limits = 1.96 * float(np.std(differences, ddof=1)) if len(differences) >= 2 else 0.0
            fig, axis = plt.subplots(figsize=(6, 4))
            axis.scatter(means, differences, alpha=0.75)
            axis.axhline(bias, color="tab:blue", label=f"Bias {bias:.1f}")
            axis.axhline(bias + limits, color="tab:red", linestyle="--")
            axis.axhline(bias - limits, color="tab:red", linestyle="--", label="95% agreement limits")
            axis.set_xlabel(f"Mean predicted/reference {target.upper()} (mmHg)")
            axis.set_ylabel("Prediction - reference (mmHg)")
            axis.set_title(f"Bland–Altman: {evaluation}, {best_model}")
            axis.legend()
            axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(output_dir / f"bland_altman_{evaluation}_{target}.png", dpi=160)
            plt.close(fig)


def plot_learning_curve(curve: pd.DataFrame, output_dir: Path, evaluation: str) -> None:
    if curve.empty:
        return
    fig, axis = plt.subplots(figsize=(6, 4))
    for target, group in curve.groupby("target"):
        group = group.sort_values("training_participant_count")
        axis.errorbar(
            group["training_participant_count"],
            group["mean_mae"],
            yerr=group["std_mae"],
            marker="o",
            capsize=3,
            label=target.upper(),
        )
    axis.set_xlabel("Training participants")
    axis.set_ylabel("MAE (mmHg)")
    axis.set_title(f"Participant-level learning curve: {evaluation}")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"learning_curve_{evaluation}.png", dpi=160)
    plt.close(fig)


def plot_single_subject_diagnostics(
    predictions: pd.DataFrame,
    selected_model_names: dict[str, str],
    output_dir: Path,
) -> None:
    """Plot locked-test predictions without performing any model selection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in ("sbp", "dbp"):
        target_rows = predictions[predictions["target"] == target].copy()
        selected_name = selected_model_names[target]
        selected = target_rows[target_rows["model"] == selected_name].sort_values("chronological_order")
        baseline = target_rows[target_rows["model"] == "zero_change"].sort_values("chronological_order")
        if selected.empty or baseline.empty:
            continue

        true = selected["true_bp"].to_numpy(float)
        predicted = selected["predicted_bp"].to_numpy(float)
        low = float(min(np.min(true), np.min(predicted)))
        high = float(max(np.max(true), np.max(predicted)))
        if math.isclose(low, high):
            low -= 1.0
            high += 1.0
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.scatter(true, predicted, alpha=0.8)
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_xlabel(f"Reference {target.upper()} (mmHg)")
        axis.set_ylabel(f"Predicted {target.upper()} (mmHg)")
        axis.set_title(f"Single-subject locked test: {selected_name}")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"single_subject_prediction_{target}.png", dpi=160)
        plt.close(fig)

        positions = np.arange(len(selected))
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(positions, true, marker="o", label="Cuff reference")
        axes[0].plot(positions, predicted, marker="o", label=selected_name)
        axes[0].plot(
            positions,
            baseline["predicted_bp"].to_numpy(float),
            linestyle="--",
            label="zero_change",
        )
        axes[0].set_ylabel(f"{target.upper()} (mmHg)")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].plot(positions, selected["error"].to_numpy(float), marker="o", label=selected_name)
        axes[1].plot(
            positions,
            baseline["error"].to_numpy(float),
            marker="o",
            linestyle="--",
            label="zero_change",
        )
        axes[1].set_xlabel("Locked test occasion (chronological)")
        axes[1].set_ylabel("Error (mmHg)")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        fig.suptitle(f"Single-subject chronological {target.upper()} errors")
        fig.tight_layout()
        fig.savefig(output_dir / f"single_subject_chronological_error_{target}.png", dpi=160)
        plt.close(fig)
