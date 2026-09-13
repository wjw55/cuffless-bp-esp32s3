#!/usr/bin/env python3
"""Train and evaluate the P001 Stage 1 usable/unusable development classifier."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from motion_quality import load_config
from motion_quality_classifier import train_and_evaluate


def _show_metric(value: float | None) -> str:
    return "na" if value is None else f"{value:.3f}"


def _show_percent(value: float | None) -> str:
    return "na" if value is None else f"{100.0 * value:.1f}%"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.output_dir or arguments.run_dir / "classifier_v1"
    try:
        if output_dir.exists():
            if not arguments.overwrite:
                raise FileExistsError(f"Output already exists: {output_dir}; use --overwrite to replace it")
            resolved_output = output_dir.resolve()
            resolved_run = arguments.run_dir.resolve()
            if resolved_output.parent != resolved_run:
                raise ValueError("Refusing to overwrite an output outside the run directory")
            shutil.rmtree(resolved_output)
        config = load_config(arguments.config)
        report = train_and_evaluate(config, arguments.config, arguments.run_dir, output_dir)
        metrics = report["selected_model_metrics"]
        print(f"Motion-quality development classifier: {output_dir.resolve()}")
        print(
            f"Windows={report['training_window_count']} "
            f"(usable={report['usable_window_count']}, unusable={report['unusable_window_count']}), "
            f"trials={len(report['trial_ids'])}"
        )
        print(
            f"Selected={report['selected_model']}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"usable_recall={metrics['usable_recall']:.3f}, "
            f"unusable_recall={metrics['unusable_recall']:.3f}"
        )
        print(f"Development gates passed: {report['development_acceptance_passed']}")
        print("Selected-model metrics by motion intensity (diagnostic only):")
        for item in report["selected_model_per_intensity_band_metrics"]:
            print(
                f"- {item['motion_intensity_band']}: windows={item['window_count']}, "
                f"balanced_accuracy={_show_metric(item['balanced_accuracy'])}, "
                f"usable_recall={_show_metric(item['usable_recall'])}, "
                f"unusable_recall={_show_metric(item['unusable_recall'])}, "
                f"reviewed_usable_time={_show_percent(item['true_usable_time_coverage'])}"
            )
        print("Independent validation: False; deployment eligible: False")
        return 0
    except (FileNotFoundError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
