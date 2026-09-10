#!/usr/bin/env python3
"""Prepare and finalize manually reviewed motion-quality windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from motion_quality import finalize_dataset, load_config, prepare_dataset


GENERATED_FILES = (
    "window_features.csv",
    "window_review.csv",
    "study_report.json",
    "reviewed_windows.csv",
    "finalization_report.json",
)


def _prepare_output(path: Path, overwrite: bool) -> None:
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    plots = list((path / "review_plots").glob("*.png")) if (path / "review_plots").exists() else []
    if (existing or plots) and not overwrite:
        raise FileExistsError(f"Output already exists in {path}; use --overwrite to replace generated files")
    if overwrite:
        for generated in existing + plots:
            generated.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Generate features and manual-review plots")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--session", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    finalize = subparsers.add_parser("finalize", help="Validate completed labels and create training rows")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "prepare":
            _prepare_output(arguments.output_dir, arguments.overwrite)
            report = prepare_dataset(config, arguments.input_dir, arguments.session, arguments.output_dir)
            print(f"Prepared motion-quality study: {arguments.output_dir.resolve()}")
            print(
                f"Trials={report['trial_count']}, candidates={report['candidate_window_count']}, "
                f"reviewable={report['reviewable_window_count']}, "
                f"protocol_excluded={report['protocol_excluded_window_count']}, "
                f"quality_rejected={report['quality_rejected_window_count']}"
            )
            print(f"Review labels: {(arguments.output_dir / 'window_review.csv').resolve()}")
            print(f"Plots: {(arguments.output_dir / 'review_plots').resolve()}")
        else:
            report = finalize_dataset(config, arguments.run_dir)
            print(f"Finalized motion-quality study: {arguments.run_dir.resolve()}")
            print(f"Reviewed={report['reviewed_window_count']}, training={report['supervised_training_window_count']}")
            print("Labels: " + json.dumps(report["label_counts"], sort_keys=True))
            for warning in report["training_readiness_warnings"]:
                print(f"WARNING: {warning}")
        return 0
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
