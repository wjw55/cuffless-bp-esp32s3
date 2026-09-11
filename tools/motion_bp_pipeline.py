"""Separate offline motion-aware BP development commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from motion_bp import ROOT, audit, chronological_split, evaluate_ablation, extract_recording, load_config, sha256


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "extract", "split", "evaluate"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/motion_bp_v1.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--identities", type=Path, help="Reviewed JSON mapping of recording subject names to actual participant IDs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--uncertainty", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--windows", type=Path, help="Reviewed derived windows for chronological splitting")
    parser.add_argument("--held-out-participant", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_dir.resolve()
    # This command creates generated data only under an already ignored directory.
    if not output.is_relative_to((ROOT / "data/processed").resolve()):
        parser.error("Output must be under data/processed (excluded from version control)")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory; existing results are immutable")
    if args.command == "evaluate" and not all((args.train, args.uncertainty, args.test)):
        parser.error("Evaluation requires --train, --uncertainty and --test")
    if args.command == 'split' and not args.windows:
        parser.error('Splitting requires --windows')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"research_only": True, "deployment_eligible": False,
                "config_sha256": sha256(args.config), "config": config,
                "command": args.command}
    if args.command in ("audit", "extract"):
        identities = json.loads(args.identities.read_text(encoding="utf-8")) if args.identities else {}
        table = audit(args.raw_dir, config, identities)
        table.to_csv(output / "recording_manifest.csv", index=False)
        manifest["recordings"] = len(table)
        manifest["paired_feature_eligible"] = int(table.paired_feature_eligible.sum()) if len(table) else 0
        if args.command == "extract":
            windows = []
            for _, row in table.iterrows():
                if not row.paired_feature_eligible:
                    continue
                metadata = json.loads(Path(row.metadata_path).read_text(encoding="utf-8"))
                extracted = extract_recording(pd.read_csv(row.ppg_path), pd.read_csv(row.imu_path), metadata, config)
                for column in ("recording_id", "participant_id", "session_id"):
                    extracted[column] = row[column]
                extracted["recording_sha256"] = row.ppg_sha256
                windows.append(extracted)
            pd.concat(windows, ignore_index=True).to_csv(output / "window_features.csv", index=False) if windows else pd.DataFrame(columns=["recording_id", "status", "rejection_reasons"]).to_csv(output / "window_features.csv", index=False)
    elif args.command == 'split':
        parts = chronological_split(pd.read_csv(args.windows), config['minimum_uncertainty_groups'])
        for name, frame in parts.items():
            frame.to_csv(output / (name + '.csv'), index=False)
        manifest['source_sha256'] = sha256(args.windows)
        manifest['split'] = {name: sorted(frame.cuff_occasion_id.unique().tolist()) for name, frame in parts.items()}
    else:
        frames = [pd.read_csv(path) for path in (args.train, args.uncertainty, args.test)]
        predictions, report = evaluate_ablation(*frames, config, held_out_participant=args.held_out_participant)
        predictions.to_csv(output / "predictions.csv", index=False)
        write_json(output / "evaluation.json", report)
        manifest["inputs"] = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in zip(("train", "uncertainty", "test"), (args.train, args.uncertainty, args.test))}
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "deployment_eligible": False}))


if __name__ == "__main__":
    main()
