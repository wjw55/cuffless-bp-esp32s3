"""Compare strict and gap-tolerant BP occasion-quality decisions.

This command extracts morphology features only.  It does not train a model,
change the strict pipeline, alter raw data, or activate the live BP viewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bp_core.config import load_config
from bp_core.datasets import _discover_local, _mark_calibration_occasions, load_recording
from bp_core.feasibility import load_feasibility_policy, process_gap_tolerant_recording
from bp_core.features import SignalQualityError, aggregate_recording_features, process_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/bp_pipeline_v1.json")
    parser.add_argument("--policy-config", default="config/bp_feasibility_v1.json")
    parser.add_argument("--participant-id", action="append", help="Repeat to select participants; default is all")
    parser.add_argument("--session-contains", help="Optional case-insensitive session filter")
    parser.add_argument("--output-dir", default="data/processed/bp_feasibility_v1")
    return parser.parse_args()


def strict_extract(recording, config):
    try:
        rows = pd.DataFrame(process_signal(recording, load_recording(recording), config))
        occasion = aggregate_recording_features(recording, rows, config)
        return occasion, rows, ""
    except (SignalQualityError, ValueError, OSError) as exc:
        return {
            "participant_id": recording.participant_id,
            "session_id": recording.session_id,
            "recording_id": recording.recording_id,
            "label_group_id": recording.label_group_id,
            "occasion_usable": False,
            "accepted_segment_count": 0,
            "total_segment_count": 0,
            "unique_clean_coverage_s": 0.0,
            "occasion_rejection_reasons": str(exc),
        }, pd.DataFrame(), str(exc)


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    policy = load_feasibility_policy(args.policy_config)
    recordings = _discover_local(config["datasets"]["local_upper_arm"])
    _mark_calibration_occasions(recordings)
    if args.participant_id:
        selected = set(args.participant_id)
        recordings = [recording for recording in recordings if recording.participant_id in selected]
    if args.session_contains:
        token = args.session_contains.lower()
        recordings = [recording for recording in recordings if token in recording.session_id.lower()]
    if not recordings:
        raise RuntimeError("No labelled local upper-arm recordings matched the requested filters")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    strict_occasions, feasibility_occasions = [], []
    strict_segments, feasibility_segments, gap_rows, comparisons = [], [], [], []
    for index, recording in enumerate(recordings, 1):
        print(f"{index}/{len(recordings)} {recording.participant_id} {recording.session_id} {recording.recording_id}", flush=True)
        strict, strict_rows, _ = strict_extract(recording, config)
        feasibility = process_gap_tolerant_recording(recording, config, policy)
        strict_occasions.append(strict)
        feasibility_occasions.append(feasibility.occasion)
        if not strict_rows.empty:
            strict_segments.extend(strict_rows.to_dict("records"))
        if not feasibility.segments.empty:
            feasibility_segments.extend(feasibility.segments.to_dict("records"))
        if not feasibility.gaps.empty:
            for row in feasibility.gaps.to_dict("records"):
                row.update(
                    participant_id=recording.participant_id,
                    session_id=recording.session_id,
                    recording_id=recording.recording_id,
                )
                gap_rows.append(row)
        comparisons.append(
            {
                "participant_id": recording.participant_id,
                "session_id": recording.session_id,
                "recording_id": recording.recording_id,
                "label_group_id": recording.label_group_id,
                "strict_usable": bool(strict.get("occasion_usable", False)),
                "strict_accepted_windows": int(strict.get("accepted_segment_count", 0)),
                "strict_clean_coverage_s": float(strict.get("unique_clean_coverage_s", 0.0)),
                "strict_rejection_reasons": str(strict.get("occasion_rejection_reasons", "")),
                "feasibility_usable": bool(feasibility.occasion.get("occasion_usable", False)),
                "feasibility_accepted_windows": int(feasibility.occasion.get("accepted_segment_count", 0)),
                "feasibility_clean_coverage_s": float(feasibility.occasion.get("unique_clean_coverage_s", 0.0)),
                "feasibility_rejection_reasons": str(feasibility.occasion.get("occasion_rejection_reasons", "")),
                "gap_event_count": int(feasibility.occasion.get("gap_event_count", 0)),
                "changed_to_usable": bool(
                    not strict.get("occasion_usable", False)
                    and feasibility.occasion.get("occasion_usable", False)
                ),
            }
        )

    tables = {
        "policy_comparison.csv": comparisons,
        "strict_occasion_features.csv": strict_occasions,
        "strict_segment_features.csv": strict_segments,
        "feasibility_occasion_features.csv": feasibility_occasions,
        "feasibility_segment_features.csv": feasibility_segments,
        "gap_events.csv": gap_rows,
    }
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / filename, index=False)

    comparison = pd.DataFrame(comparisons)
    report = {
        "policy_name": policy["policy_name"],
        "policy_parameters": policy,
        "research_only": True,
        "medical_device_validation": False,
        "strict_policy_modified": False,
        "bp_labels_used_for_quality_decisions": False,
        "config_path": str(Path(config_path)),
        "policy_config_path": str(Path(args.policy_config)),
        "recordings": len(comparison),
        "strict_usable": int(comparison["strict_usable"].sum()),
        "feasibility_usable": int(comparison["feasibility_usable"].sum()),
        "changed_to_usable": int(comparison["changed_to_usable"].sum()),
        "newly_usable_recordings": comparison.loc[
            comparison["changed_to_usable"], ["participant_id", "session_id", "recording_id"]
        ].to_dict("records"),
        "feasibility_rejected_recordings": comparison.loc[
            ~comparison["feasibility_usable"],
            ["participant_id", "session_id", "recording_id", "feasibility_rejection_reasons"],
        ].to_dict("records"),
        "limitations": [
            "Feasibility acceptance is not medical validation.",
            "Recordings accepted only by this policy remain development data.",
            "No window crosses an acquisition gap or planned movement interval.",
            "BP accuracy must be evaluated separately without tuning this gate on test-label errors.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Strict usable={report['strict_usable']}/{report['recordings']}; "
        f"feasibility usable={report['feasibility_usable']}/{report['recordings']}; "
        f"newly usable={report['changed_to_usable']}"
    )
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
