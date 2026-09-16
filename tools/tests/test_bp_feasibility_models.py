from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_bp_feasibility_models import (
    balanced_p001_combination,
    detailed_metrics,
    prepare_evaluation,
)


def occasions(participants=4):
    rows = []
    for participant_index in range(participants):
        participant = f"P{participant_index:03d}"
        for occasion in range(3):
            role = ["calibration", "stationary", "recovery"][occasion]
            rows.append(
                {
                    "dataset_id": "local_upper_arm",
                    "participant_id": participant,
                    "session_id": "day1",
                    "label_group_id": f"local:{participant}:{role}_{occasion}",
                    "chronological_order": f"2026-09-16T10:{occasion:02d}:00+08:00",
                    "sbp": 110 + participant_index + occasion,
                    "dbp": 70 + participant_index + occasion,
                    "calibration_occasion": occasion == 0,
                    "occasion_usable": True,
                    "quality_policy": "frozen_feasibility_v1",
                    "gap_tolerant": occasion == 2,
                    "median__feature__pulse_rate_bpm": 65 + participant_index + occasion,
                    "median__feature__rise_time_s": 0.2 + occasion * 0.01,
                }
            )
    return pd.DataFrame(rows)


class FeasibilityModelEvaluationTests(unittest.TestCase):
    def test_builds_one_calibration_and_two_examples_per_participant(self):
        examples, calibrations, manifest = prepare_evaluation(occasions())
        self.assertEqual(len(calibrations), 4)
        self.assertEqual(len(examples), 8)
        self.assertEqual(manifest["participant_count"], 4)
        self.assertFalse(examples["label_group_id"].str.contains("calibration").any())
        self.assertEqual(set(examples["recording_role"]), {"stationary", "recovery"})
        self.assertEqual(int(examples["gap_tolerant"].sum()), 4)

    def test_duplicate_occasion_is_rejected(self):
        frame = occasions()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate cuff occasions"):
            prepare_evaluation(frame)

    def test_requires_participants_and_post_calibration_data(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            prepare_evaluation(occasions(participants=2))
        frame = occasions()
        participant = frame.participant_id.iloc[0]
        frame = frame[~((frame.participant_id == participant) & ~frame.calibration_occasion)]
        with self.assertRaisesRegex(ValueError, "no usable post-calibration"):
            prepare_evaluation(frame)

    def test_requires_one_frozen_policy(self):
        frame = occasions()
        frame.loc[0, "quality_policy"] = "changed_after_results"
        with self.assertRaisesRegex(ValueError, "exactly one frozen"):
            prepare_evaluation(frame)

    def test_rejected_row_without_chronology_does_not_block_evaluation(self):
        frame = occasions()
        rejected = frame.iloc[[0]].copy()
        rejected["label_group_id"] = "rejected:empty"
        rejected["occasion_usable"] = False
        rejected["chronological_order"] = None
        examples, _, _ = prepare_evaluation(pd.concat([frame, rejected], ignore_index=True))
        self.assertEqual(len(examples), 8)

    def test_participant_balanced_mae_gives_each_person_equal_weight(self):
        rows = [
            {
                "participant_id": "P001", "target": "sbp", "model": "ridge",
                "recording_role": "stationary", "true_bp": 100.0, "predicted_bp": 200.0,
            }
        ]
        rows.extend(
            {
                "participant_id": "P002", "target": "sbp", "model": "ridge",
                "recording_role": "stationary", "true_bp": 100.0, "predicted_bp": 100.0,
            }
            for _ in range(10)
        )
        metrics = detailed_metrics(pd.DataFrame(rows))
        overall = metrics[metrics.scope == "overall"].set_index("aggregation")
        self.assertAlmostEqual(overall.loc["occasion_weighted", "mae"], 100 / 11)
        self.assertAlmostEqual(overall.loc["participant_balanced", "mae"], 50.0)

    def test_balanced_combination_caps_p001_and_prefers_protocol_roles(self):
        ah = occasions(participants=4)
        ah["participant_id"] = ah["participant_id"].map(
            {f"P{index:03d}": f"P{index + 100:03d}" for index in range(4)}
        )
        ah["label_group_id"] = [
            f"local:{participant}:{role}_{index % 3}"
            for index, (participant, role) in enumerate(
                zip(ah["participant_id"], ["calibration", "stationary", "recovery"] * 4)
            )
        ]
        p001 = occasions(participants=1).copy()
        p001["participant_id"] = "P001"
        p001["label_group_id"] = [
            "local:P001:calibration_000",
            "local:P001:stationary_001",
            "local:P001:recovery_002",
        ]
        combined, manifest = balanced_p001_combination(ah, p001)
        selected = manifest["selected_p001_followup_ids"]
        self.assertEqual(manifest["p001_followup_cap"], 2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(any("stationary" in value for value in selected))
        self.assertTrue(any("recovery" in value for value in selected))
        self.assertEqual((combined.participant_id == "P001").sum(), 3)


if __name__ == "__main__":
    unittest.main()
