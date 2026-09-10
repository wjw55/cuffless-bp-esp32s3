import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion_study_protocol import (  # noqa: E402
    ANNOTATION_COLUMNS,
    MOTION_QUALITY_V1,
    MotionBlock,
    MotionProtocol,
    MotionProtocolRun,
    format_protocol_schedule,
    validate_motion_protocol,
    write_motion_annotations,
)


class MotionStudyProtocolTests(unittest.TestCase):
    def test_v1_is_continuous_and_240_seconds(self):
        validate_motion_protocol(MOTION_QUALITY_V1)

        self.assertEqual(MOTION_QUALITY_V1.duration_s, 240.0)
        self.assertEqual(len(MOTION_QUALITY_V1.blocks), 8)
        self.assertEqual(MOTION_QUALITY_V1.blocks[0].activity_label, "still")
        self.assertEqual(MOTION_QUALITY_V1.blocks[-1].activity_label, "sensor_contact_disturbance")

    def test_rejects_gap_between_blocks(self):
        invalid = MotionProtocol(
            name="invalid",
            description="contains a gap",
            blocks=(
                MotionBlock(0.0, 10.0, "one", "still", "Still"),
                MotionBlock(11.0, 20.0, "two", "moving", "Move"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "continuous"):
            validate_motion_protocol(invalid)

    def test_emits_each_cue_once_and_records_actual_elapsed_time(self):
        run = MotionProtocolRun(MOTION_QUALITY_V1)
        messages = []

        run.update(0.0, messages.append)
        run.update(10.0, messages.append)
        run.update(30.25, messages.append)

        self.assertEqual(len(messages), 2)
        self.assertEqual(run.cue_elapsed_s[0], 0.0)
        self.assertEqual(run.cue_elapsed_s[1], 30.25)
        self.assertIn("Move the sensor arm slowly", messages[-1])

    def test_marks_interrupted_blocks_without_inventing_completion(self):
        run = MotionProtocolRun(MOTION_QUALITY_V1)
        run.update(0.0, lambda _message: None)
        run.update(30.1, lambda _message: None)

        rows = run.annotation_rows("P001", "motion_quality_v1", "motion_001", 45.0)

        self.assertEqual(rows[0]["completion_status"], "complete")
        self.assertEqual(rows[1]["completion_status"], "partial")
        self.assertEqual(rows[2]["completion_status"], "not_started")
        self.assertIsNone(rows[2]["cue_elapsed_s"])

    def test_writes_stable_annotation_schema(self):
        run = MotionProtocolRun(MOTION_QUALITY_V1)
        for elapsed in range(0, 240, 30):
            run.update(float(elapsed), lambda _message: None)
        rows = run.annotation_rows("P001", "motion_quality_v1", "motion_001", 240.0)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.csv"
            write_motion_annotations(path, rows)
            with path.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(list(saved[0]), ANNOTATION_COLUMNS)
        self.assertEqual(len(saved), 8)
        self.assertTrue(all(row["completion_status"] == "complete" for row in saved))
        self.assertEqual(saved[-1]["block_name"], "contact_disturbance")

    def test_formats_human_readable_schedule(self):
        lines = format_protocol_schedule(MOTION_QUALITY_V1)

        self.assertEqual(len(lines), 8)
        self.assertIn("0-30", lines[0])
        self.assertIn("contact_disturbance", lines[-1])


if __name__ == "__main__":
    unittest.main()
