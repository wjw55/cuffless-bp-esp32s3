"""Timed activity protocols and annotations for synchronized PPG/IMU studies."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


PROTOCOL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MotionBlock:
    start_s: float
    end_s: float
    block_name: str
    activity_label: str
    instruction: str


@dataclass(frozen=True)
class MotionProtocol:
    name: str
    description: str
    blocks: tuple[MotionBlock, ...]

    @property
    def duration_s(self) -> float:
        return self.blocks[-1].end_s


MOTION_QUALITY_V1 = MotionProtocol(
    name="motion_quality_v1",
    description="Upper-arm PPG/IMU motion-quality development protocol",
    blocks=(
        MotionBlock(0.0, 30.0, "still_baseline", "still", "Remain completely still with the arm supported."),
        MotionBlock(
            30.0,
            60.0,
            "gentle_arm_motion",
            "gentle_arm_motion",
            "Move the sensor arm slowly without touching the strap or cable.",
        ),
        MotionBlock(60.0, 90.0, "still_recovery_1", "still", "Return the arm to support and remain still."),
        MotionBlock(
            90.0,
            120.0,
            "typing",
            "object_handling",
            "Type naturally while keeping the sensor and strap untouched.",
        ),
        MotionBlock(120.0, 150.0, "still_recovery_2", "still", "Return the arm to support and remain still."),
        MotionBlock(
            150.0,
            180.0,
            "seated_body_motion",
            "whole_body_motion",
            "While seated, repeatedly lean the torso and move both shoulders; do not stand near the cable.",
        ),
        MotionBlock(180.0, 210.0, "still_recovery_3", "still", "Return to the original supported posture and remain still."),
        MotionBlock(
            210.0,
            240.0,
            "contact_disturbance",
            "sensor_contact_disturbance",
            "Gently press and release the strap edge; do not disconnect or reposition the electronics.",
        ),
    ),
)

MOTION_PROTOCOLS = {MOTION_QUALITY_V1.name: MOTION_QUALITY_V1}

ANNOTATION_COLUMNS = [
    "schema_version",
    "participant_id",
    "session_id",
    "trial_id",
    "protocol",
    "block_index",
    "block_name",
    "scheduled_start_s",
    "scheduled_end_s",
    "cue_elapsed_s",
    "activity_label",
    "instruction",
    "completion_status",
    "label_source",
]


def get_motion_protocol(name: str) -> MotionProtocol:
    try:
        return MOTION_PROTOCOLS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown motion-study protocol: {name}") from exc


def validate_motion_protocol(protocol: MotionProtocol) -> None:
    if not protocol.blocks:
        raise ValueError("Motion-study protocol must contain at least one block")
    expected_start = 0.0
    for block in protocol.blocks:
        if block.start_s != expected_start:
            raise ValueError("Motion-study blocks must be continuous and start at 0 seconds")
        if block.end_s <= block.start_s:
            raise ValueError("Motion-study block end must be after its start")
        expected_start = block.end_s


def format_protocol_schedule(protocol: MotionProtocol) -> list[str]:
    return [
        f"{block.start_s:>5.0f}-{block.end_s:<5.0f} s  {block.block_name}: {block.instruction}"
        for block in protocol.blocks
    ]


@dataclass
class MotionProtocolRun:
    protocol: MotionProtocol
    cue_elapsed_s: dict[int, float] = field(default_factory=dict)
    last_cued_block: int = -1

    def update(self, elapsed_s: float, output_func: Callable[[str], None] = print) -> None:
        """Emit every newly reached cue and retain its actual PC elapsed time."""
        elapsed_s = max(0.0, float(elapsed_s))
        reached = [
            index
            for index, block in enumerate(self.protocol.blocks)
            if block.start_s <= elapsed_s < block.end_s
        ]
        if not reached:
            return
        current = reached[-1]
        for index in range(self.last_cued_block + 1, current + 1):
            block = self.protocol.blocks[index]
            cue_time = elapsed_s if index == current else block.start_s
            self.cue_elapsed_s[index] = cue_time
            output_func(
                f"\n[MOTION {index + 1}/{len(self.protocol.blocks)} | "
                f"{block.start_s:.0f}-{block.end_s:.0f} s] {block.instruction}"
            )
        self.last_cued_block = current

    def annotation_rows(
        self,
        participant_id: str,
        session_id: str,
        trial_id: str,
        observed_duration_s: float,
    ) -> list[dict]:
        observed_duration_s = max(0.0, float(observed_duration_s))
        rows: list[dict] = []
        for index, block in enumerate(self.protocol.blocks):
            if observed_duration_s >= block.end_s:
                completion = "complete"
            elif observed_duration_s > block.start_s:
                completion = "partial"
            else:
                completion = "not_started"
            rows.append(
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "protocol": self.protocol.name,
                    "block_index": index + 1,
                    "block_name": block.block_name,
                    "scheduled_start_s": block.start_s,
                    "scheduled_end_s": block.end_s,
                    "cue_elapsed_s": self.cue_elapsed_s.get(index),
                    "activity_label": block.activity_label,
                    "instruction": block.instruction,
                    "completion_status": completion,
                    "label_source": "timed_protocol_cue",
                }
            )
        return rows


def write_motion_annotations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


validate_motion_protocol(MOTION_QUALITY_V1)
