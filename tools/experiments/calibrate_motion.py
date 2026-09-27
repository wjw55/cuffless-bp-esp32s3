"""Calibrate and validate the causal ADXL345 Still/Moving classifier.

Each input must be an IMU CSV from a 90-second controlled-motion trial:
0-20 still, 20-30 gentle motion, 30-45 still, 45-55 large motion,
55-70 still, 70-80 sensor disturbance, and 80-90 still.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import glob
import json
import math
from pathlib import Path


SCALE_G_PER_LSB = 0.0039
GRAVITY_ALPHA = 0.01
RMS_WINDOW_SAMPLES = 100
WARMUP_SECONDS = 2.0
STILL_THRESHOLD_RATIO = 0.70
MOVING_DWELL_SECONDS = 0.20
STILL_DWELL_SECONDS = 1.0
EXPECTED_RATE_HZ = 100.0
MAX_CONFIG_THRESHOLD_MG = 4000
STATIONARY_GATE = 0.95
MOVEMENT_GATE = 0.90

# Guard bands exclude cue timing, the one-second RMS history, and filter recovery.
CLEAN_STILL_INTERVALS = ((4.0, 18.0), (34.0, 43.0), (59.0, 68.0), (84.0, 89.0))
REPRESENTATIVE_GENTLE_INTERVALS = ((22.0, 28.0),)
MOVEMENT_INTERVALS = ((22.0, 28.0), (47.0, 53.0), (72.0, 78.0))
MOVEMENT_LABELS = ("gentle_motion", "large_motion", "sensor_disturbance")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_imu_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"imu_seq", "timestamp_ms", "x_raw", "y_raw", "z_raw"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required IMU columns: {sorted(required)}")
        for row in reader:
            rows.append(
                {
                    "imu_seq": int(row["imu_seq"]),
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "x_raw": int(row["x_raw"]),
                    "y_raw": int(row["y_raw"]),
                    "z_raw": int(row["z_raw"]),
                }
            )
    if len(rows) < 2:
        raise ValueError(f"{path} contains fewer than two IMU samples")
    for previous, current in zip(rows, rows[1:]):
        if current["imu_seq"] != previous["imu_seq"] + 1:
            raise ValueError(f"{path} contains missing or non-increasing IMU sequences")
        if current["timestamp_ms"] <= previous["timestamp_ms"]:
            raise ValueError(f"{path} contains non-increasing IMU timestamps")
    duration_s = (rows[-1]["timestamp_ms"] - rows[0]["timestamp_ms"]) / 1000.0
    estimated_rate_hz = (len(rows) - 1) / duration_s if duration_s > 0 else 0.0
    if not 95.0 <= estimated_rate_hz <= 105.0:
        raise ValueError(f"{path} IMU rate {estimated_rate_hz:.2f} Hz is outside 95-105 Hz")
    return rows


def rolling_rms(squared_values: list[float], window_samples: int = RMS_WINDOW_SAMPLES) -> list[float]:
    if window_samples <= 0:
        raise ValueError("RMS window must contain at least one sample")
    window: deque[float] = deque()
    running_sum = 0.0
    result: list[float] = []
    for value in squared_values:
        window.append(value)
        running_sum += value
        if len(window) > window_samples:
            running_sum -= window.popleft()
        result.append(math.sqrt(max(running_sum / len(window), 0.0)))
    return result


def causal_activity(rows: list[dict]) -> list[tuple[float, float]]:
    origin_ms = rows[0]["timestamp_ms"]
    gravity: list[float] | None = None
    times: list[float] = []
    dynamic_squared_values: list[float] = []

    for row in rows:
        axes = [row[f"{axis}_raw"] * SCALE_G_PER_LSB for axis in ("x", "y", "z")]
        if gravity is None:
            gravity = list(axes)
        gravity = [
            previous + GRAVITY_ALPHA * (value - previous)
            for previous, value in zip(gravity, axes)
        ]
        dynamic_squared_values.append(
            sum((value - baseline) ** 2 for value, baseline in zip(axes, gravity))
        )
        times.append((row["timestamp_ms"] - origin_ms) / 1000.0)

    activities = rolling_rms(dynamic_squared_values)
    return list(zip(times, activities))


def in_intervals(time_s: float, intervals: tuple[tuple[float, float], ...]) -> bool:
    return any(start <= time_s < end for start, end in intervals)


def classify_trial(activity: list[tuple[float, float]], moving_threshold_g: float) -> list[tuple[float, str]]:
    still_threshold_g = moving_threshold_g * STILL_THRESHOLD_RATIO
    status = "calibrating"
    above_started: float | None = None
    below_started: float | None = None
    classified: list[tuple[float, str]] = []

    for time_s, value in activity:
        if time_s < WARMUP_SECONDS:
            classified.append((time_s, status))
            continue
        if value > moving_threshold_g:
            above_started = time_s if above_started is None else above_started
            below_started = None
            if time_s - above_started >= MOVING_DWELL_SECONDS:
                status = "moving"
        elif value < still_threshold_g:
            below_started = time_s if below_started is None else below_started
            above_started = None
            if time_s - below_started >= STILL_DWELL_SECONDS:
                status = "still"
        else:
            above_started = None
            below_started = None
        classified.append((time_s, status))
    return classified


def acceptance_gates(stationary_fraction: float, detected_blocks: int, total_blocks: int) -> dict:
    block_fraction = detected_blocks / total_blocks if total_blocks else 0.0
    passes_stationary = stationary_fraction >= STATIONARY_GATE
    passes_movement = block_fraction >= MOVEMENT_GATE
    return {
        "passes_stationary_gate": passes_stationary,
        "passes_movement_gate": passes_movement,
        "passes_all_gates": passes_stationary and passes_movement,
    }


def validate_classifier(
    trials: list[list[tuple[float, float]]],
    moving_threshold_g: float,
    trial_names: list[str] | None = None,
) -> dict:
    names = trial_names or [f"trial_{index + 1}" for index in range(len(trials))]
    classified_trials = [classify_trial(trial, moving_threshold_g) for trial in trials]
    per_trial: list[dict] = []
    total_stationary = 0
    total_stationary_still = 0
    detected_blocks = 0
    block_results: list[dict] = []

    for name, classified in zip(names, classified_trials):
        stationary_labels = [
            status for time_s, status in classified if in_intervals(time_s, CLEAN_STILL_INTERVALS)
        ]
        stationary_still = sum(status == "still" for status in stationary_labels)
        total_stationary += len(stationary_labels)
        total_stationary_still += stationary_still
        per_trial.append(
            {
                "input_file": name,
                "stationary_still_fraction": stationary_still / len(stationary_labels)
                if stationary_labels
                else 0.0,
            }
        )

        for label, (start, end) in zip(MOVEMENT_LABELS, MOVEMENT_INTERVALS):
            detected = any(
                start <= time_s < end and status == "moving" for time_s, status in classified
            )
            detected_blocks += int(detected)
            block_results.append(
                {
                    "input_file": name,
                    "label": label,
                    "interval_seconds": [start, end],
                    "detected": detected,
                }
            )

    stationary_fraction = total_stationary_still / total_stationary if total_stationary else 0.0
    total_blocks = len(block_results)
    block_fraction = detected_blocks / total_blocks if total_blocks else 0.0
    return {
        "stationary_still_fraction": stationary_fraction,
        "per_trial_results": per_trial,
        "movement_block_results": block_results,
        "movement_blocks_detected": detected_blocks,
        "movement_blocks_total": total_blocks,
        "movement_block_detection_fraction": block_fraction,
        **acceptance_gates(stationary_fraction, detected_blocks, total_blocks),
    }


def calculate_threshold(
    trials: list[list[tuple[float, float]]], trial_names: list[str] | None = None
) -> dict:
    stationary = [
        value
        for trial in trials
        for time_s, value in trial
        if in_intervals(time_s, CLEAN_STILL_INTERVALS)
    ]
    gentle = [
        value
        for trial in trials
        for time_s, value in trial
        if in_intervals(time_s, REPRESENTATIVE_GENTLE_INTERVALS)
    ]
    if not stationary or not gentle:
        raise ValueError("controlled trials do not contain the required guarded intervals")

    maximum_candidate_mg = min(MAX_CONFIG_THRESHOLD_MG, max(1, math.ceil(max(gentle) * 1000.0)))
    selected_validation: dict | None = None
    selected_threshold_mg: int | None = None
    for threshold_mg in range(1, maximum_candidate_mg + 1):
        validation = validate_classifier(trials, threshold_mg / 1000.0, trial_names)
        if validation["passes_all_gates"]:
            selected_threshold_mg = threshold_mg
            selected_validation = validation
            break

    result = {
        "stationary_p99_g": percentile(stationary, 0.99),
        "gentle_motion_p25_g": percentile(gentle, 0.25),
        "threshold_search_min_mg": 1,
        "threshold_search_max_mg": maximum_candidate_mg,
        "moving_threshold_g": selected_threshold_mg / 1000.0
        if selected_threshold_mg is not None
        else None,
        "still_threshold_g": selected_threshold_mg / 1000.0 * STILL_THRESHOLD_RATIO
        if selected_threshold_mg is not None
        else None,
        "config_motion_threshold_mg": selected_threshold_mg,
    }
    if selected_validation is not None:
        result.update(selected_validation)
    else:
        result.update(
            {
                "passes_stationary_gate": False,
                "passes_movement_gate": False,
                "passes_all_gates": False,
            }
        )
    return result


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return list(dict.fromkeys(paths))


def intervals_for_report(intervals: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[start, end] for start, end in intervals]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Still/Moving from controlled 90-second IMU trials.")
    parser.add_argument("imu_csv", nargs="+", help="Three controlled-trial IMU CSV paths or glob patterns")
    parser.add_argument("--output", default="motion_calibration.json", help="Calibration report JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = expand_inputs(args.imu_csv)
    if len(paths) < 3:
        print("ERROR: Provide at least three controlled-motion IMU CSV files.")
        return 2

    try:
        activities = [causal_activity(read_imu_csv(path)) for path in paths]
        for path, activity in zip(paths, activities):
            if activity[-1][0] < 89.0:
                raise ValueError(f"{path} is shorter than the required 90-second protocol")
        calibration = calculate_threshold(activities, [str(path) for path in paths])
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    report = {
        "protocol": "controlled_motion_90s_v2",
        "input_files": [str(path) for path in paths],
        "algorithm": {
            "gravity_alpha": GRAVITY_ALPHA,
            "activity_method": "causal_rolling_rms",
            "rms_window_samples": RMS_WINDOW_SAMPLES,
            "expected_sample_rate_hz": EXPECTED_RATE_HZ,
            "moving_dwell_seconds": MOVING_DWELL_SECONDS,
            "still_dwell_seconds": STILL_DWELL_SECONDS,
            "still_threshold_ratio": STILL_THRESHOLD_RATIO,
        },
        "guarded_intervals_seconds": {
            "clean_stationary": intervals_for_report(CLEAN_STILL_INTERVALS),
            "representative_gentle_motion": intervals_for_report(REPRESENTATIVE_GENTLE_INTERVALS),
            "movement_scoring": intervals_for_report(MOVEMENT_INTERVALS),
        },
        "acceptance_requirements": {
            "stationary_still_fraction": STATIONARY_GATE,
            "movement_block_detection_fraction": MOVEMENT_GATE,
        },
        "bpm_suppression_enabled": False,
        **calibration,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration report: {output_path}")
    if report["passes_all_gates"]:
        print(f"Validated threshold: CONFIG_MOTION_THRESHOLD_MG={report['config_motion_threshold_mg']}")
        print("BPM suppression remains disabled; motion status is validation-only.")
        return 0
    print("FAILED: no threshold passed both stationary and movement-detection gates.")
    print("Do not configure motion-based BPM suppression.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
