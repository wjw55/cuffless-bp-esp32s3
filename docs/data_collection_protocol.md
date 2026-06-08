# CDE3301 PPG and Omron Pilot Data Collection Protocol

## Purpose

This protocol describes how to collect raw PPG recordings from an ESP32-S3 and MAX30102 sensor while recording reference blood pressure readings from an Omron cuff monitor. The current goal is to validate stable PPG acquisition and trial metadata before using any data for cuffless BP model training.

Raw PPG CSV format is fixed:

```csv
sample_seq,timestamp_ms,red,ir
```

Blood pressure labels and collection details are stored in the trial metadata JSON, not in the raw signal CSV.

## Hardware Setup

- Microcontroller: ESP32-S3.
- PPG sensor: MAX30102.
- Reference device: Omron cuff BP monitor.
- Firmware output: serial CSV rows plus comment-prefixed debug/status lines.
- Collection tool: `tools/collect_ppg.py`.
- Analysis tool: `tools/analyze_trials.py`.

Typical MAX30102 wiring:

| MAX30102 breakout | ESP32-S3 |
| --- | --- |
| VIN or 3V3 | 3V3 |
| GND | GND |
| SDA | GPIO 8 |
| SCL | GPIO 9 |
| INT | Not used |

## Subject Posture

- Seated posture.
- Back supported.
- Feet flat on the floor.
- Both arms supported on a table.
- Avoid talking or moving during recording.
- Rest for 1-2 minutes between trials.

## Cuff and Sensor Placement

- Omron cuff on the left upper arm.
- MAX30102 sensor on the right index finger or right middle finger.
- Keep the PPG sensor stable and avoid changing finger pressure during the trial.
- Keep the cuff arm and PPG hand supported to reduce motion artifacts.

## Timing Procedure

1. Start the PPG recording first.
2. Record for 90 seconds per trial.
3. Start the Omron measurement around 20-30 seconds after PPG recording begins.
4. Record the cuff reading time when the Omron result is available.
5. Enter systolic BP, diastolic BP, cuff HR, cuff timing, and notes into metadata.
6. Repeat only a small number of trials first, then inspect quality before collecting more.

Example command:

```powershell
python tools\collect_ppg.py --port COM3 --duration 90 --subject test --session omron_pilot_001 --trial-id omron_001 --posture seated --sensor-location right_index_finger --cuff-arm left --ppg-hand right --cuff-start-time-s 25 --notes "Omron pilot trial 1" --prompt-bp-after
```

## Metadata Fields Recorded

Core trial identity:

- `subject_id`
- `session_id`
- `trial_id`
- `output_csv_filename`
- `output_csv_path`
- `firmware_git_commit`

Protocol metadata:

- `posture`
- `sensor_location`
- `cuff_arm`
- `ppg_hand`
- `notes`

Recording metadata:

- `port`
- `baud_rate`
- `duration_seconds`
- `recording_start_time`
- `sample_count`
- `sample_sequence_start`
- `sample_sequence_end`
- `missing_sample_sequences`

Timing diagnostics:

- `median_sample_interval_ms`
- `mean_sample_interval_ms`
- `min_sample_interval_ms`
- `max_sample_interval_ms`
- `p95_sample_interval_ms`
- `p99_sample_interval_ms`
- `timestamp_gaps_gt_15ms`
- `timestamp_gaps_gt_20ms`
- `non_increasing_timestamp_count`
- `timing_quality`
- `timing_quality_reason`
- `warnings`

Optional Omron fields:

- `systolic_mmHg`
- `diastolic_mmHg`
- `cuff_hr_bpm`
- `cuff_start_time_s`
- `cuff_reading_time_s`
- `cuff_timestamp`

## Timing Quality Rules

The collection pipeline classifies timestamp quality using sample sequence and timestamp diagnostics.

- `good`: no missing sample sequences, no non-increasing timestamps, no gaps greater than 15 ms, and maximum sample interval at or below 15 ms.
- `usable`: no missing sample sequences, no non-increasing timestamps, no gaps greater than 20 ms, and maximum sample interval at or below 20 ms.
- `borderline`: no missing sample sequences, no non-increasing timestamps, at most 5 gaps greater than 20 ms, and maximum sample interval at or below 40 ms.
- `reject`: missing sample sequences, non-increasing timestamps, more than 5 gaps greater than 20 ms, or maximum sample interval above 40 ms.

For this ESP32/MAX30102 logger, occasional 20 ms intervals can occur during FIFO draining or timestamp reconstruction. These are acceptable when no samples are missing and no gap exceeds 20 ms.

## Analysis Quality Rules

The analysis script generates `analysis_quality` for each trial.

- `usable`: timing is `good` or `usable`, no missing samples, enough IR peaks are detected, and PPG-derived HR is plausible.
- `borderline`: timing is `borderline`, non-timing metadata warnings exist, PPG HR differs from cuff HR by more than 10 bpm, signal span is small, or the analysis has a clear limitation.
- `reject`: CSV or metadata is missing, sample sequences are missing, timing quality is `reject`, too few PPG peaks are detected, or PPG HR is unavailable or impossible.

Legacy timestamp warning strings may be preserved in the summary but ignored for classification when `timing_quality` is already `good` or `usable`.

## Limitations

- The Omron cuff provides practical reference labels for a class pilot, not clinical-grade arterial BP ground truth.
- A small pilot session is not enough for BP model training.
- The current PPG HR estimate is simple and intended for signal quality checking, not medical use.
- Motion, finger pressure, ambient light, sensor placement, and cuff timing can affect data quality.
- Only `usable` trials should be considered for later modeling, and even those should be manually reviewed with plots before inclusion.
