# Paired PPG + IMU feasibility

This stage asks whether mild movement leaves usable PPG morphology. It does not
train BP, estimate BP during movement, or alter any live gate. Keep main firmware
at the stable build; do not use rtos-trial. Lighting investigation remains deferred.

## Available data

Inspection on 2026-09-12 found 93 metadata records: 71 paired PPG+IMU,
18 PPG-only, and four empty captures. No IMU-only raw files were found locally.
The five previously described IMU-only movement recordings remain unidentified;
if recovered, use them only for IMU/movement analysis, never PPG correction or BP.

Movement prefixes in `data/raw` (each paired with `_ppg.csv`, `_imu.csv` and
`_metadata.json`):

| Prefix | Recordings | Current alignment assessment |
| --- | --- | --- |
| P001_motion_quality_v1_motion_quality_ | 001–003 | Timestamp lag uncertainty; quarantine paired analysis |
| P001_motion_quality_validation_v1_validation_ | 001–006 | Timestamp lag uncertainty; 005 also has a PPG sequence gap |
| test_motion_calibration_motion_ | 001–004 | Timestamp lag uncertainty; quarantine paired analysis |
| P001_motion_timing_development_20260912_movement_001 | One 300-second pilot | Passes software timing checks; manual cues delayed, unsuitable for precise block labels |
| P001_movement_pilot_20260912_seated_001 | One 300-second pilot | Passes software timing checks; guarded spoken-cue intervals available |
| P001_movement_pilot_20260912_preflight_001 | One 8-second preflight | Timing checks pass, too short for a movement trial |

Passing means monotonic timestamps, sequence continuity, acceptable gaps and
recorded health/clock diagnostics. Alignment uses device timestamps, never row
numbers. These checks do not independently measure physical sensor latency.
The seated pilot's host-to-device cue mapping had approximately 10.5 ms spread
(95th–5th percentile), but spoken instructions are not observations of action onset.

## Run the focused offline comparison

From the project directory:

```powershell
python tools/motion_feasibility.py --metadata data/raw/P001_movement_pilot_20260912_seated_001_metadata.json --activities data/processed/motion_bp_v1/seated_pilot_20260912/guarded_activity_intervals.csv --model-package data/processed/motion_quality_v1/stage1/classifier_v1/motion_quality_classifier.joblib --output-dir data/processed/motion_feasibility_v1/seated_pilot
```

Only load trusted local model packages. Outputs stay in ignored `data/processed`:
`summary.json`, `window_features.csv`, and `window_review.csv`. For a reviewed
rerun, supply `--reviews` and use a new output directory to preserve the original.
Omit `--model-package` when unavailable; unscored classifier coverage is reported
separately, and acceptance/error rates without observations are null.

The tool reuses existing 8-second windows, 4-second steps, timestamp alignment,
PPG morphology and IMU features. Provisional dynamic acceleration RMS boundaries
are 0.02, 0.08 and 0.20 g: stationary, mild, moderate, severe. These are intensity
categories, not BP measurements or validated corruption labels.

Still/Moving time assigns overlapping windows exclusively at center midpoints.
Unsupported time is Unknown. Accepted-window coverage is the union of accepted
window intervals, divided by the full reporting interval. Both use timestamps,
not sample counts. Activity comparisons include only windows wholly contained in
the supplied guarded intervals. Whole-recording results include transitions;
activity results exclude them. Contact-step, low-contact and clipping flag rates
integrate raw sample intervals and are diagnostic flags, not confirmed causes.

Three decisions remain separate: PPG-only candidate morphology, the existing
PPG+motion safety gate, and the frozen quality classifier in shadow mode. The
classifier uses both PPG and IMU features; it is not an IMU-only intensity model.
All BP output is suppressed in this tool, including severe motion and poor contact.

Review existing raw plots/zoom plots and the corresponding timestamped waveform
segments. Fill `window_review.csv` with `clean`, `motion_corrupted`,
`contact_corrupted`, or `uncertain`, plus a reviewer. Do not use cuff values,
classifier predictions, or the instruction to move as waveform truth. Mark
ambiguous mechanisms uncertain. False acceptance is accepted/corrupted reviewed
windows; false rejection is rejected/clean reviewed windows. Unscored and
uncertain counts remain separate. Overlapping windows are correlated; these
counts are descriptive, not independent validation sample sizes. IMU severity
does not establish severe waveform corruption.

## Initial seated-pilot result

| Guarded activity | Still / Moving / Unknown time (%) | PPG candidate coverage (%) | Frozen classifier acceptance coverage (%) | Contact-step flag time (%) |
| --- | --- | --- | --- | --- |
| Still baseline | 93.7 / 0 / 6.3 | 17.0 | 0 | 0 |
| Gentle arm | 0 / 93.5 / 6.5 | 76.5 | 0 | 42.7 |
| Object handling | 0 / 91.4 / 8.6 | 91.4 | 0 | 87.6 |
| Seated torso | 0 / 88.4 / 11.6 | 79.5 | 0 | 0 |
| Still recovery | 90.4 / 0 / 9.6 | 49.3 | 16.4 | 0 |

Low-contact and clipping flag rates were zero in these guarded blocks. Across
the whole recording, 26 windows were stationary, 47 mild, and none moderate or
severe. PPG candidate and existing gated coverage matched here because there
were no severe windows. High morphology acceptance during contact-step flags
and disagreement with the frozen classifier require waveform review; neither
decision establishes accuracy. There are no reviewed labels for this pilot yet,
so false acceptance/rejection rates are unknown. The classifier was trained on
the same participant; this is not participant-held-out validation.

## Minimum next collection

First review a small balanced selection from the existing pilot: still, gentle,
object handling, torso and recovery, including candidate acceptances and
rejections. This can be done without wearing the sensors. Keep uncertainty
explicit and expand review before claiming classifier error rates for all time.

Then collect three paired five-minute recordings across at least two sessions:
one development recording and two later repeatability checks. This is a practical
feasibility minimum, not enough for generalization or BP accuracy claims.

| Seconds | Instruction |
| --- | --- |
| 0–60 | Still, seated, supported arm |
| 60–120 | Gentle comfortable sensor-arm movement |
| 120–180 | Still recovery |
| 180–240 | Larger comfortable seated arm/shoulder movement |
| 240–300 | Still recovery |

Use the same comfortably snug mounting, cable slack and stable main firmware.
Record spoken-cue timestamps and actual completion/problems as sidecar metadata;
keep raw CSV formats unchanged. Exclude the first five seconds after each spoken
cue finishes and the last two seconds before the next cue starts. Label intensity
from IMU measurements; if larger movement remains mild, report the missing
moderate data rather than relabelling it. No deliberate sensor dislodging is
needed for this first feasibility repeat. Severe/contact-specific error rates
will remain unassessed unless independently reviewed examples are available.

Cuff readings are unnecessary for this signal-only milestone. Any collected
Omron reading stays a calibration/occasion reference and must not be copied onto
movement windows. Preserve recording/session separation for later checks; do not
randomly split overlapping windows or tune thresholds on the later recordings.
Proceed toward BP work only if repeatable mild-motion morphology survives review;
do not create a motion-BP viewer or claim motion-compensated BP accuracy here.

## Essential changes

- `tools/motion_feasibility.py`: one offline report using existing features.
- `tools/tests/test_motion_feasibility.py`: time accounting, guards and review tests.
- This note: inventory, interpretation and short collection protocol.

Firmware, raw formats, existing BP architecture, quality thresholds and viewer are
unchanged. Existing unrelated worktree changes are outside this milestone.
