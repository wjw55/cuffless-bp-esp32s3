# Offline motion-aware PPG/BP development

This path is separate from `bp_pipeline.py`, the stationary models and all live
viewers. It does not change firmware, raw CSV formats or the shadow classifier.
Every run is research-only and deployment-ineligible. No motion BP model has yet
been validated. The RTOS trial must not be used for participant collection.

## Current evidence and admission

At implementation, local motion sessions contain three development and six
validation paired red/IR+IMU CSVs. This does not identify the five IMU-only
recordings mentioned in the study background. Resolve their filenames before
admission. IMU-only recordings cannot train PPG quality, artifact correction or
BP models; they can characterize motion or validate an IMU-only classifier.

The existing paired motion recordings contain timestamp-lag warnings. Shared
timestamp units alone do not establish physical synchronization. The new audit
quarantines recordings exceeding a provisional 20 ms lag limit, missing health
counters, discontinuous sequences, nonfinite data or timestamp gaps. The lag
warning is evidence of uncertainty, not a measured inter-sensor offset. Do not
correct it by maximizing PPG/IMU correlation. A future correction needs an
independently evidenced clock mapping, retained alongside original timestamps.

All limits in `config/motion_bp_v1.json`, including severity, contact and
uncertainty limits, are provisional engineering settings. They are not calibrated
upper-arm thresholds or acceptance criteria for deployment. The current 50,000
count contact threshold is inherited conservatively and needs upper-arm review.

Run from the repository root:

```powershell
& C:\wjw\Anaconda\python.exe tools\motion_bp_pipeline.py audit --output-dir data/processed/motion_bp_v1/audit_001
& C:\wjw\Anaconda\python.exe tools\motion_bp_pipeline.py extract --output-dir data/processed/motion_bp_v1/features_001
```

Supply `--identities data/processed/motion_bp_v1/identities.json` when a human has
resolved subject aliases. It maps raw names to actual participant identities,
for example `{"P001": "P001"}`. Never assume `self`, `test` and `P001` are different
people. Missing mappings stay unresolved; extraction is diagnostic only.

Outputs contain file hashes, admission reasons and configuration snapshots.
Outputs must be under ignored `data/processed`; existing run directories cannot
be overwritten. Raw data and trained models must never be added to Git. Historical
tracked label files are left unchanged. No model package is exported by this CLI.

## Collection protocol

Use only archived `participant-study-fw-v1.0`; explicitly record
`firmware=participant-study-fw-v1.0`. The automatically recorded PC Git commit
does not verify the binary on the board. Verify actual PPG and IMU values,
sequence continuity, rates and timestamp-lag warnings before each session. If
the approved firmware cannot meet synchronization requirements, stop paired
collection and investigate separately; do not silently substitute RTOS firmware.

Proposed 300-second cue schedule, initially externally cued with a sidecar log:

| Seconds | Activity |
|---|---|
| 0–60 | Supported still baseline |
| 60–90 | Gentle arm motion |
| 90–120 | Still recovery |
| 120–150 | Moderate seated arm/body motion |
| 150–180 | Still recovery |
| 180–210 | Typing/object handling |
| 210–240 | Still recovery |
| 240–270 | Controlled strap/contact disturbance |
| 270–300 | Still recovery |

The existing collector's 240-second protocol is unchanged. For this schedule,
use ordinary 300-second raw collection and an observer cue log until a separate
collector protocol extension is reviewed. Record planned and actual cue times,
their clock relationship to sensor timestamps, completion and mounting details.
Keep sensor site, axis orientation, strap tension and LED current documented.
Begin with three development repetitions and three later locked repetitions
across sessions. This is a quality pilot, not a BP validation sample size.

Review transition and recovery periods explicitly; do not derive waveform labels
from cue names. Reviewers must not see cuff values, continuous BP values or model
errors. Store `reviewed_quality` separately using `preserved`,
`potentially_recoverable`, `severely_corrupted`, `contact_corrupted` or `uncertain`,
with reviewer, notes and feature-manifest hash. Unreviewed/uncertain windows must
not be counted as evidence of low false acceptance. The initial extractor does
not train a new quality classifier or silently use the existing shadow model.

## Features and quality

Windows are selected by common sensor timestamps, initially 8 s wide with a 4 s
step. Unequal stream lengths and rates are expected. No row matching,
extrapolation or interpolation through rejected gaps is allowed. IMU gravity
history is causal; record its full support when creating split manifests.

Existing features supply morphology, PPG distortion, RMS acceleration, jerk,
frequency, orientation, lag correlation and spectral overlap. New movement
duration and longest-bout features integrate elapsed time rather than sample
counts. Duration is clipped to each window. More complex coherence, axis coupling
and waveform correction remain deferred until timing is trustworthy.

Motion intensity and PPG recoverability are separate fields. Severe intensity,
poor contact, clipping and missing/timing-invalid data cannot be accepted. A
potentially recoverable window only passes a provisional signal check; its BP
state remains `Low confidence` without validated reference evidence. Quality
code never reads BP labels or errors. IMU is contextual information, not a direct
measurement of BP. Do not remove every movement-correlated component: movement
can alter real physiology as well as corrupt the sensor signal.

## Reference labels and modelling

Intermittent Omron readings support personal calibration and individual
stationary occasions. They cannot label the intervening movement sequence.
Do not interpolate cuff readings or copy them onto movement windows. Continue
using the existing stationary pipeline for cuff-labelled baseline experiments.

Motion BP evaluation requires independently reviewed continuous-reference
windows. Reference quality, device latency, synchronization uncertainty and
reference-valid time must be documented independently of PPG quality and model
errors. A continuous-reference table must contain:

- `reference_kind=continuous`, `reference_valid=true`,
  `reference_alignment_verified=true`, `reference_source_sha256`;
- `reference_start_ms`, `reference_end_ms`, `true_sbp`, `true_dbp`;
- `start_timestamp_ms`, `end_timestamp_ms`, covered fully by the reference;
- resolved `participant_id`, `recording_id`, `recording_sha256`,
  `cuff_occasion_id` (the independent reference-occasion grouping ID);
- absolute `support_start_utc`, `support_end_utc` covering all filter/IMU context;
- one `calibration_occasion_id`, `calibration_end_utc`, `baseline_sbp`,
  `baseline_dbp`, and `calibration__<morphology feature>` values;
- `morph__<feature>`, IMU/cross-modal features, signal-only `signal_eligible`,
  `severity`, and optionally independent `reviewed_quality`.

These are derived tables, not new raw formats. Admission flags require actual
review; setting them in a CSV is not evidence by itself. Preserve the source and
review provenance. This version does not import a continuous-reference device.

Use `chronological_split` to lock complete occasions into fit/uncertainty/test
sets (60/20/20 with at least ten independent occasions each). This is intentionally
separate from the stationary pipeline's existing 70/30 development/test design.
Split before inspecting BP outcomes. The additional uncertainty partition must
never be used to fit the regressor. Historical inspected tests are development
data; collect a new locked test. If tuning is later added, use forward-only folds
inside the fit partition. Freeze thresholds and feature choices before testing.

```powershell
& C:\wjw\Anaconda\python.exe tools\motion_bp_pipeline.py split `
  --windows data/processed/motion_bp_v1/reviewed_reference_windows.csv `
  --output-dir data/processed/motion_bp_v1/split_001
```

```powershell
& C:\wjw\Anaconda\python.exe tools\motion_bp_pipeline.py evaluate `
  --train data/processed/motion_bp_v1/train.csv `
  --uncertainty data/processed/motion_bp_v1/uncertainty.csv `
  --test data/processed/motion_bp_v1/test.csv `
  --output-dir data/processed/motion_bp_v1/evaluation_001
```

For participant-held-out evaluation, add `--held-out-participant` and supply a
test participant absent from both fitting and uncertainty estimation. Repeat
with frozen choices for each held-out participant. Only their one predefined,
earlier calibration is permitted. All windows of a recording, repeated file
hash and reference/cuff occasion remain in one partition. Full signal-support
intervals must not overlap between chronological partitions. Personal calibration
values/features must stay fixed across partitions. Calibration outcomes are
excluded from evaluation.

The first ablation deliberately uses fixed regularized Ridge models, predicting
BP change from calibration, with preprocessing fitted only on training data:

1. PPG morphology plus personal calibration.
2. The same plus dynamic acceleration RMS and movement duration.
3. The same plus the full allowlisted IMU/cross-modal feature set.
4. PPG prediction with IMU used only in the shared rejection gate. This is
   explicitly the same predictor as (1), serving as the rejection-only control.
5. Personalized zero-change, evaluated on the PPG baseline's accepted windows.

Keep the frozen stationary pipeline as an additional separately reported
baseline; this command does not relabel its occasion predictions as window BP.
No current real motion-BP results are available. Never report simulated test
fixtures as experimental accuracy.

## Reporting and release gate

Reports include per-target MAE/RMSE/bias/max error, stationary/motion/severity
strata, unique-time reference-labelled coverage, interval coverage, threshold
error/coverage curves, reviewed corruption false acceptance, and comparisons on
common accepted windows. Report operational coverage separately using the full
acquisition timeline including missing data, warm-up and rejected intervals.
The evaluator's window-union denominator is explicitly not total session time.
With eight-second windows stepped every four seconds it measures supported signal
coverage, not the percentage of wall-clock time a causal viewer displayed BP.

Uncertainty uses independent occasion-max residuals within each severity stratum.
A stratum with fewer than ten independent uncertainty occasions receives an
unbounded interval and cannot produce an accepted estimate. Nominal interval coverage is
not guaranteed under temporal or participant distribution shift. Small groups,
missing labels or all-rejected subsets must produce unavailable metrics, not
zero errors. Use grouped confidence intervals for final superiority claims;
current point estimates alone do not prove improvement. Match coverage and
compare common windows against PPG rejection-only and zero-change. Review all
severe/contact false acceptances. Predeclare clinically unclaimed engineering
improvement, stationary non-regression, coverage and false-acceptance limits
before collecting the final test. Insufficient evidence means no release.

Only after independent offline validation may a separate viewer change display
`Stable estimate`, `Motion-compensated estimate`, `Low confidence`,
`Motion too severe`, `Poor contact`, or `Insufficient data`. Invalid states
suppress current numbers. Last-valid BP, if displayed, must have a separate label
and continuously updated age. Never count it as a current estimate. This code
does not enable viewer integration, medical-grade claims or continuous-monitoring
claims.

## Verification

```powershell
& C:\wjw\Anaconda\python.exe -B -m unittest discover -s tools -p 'test_*.py'
```

Tests cover unequal-rate timestamp extraction, nonfinite/gapped streams,
IMU-only admission, duration units, BP-independent gates, continuous-reference
admission, grouped chronological/participant splits, calibration consistency,
held-out label perturbation, error metrics, overlap-safe coverage and explicit
all-rejected/false-acceptance results. Existing stationary and viewer tests remain
part of the regression suite.
