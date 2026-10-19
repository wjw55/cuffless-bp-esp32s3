# Offline pulse-detector comparison

This focused development experiment investigates harmonic-driven extra peak
detection. It does not change firmware, Ridge packages, calibration, BP features,
quality thresholds or the detector selected by any live viewer. It produces no
BP estimate. Results and participant files remain in ignored `data/processed`.

## What changes in the candidate

The original detector uses the strongest spectral peak to set peak spacing. A
second or third harmonic can therefore permit more peaks than the underlying
pulse rate. That is a hypothesis to test, not proof that every high-frequency
component is interference.

`tools/experiments/compare_pulse_detectors.py` compares the original estimator with
`autocorrelation_guided_peaks_v1_offline`:

1. Estimate a period from the existing autocorrelation implementation.
2. Require autocorrelation at that period >= 0.30 and spectral power near its
   fundamental >= 10% of the strongest in-band spectral power. The latter is a
   veto against unsupported subharmonics, not another independent HR vote.
3. Use a second-order, zero-phase low-pass at `min(3 Hz, 1.5 * fundamental)`
   for **peak detection only**. Set peak spacing to the greater of 0.4 seconds
   and 70% of the autocorrelation period.
4. Check intervals and templates against the original band-passed IR, not a
   smoothed replacement waveform. Require detected-interval HR to agree with
   autocorrelation within 6 BPM.

The 8-peak, 0.20 interval-CV, 0.70 valid-interval-fraction and 0.60 template
correlation thresholds remain unchanged. Red/IR agreement is not a gate. The
candidate does not currently calculate the optional red/IR diagnostic value.

Both methods run through `analyze_upper_arm_ppg`. A keyword-only
`window_estimator` argument allows the offline experiment to replace the window
estimator without duplicating motion/contact masks, timing and health checks,
20-second window boundaries, unique coverage or trial consensus. Existing
callers omit it and continue using the original `estimate_window`.

Autocorrelation-guided detection and autocorrelation HR are **not independent**.
This dependence must be considered when evaluating apparent agreement. Periodic
artifacts can still pass. A good HR estimate also does not establish preserved
pulse morphology or BP accuracy.

## Run

From the project root, with NumPy, Pandas and SciPy available:

```powershell
python tools\experiments\compare_pulse_detectors.py `
  --session housing_v1_development_20261020 `
  --session upper_arm_hr_validation_001 `
  --session upper_arm_hr_validation_002 `
  --session movement_pilot_20260912 `
  --session fingertip_control_20260912_233515 `
  --review-csv data\processed\motion_feasibility_v1\visual_review\window_review.csv `
  --review-recording P001_movement_pilot_20260912_seated_001 `
  --output-dir data\processed\pulse_detector_comparison_v1
```

Sessions are exact metadata matches. Repeat `--session` to add replay sources.
Review options are optional and must be supplied together. Existing output
directories are refused: use a new directory for each run. `--input-dir` and
`--labels-dir` default to the project's local raw and label directories.

## Outputs and interpretation

- `recording_comparison.csv`: original/candidate trial statuses, accepted-window
  counts, unique coverage, estimated HR and available reference-HR errors.
- `window_comparison.csv`: all windows, including rejected windows and reasons.
- `review_overlap.csv`: timestamp overlap between accepted windows and existing
  clean, corrupted or uncertain annotations. Overlap is unioned, not double
  counted. Empty when no review file is supplied.
- `report.json`: frozen candidate constants, versions, source hashes, inputs,
  errors and limitations. A partial run with recording errors exits nonzero.

BP/HR labels, notes and activity labels never enter the detectors. Reference HR
is joined by participant/session/trial only **after** decisions. Conflicting
metadata/CSV reference HR is an error. After-PPG cuff HR is an approximate
trial-level comparator, not synchronized beat-to-beat truth.

The existing reviewed intervals are 8 seconds, while this experiment uses
20-second windows. Overlap warnings are therefore **not** exact window-level
false-acceptance rates. A motion mask may prevent overlap before either pulse
detector runs. Zero overlap does not prove the candidate itself detects motion
corruption. Uncertain intervals must never be counted as known-clean controls.
These annotations were Codex-assisted, not independently adjudicated.

Do not select a winner from increased coverage or lower HR error alone. Check
losses of previously accepted trials and acceptance of uncertain/corrupted
regions. Keep the candidate offline if any trade-off remains unresolved. Further
parameter changes make the same recordings development data, not fresh tests.

## Tests

```powershell
python -m unittest discover -s tools\tests -p test_compare_pulse_detectors.py
python -m unittest discover -s tools\tests -p 'test_*.py'
```

Tests cover native sample rates, dominant harmonics, genuine fast pulses,
unsupported subharmonics, noise and invalid input, original-signal template
checks, unchanged default behavior, masks/faults, unique coverage, reference
independence/conflicts, review-overlap accounting, reproducible output and raw
byte preservation. No model or live-detector promotion is automatic.
