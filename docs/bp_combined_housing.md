# Setup-aware personal BP development

This separate offline mode pools declared legacy and new-housing P001 recordings
using an explicit calibration per setup episode. Existing participant-level
evaluation, `single-subject`, normal `predict`, firmware, current model and live
viewers are unchanged. No research package automatically enables live BP.

## Cohort and calibration

`config/bp_combined_housing_v1.json` declares exact IDs, not automatic selection of
all P001 data. `legacy_mount` uses the existing package's named calibration and
post-calibration development/previously examined test occasions. `housing_v1`
uses `stationary_001` as calibration and 002–005 as development. The remaining
five IDs are reserved for future prospective recordings. Failed explicit
calibration stops preparation; there is no error-minimizing selection/fallback.

Each episode's calibration BP **and waveform features** produce current,
calibration and change predictors. Models learn BP change, then add the target
episode's calibration BP. Calibration rows never enter prediction metrics.
Episodes retain P001; they are not independent participants. Setup assignments
are declarations, not proof of identical mounting across legacy sessions. Do
not invent retrospective hardware details or periodic recalibration evidence.

## Frozen processing and chronology

All selected recordings are reprocessed with the unchanged feasibility
implementation: three accepted 8-second windows, 24 unique clean seconds, its
existing artifact/health checks and guarded synchronized-gap handling. This is
not the strict BP policy or proof of preserved BP morphology. Labels do not
influence waveform quality. Original files and thresholds are never modified.

Preparation reports failed occasions, rejects missing/unlabelled selected IDs,
identity/reference conflicts, invalid/tied timestamps, wrong sensor location
or profile, pre-calibration examples and duplicate raw PPG contents. Missing IMU
and sensor faults remain quality failures. Calibrations must pass quality.

Forward folds validate only later new-housing follow-ups. Combined training
adds historical examples strictly earlier than each validation fold. Each fold
refits its own imputer, variance filter, scaler and Ridge. Setup names, true BP,
reference HR and scheduled activity are not predictors. Calibration BP is an
intentional predictor. Whole cuff occasions stay together; windows never become
independent labelled examples.

Both targets use Ridge, with alpha chosen by development MAE and deterministic
ties. Zero-change, training mean-change and HR-only baselines use the same
validation occasions but are not saved model-family candidates. The deployment
candidate is fixed to `combined` before testing. A new-housing-only CV comparison
is attempted when five target follow-ups are available; the pilot has only four.
Combined fitting requires at least five total usable development occasions,
three target follow-ups and two safe forward folds. Existing evaluation minima
are unchanged. CV scores used to choose alpha are development scores, not
unbiased accuracy; three scored target occasions are especially uncertain.

## Training / preparation

From the project root, using the installed Python:

```powershell
python tools\bp_pipeline.py combined-subject `
  --config config\bp_pipeline_v1.json `
  --policy-config config\bp_feasibility_v1.json `
  --study-config config\bp_combined_housing_v1.json `
  --output-dir data\processed\bp\P001_combined_housing_v1_run001
```

Use a new directory for every run; overwrite is refused. `--prepare-only`
produces cohort/features/split reports without models. Later fitting needs a new
directory. Freeze the chosen run before prospective collection.

Outputs: all occasion/segment quality decisions, development examples, requested
and rejected IDs, explicit calibrations, reserved IDs, chronological cutoff,
forward folds/CV predictions/metrics, selected alphas, software versions, raw
hashes, frozen config/policy/study/calibration and two research Ridge files.
`combined_model_manifest.json` checks configuration, feature-table, calibration,
model, participant/schema and processing-code integrity. Code changes fail
closed rather than silently changing inference. Only load trusted local Joblib
packages; checksums are not a defence against malicious pickle contents.

Python sources use LF line endings through the repository's `.gitattributes`,
including Windows checkouts with `core.autocrlf=true`. Processing fingerprints
remain exact byte hashes: the loader does not ignore code changes or rewrite
saved manifests. An editor that saves processing sources as CRLF can therefore
cause a compatibility failure even when the visible code is unchanged. Restore
LF only after confirming that the normalized source matches the frozen hash;
do not refresh manifest hashes to hide a mismatch. Raw signals, labels, model
weights and frozen data/configuration files are never newline-normalized.

## Offline prediction

Both raw PPG and IMU are required. Training and prediction call the same
feasibility extraction and calibration-relative construction. Setup ID is an
operator declaration, not detected from PPG; known legacy IDs cannot be passed
as the new episode.

```powershell
python tools\bp_pipeline.py combined-predict `
  --run-dir data\processed\bp\P001_combined_housing_v1_run001 `
  --setup-id housing_v1 `
  --ppg-csv data\raw\P001_housing_v1_development_20261020_stationary_004_ppg.csv `
  --imu-csv data\raw\P001_housing_v1_development_20261020_stationary_004_imu.csv `
  --metadata-json data\raw\P001_housing_v1_development_20261020_stationary_004_metadata.json
```

Default numeric output is hidden as `model_validation_pending`. Add
`--allow-unvalidated` only for a labelled offline development estimate. A replay
of a training occasion is flagged, not validation. Bad quality, non-finite BP
and SBP <= DBP are rejected, never clipped. Calibration is not a prediction
example. Reserved test IDs cannot be examined through this command.

## Future evaluation, without retraining

The example reserves P001 session `housing_v1_prospective_test_001`, trials
`validation_001` through `validation_005`. These must be fresh recordings, not
renamed pilot files. Collect over multiple days with unchanged mounting and a
cuff reading immediately after PPG. Do not tune using their labels/results.

```powershell
python tools\bp_pipeline.py combined-evaluate `
  --run-dir data\processed\bp\P001_combined_housing_v1_run001
```

All named tests must be present and labelled, strictly later than development,
and free from duplicated development raw content. Frozen models are loaded
without fitting. Output retains rejected occasions in coverage, reports
true/predicted BP and zero-change metrics, and requires at least five accepted
predictions plus lower MAE for both targets for the baseline gate. Missing or
failed occasions are not silently replaced. Completed evaluation refuses
overwrite/re-evaluation; changing parameters after seeing results requires a
fresh test batch. These are workflow guardrails, not a security boundary.

Even a passed baseline gate does not activate live BP or establish population
or medical validation. Research packages deliberately omit `model_manifest.json`,
so ordinary prediction and all existing viewers reject them, even with override.
Do not rename the manifest to bypass this boundary. Future live integration
requires separate review. Existing model directories remain untouched.

## Tests

```powershell
python -m unittest discover -s tools\tests -p test_bp_combined.py
python -m unittest discover -s tools\tests -p 'test_*.py'
```

Coverage includes episode calibration, folds/leakage, Ridge-only fitting,
raw-byte preservation, real raw training/offline-inference equivalence, pending
mode/override, faults, corrupt packages and one-time evaluation without refit.
Recordings, models and generated reports remain in ignored local directories.
