# Offline PPG-to-BP Research Pipeline

`tools/bp_pipeline.py` audits labelled PPG datasets, extracts pulse-morphology features, and evaluates one-time-calibrated SBP/DBP baselines. It is PC-side research software. It does not modify firmware, acquire serial data, or provide a medical measurement.

## Data locations

Keep downloaded datasets outside version control:

```text
data/external/one_month_wrist/
data/external/ppg_bp/
```

The One-Month source must be the repository containing participant folders such as `A1/week1/session1/...`. Only the header-labelled `ESP...(PPG-IMU).csv` files are used for modelling. The headerless processed files are audited but never used for training.

Local upper-arm recordings are discovered from the existing `data/raw` metadata and `data/labels` files. Only records with `ppg_profile=upper_arm_experimental` and non-conflicting SBP/DBP labels are eligible.

## Commands

From the `ppg_logger` directory:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\bp_pipeline.py audit `
  --config config\bp_pipeline_v1.json

& "C:\wjw\Anaconda\python.exe" tools\bp_pipeline.py run `
  --config config\bp_pipeline_v1.json
```

If a run is interrupted after feature extraction, reuse its saved features without altering raw data:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\bp_pipeline.py run `
  --config config\bp_pipeline_v1.json `
  --run-id <existing_run_id> `
  --resume
```

To reproduce reports from saved predictions:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\bp_pipeline.py evaluate `
  --run-dir data\processed\bp\<run_id>
```

For retrospective development with one local upper-arm participant, use an existing run's frozen occasion features:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\bp_pipeline.py single-subject `
  --config config\bp_pipeline_v1.json `
  --participant-id test `
  --run-dir data\processed\bp\<run_id>
```

This mode is separate from the participant-level evaluation. It selects the first explicitly marked valid calibration occasion (or the first chronological usable occasion), uses the earliest 70% of subsequent occasions for development, and locks the latest 30% for one final test. At least five development and five test occasions are required. Ridge, Elastic Net, and an HR-only linear model are selected using forward-only folds within the development period. The locked test never participates in imputation, scaling, tuning, fitting, or model selection.

Outputs are written under `single_subject/<participant_id>/` and include the exact split, development-fold scores, locked-test predictions and metrics, selected research model files, and prediction/error plots. The pass criterion compares the model selected on development data against `zero_change` for both SBP and DBP. Results always remain labelled `single_subject_development=true` and `population_validated=false`.

The run folder contains the canonical manifest, segment decisions, occasion-level features, personalized examples, participant-level split definitions, predictions, metrics, fitted research models, plots, and a reproducibility manifest.

## Experimental BP prediction and viewer

Check one saved upper-arm recording with the same feature extraction and model package used during training:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\bp_pipeline.py predict `
  --config config\bp_pipeline_v1.json `
  --model-dir data\processed\bp\<run_id>\single_subject\P001 `
  --ppg-csv data\raw\<recording>_ppg.csv `
  --metadata-json data\raw\<recording>_metadata.json
```

The command reports signal quality, accepted windows, calibration BP, predicted changes, and reconstructed SBP/DBP. By default it refuses a model unless both its SBP and DBP models beat the zero-change baseline on the locked test. `--allow-unvalidated` permits a research-only check and marks the result as unvalidated. It never silently clips non-finite or physiologically inconsistent output.

The terminal viewer can be used immediately without a model. Pending mode displays sensor health and buffer progress but never a numeric BP:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\view_live_bp.py `
  --port COM5 `
  --participant-id P001 `
  --calibration-sbp 116 `
  --calibration-dbp 72
```

After creating a model package with `single-subject`, connect it without rebuilding the viewer:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\view_live_bp.py `
  --port COM5 `
  --participant-id P001 `
  --model-dir data\processed\bp\<run_id>\single_subject\P001
```

The viewer requires approximately 85 seconds of continuous stationary upper-arm PPG within a 90-second rolling buffer. Movement, timestamp restarts, I2C errors, or FIFO overflows clear the usable buffer and hide the previous result. A failing model remains hidden unless `--allow-unvalidated` is supplied; that override displays a prominent `UNVALIDATED DEVELOPMENT ESTIMATE` warning.

Only one program can own the serial port. Close `idf.py monitor`, the HR viewer, or any other serial program before starting the BP viewer.

### Optional reproducible live capture

To save normal raw data and the quality-gated BP updates from the same serial session, add a model to the collector:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\collect_ppg.py `
  --port COM5 `
  --duration 90 `
  --subject P001 `
  --session bp_live_validation_001 `
  --trial-id validation_001 `
  --posture seated `
  --sensor-location left_outer_upper_arm_5cm_above_elbow `
  --ppg-profile upper_arm_experimental `
  --imu-location left_upper_arm_adjacent_to_ppg `
  --imu-orientation x_distal_y_left_z_outward `
  --cuff-arm right `
  --live-bp-model-dir data\processed\bp\<run_id>\single_subject\P001 `
  --prompt-labels
```

This adds a separate `*_live_bp.csv` and model identifiers in metadata while leaving the raw PPG and IMU formats unchanged. The same default eligibility gate applies; use `--allow-unvalidated` only for an explicitly labelled development capture.

## Interpretation

The primary comparison is against `zero_change`, which always predicts the participant's calibration BP. A morphology model is scientifically useful only if it reduces held-out-participant MAE for both SBP and DBP. Wrist/fingertip results do not validate upper-arm performance, and generated models must not be presented as medical devices.

The single-subject command is a retrospective feasibility check, not evidence that the method generalizes to another person. Do not choose a different model after viewing locked-test errors; doing so turns those test occasions into development data and requires a new future test set.
