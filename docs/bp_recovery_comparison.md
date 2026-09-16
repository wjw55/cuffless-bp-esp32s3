# Offline shorter-window BP comparison

Run from the `ppg_logger` directory:

```powershell
& 'C:\wjw\Anaconda\python.exe' tools\compare_bp_recovery.py
```

This reads existing P001 cuff-labelled upper-arm recordings and the timestamp-
checked September 12 seated movement pilot. It does not connect to the board.
Outputs are CSV files and `conclusion.md` under ignored
`data/processed/bp_recovery_comparison`. The saved September 5 model, calibration,
feature definitions, preprocessing, firmware and live viewer remain unchanged.

The baseline uses the original model configuration, requires 85 seconds of fresh
stationary signal and uses a buffer capped at 90 seconds. Experimental 20-, 24-,
30- and 60-second policies reuse the model in memory but require three accepted
eight-second windows and 80% unique clean coverage. The whole-recording upper-arm
analyzer is diagnostic for candidates, not a gate; all original per-window checks
remain. Candidate predictions are always labelled unvalidated. Outputs before the
initial long warm-up are diagnostic experiments, not a proposed change to startup.

Firmware motion messages determine fresh still intervals, expiring after three
seconds. The movement pilot additionally requires a guarded instructed-still or
recovery interval; it never scores instructed movement. Timestamp gaps and sensor
health faults exclude complete recordings conservatively. No future samples enter
a prediction. Metadata health screening is an offline admission check, not a claim
to have replayed the exact timing of historical health messages.

Each cuff occasion contributes only its recording-end prediction, and only a
documented `after_ppg` reference is admitted to cuff error comparisons. Calibration
and model-fitting occasions cannot select a duration. Previously examined test
occasions are development evidence. Short/long comparisons use matched endpoints;
separate all-accepted metrics and candidate-only update counts disclose differences
in acceptance. Do not attach these cuff labels to intermediate or moving windows.

Selection uses the agreed engineering criteria: both targets have MAE no more than
2 mmHg worse and maximum absolute error no more than 5 mmHg worse than the baseline
on matched occasions, and at least 80% of baseline-accepted update times are retained.
Missing baseline evidence fails selection. Choose the shortest passing duration;
60 seconds is the fallback. Counts accompany every metric because the sample is small.

`recovery_latency.csv` measures potential resumption after the later of a firmware
Still boundary and a guarded recovery cue. No recovery BP accuracy is inferred.
The pilot starts with less than 85 seconds still, so it cannot confirm the complete
startup-then-recovery workflow. New recordings and viewer integration remain gated
on the offline findings and the subsequent six-occasion confirmation.

Essential additions: the comparison script, its regression tests, and this note.
No trained models or generated results belong in version control.

## Separate gap-tolerant feasibility policy

The strict policy above remains unchanged. For school-project feasibility reporting,
one brief synchronized PPG/IMU interruption may instead be evaluated with the
separate `config/bp_feasibility_v1.json` policy:

```powershell
& 'C:\wjw\Anaconda\python.exe' tools\evaluate_bp_feasibility.py `
  --session-contains ah_staff_day1_20260916 `
  --output-dir data\processed\bp_feasibility_v1\ah_staff_day1
```

This policy permits at most one synchronized gap no longer than 200 ms, removes a
one-second margin on both sides, splits processing into continuous intervals, and
never interpolates or constructs an eight-second window across the break. For a
recovery trial, the planned 90–110 second movement block plus a one-second margin
is also excluded even if firmware reports `still`. It requires at least three
accepted windows and 24 seconds of non-overlapping clean coverage. Positive I2C or
FIFO error counters, non-monotonic data, unsynchronized gaps, repeated gaps and
long gaps still reject the recording.

Outputs include strict and feasibility feature tables, explicit gap events and a
side-by-side `policy_comparison.csv`. A recording accepted only by the feasibility
policy is development evidence; it does not become strict training or validation
data. BP labels are carried into the feature table for later evaluation but never
influence the signal-quality decision.

After freezing the feasibility gate, run the evaluation-only personalized model
comparison:

```powershell
& 'C:\wjw\Anaconda\python.exe' tools\evaluate_bp_feasibility_models.py `
  --config config\bp_pipeline_v1.json `
  --occasion-features data\processed\bp_feasibility_v1\ah_staff_day1\feasibility_occasion_features.csv `
  --output-dir data\processed\bp_feasibility_v1\ah_staff_day1_model_evaluation
```

The first explicitly marked usable occasion, or otherwise the first chronological
usable occasion, supplies each participant's calibration BP and morphology. It is
excluded from metrics. Leave-one-participant-out folds train on the other people
and predict every later occasion for the held-out person using only that person's
calibration. Zero-change, mean-change, HR-only, Ridge, Elastic Net and constrained
histogram gradient boosting are compared. The command writes predictions, fold
parameters, metrics and leakage-auditable split/calibration manifests, but fits and
saves no final deployment model. These same-day results remain pilot development
evidence; model selection must be frozen before a new multi-day validation batch.

To compare the AH-only dataset with all compatible P001 data and a balanced P001
subset, run:

```powershell
& 'C:\wjw\Anaconda\python.exe' tools\evaluate_bp_feasibility_models.py `
  --config config\bp_pipeline_v1.json `
  --ah-occasion-features data\processed\bp_feasibility_v1\ah_staff_day1\feasibility_occasion_features.csv `
  --p001-occasion-features data\processed\bp_feasibility_v1\P001_inventory\feasibility_occasion_features.csv `
  --output-dir data\processed\bp_feasibility_v1\combined_model_comparison
```

This creates `ah_only`, `combined_all` and `combined_balanced` evaluations. The
balanced experiment retains P001's calibration and caps P001 follow-ups to the
median AH follow-up count, preferring one stationary and one recovery occasion.
All experiments report both occasion-weighted and participant-balanced metrics;
the latter is the primary pass decision so P001's larger history cannot dominate
the result. No final model is fitted or saved.
