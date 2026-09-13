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
