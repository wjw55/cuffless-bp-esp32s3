# Omron Pilot 003 Summary

## Protocol Summary

- Test date: 2026-06-20
- Session: `omron_pilot_003`
- Subject: `self`
- PPG sensor: MAX30102 on right index finger
- Reference cuff: Omron on left upper arm
- Posture: seated
- Raw PPG CSV format: `sample_seq,timestamp_ms,red,ir`
- Label file: `data/labels/omron_pilot_003_labels.csv`
- Processed summary CSV: `data/processed/omron_pilot_003_summary.csv`

This pilot checks repeatability of the labelled finger-PPG workflow after `omron_pilot_002`. It joins separately stored Omron SBP/DBP/HR labels with raw PPG timing and simple PPG feature metrics. It does not train or evaluate a blood pressure model.

## Label Table

| Trial | SBP | DBP | Omron HR | Label Timing | Label Timing Quality | Label Quality |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `trial_001` | 104 | 57 | 70 | during_ppg | good | good |
| `trial_002` | 105 | 62 | 68 | during_ppg | good | good |
| `trial_003` | 106 | 63 | 67 | during_ppg | good | good |
| `trial_004` | 112 | 54 | 67 | during_ppg | reject | reject |
| `trial_005` | 106 | 50 | 74 | during_ppg | good | good |

Observed clean-label range: SBP 104-106 mmHg, DBP 50-63 mmHg. `trial_004` is marked `reject` and is excluded from clean aggregate statistics.

## PPG Timing and Data Quality

All five referenced PPG CSV and metadata JSON files exist. Trials `trial_001`, `trial_002`, `trial_003`, and `trial_005` meet the requested timing checks: `timing_quality` is `good`, missing sample sequences are 0, non-increasing timestamp counts are 0, and timestamp gaps greater than 20 ms are 0. `trial_004` contains no valid CSV samples and is rejected.

| Trial | Samples | Duration (s) | Timing Quality | Missing Seq | Non-Increasing TS | Gaps >20 ms | Analyzer Quality |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `trial_001` | 8974 | 89.73 | good | 0 | 0 | 0 | usable |
| `trial_002` | 8973 | 89.72 | good | 0 | 0 | 0 | usable |
| `trial_003` | 8973 | 89.72 | good | 0 | 0 | 0 | usable |
| `trial_004` | 0 | n/a | reject | 0 | 0 | 0 | reject |
| `trial_005` | 8973 | 89.72 | good | 0 | 0 | 0 | usable |

## PPG Features

IR peak-to-peak is reported as `ir_max - ir_min` over the full trial. Saturation uses the MAX30102 18-bit ADC range with a 1000-count margin; flat-signal warning uses IR span below 1000 counts. Empty rejected trials are reported as not available rather than as flat signals.

| Trial | PPG HR | Signal Quality | Red Min-Max | IR Min-Max | IR Peak-to-Peak | Saturation | Flat Signal |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `trial_001` | 71.43 | usable | 115594-125807 | 136132-154322 | 18190 | no | no |
| `trial_002` | 70.59 | usable | 124030-127270 | 148394-155489 | 7095 | no | no |
| `trial_003` | 69.77 | usable | 121906-125336 | 145271-153084 | 7813 | no | no |
| `trial_004` | n/a | reject_no_samples | n/a | n/a | n/a | n/a | n/a |
| `trial_005` | 74.07 | usable | 120731-126142 | 145348-154519 | 9171 | no | no |

## PPG HR vs Omron HR

| Trial | Omron HR | PPG HR | Error | Abs Error | Clean Aggregate |
| --- | ---: | ---: | ---: | ---: | --- |
| `trial_001` | 70 | 71.43 | +1.43 | 1.43 | yes |
| `trial_002` | 68 | 70.59 | +2.59 | 2.59 | yes |
| `trial_003` | 67 | 69.77 | +2.77 | 2.77 | yes |
| `trial_004` | 67 | n/a | n/a | n/a | no |
| `trial_005` | 74 | 74.07 | +0.07 | 0.07 | yes |

All-trial HR summary: 4 evaluable comparisons, MAE 1.72 bpm. `trial_004` is included in the table but has no PPG HR because no samples were recorded.

Clean-trial HR summary: 4 clean comparisons, mean error +1.72 bpm, MAE 1.72 bpm, max absolute error 2.77 bpm.

## Clean-Trial Aggregate Summary

- Clean labelled trials: 4
- Rejected trials: 1 (`trial_004`)
- Clean SBP range: 104-106 mmHg
- Clean DBP range: 50-63 mmHg
- Clean PPG timing: all clean trials have 0 missing sample sequences, 0 non-increasing timestamps, and 0 timestamp gaps greater than 20 ms.
- Clean PPG signal quality: all clean trials are classified `usable` by the analyzer, with no saturation or flat-signal warnings.

## Workflow Conclusion

This pilot supports the labelled finger-PPG workflow: four trials produced clean 100 Hz timing, usable PPG signal features, and close PPG HR agreement with Omron HR. The rejected zero-sample trial is correctly separated from clean aggregate statistics.

## Limitations

- One subject.
- Finger PPG only.
- Narrow BP range.
- One rejected trial.
- Not enough data for BP model training.
- Not final upper-arm placement validation.
- Omron labels are practical pilot labels, not invasive or clinical-grade reference measurements.

## Recommended Next Steps

1. Keep collecting small labelled finger-PPG batches until the full collection workflow feels routine.
2. Preserve the separate label CSV workflow and unchanged raw PPG CSV row format.
3. Investigate any future zero-sample rejected trials at collection time, especially serial connection and recording-start behavior.
4. Proceed to upper-arm feasibility testing only with the same timing-quality gates and reject handling.
5. Continue deferring BP model training until the dataset includes more subjects, wider BP range, and validated placement data.
