# Omron Pilot 002 Summary

## Protocol Summary

- Session: `omron_pilot_002`
- Subject: `self`
- PPG sensor: MAX30102 on right index finger
- Reference cuff: Omron on left upper arm
- Posture: seated
- Raw PPG CSV format: `sample_seq,timestamp_ms,red,ir`
- Label file: `data/labels/omron_pilot_002_labels.csv`
- Processed summary CSV: `data/processed/omron_pilot_002_summary.csv`

This pilot checks whether the collection workflow can pair clean finger PPG recordings with separate Omron SBP/DBP/HR labels. It does not train or evaluate a blood pressure model.

## Label Table

| Trial | SBP | DBP | Omron HR | Label Timing | Label Quality |
| --- | ---: | ---: | ---: | --- | --- |
| `trial_001` | 108 | 61 | 69 | unknown | good |
| `trial_002` | 107 | 64 | 68 | during_ppg | good |
| `trial_003` | 107 | 66 | 74 | during_ppg | good |
| `trial_004` | 98 | 61 | 69 | during_ppg | good |
| `trial_005` | 101 | 62 | 69 | during_ppg | good |

Observed label range: SBP 98-108 mmHg, DBP 61-66 mmHg.

## PPG Timing and Data Quality

All five referenced PPG CSV and metadata JSON files exist. All five trials meet the requested timing checks: `timing_quality` is `good`, missing sample sequences are 0, non-increasing timestamp counts are 0, and timestamp gaps greater than 20 ms are 0.

| Trial | Samples | Duration (s) | Timing Quality | Missing Seq | Non-Increasing TS | Gaps >20 ms | Analyzer Quality |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `trial_001` | 8973 | 89.72 | good | 0 | 0 | 0 | usable |
| `trial_002` | 8973 | 89.72 | good | 0 | 0 | 0 | usable |
| `trial_003` | 8971 | 89.70 | good | 0 | 0 | 0 | usable |
| `trial_004` | 8972 | 89.71 | good | 0 | 0 | 0 | usable |
| `trial_005` | 8971 | 89.70 | good | 0 | 0 | 0 | usable |

## PPG Features

IR peak-to-peak is reported as `ir_max - ir_min` over the full trial. Saturation uses the MAX30102 18-bit ADC range with a 1000-count margin; flat-signal warning uses IR span below 1000 counts.

| Trial | PPG HR (bpm) | Signal Quality | Red Min-Max | IR Min-Max | IR Peak-to-Peak | Saturation | Flat Signal |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `trial_001` | 72.29 | usable | 122304-125926 | 145737-152161 | 6424 | no | no |
| `trial_002` | 73.17 | usable | 121681-125010 | 146843-154277 | 7434 | no | no |
| `trial_003` | 76.92 | usable | 123953-126746 | 147508-154020 | 6512 | no | no |
| `trial_004` | 73.17 | usable | 122771-126614 | 143097-154137 | 11040 | no | no |
| `trial_005` | 71.86 | usable | 122801-126164 | 146889-155521 | 8632 | no | no |

## PPG HR vs Omron HR

| Trial | Omron HR | PPG HR | Error (PPG - Omron) | Absolute Error |
| --- | ---: | ---: | ---: | ---: |
| `trial_001` | 69 | 72.29 | +3.29 | 3.29 |
| `trial_002` | 68 | 73.17 | +5.17 | 5.17 |
| `trial_003` | 74 | 76.92 | +2.92 | 2.92 |
| `trial_004` | 69 | 73.17 | +4.17 | 4.17 |
| `trial_005` | 69 | 71.86 | +2.86 | 2.86 |

Mean absolute HR error across the five trials: **3.68 bpm**.

## Workflow Conclusion

This pilot validates the data-collection workflow for this stage: raw PPG timing is clean, labels are stored separately from the raw CSV rows, and the analyzer can derive plausible PPG heart-rate estimates that are close to Omron HR. The clean timing and label join make these trials useful for pipeline validation and later exploratory feature work.

## Limitations

- Only 5 trials.
- One subject.
- Finger PPG only.
- Narrow BP range: SBP 98-108 mmHg and DBP 61-66 mmHg.
- Not enough data for BP model training.
- Not final upper-arm placement validation.
- Omron labels are practical pilot labels, not invasive or clinical-grade reference measurements.

## Recommended Next Steps

1. Keep collecting small labelled batches with the same raw CSV format and separate label CSV workflow.
2. Add more sessions only when each batch passes the same timing checks used here.
3. Include repeated rest periods and consistent cuff timing so labels are easier to interpret.
4. Review IR peak diagnostic plots for each usable trial before including it in any feature dataset.
5. Expand subjects, BP range, postures, and sensor placements only after the finger-PPG workflow remains stable across more sessions.
6. Defer BP model training until there are many more clean labelled trials and a clear validation split.
