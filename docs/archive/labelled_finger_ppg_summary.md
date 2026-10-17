# Labelled Finger PPG Summary

## Scope

This document summarizes labelled right-index-finger PPG sessions collected with a MAX30102 sensor and separate Omron left-upper-arm cuff labels. Raw PPG CSV rows remain unchanged as `sample_seq,timestamp_ms,red,ir`; Omron labels remain in separate `data/labels/<session>_labels.csv` files.

## Session Summary

| Session | Clean Trials | Rejected Trials | Clean SBP Range | Clean DBP Range | Clean HR MAE |
| --- | --- | --- | --- | --- | --- |
| `omron_pilot_002` | 5 | 0 | 98-108 | 61-66 | 3.68 bpm |
| `omron_pilot_003` | 4 | 1 | 104-106 | 50-63 | 1.72 bpm |
| Combined clean | 9 | 1 total rejected outside clean set | 98-108 | 50-66 | 2.81 bpm |

## Combined Clean Dataset

- `omron_pilot_002`: 5 clean labelled trials.
- `omron_pilot_003`: 4 clean labelled trials plus 1 rejected trial (`trial_004`).
- Total clean labelled finger trials: 9.
- BP range across clean trials: SBP 98-108 mmHg, DBP 50-66 mmHg.
- HR agreement across clean trials: 9 evaluable comparisons, mean error +2.81 bpm, MAE 2.81 bpm.

## Interpretation

The labelled finger-PPG workflow is repeatable enough to proceed to upper-arm feasibility testing. Across two sessions, clean trials preserve 100 Hz timing quality with no missing sample sequences, no non-increasing timestamps, and no timestamp gaps greater than 20 ms. PPG-derived HR remains close enough to Omron HR for workflow validation.

This conclusion validates data-acquisition and labelling workflow repeatability for finger PPG. It does not validate final upper-arm placement, cuffless BP accuracy, or model performance.

## Recommended Next Steps

1. Run the same labelled workflow during upper-arm feasibility testing while keeping raw PPG rows unchanged.
2. Keep reject handling explicit so failed or zero-sample trials do not contaminate aggregate statistics.
3. Track HR agreement, timing quality, saturation, and flat-signal warnings for every new session.
4. Continue collecting before BP model training; 9 clean trials from one subject and a narrow BP range are still only workflow-validation data.
