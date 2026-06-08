# Omron Pilot 001 Summary

## Session Overview

- Session ID: `omron_pilot_001`
- Subject ID: `test`
- Recording duration: about 90 seconds per trial
- Raw CSV format: `sample_seq,timestamp_ms,red,ir`
- Trials found: 6
- Usable trials: 4
- Borderline trials: 0
- Rejected trials: 2
- Clean pilot candidates: `omron_002`, `omron_004`, `omron_005`, `omron_006`

This session is a pilot validation dataset. It is useful for checking the collection pipeline, timing stability, metadata flow, and PPG signal quality. It is not large enough for BP model training.

## Trial Summary

| Trial | BP (mmHg) | Cuff HR (bpm) | PPG HR (bpm) | HR Error (bpm) | Timing Quality | Analysis Quality | Reason |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `omron_001` | 106/71 | 71 | 72.29 | 1.29 | reject | reject | timing_quality=reject |
| `omron_002` | 119/66 | 66 | 68.97 | 2.97 | usable | usable | usable timing and plausible PPG HR |
| `omron_003` | 109/66 | 68 | 71.43 | 3.43 | reject | reject | timing_quality=reject |
| `omron_004` | 106/68 | 68 | 69.77 | 1.77 | usable | usable | usable timing and plausible PPG HR |
| `omron_005` | 103/71 | 66 | 69.77 | 3.77 | usable | usable | usable timing and plausible PPG HR |
| `omron_006` | 105/72 | 68 | 68.18 | 0.18 | usable | usable | usable timing and plausible PPG HR |

## Timing and Sampling Notes

All six trials had:

- Missing sample sequences: 0
- Median sample interval: 10.0 ms
- Approximate sampling rate: 100.0 Hz
- Gaps greater than 20 ms: 0

Rejected trials:

- `omron_001`: rejected because `non_increasing_timestamp_count=1`.
- `omron_003`: rejected because `non_increasing_timestamp_count=1`.

Usable trials:

- `omron_002`, `omron_004`, `omron_005`, and `omron_006` had usable timing and plausible PPG HR estimates.
- `omron_005` and `omron_006` preserved legacy timestamp warning strings, but those warnings were ignored for classification because their computed `timing_quality` was usable.

## What Worked

- Trial filenames included subject, session, and trial ID, so recordings did not overwrite each other.
- Each trial produced a raw CSV and matching metadata JSON.
- Sequence numbers showed no missing samples across all six trials.
- The analyzer produced a session summary and IR peak diagnostic plots.
- PPG-derived HR was close to Omron cuff HR for the clean pilot candidates:
  - `omron_002`: 2.97 bpm error
  - `omron_004`: 1.77 bpm error
  - `omron_005`: 3.77 bpm error
  - `omron_006`: 0.18 bpm error

## What Failed

- Two trials were rejected due to one non-increasing timestamp each.
- Older metadata warning strings were too broad before `timing_quality` became the primary timing source of truth.
- The session is still too small and too single-subject to support BP modeling.

## Lessons Learned

- `sample_seq` is useful for confirming that sample data was not dropped.
- Detailed timestamp diagnostics are necessary; a generic timestamp warning is not enough to decide whether to keep a trial.
- `timing_quality` should drive timing decisions, while raw warning strings should remain visible for traceability.
- PPG HR agreement with cuff HR is a practical sanity check before labeling trials as clean candidates.

## Next Steps

- Repeat or replace rejected trials before expanding the dataset.
- Continue using the same posture, cuff placement, sensor placement, and 90-second timing procedure.
- Review diagnostic IR peak plots for all usable trials before using them later.
- Collect more sessions only after the timestamp issue causing non-increasing timestamps is understood or shown to be rare.
- Do not start BP model training until there are many more clean trials across more conditions and subjects.
