# Graphene PPG/Finapres short-window experiment

This is a separate source-domain feasibility experiment. It uses only raw fingertip PPG and synchronized continuous Finapres blood pressure from the [Graphene blood-pressure dataset](https://physionet.org/content/bp-graphene-bioimpedance/1.0.0/). ECG, Bio-Z, PAT and PTT are not inputs.

The experiment asks one narrow question: can PPG morphology extracted from 20, 30 or 60 seconds retain useful BP information compared with an approximately 85-second observation? It does not validate upper-arm BP, recovery after motion, or BP during movement.

## Data placement

Place the original dataset tree below:

```text
data/external/graphene_bp/
  subject1_day1/
    setup01_baseline/
      data_trial01_ppg.csv
      data_trial01_finapresBP.csv
```

The directory is ignored by Git. Keep the PhysioNet version and licence with the local download.

The complete archive is approximately 751 MB compressed. This resumable command downloads it; extraction requires approximately 2.3 GB plus the archive:

```powershell
curl.exe -L -C - --output data\external\graphene_bp_v1.0.0.zip `
  https://physionet.org/content/bp-graphene-bioimpedance/get-zip/1.0.0/

Expand-Archive data\external\graphene_bp_v1.0.0.zip `
  -DestinationPath data\external\graphene_bp -Force
```

## Commands

Audit pairing and participant discovery:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\graphene_ppg_bp.py audit `
  --config config\graphene_ppg_bp_v1.json `
  --output-dir data\processed\graphene_ppg_bp\audit_001
```

Run the experiment:

```powershell
& "C:\wjw\Anaconda\python.exe" tools\graphene_ppg_bp.py run `
  --config config\graphene_ppg_bp_v1.json `
  --output-dir data\processed\graphene_ppg_bp\run_001
```

Use `--max-recordings 1` for a quick adapter check before processing the complete download.

## Processing and safeguards

- PPG and Finapres are aligned by their original timestamps, never by row number.
- Native PPG is anti-alias downsampled to 125 Hz for analysis.
- The same normalized morphology extractor used by the main BP research pipeline processes 8-second segments.
- Overlapping accepted segments are unioned when calculating clean coverage.
- Finapres systolic peaks and intervening diastolic troughs provide the reference for each observation.
- Every duration ends at the same recording timestamp and uses the same trailing 10-second Finapres label interval.
- PPG quality decisions are made before, and independently of, BP values.
- The first usable recording for each public participant supplies calibration; every observation from that recording is excluded from prediction metrics.
- Accuracy is evaluated only on matched recording endpoints accepted at every duration. Five-second rolling candidates remain only for coverage analysis.

Outputs include the source manifest, every accepted/rejected observation and reason, calibration records, held-out predictions, per-duration metrics, and a reproducibility manifest.

No fitted model is exported. `deployment_eligible` is always false, and the tool never modifies the existing upper-arm model or live viewer. A promising result would justify testing the same frozen short-window policy on newly collected local upper-arm recovery data; it would not itself prove that the policy works on the arm.
