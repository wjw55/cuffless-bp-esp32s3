# Tool catalogue

The scripts kept directly in `tools/` are current command-line entry points. Completed, reproducible research experiments live in `experiments/`. Shared BP implementation lives in `bp_core/`, and all regression tests live in `tests/`.

For the current team workflow, focus on `collect_ppg.py` for recording, `analyze_trials.py` for signal/HR review, `evaluate_bp_feasibility_models.py` for offline BP feasibility evaluation, and `view_live_bp_desktop.py` for the display-only live preview. The viewer does not collect or save study data.

## Acquisition and trial analysis

| Tool | Purpose |
| --- | --- |
| `collect_ppg.py` | Record synchronized raw PPG and IMU data, metadata and optional reference labels over USB serial or BLE. |
| `analyze_trials.py` | Inspect recorded trials and run finger or upper-arm HR analysis. |
| `upper_arm_hr.py` | Shared upper-arm HR and waveform-quality implementation used by analysis and viewers. |
| `line_transport.py` | Shared blocking USB serial/BLE line source used by the collector and BP viewer. |

## Live presentation

| Tool | Purpose |
| --- | --- |
| `view_live_hr.py` | Firmware live-HR terminal display, primarily for the finger profile. |
| `view_live_upper_arm_hr.py` | PC rolling upper-arm HR preview. |
| `view_live_bp.py` | Experimental PC rolling BP preview over USB serial or BLE, with safe model gating and an opt-in 30-second development policy. |
| `view_live_bp_desktop.py` | Display-only Windows BP window with USB/BLE setup, waveform, motion, health and the same inference gates as the terminal viewer. |

## Motion and signal quality

| Tool | Purpose |
| --- | --- |
| `motion_study_protocol.py` | Define and validate prompted motion-study schedules. |
| `prepare_motion_quality.py` | Prepare and finalize manually reviewed motion-quality windows. |
| `train_motion_quality.py` | Train the usable/unusable classifier. |
| `evaluate_motion_quality.py` | Evaluate a frozen classifier without retraining. |
| `motion_quality.py` | Shared synchronization and feature extraction. |
| `motion_quality_classifier.py` | Shared classifier training and evaluation. |
| `motion_quality_shadow.py` | Run the frozen classifier in non-controlling shadow mode. |

## Blood-pressure research

| Tool | Purpose |
| --- | --- |
| `bp_pipeline.py` | Main stationary personalized PPG-to-BP pipeline. |
| `bp_core/` | Shared dataset, feature, model, inference and reporting modules. |
| `evaluate_bp_feasibility.py` | Compare the unchanged strict BP gate with the gap-tolerant school-project feasibility gate. |
| `evaluate_bp_feasibility_models.py` | Run evaluation-only participant-held-out personalized BP models from frozen feasibility occasion features. |

`bp_pipeline.py` remains the main upper-arm stationary path. The active feasibility evaluators do not change the live BP algorithm.

## Archived experiments

These five completed experiments remain runnable for reproducing earlier findings; invoke them from the project root using their `tools/experiments/` paths.

| Tool | Purpose |
| --- | --- |
| `experiments/calibrate_motion.py` | Original still/moving RMS threshold calibration. |
| `experiments/compare_bp_recovery.py` | Shorter post-motion recovery-window comparison. |
| `experiments/graphene_ppg_bp.py` | Public fingertip PPG/Finapres short-window experiment. |
| `experiments/motion_bp_pipeline.py` and `experiments/motion_bp.py` | PPG-only versus PPG+IMU motion-band experiment and its shared implementation. |
| `experiments/motion_feasibility.py` | Offline review of PPG usability across motion categories. |

## Tests

Run every Python and host-side firmware-clock regression test from the project root:

```powershell
& "C:\wjw\Anaconda\python.exe" -m unittest discover -s tools\tests -p "test_*.py"
```

Do not place generated reports or datasets in `tools/`; they belong under the ignored `data/processed/` and `data/external/` directories.
