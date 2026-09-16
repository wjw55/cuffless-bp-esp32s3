# Tool catalogue

The scripts kept directly in `tools/` are stable command-line entry points. Shared BP implementation lives in `bp_core/`, and all regression tests live in `tests/`.

## Acquisition and trial analysis

| Tool | Purpose |
| --- | --- |
| `collect_ppg.py` | Record synchronized raw PPG and IMU data, metadata and optional reference labels. |
| `analyze_trials.py` | Inspect recorded trials and run finger or upper-arm HR analysis. |
| `upper_arm_hr.py` | Shared upper-arm HR and waveform-quality implementation used by analysis and viewers. |

## Live presentation

| Tool | Purpose |
| --- | --- |
| `view_live_hr.py` | Firmware live-HR terminal display, primarily for the finger profile. |
| `view_live_upper_arm_hr.py` | PC rolling upper-arm HR preview. |
| `view_live_bp.py` | Experimental PC rolling BP preview with safe model gating and an opt-in 30-second development policy. |

## Motion and signal quality

| Tool | Purpose |
| --- | --- |
| `calibrate_motion.py` | Calibrate the original still/moving RMS threshold. |
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
| `compare_bp_recovery.py` | Compare shorter post-motion recovery windows with the stationary policy. |
| `evaluate_bp_feasibility.py` | Compare the unchanged strict BP gate with the gap-tolerant school-project feasibility gate. |
| `evaluate_bp_feasibility_models.py` | Run evaluation-only participant-held-out personalized BP models from frozen feasibility occasion features. |
| `motion_bp_pipeline.py` | Run the PPG-only versus PPG+IMU motion-band experiment. |
| `motion_bp.py` | Shared motion-aware BP experiment implementation. |
| `motion_feasibility.py` | Review whether PPG remains usable across motion categories. |
| `graphene_ppg_bp.py` | Separate public fingertip PPG/Finapres short-window experiment. |

The BP research tools are not interchangeable: `bp_pipeline.py` remains the main upper-arm stationary path, while the recovery, motion and Graphene tools are isolated experiments.

## Tests

Run every Python and host-side firmware-clock regression test from the project root:

```powershell
& "C:\wjw\Anaconda\python.exe" -m unittest discover -s tools\tests -p "test_*.py"
```

Do not place generated reports or datasets in `tools/`; they belong under the ignored `data/processed/` and `data/external/` directories.
