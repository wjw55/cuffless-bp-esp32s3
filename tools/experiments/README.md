# Archived research experiments

These completed experiments are retained to reproduce earlier findings. They do
not replace the collector, the stationary BP pipeline, or the live dashboard.
No new dataset, model or threshold change is implied by keeping them here.

Run from the project root, using the existing paths:

| Entry point | Purpose | Guide |
| --- | --- | --- |
| `tools/experiments/calibrate_motion.py` | Original rolling-RMS still/moving calibration | [Project protocol](../../README.md) |
| `tools/experiments/compare_bp_recovery.py` | Shorter-window recovery comparison | [Recovery comparison](../../docs/bp_recovery_comparison.md) |
| `tools/experiments/graphene_ppg_bp.py` | Public PPG/Finapres source-domain comparison | [Graphene experiment](../../docs/graphene_ppg_bp.md) |
| `tools/experiments/motion_feasibility.py` | Offline waveform/motion feasibility review | [Motion feasibility](../../docs/motion_feasibility.md) |
| `tools/experiments/motion_bp_pipeline.py` | Offline PPG versus PPG+IMU model comparison | [Motion BP pipeline](../../docs/motion_bp_pipeline.md) |

`motion_bp.py` is shared implementation for the motion-BP experiment, not an
additional everyday command. The five experiments comprise six Python
implementation files plus the package marker. Their regression tests remain in
`tools/tests/` and are included in the full suite.

See [the active tool catalogue](../README.md) for routine commands. Do not move
active shared code into this directory just because it originated in research.

## Active offline investigation

`compare_pulse_detectors.py` is a separate, offline-only comparison of the existing
upper-arm pulse detector and an autocorrelation-guided candidate. It is not a
sixth completed experiment or a replacement live algorithm. See the
[comparison guide](../../docs/pulse_detector_comparison.md). Generated comparisons
remain local; no BP model is fitted or changed.
