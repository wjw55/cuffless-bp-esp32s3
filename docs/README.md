# Documentation index

## Current system and team workflow

- [Project overview, wiring and commands](../README.md)
- [Tool catalogue: primary commands, fallbacks and shared modules](../tools/README.md)
- [Data collection and labelling protocol](data_collection_protocol.md)
- [USB/BLE transport and validation procedure](ble_transport.md)
- [Offline BP pipeline, model packages and prediction](bp_pipeline.md)
- [Strict and gap-tolerant feasibility policies](bp_recovery_comparison.md#separate-gap-tolerant-feasibility-policy)
- [PPG clock tracking: implementation, bench evidence and limitations](ppg_timestamp_tracking.md)
- [Branches, testing and local-file hygiene](development_workflow.md)

The current live BP path is stationary and quality gated. The opt-in 30-second
policy is experimental. Historical results do not validate a newly soldered unit,
new housing, a new participant or BP during motion.

## Optional offline research

These guides describe separate experiments, not additional live requirements:

- [Short-window recovery comparison](bp_recovery_comparison.md)
- [Public Graphene PPG/Finapres experiment](graphene_ppg_bp.md)
- [PPG/IMU motion-feasibility investigation](motion_feasibility.md)
- [Motion-aware BP research pipeline](motion_bp_pipeline.md)
- [Archived experiment commands](../tools/experiments/README.md)

## Historical evidence

Completed finger-pilot reports and the original soak-test report are retained in
[archive/](archive/README.md). Their original settings and results are preserved
for provenance; use the current collection guide for new recordings.

Recordings, participant labels, fitted models and generated reports stay in the
ignored local data directories or secure backup, not in documentation or branches.
