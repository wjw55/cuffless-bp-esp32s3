# PPG timestamp tracking development fix

This is a development build on main, not a new participant firmware freeze.
`participant-study-fw-v1.0` and `rtos-trial` remain unchanged. Restore the archived
participant firmware before participant collection unless a new freeze has been
explicitly validated and approved.

## Problem and approach

The old PPG cursor advanced exactly 10,000 us per sample regardless of the sensor's
actual clock. The IMU already tracked ESP time. A five-minute bench capture showed
708.897 ms reported PPG lag, approximately 2.50 ms/s growth, with no sequence gaps,
reported I2C errors or FIFO overflows. The actual PPG output rate was approximately
99.75 Hz relative to ESP time while the raw timestamps appeared exactly 100 Hz.

`main/ppg_clock.c` now backdates the initial FIFO timestamp to the oldest queued
sample. For each subsequent FIFO snapshot it compares the cursor to observed
ESP read time, accounting for queued samples. A 1/16 phase correction is limited
to 250 us per FIFO observation. This removes accumulated clock drift without
replacing timestamps with irregular serial receipt times. Within each FIFO batch
the nominal 10 ms spacing is preserved.

Observations with phase errors above 40 ms or invalid FIFO depths are rejected
without shifting the cursor. Such faults produce explicit diagnostics rather
than being absorbed into a clock correction. Existing overflow recovery resets
the observation origin. A 1 ms monotonic minimum replaces the previous fixed
10 ms floor, allowing tracking in either clock direction without duplicate
millisecond timestamps. Normal per-observation corrections yield approximately
9–11 ms CSV intervals, not deliberately uniform 10 ms intervals.

The raw PPG and IMU CSV headers/columns are unchanged. Existing `# stats` lines
add `clock_adjustments`, `clock_adjustment_us`, `clock_phase_error_us` and
`clock_rejected_observations`. The collector retains these in metadata. The
offline motion-BP gate rejects clock-observation faults. The live motion BP
policy, stationary models and shadow classifier are unchanged.

## Limits

FIFO polling and I2C latency mean the observed time is not an exact optical
data-ready timestamp. Small residual phase error and the absence of lag warnings
do not prove sub-millisecond PPG/IMU alignment. The fixed controller is tested
for regular acquisition with modest FIFO batches; sustained backlog, large
clock differences or long stalls can still trigger rejection. Additional load,
long-uptime and independent data-ready timing checks are needed before freezing
participant firmware or using cross-modal BP results.

## Verification

`tools/tests/test_ppg_clock.c` executes the actual firmware clock implementation over
ten-minute simulated slow/fast clocks, polling jitter and FIFO batches, and checks
monotonicity, bounded timestamp error, invalid depth, large latency rejection and
reinitialization. `tools/tests/test_ppg_clock.py` builds/runs it using GCC or the installed
MSVC host compiler as part of the Python regression suite. It explicitly skips
when no host compiler is available; require a successful C test before flashing.

The development firmware is built with ESP-IDF v5.5.2 and flashed only to the
application partition. Capture raw serial lines with host monotonic receipt times,
startup build identity and all status lines. Retain full captures and binary
hashes under ignored `data/processed/motion_bp_v1/`. Compare sequence continuity,
timestamp intervals, reported sensor faults, phase residuals, cumulative clock
correction and sensor-time duration against host receipt duration. Host receipt
latency is supporting evidence, not a ground-truth synchronization reference.

## Five-minute bench result, 2026-09-12

The corrected application was built, flashed to COM5 with flash hash verification,
and its startup ELF hash matched the built application. All 193 regression tests
passed, including the host C simulations. The targeted motion-BP tests were also
rerun after adding clock-fault rejection assertions.

| Measure | Original run | Corrected run |
|---|---:|---:|
| Duration | 300 s | 300 s |
| PPG lag warnings | 14 | 0 |
| Last reported untracked lag | 708.897 ms | No lag warning |
| Absolute PPG timestamp-span vs host receipt-span difference | 728.984 ms | 11.938 ms |
| PPG/IMU sequence discontinuities | 0 / 0 | 0 / 0 |
| Reported I2C errors / FIFO overflows | 0 / 0 | 0 / 0 |
| Rejected PPG clock observations | Not available | 0 |

In the corrected run, the 59 five-second status samples had phase errors from
-15 us to 8,760 us (95th percentile absolute error 7,281 us). This is sampled
phase telemetry, not a bound on every sample's alignment error. The last status
reported 749,827 us of cumulative PPG clock correction. PPG timestamps remained
strictly increasing, with maximum interval 11 ms. This confirms removal of the
observed accumulated drift in this run; it does not establish exact optical/IMU
alignment or participant-study readiness.

Corrected binary SHA256:
`5771133987380121f5b705a0e5889caf93f226cc4d3393fe862974a9ed0bda69`.
ELF SHA256:
`e3c9f13afe22323564811d4739325edce65f24068e32b777194f14c5254a67c0`.
The startup app-version string was `1`; use the hashes and source diff to identify
this development build, not that generic version string.

Local evidence folders (ignored by Git):
`data/processed/motion_bp_v1/bench_20260912T005524/` and
`data/processed/motion_bp_v1/fixed_bench_20260912T011029/`.
COM5 was closed after capture. The board remains on the corrected development
application; the archived participant freeze was neither overwritten nor retagged.
