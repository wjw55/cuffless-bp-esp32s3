# Soak Test 001 Summary

## Test Date

June 20, 2026

## Hardware Setup

- Microcontroller: ESP32-S3.
- PPG sensor: MAX30102.
- Sensor placement for this validation: finger PPG.
- I2C wiring: SDA on GPIO 8, SCL on GPIO 9.
- MAX30102 interrupt pin: not used.
- Host collection path: ESP-IDF monitor-compatible serial output collected by `tools/collect_ppg.py`.

## Firmware Purpose

The firmware records raw red/IR MAX30102 PPG samples for later cuffless blood pressure modelling work. The raw CSV data row format remains fixed:

```csv
sample_seq,timestamp_ms,red,ir
```

Lines beginning with `#` are firmware status or diagnostic comments and are not raw sample rows.

## Timestamp-Resync Bug

A 90-second sanity test before this fix showed four `timestamp_resync` events with `lag_us=60000`. Each event inserted one artificial 70 ms timestamp interval even though:

- `sample_seq` stayed continuous.
- FIFO overflow count was 0.
- I2C error count was 0.
- Red/IR values were smooth across the affected rows.

This indicated firmware-side timestamp reconstruction drift, not true sensor sample loss or serial collection loss. Ordinary lag between `esp_timer_get_time()` and the nominal timestamp cursor was being treated as a timestamp resync and rebased into one emitted CSV row.

## Fix Summary

Ordinary timestamp lag is now diagnostic only. When lag exceeds the warning threshold, firmware emits `# warning event=timestamp_lag ...` and increments `timestamp_lag_warning_count`, but it does not modify `next_sample_timestamp_us`.

The monotonic timestamp guard remains active. FIFO overflow recovery still invalidates the timestamp cursor because that is an explicit acquisition discontinuity. Firmware stats now distinguish:

- `timestamp_resyncs`: true cursor resyncs after discontinuity.
- `timestamp_lag_warnings`: diagnostic lag warnings that do not rebase timestamps.
- `timestamp_corrections`: monotonic guard corrections, if any.

## 90-Second Sanity Test

Trial ID: `sanity_002`

| Metric | Result |
| --- | ---: |
| Sample count | 8973 |
| Data duration | 89.72 s |
| Median dt | 10.00 ms |
| Mean dt | 10.00 ms |
| dt min / p95 / p99 / max | 10.00 / 10.00 / 10.00 / 10.00 ms |
| Estimated sampling rate | 100.00 Hz |
| Missing sample sequences | 0 |
| Timestamp gaps >15 ms | 0 |
| Timestamp gaps >20 ms | 0 |
| Non-increasing timestamps | 0 |
| Timing quality | good |
| Warnings | none |

## 15-Minute Soak Test

Trial ID: `soak_001`

| Metric | Result |
| --- | ---: |
| Sample count | 89704 |
| Requested duration | 900.00 s |
| Data duration | 897.03 s |
| Median dt | 10.00 ms |
| Mean dt | 10.00 ms |
| dt min / p95 / p99 / max | 10.00 / 10.00 / 10.00 / 10.00 ms |
| Estimated sampling rate | 100.00 Hz |
| Missing sample sequences | 0 |
| Timestamp gaps >15 ms | 0 |
| Timestamp gaps >20 ms | 0 |
| Non-increasing timestamps | 0 |
| Timing quality | good |
| Warnings | none |

## Conclusion

Pass. The timestamp fix eliminated artificial lag-based timestamp gaps during continuous acquisition while preserving the fixed raw CSV row format.

This validates firmware and data-acquisition timing reliability for finger PPG collection with this ESP32-S3 and MAX30102 setup. It does not validate final upper-arm wearable placement, cuffless blood pressure model accuracy, or clinical blood pressure performance.
