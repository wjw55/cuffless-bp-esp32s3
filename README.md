# ESP32-S3 MAX30102 PPG + ADXL345 Motion Logger

Synchronized raw PPG and motion acquisition firmware for an ESP32-S3, MAX30102, and optional ADXL345. The ADXL345 is used to flag general body/arm motion that may corrupt PPG; it is not a blood-pressure model input.

## Wiring

| MAX30102 breakout | ESP32-S3 |
| --- | --- |
| VIN or 3V3 | 3V3 |
| GND | GND |
| SDA | GPIO 8 |
| SCL | GPIO 9 |
| INT | Not used |

The firmware enables internal I2C pullups, but many MAX30102 breakout boards already include pullups. If using a bare sensor board, add suitable external pullups, for example 4.7 kOhm to 3V3.

The GY-291/ADXL345 shares the same I2C bus:

| GY-291 ADXL345 | ESP32-S3 |
| --- | --- |
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO 8 |
| SCL | GPIO 9 |
| CS | High / I2C mode |
| SDO or ALT ADDRESS | GND, selecting `0x53` |
| INT1 / INT2 | Not used |

Use 3.3 V power and logic. Both breakouts may contain pullups; if communication is unreliable, measure the combined pullup resistance before adding more. The firmware expects the MAX30102 at `0x57` and verifies the ADXL345 device ID at `0x53`. If the IMU is absent, it warns and continues in PPG-only mode.

## ESP-IDF Environment

Open an ESP-IDF terminal before building. On Windows, use the ESP-IDF PowerShell or ESP-IDF Command Prompt installed by Espressif. From the repository root:

```powershell
cd C:\wjw\cuffless_bp_idf\ppg_logger
idf.py set-target esp32s3
```

## Build

```powershell
idf.py build
```

## Flash

Replace `COM3` with your board port:

```powershell
idf.py -p COM3 flash
```

## Monitor

```powershell
idf.py -p COM3 monitor
```

The firmware prints CSV samples and comment-prefixed status lines. Press `Ctrl+]` to exit the ESP-IDF monitor.

## Expected Serial Output

```csv
sample_seq,timestamp_ms,red,ir
0,12345,48231,53120
imu,0,12346,-12,8,258
1,12355,48244,53145
# stats samples=500 captured_samples=500 rate_hz=99.8 effective_rate_hz=99.9 fifo_avail=2 ovf=0 i2c_errors=0 timestamp_resyncs=0 timestamp_corrections=0 overflow_recoveries=0
# imu_stats samples=500 rate_hz=100.0 effective_rate_hz=99.9 fifo_entries=1 fifo_overflows=0 i2c_errors=0 timestamp_resyncs=0 timestamp_corrections=0 clock_adjustments=480 clock_adjustment_us=-11840
# hr timestamp_ms=20420 bpm=72.4 status=stable beats=6
```

Four-column numeric rows remain backward-compatible PPG records. Six-column rows tagged `imu` are ADXL345 records. Lines beginning with `#` are status/debug records.

## Live Heart Rate

The firmware prints a live heart-rate status approximately once per second without changing either raw stream. It waits eight seconds after detecting a finger, causally removes the IR baseline, smooths the signal, detects plausible peaks between 40 and 180 BPM, and reports the median of up to seven recent beat intervals.

```text
# hr timestamp_ms=10420 bpm=na status=warming_up beats=2
# hr timestamp_ms=20420 bpm=72.4 status=stable beats=6
```

Possible states are `warming_up`, `stable`, `poor_signal`, `no_finger`, and `insufficient_beats`. Only `stable` contains a BPM. Live BPM is a demonstration and signal-quality estimate; saved offline analysis remains authoritative until repeated comparisons against offline and Omron HR are acceptable.

## Collect A 90-Second PPG-Only Recording

Close `idf.py monitor` first because only one program can use the serial port at a time.

```powershell
python tools\collect_ppg.py `
  --port COM3 `
  --duration 90 `
  --subject test `
  --session baseline_001 `
  --trial-id ppg_only_001 `
  --posture seated `
  --sensor-location right_index_finger `
  --imu-location right_forearm `
  --imu-orientation x_distal_y_left_z_outward `
  --ppg-hand right `
  --cuff-arm left `
  --notes "Unlabeled stability check before BP collection"
```

The collector saves:

- `data/raw/<subject>_<session>_<trial_id>_ppg.csv`
- `data/raw/<subject>_<session>_<trial_id>_imu.csv`
- `data/raw/<subject>_<session>_<trial_id>_metadata.json`
- `data/raw/<subject>_<session>_<trial_id>_plot.png`
- `data/raw/<subject>_<session>_<trial_id>_zoom_plot.png`
- `data/raw/<subject>_<session>_<trial_id>_motion_plot.png`

Existing output files are not overwritten by default. Use `--overwrite` only when you intentionally want to replace a previous recording.

## CSV Format

```csv
sample_seq,timestamp_ms,red,ir
```

- `sample_seq`: monotonically increasing firmware sample counter.
- `timestamp_ms`: ESP timer timestamp in milliseconds. The first captured sample initializes the firmware timestamp cursor, then each emitted sample advances by the nominal 10 ms period.
- `red`: raw MAX30102 red channel ADC value.
- `ir`: raw MAX30102 IR channel ADC value.

The raw IMU CSV remains separate:

```csv
imu_seq,timestamp_ms,x_raw,y_raw,z_raw
```

The axes are signed raw readings converted during analysis using approximately `0.0039 g/LSB`. PPG and IMU timestamps use the same ESP timer, while each stream retains its own sequence counter and FIFO timing. The IMU cursor continuously tracks the observed ADXL345 clock because its real output rate can differ from the nominal 100 Hz; `clock_adjustments` and cumulative `clock_adjustment_us` appear in `# imu_stats`.

The motion plot contains PPG, acceleration magnitude, gravity-removed dynamic acceleration, and exploratory motion candidates. Its threshold is calculated independently for each recording as the median plus six scaled median absolute deviations. Validate these flags with labelled motion periods before using them to reject data.

## IMU Validation Order

1. Confirm startup finds `0x57` and verifies the ADXL345 at `0x53`.
2. Point each axis upward and downward; the gravity axis should read near `+1 g` or `-1 g`, with the other axes near zero.
3. Make a stationary 90-second recording. Require approximately 100 Hz, monotonic timestamps, no missing sequences, no FIFO overflows, and no I2C errors for both streams.
4. Record labelled stillness, gentle arm movement, larger movement, and deliberate sensor disturbance. Confirm flagged movement aligns with PPG artifacts.

An arm- or torso-mounted ADXL345 measures general motion and may miss local finger movement. Keep its mounting position and axis orientation consistent and record both for every trial.

## AD8232 Is Deferred

Do not add the AD8232 until combined PPG+IMU capture passes the validation sequence. ECG integration needs a verified ESP32-S3 ADC pin, 250-500 Hz acquisition, lead-off inputs, filtering, synchronization, and noise tests. A hobby AD8232 module is not medically isolated; define a battery-powered or properly isolated on-body setup before attaching electrodes, and do not create a body-to-mains/USB-ground path.

## Before Omron-Labeled Collection

Do not start Omron-labeled BP data collection until raw PPG capture is stable. First verify:

- `# stats` reports approximately 100 Hz over a 90-second recording.
- `ovf=0` or rare, explainable FIFO overflows.
- `i2c_errors=0` during normal contact.
- The quick-look and zoom plots show a visible pulse waveform without saturation.
- The metadata JSON has the correct subject, session, trial, posture, sensor location, cuff arm, and PPG hand.

## Omron-Labeled Pilot Setup

For the first pilot, keep the protocol deliberately small and repeatable:

- Omron cuff on left upper arm.
- MAX30102 PPG sensor on right index finger or right middle finger.
- Seated posture, back supported, feet flat.
- Both arms supported on a table.
- Start the Omron measurement around 20-30 seconds after PPG recording begins.
- Do not talk or move during recording.
- Rest 1-2 minutes between trials.
- Do only 3 labeled trials first.

## Collect A 90-Second Omron-Labeled Pilot Recording

```powershell
python tools\collect_ppg.py --port COM3 --duration 90 --subject test --session omron_pilot_001 --trial-id omron_001 --posture seated --sensor-location right_index_finger --cuff-arm left --ppg-hand right --cuff-start-time-s 25 --notes "Omron pilot trial 1" --prompt-bp-after
```

After recording finishes, enter the Omron systolic BP, diastolic BP, cuff HR, cuff reading time, and any extra notes when prompted. Leave a prompt blank to omit that field or keep a value already supplied by CLI.

The CSV remains raw signal only:

```csv
sample_seq,timestamp_ms,red,ir
```

BP and protocol fields are saved in the metadata JSON next to the CSV, not mixed into the signal file.

## Analyze Omron Pilot Trials

After collecting a small pilot session, summarize trial quality before using any data for later modeling:

```powershell
python tools\analyze_trials.py --input-dir data\raw --session omron_pilot_001
```

Optional filters and diagnostics:

```powershell
python tools\analyze_trials.py --input-dir data\raw --session omron_pilot_001 --subject test --make-plots --verbose
```

The analyzer is for pilot data validation only. It does not train a blood pressure model and does not predict BP. It reads the raw CSV and matching metadata JSON files, then writes:

- `data/processed/<session_id>/session_summary.csv`
- `data/processed/<session_id>/session_summary.json`
- `data/processed/<session_id>/plots/<trial_id>_ir_peaks.png` when `--make-plots` is used

`analysis_quality` is a practical triage label:

- `usable`: timing is good or usable, no missing samples, enough IR peaks, and PPG HR is plausible.
- `borderline`: timing is borderline, metadata warnings exist, signal span is small, or PPG HR differs from Omron HR by more than 10 bpm.
- `reject`: missing CSV/metadata, missing sample sequences, rejected timing, too few peaks, or no plausible PPG HR estimate.

For future modeling, start with only `usable` trials. Review `borderline` trials manually with the generated peak plots before deciding whether to keep them.
