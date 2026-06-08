# ESP32-S3 MAX30102 PPG Logger

Raw PPG acquisition firmware for an ESP32-S3 and MAX30102 sensor. The current goal is reliable red/IR PPG capture before collecting Omron cuff blood pressure labels.

## Wiring

| MAX30102 breakout | ESP32-S3 |
| --- | --- |
| VIN or 3V3 | 3V3 |
| GND | GND |
| SDA | GPIO 8 |
| SCL | GPIO 9 |
| INT | Not used |

The firmware enables internal I2C pullups, but many MAX30102 breakout boards already include pullups. If using a bare sensor board, add suitable external pullups, for example 4.7 kOhm to 3V3.

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
1,12355,48244,53145
# stats samples=500 rate_hz=99.8 fifo_avail=2 ovf=0 i2c_errors=0
```

Only rows without a leading `#` are data samples. Lines beginning with `#` are status/debug comments and are safe for the Python collector to ignore.

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
  --ppg-hand right `
  --cuff-arm left `
  --notes "Unlabeled stability check before BP collection"
```

The collector saves:

- `data/raw/<subject>_<session>_<trial_id>_ppg.csv`
- `data/raw/<subject>_<session>_<trial_id>_metadata.json`
- `data/raw/<subject>_<session>_<trial_id>_plot.png`
- `data/raw/<subject>_<session>_<trial_id>_zoom_plot.png`

Existing output files are not overwritten by default. Use `--overwrite` only when you intentionally want to replace a previous recording.

## CSV Format

```csv
sample_seq,timestamp_ms,red,ir
```

- `sample_seq`: monotonically increasing firmware sample counter.
- `timestamp_ms`: ESP timer timestamp in milliseconds, estimated at 100 Hz when draining FIFO batches.
- `red`: raw MAX30102 red channel ADC value.
- `ir`: raw MAX30102 IR channel ADC value.

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
