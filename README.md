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

## Collect A 90-Second PPG Recording

Close `idf.py monitor` first because only one program can use the serial port at a time.

```powershell
python tools\collect_ppg.py `
  --port COM3 `
  --duration 90 `
  --subject S01 `
  --session baseline_001 `
  --trial-id T01 `
  --posture seated `
  --sensor-location index_finger `
  --ppg-hand right `
  --cuff-arm left `
  --notes "Unlabeled stability check before BP collection"
```

The collector saves:

- `data/raw/<subject>_<session>_ppg.csv`
- `data/raw/<subject>_<session>_metadata.json`
- `data/raw/<subject>_<session>_plot.png`
- `data/raw/<subject>_<session>_zoom_plot.png`

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

Once stable, optional BP fields can be added to the collector command:

```powershell
python tools\collect_ppg.py --port COM3 --duration 90 --subject S01 --session bp_001 --trial-id T01 --posture seated --sensor-location index_finger --ppg-hand right --cuff-arm left --systolic-mmhg 118 --diastolic-mmhg 76 --cuff-hr-bpm 72 --cuff-timestamp 2026-06-07T20:15:00+08:00
```
