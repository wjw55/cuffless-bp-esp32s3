# BLE telemetry transport

BLE is an optional transport for the XIAO ESP32-S3 logger. It does not change sensor acquisition, timestamps, sequence counters, signal-quality rules, BP inference, or raw CSV columns. USB serial remains available for flashing, debugging, and fallback collection.

## Firmware behavior

- Acquisition starts automatically at boot whether or not a BLE client is present.
- The peripheral name is `PPG-LOGGER-<last-six-MAC-digits>`.
- NimBLE runs as an unpaired, unbonded peripheral with one connection.
- Service UUID: `9f5c0001-6f5a-4b7d-9a21-3d4e5f607182`.
- Notify characteristic UUID: `9f5c0002-6f5a-4b7d-9a21-3d4e5f607182`.
- Preferred ATT MTU: 247 bytes, while the compact protocol also supports the default 23-byte MTU used by Windows on the development laptop.
- PPG and IMU notifications contain up to 15 losslessly packed samples when the negotiated MTU allows it. Every packet carries its own starting sequence and timestamp, so a missing notification creates an explicit sequence gap without corrupting later samples. All raw values, sequence numbers and millisecond timestamps are reconstructed before the existing PC parsers run.
- Separate non-blocking PPG, IMU and status queues decouple acquisition from BLE transmission. The sender runs at low priority on the second ESP32-S3 core, batches queued samples and adapts its payload to the negotiated MTU. A temporarily failed notification is retried without blocking acquisition or silently losing its payload.
- Data is queued only while notifications are subscribed. Disconnecting discards stale queued data and restarts advertising.
- USB serial uses a buffered, zero-wait writer. An attached but unread USB port—or a power-only USB source—can therefore never block sensor acquisition. BLE and USB carry the same records when both clients are actively draining data; if USB is not being read, excess USB bytes are discarded without affecting BLE or the sensors.
- Status notifications use sequenced UTF-8 fragments. The PC decoder discards an incomplete status line if a fragment is lost, combines intact fragments and converts compact sample packets back to the original newline-delimited PPG and IMU rows. The decoder also accepts the earlier stateful compact and unframed-text BLE firmware for compatibility.
- `# ble_stats` is emitted approximately every five seconds together with the existing sensor statistics.

`sdkconfig.defaults` contains the tracked BLE-only NimBLE configuration. Classic Bluetooth, pairing, bonding, central mode, and observer mode are disabled.

## PC setup

Install Bleak into the same Python environment used for collection:

```powershell
& "C:\wjw\Anaconda\python.exe" -m pip install -r requirements-ble.txt
```

Both commands default to serial. For BLE, pass `--transport ble`; `--port` must then be omitted. `--ble-device` accepts the exact advertised name or OS-specific address. If it is omitted, automatic selection succeeds only when exactly one `PPG-LOGGER-*` device is discovered.

```powershell
& "C:\wjw\Anaconda\python.exe" tools\view_live_bp.py `
  --transport ble `
  --ble-device PPG-LOGGER-A1B2C3 `
  --participant-id P001 `
  --model-dir data\processed\bp\20260905T235649\single_subject\P001 `
  --experimental-fast-window 30 `
  --last-validated-max-age 300
```

BLE transport metadata is additive. PPG and IMU CSV schemas remain unchanged. The collector never transmits the participant ID to the ESP32.

## Disconnect policy

- Collector: stop at the first disconnect, save received samples, and mark the attempt incomplete. Pre- and post-reconnection data are never silently combined.
- BP viewer: reconnect automatically. Silence makes sensor data stale and hides both current and held BP. A partial line from the old connection is discarded; the resulting sequence discontinuity restarts the clean buffer.

## Hardware validation

Complete these checks after flashing the BLE firmware:

1. Record 90 seconds over USB with no BLE client connected.
2. Record 90 seconds over BLE.
3. Capture USB and BLE simultaneously with a diagnostic receiver, then compare rows sharing the same PPG or IMU sequence number.
4. Disconnect and reconnect the BP viewer deliberately; confirm BP is hidden and the clean buffer restarts.
5. Run a 30-minute BLE recording.

Accept the transport when both streams remain approximately 100 Hz with at least 99.5% completeness, matching USB/BLE sequence numbers have identical values, and BLE queue drops, I2C errors, and FIFO overflows all remain zero. This first BLE version is unpaired and must not carry identifiable clinical information.
