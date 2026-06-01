# Cuffless BP ESP32-S3

This repository contains the ESP-IDF firmware for a CDE3301 cuffless blood pressure wearable prototype.

Current hardware:
- ESP32-S3
- MAX30102 PPG sensor

Current features:
- I2C communication with MAX30102
- RED and IR raw PPG logging
- Simple finger detection using IR threshold
- CSV-style serial output for later data collection

Current CSV output format:

```csv
timestamp_ms,red,ir,finger