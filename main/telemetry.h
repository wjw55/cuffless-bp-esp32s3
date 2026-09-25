#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <stdint.h>

#include "esp_err.h"

esp_err_t telemetry_init(void);
int telemetry_printf(const char *format, ...);
void telemetry_ppg_sample(uint64_t sequence, int64_t timestamp_ms, uint32_t red, uint32_t ir);
void telemetry_imu_sample(
    uint64_t sequence,
    int64_t timestamp_ms,
    int16_t x,
    int16_t y,
    int16_t z);
void telemetry_print_ble_stats(void);

#endif
