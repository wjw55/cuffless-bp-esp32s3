#ifndef ADXL345_H
#define ADXL345_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

#define ADXL345_I2C_ADDRESS 0x53
#define ADXL345_SAMPLE_RATE_HZ 100
#define ADXL345_SAMPLE_PERIOD_US (1000000 / ADXL345_SAMPLE_RATE_HZ)
#define ADXL345_RANGE_G 4
#define ADXL345_SCALE_G_PER_LSB 0.0039f

bool adxl345_is_connected(void);
esp_err_t adxl345_init(void);
esp_err_t adxl345_get_fifo_entries(uint8_t *entries);
esp_err_t adxl345_read_fifo_sample(int16_t *x, int16_t *y, int16_t *z);
esp_err_t adxl345_reset_fifo(void);
uint32_t adxl345_get_i2c_error_count(void);

#endif
