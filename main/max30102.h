#ifndef MAX30102_H
#define MAX30102_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

#define MAX30102_SAMPLE_RATE_HZ 100
#define MAX30102_SAMPLE_PERIOD_MS (1000 / MAX30102_SAMPLE_RATE_HZ)
#define MAX30102_FINGER_IR_THRESHOLD 50000
#define MAX30102_ADC_MAX_VALUE 0x3FFFF

bool max30102_is_connected(void);
esp_err_t max30102_init(void);
uint8_t max30102_available_samples(void);
esp_err_t max30102_read_fifo_sample(uint32_t *red, uint32_t *ir);

#endif
