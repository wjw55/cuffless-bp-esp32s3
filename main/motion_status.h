#ifndef MOTION_STATUS_H
#define MOTION_STATUS_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    MOTION_STATUS_CALIBRATING,
    MOTION_STATUS_STILL,
    MOTION_STATUS_MOVING,
    MOTION_STATUS_IMU_UNAVAILABLE,
} motion_status_value_t;

#define MOTION_RMS_WINDOW_SAMPLES 100

typedef struct {
    int64_t timestamp_ms;
    motion_status_value_t status;
    float activity_g;
    float moving_threshold_g;
} motion_status_report_t;

typedef struct {
    bool filter_initialized;
    bool report_clock_initialized;
    motion_status_value_t status;
    int64_t first_sample_ms;
    int64_t next_report_ms;
    float gravity_x_g;
    float gravity_y_g;
    float gravity_z_g;
    float dynamic_squared_window[MOTION_RMS_WINDOW_SAMPLES];
    float dynamic_squared_sum;
    uint16_t dynamic_squared_index;
    uint16_t dynamic_squared_count;
    uint16_t above_moving_samples;
    uint16_t below_still_samples;
} motion_status_state_t;

void motion_status_init(motion_status_state_t *state);
bool motion_status_process_sample(
    motion_status_state_t *state,
    int64_t timestamp_ms,
    int16_t x_raw,
    int16_t y_raw,
    int16_t z_raw,
    motion_status_report_t *report);
const char *motion_status_name(motion_status_value_t status);

#endif
