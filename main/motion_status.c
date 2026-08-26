#include "motion_status.h"

#include <math.h>
#include <string.h>

#include "adxl345.h"
#include "sdkconfig.h"

#define MOTION_REPORT_PERIOD_MS 1000
#define MOTION_FILTER_WARMUP_MS 2000
#define MOTION_GRAVITY_ALPHA 0.01f
#define MOTION_STILL_THRESHOLD_RATIO 0.70f
#define MOTION_MOVING_DWELL_SAMPLES 21
#define MOTION_STILL_DWELL_SAMPLES 101

static float configured_moving_threshold_g(void)
{
    return (float)CONFIG_MOTION_THRESHOLD_MG / 1000.0f;
}

void motion_status_init(motion_status_state_t *state)
{
    memset(state, 0, sizeof(*state));
    state->status = MOTION_STATUS_CALIBRATING;
}

static void update_state(motion_status_state_t *state, float activity_g)
{
    const float moving_threshold_g = configured_moving_threshold_g();
    if (moving_threshold_g <= 0.0f) {
        state->status = MOTION_STATUS_CALIBRATING;
        return;
    }

    const float still_threshold_g = moving_threshold_g * MOTION_STILL_THRESHOLD_RATIO;
    if (activity_g > moving_threshold_g) {
        state->above_moving_samples++;
        state->below_still_samples = 0;
        if (state->above_moving_samples >= MOTION_MOVING_DWELL_SAMPLES) {
            state->status = MOTION_STATUS_MOVING;
        }
    } else if (activity_g < still_threshold_g) {
        state->below_still_samples++;
        state->above_moving_samples = 0;
        if (state->below_still_samples >= MOTION_STILL_DWELL_SAMPLES) {
            state->status = MOTION_STATUS_STILL;
        }
    } else {
        state->above_moving_samples = 0;
        state->below_still_samples = 0;
    }
}

bool motion_status_process_sample(
    motion_status_state_t *state,
    int64_t timestamp_ms,
    int16_t x_raw,
    int16_t y_raw,
    int16_t z_raw,
    motion_status_report_t *report)
{
    const float x_g = (float)x_raw * ADXL345_SCALE_G_PER_LSB;
    const float y_g = (float)y_raw * ADXL345_SCALE_G_PER_LSB;
    const float z_g = (float)z_raw * ADXL345_SCALE_G_PER_LSB;

    if (!state->filter_initialized) {
        state->filter_initialized = true;
        state->first_sample_ms = timestamp_ms;
        state->gravity_x_g = x_g;
        state->gravity_y_g = y_g;
        state->gravity_z_g = z_g;
    }
    if (!state->report_clock_initialized) {
        state->report_clock_initialized = true;
        state->next_report_ms = timestamp_ms + MOTION_REPORT_PERIOD_MS;
    }

    state->gravity_x_g += MOTION_GRAVITY_ALPHA * (x_g - state->gravity_x_g);
    state->gravity_y_g += MOTION_GRAVITY_ALPHA * (y_g - state->gravity_y_g);
    state->gravity_z_g += MOTION_GRAVITY_ALPHA * (z_g - state->gravity_z_g);

    const float dynamic_x_g = x_g - state->gravity_x_g;
    const float dynamic_y_g = y_g - state->gravity_y_g;
    const float dynamic_z_g = z_g - state->gravity_z_g;
    const float dynamic_squared =
        (dynamic_x_g * dynamic_x_g) + (dynamic_y_g * dynamic_y_g) + (dynamic_z_g * dynamic_z_g);
    if (state->dynamic_squared_count == MOTION_RMS_WINDOW_SAMPLES) {
        state->dynamic_squared_sum -=
            state->dynamic_squared_window[state->dynamic_squared_index];
    } else {
        state->dynamic_squared_count++;
    }
    state->dynamic_squared_window[state->dynamic_squared_index] = dynamic_squared;
    state->dynamic_squared_sum += dynamic_squared;
    state->dynamic_squared_index =
        (state->dynamic_squared_index + 1U) % MOTION_RMS_WINDOW_SAMPLES;
    const float activity_g = sqrtf(fmaxf(
        state->dynamic_squared_sum / (float)state->dynamic_squared_count,
        0.0f));

    if ((timestamp_ms - state->first_sample_ms) < MOTION_FILTER_WARMUP_MS) {
        state->status = MOTION_STATUS_CALIBRATING;
    } else {
        update_state(state, activity_g);
    }

    if (timestamp_ms < state->next_report_ms) {
        return false;
    }
    while (state->next_report_ms <= timestamp_ms) {
        state->next_report_ms += MOTION_REPORT_PERIOD_MS;
    }

    report->timestamp_ms = timestamp_ms;
    report->status = state->status;
    report->activity_g = activity_g;
    report->moving_threshold_g = configured_moving_threshold_g();
    return true;
}

const char *motion_status_name(motion_status_value_t status)
{
    switch (status) {
    case MOTION_STATUS_CALIBRATING:
        return "calibrating";
    case MOTION_STATUS_STILL:
        return "still";
    case MOTION_STATUS_MOVING:
        return "moving";
    case MOTION_STATUS_IMU_UNAVAILABLE:
        return "imu_unavailable";
    default:
        return "unknown";
    }
}
