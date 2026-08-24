#ifndef LIVE_HR_H
#define LIVE_HR_H

#include <stdbool.h>
#include <stdint.h>

#define LIVE_HR_MAX_INTERVALS 7

typedef enum {
    LIVE_HR_WARMING_UP,
    LIVE_HR_STABLE,
    LIVE_HR_POOR_SIGNAL,
    LIVE_HR_NO_FINGER,
    LIVE_HR_INSUFFICIENT_BEATS,
} live_hr_status_t;

typedef struct {
    bool report_ready;
    int64_t timestamp_ms;
    live_hr_status_t status;
    float bpm;
    uint8_t beats;
} live_hr_report_t;

typedef struct {
    bool report_clock_initialized;
    bool filter_initialized;
    bool finger_present;
    int64_t next_report_ms;
    int64_t finger_start_ms;
    int64_t previous_sample_ms;
    int64_t last_peak_ms;
    float baseline;
    float smoothed;
    float previous_smoothed;
    float previous_previous_smoothed;
    float variance_ema;
    float signal_envelope;
    uint16_t intervals_ms[LIVE_HR_MAX_INTERVALS];
    uint8_t interval_count;
    uint8_t interval_write_index;
} live_hr_state_t;

void live_hr_init(live_hr_state_t *state);
bool live_hr_process_sample(
    live_hr_state_t *state,
    int64_t timestamp_ms,
    uint32_t ir,
    bool finger_present,
    live_hr_report_t *report);
const char *live_hr_status_name(live_hr_status_t status);

#endif
