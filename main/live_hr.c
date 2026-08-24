#include "live_hr.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define LIVE_HR_REPORT_PERIOD_MS 1000
#define LIVE_HR_WARMUP_MS 8000
#define LIVE_HR_MIN_BPM 40.0f
#define LIVE_HR_MAX_BPM 180.0f
#define LIVE_HR_MIN_INTERVAL_MS ((uint16_t)(60000.0f / LIVE_HR_MAX_BPM))
#define LIVE_HR_MAX_INTERVAL_MS ((uint16_t)(60000.0f / LIVE_HR_MIN_BPM))
#define LIVE_HR_BASELINE_ALPHA 0.01f
#define LIVE_HR_SMOOTHING_ALPHA 0.20f
#define LIVE_HR_VARIANCE_ALPHA 0.01f
#define LIVE_HR_ENVELOPE_DECAY 0.995f
#define LIVE_HR_THRESHOLD_STD_FRACTION 0.25f
#define LIVE_HR_MIN_PEAK_THRESHOLD 50.0f
#define LIVE_HR_MIN_SIGNAL_AMPLITUDE 500.0f
#define LIVE_HR_MIN_INTERVALS_FOR_REPORT 3

static void reset_signal_state(live_hr_state_t *state)
{
    state->filter_initialized = false;
    state->finger_present = false;
    state->finger_start_ms = 0;
    state->previous_sample_ms = 0;
    state->last_peak_ms = 0;
    state->baseline = 0.0f;
    state->smoothed = 0.0f;
    state->previous_smoothed = 0.0f;
    state->previous_previous_smoothed = 0.0f;
    state->variance_ema = 0.0f;
    state->signal_envelope = 0.0f;
    state->interval_count = 0;
    state->interval_write_index = 0;
    memset(state->intervals_ms, 0, sizeof(state->intervals_ms));
}

void live_hr_init(live_hr_state_t *state)
{
    memset(state, 0, sizeof(*state));
}

static void store_interval(live_hr_state_t *state, uint16_t interval_ms)
{
    state->intervals_ms[state->interval_write_index] = interval_ms;
    state->interval_write_index = (state->interval_write_index + 1) % LIVE_HR_MAX_INTERVALS;
    if (state->interval_count < LIVE_HR_MAX_INTERVALS) {
        state->interval_count++;
    }
}

static float median_interval_ms(const live_hr_state_t *state)
{
    uint16_t sorted[LIVE_HR_MAX_INTERVALS] = {0};
    for (uint8_t index = 0; index < state->interval_count; index++) {
        sorted[index] = state->intervals_ms[index];
    }

    for (uint8_t index = 1; index < state->interval_count; index++) {
        uint16_t value = sorted[index];
        int position = index - 1;
        while (position >= 0 && sorted[position] > value) {
            sorted[position + 1] = sorted[position];
            position--;
        }
        sorted[position + 1] = value;
    }

    uint8_t middle = state->interval_count / 2;
    if ((state->interval_count % 2) != 0) {
        return (float)sorted[middle];
    }
    return ((float)sorted[middle - 1] + (float)sorted[middle]) / 2.0f;
}

static void process_peak_candidate(live_hr_state_t *state, int64_t peak_timestamp_ms)
{
    if (state->last_peak_ms == 0) {
        state->last_peak_ms = peak_timestamp_ms;
        return;
    }

    int64_t interval_ms = peak_timestamp_ms - state->last_peak_ms;
    if (interval_ms < LIVE_HR_MIN_INTERVAL_MS) {
        return;
    }

    if (interval_ms > LIVE_HR_MAX_INTERVAL_MS) {
        state->interval_count = 0;
        state->interval_write_index = 0;
        memset(state->intervals_ms, 0, sizeof(state->intervals_ms));
    } else {
        store_interval(state, (uint16_t)interval_ms);
    }

    state->last_peak_ms = peak_timestamp_ms;
}

static void update_filter_and_peaks(live_hr_state_t *state, int64_t timestamp_ms, uint32_t ir)
{
    float ir_value = (float)ir;
    if (!state->filter_initialized) {
        state->filter_initialized = true;
        state->baseline = ir_value;
        state->previous_sample_ms = timestamp_ms;
        return;
    }

    state->baseline += LIVE_HR_BASELINE_ALPHA * (ir_value - state->baseline);
    float detrended = ir_value - state->baseline;
    state->smoothed += LIVE_HR_SMOOTHING_ALPHA * (detrended - state->smoothed);
    state->variance_ema += LIVE_HR_VARIANCE_ALPHA *
                           ((state->smoothed * state->smoothed) - state->variance_ema);

    float absolute_signal = fabsf(state->smoothed);
    state->signal_envelope *= LIVE_HR_ENVELOPE_DECAY;
    if (absolute_signal > state->signal_envelope) {
        state->signal_envelope = absolute_signal;
    }

    float threshold = LIVE_HR_THRESHOLD_STD_FRACTION * sqrtf(fmaxf(state->variance_ema, 0.0f));
    if (threshold < LIVE_HR_MIN_PEAK_THRESHOLD) {
        threshold = LIVE_HR_MIN_PEAK_THRESHOLD;
    }

    bool is_peak = state->previous_smoothed > threshold &&
                   state->previous_smoothed >= state->previous_previous_smoothed &&
                   state->previous_smoothed > state->smoothed;
    if (is_peak) {
        process_peak_candidate(state, state->previous_sample_ms);
    }

    state->previous_previous_smoothed = state->previous_smoothed;
    state->previous_smoothed = state->smoothed;
    state->previous_sample_ms = timestamp_ms;
}

static void build_report(const live_hr_state_t *state, int64_t timestamp_ms, live_hr_report_t *report)
{
    report->report_ready = true;
    report->timestamp_ms = timestamp_ms;
    report->bpm = 0.0f;
    report->beats = state->interval_count > 0 ? state->interval_count + 1 : 0;

    if (!state->finger_present) {
        report->status = LIVE_HR_NO_FINGER;
        return;
    }
    if ((timestamp_ms - state->finger_start_ms) < LIVE_HR_WARMUP_MS) {
        report->status = LIVE_HR_WARMING_UP;
        return;
    }
    if (state->signal_envelope < LIVE_HR_MIN_SIGNAL_AMPLITUDE) {
        report->status = LIVE_HR_POOR_SIGNAL;
        return;
    }
    if (state->interval_count < LIVE_HR_MIN_INTERVALS_FOR_REPORT ||
        state->last_peak_ms == 0 ||
        (timestamp_ms - state->last_peak_ms) > LIVE_HR_MAX_INTERVAL_MS) {
        report->status = LIVE_HR_INSUFFICIENT_BEATS;
        return;
    }

    float interval_ms = median_interval_ms(state);
    float bpm = interval_ms > 0.0f ? 60000.0f / interval_ms : 0.0f;
    if (bpm < LIVE_HR_MIN_BPM || bpm > LIVE_HR_MAX_BPM) {
        report->status = LIVE_HR_INSUFFICIENT_BEATS;
        return;
    }

    report->status = LIVE_HR_STABLE;
    report->bpm = bpm;
}

bool live_hr_process_sample(
    live_hr_state_t *state,
    int64_t timestamp_ms,
    uint32_t ir,
    bool finger_present,
    live_hr_report_t *report)
{
    report->report_ready = false;
    if (!state->report_clock_initialized) {
        state->report_clock_initialized = true;
        state->next_report_ms = timestamp_ms + LIVE_HR_REPORT_PERIOD_MS;
    }

    if (!finger_present) {
        if (state->finger_present || state->filter_initialized) {
            reset_signal_state(state);
        }
    } else {
        if (!state->finger_present) {
            reset_signal_state(state);
            state->finger_present = true;
            state->finger_start_ms = timestamp_ms;
        }
        update_filter_and_peaks(state, timestamp_ms, ir);
    }

    if (timestamp_ms < state->next_report_ms) {
        return false;
    }

    while (state->next_report_ms <= timestamp_ms) {
        state->next_report_ms += LIVE_HR_REPORT_PERIOD_MS;
    }
    build_report(state, timestamp_ms, report);
    return true;
}

const char *live_hr_status_name(live_hr_status_t status)
{
    switch (status) {
    case LIVE_HR_WARMING_UP:
        return "warming_up";
    case LIVE_HR_STABLE:
        return "stable";
    case LIVE_HR_POOR_SIGNAL:
        return "poor_signal";
    case LIVE_HR_NO_FINGER:
        return "no_finger";
    case LIVE_HR_INSUFFICIENT_BEATS:
        return "insufficient_beats";
    default:
        return "unknown";
    }
}
