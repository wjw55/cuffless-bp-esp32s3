#include "ppg_clock.h"

int64_t ppg_clock_oldest_us(int64_t read_us, unsigned fifo_samples)
{
    return read_us - ((int64_t)fifo_samples - 1) * PPG_CLOCK_PERIOD_US;
}

ppg_clock_observation_t ppg_clock_observe(int64_t next_us, int64_t last_us,
                                         int64_t read_us, unsigned fifo_samples)
{
    ppg_clock_observation_t result;
    result.next_us = next_us;
    result.phase_error_us = 0;
    result.adjustment_us = 0;
    result.rejected = false;
    if (fifo_samples == 0 || fifo_samples >= 32) {
        result.rejected = true;
        return result;
    }
    result.phase_error_us = ppg_clock_oldest_us(read_us, fifo_samples) - next_us;
    if (result.phase_error_us > PPG_CLOCK_MAX_PHASE_ERROR_US ||
        result.phase_error_us < -PPG_CLOCK_MAX_PHASE_ERROR_US) {
        /* A long stall/reset/unknown sample loss must remain visible. */
        result.rejected = true;
        return result;
    }
    int64_t adjustment = result.phase_error_us / 16;
    if (adjustment > PPG_CLOCK_MAX_ADJUSTMENT_US) {
        adjustment = PPG_CLOCK_MAX_ADJUSTMENT_US;
    } else if (adjustment < -PPG_CLOCK_MAX_ADJUSTMENT_US) {
        adjustment = -PPG_CLOCK_MAX_ADJUSTMENT_US;
    }
    if (last_us >= 0 && next_us + adjustment < last_us + PPG_CLOCK_MIN_SPACING_US) {
        result.rejected = true;
        return result;
    }
    result.next_us += adjustment;
    result.adjustment_us = adjustment;
    return result;
}
