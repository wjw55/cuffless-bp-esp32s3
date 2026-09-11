#ifndef PPG_CLOCK_H
#define PPG_CLOCK_H

#include <stdbool.h>
#include <stdint.h>

/* FIFO reads bound sample time only approximately: polling and I2C latency
 * remain. This tracks clock phase, not optical data-ready accuracy. */
#define PPG_CLOCK_PERIOD_US 10000
#define PPG_CLOCK_MAX_ADJUSTMENT_US 250
#define PPG_CLOCK_MAX_PHASE_ERROR_US 40000
#define PPG_CLOCK_MIN_SPACING_US 1000

typedef struct {
    int64_t next_us;
    int64_t phase_error_us;
    int64_t adjustment_us;
    bool rejected;
} ppg_clock_observation_t;

int64_t ppg_clock_oldest_us(int64_t read_us, unsigned fifo_samples);
ppg_clock_observation_t ppg_clock_observe(int64_t next_us, int64_t last_us,
                                         int64_t read_us, unsigned fifo_samples);

#endif
