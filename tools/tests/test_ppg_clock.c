/* Freestanding host test: return the failed check number, zero on success.
 * Executes the same C implementation used by the ESP32 firmware. */
#include "ppg_clock.h"

static int simulate(int period_us, int batch_size)
{
    int64_t next = 0, last = -1;
    for (int sample = 0; sample < 60000; sample += batch_size) {
        int64_t read = 1000000 + (int64_t)(sample + batch_size - 1) * period_us;
        /* Vary polling latency without changing the physical sample clock. */
        read += (sample % 9) * 900;
        if (sample == 0) {
            next = ppg_clock_oldest_us(read, batch_size);
        } else {
            ppg_clock_observation_t value = ppg_clock_observe(next, last, read, batch_size);
            if (value.rejected) return 1;
            if (value.adjustment_us > 250 || value.adjustment_us < -250) return 2;
            next = value.next_us;
        }
        for (int i = 0; i < batch_size; i++) {
            int64_t true_time = 1000000 + (int64_t)(sample + i) * period_us;
            if (next <= last || next - true_time > 15000 || true_time - next > 15000) return 3;
            last = next;
            next += PPG_CLOCK_PERIOD_US;
        }
    }
    return 0;
}

int main(void)
{
    if (ppg_clock_oldest_us(100000, 4) != 70000) return 10;
    if (simulate(10025, 1)) return 11;
    if (simulate(9975, 1)) return 12;
    if (simulate(10025, 4)) return 13;
    if (simulate(9975, 4)) return 14;
    ppg_clock_observation_t late = ppg_clock_observe(100000, 90000, 200000, 1);
    if (!late.rejected || late.next_us != 100000 || late.adjustment_us != 0) return 15;
    if (!ppg_clock_observe(10000, 9900, 10000, 1).rejected) return 16;
    if (!ppg_clock_observe(10000, 0, 10000, 0).rejected) return 17;
    if (!ppg_clock_observe(10000, 0, 10000, 32).rejected) return 18;
    /* Resynchronization starts a new cursor from the observed FIFO, not old phase. */
    if (ppg_clock_oldest_us(1000000, 3) != 980000) return 19;
    return 0;
}
