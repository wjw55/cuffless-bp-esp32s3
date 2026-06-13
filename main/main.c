#include <inttypes.h>
#include <stdio.h>

#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "max30102.h"

#define I2C_MASTER_PORT I2C_NUM_0
#define I2C_MASTER_SDA_IO 8
#define I2C_MASTER_SCL_IO 9
#define I2C_MASTER_FREQ_HZ 100000

#define DEBUG_SIGNAL_QUALITY 0
#define MAX30102_POLL_DELAY_MS 20
#define ACQUISITION_STATS_PERIOD_MS 5000
#define SIGNAL_QUALITY_STATUS_PERIOD_MS 1000
#define SIGNAL_QUALITY_SATURATION_MARGIN 1000
#define TIMESTAMP_RESYNC_THRESHOLD_US (5 * MAX30102_SAMPLE_PERIOD_US)
#define I2C_WARNING_THROTTLE_MS 1000

static esp_err_t i2c_master_init(void)
{
    i2c_config_t config = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags = 0,
    };

    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_PORT, &config));

    return i2c_driver_install(I2C_MASTER_PORT, config.mode, 0, 0, 0);
}

static void print_i2c_warning_throttled(
    const char *event,
    esp_err_t result,
    int64_t now_us,
    int64_t *last_warning_time_us)
{
    if (*last_warning_time_us != 0 &&
        (now_us - *last_warning_time_us) < (I2C_WARNING_THROTTLE_MS * 1000)) {
        return;
    }

    printf("# warning event=%s error=%s i2c_errors=%" PRIu32 "\n",
           event,
           esp_err_to_name(result),
           max30102_get_i2c_error_count());
    *last_warning_time_us = now_us;
}

void app_main(void)
{
    ESP_ERROR_CHECK(i2c_master_init());

    if (!max30102_is_connected()) {
        return;
    }

    ESP_ERROR_CHECK(max30102_init());

    printf("sample_seq,timestamp_ms,red,ir\n");

    uint64_t sample_seq = 0;
    uint64_t last_stats_sample_seq = 0;
    uint32_t overflow_count_total = 0;
    uint32_t overflow_recovery_count = 0;
    uint32_t timestamp_resync_count = 0;
    uint32_t timestamp_correction_count = 0;
    uint8_t latest_fifo_available = 0;
    bool timestamp_initialized = false;
    int64_t acquisition_start_time_us = 0;
    int64_t next_sample_timestamp_us = 0;
    int64_t last_emitted_timestamp_us = -1;
    int64_t last_stats_time_us = esp_timer_get_time();
    int64_t last_fifo_pointer_warning_time_us = 0;
    int64_t last_overflow_counter_warning_time_us = 0;

#if DEBUG_SIGNAL_QUALITY
    uint32_t ir_min = MAX30102_ADC_MAX_VALUE;
    uint32_t ir_max = 0;
    uint32_t quality_sample_count = 0;
    int64_t next_quality_status_ms = (esp_timer_get_time() / 1000) + SIGNAL_QUALITY_STATUS_PERIOD_MS;
#endif

    while (true) {
        uint8_t samples = 0;
        bool skip_fifo_drain = false;

        uint8_t overflow_count = 0;
        esp_err_t overflow_result = max30102_read_overflow_count(&overflow_count);
        if (overflow_result == ESP_OK) {
            if (overflow_count > 0) {
                skip_fifo_drain = true;
                timestamp_initialized = false;
                latest_fifo_available = 0;

                esp_err_t reset_result = max30102_reset_fifo();
                if (reset_result == ESP_OK) {
                    overflow_count_total += overflow_count;
                    overflow_recovery_count++;
                    timestamp_resync_count++;

                    printf("# warning event=fifo_overflow count=%u total=%" PRIu32 "\n",
                           (unsigned)overflow_count,
                           overflow_count_total);
                    printf("# warning event=fifo_overflow_recovery count=%" PRIu32
                           " total=%" PRIu32 " sample_seq=%" PRIu64
                           " action=fifo_reset timestamp_cursor=fresh\n",
                           overflow_recovery_count,
                           overflow_recovery_count,
                           sample_seq);
                } else {
                    printf("# warning event=fifo_overflow_recovery_failed count=%u total=%" PRIu32
                           " sample_seq=%" PRIu64
                           " error=%s i2c_errors=%" PRIu32 "\n",
                           (unsigned)overflow_count,
                           overflow_count_total,
                           sample_seq,
                           esp_err_to_name(reset_result),
                           max30102_get_i2c_error_count());
                }
            }
        } else {
            skip_fifo_drain = true;
            print_i2c_warning_throttled(
                "overflow_counter_read_failed",
                overflow_result,
                esp_timer_get_time(),
                &last_overflow_counter_warning_time_us);
        }

        if (!skip_fifo_drain) {
            esp_err_t samples_result = max30102_get_available_samples(&samples);
            if (samples_result == ESP_OK) {
                latest_fifo_available = samples;
            } else {
                print_i2c_warning_throttled(
                    "fifo_pointer_read_failed",
                    samples_result,
                    esp_timer_get_time(),
                    &last_fifo_pointer_warning_time_us);
            }
        }

        if (samples > 0 && timestamp_initialized) {
            int64_t read_time_us = esp_timer_get_time();
            int64_t expected_latest_timestamp_us =
                next_sample_timestamp_us + ((int64_t)(samples - 1) * MAX30102_SAMPLE_PERIOD_US);
            int64_t lag_us = read_time_us - expected_latest_timestamp_us;

            if (lag_us > TIMESTAMP_RESYNC_THRESHOLD_US) {
                int64_t corrected_next_timestamp_us =
                    read_time_us - ((int64_t)(samples - 1) * MAX30102_SAMPLE_PERIOD_US);

                if (last_emitted_timestamp_us >= 0) {
                    int64_t minimum_next_timestamp_us =
                        last_emitted_timestamp_us + MAX30102_SAMPLE_PERIOD_US;
                    if (corrected_next_timestamp_us < minimum_next_timestamp_us) {
                        corrected_next_timestamp_us = minimum_next_timestamp_us;
                    }
                }

                printf("# warning event=timestamp_resync sample_seq=%" PRIu64
                       " reason=%s lag_us=%" PRId64 " old_next_us=%" PRId64
                       " new_next_us=%" PRId64 " count=%" PRIu32 "\n",
                       sample_seq,
                       "lag",
                       lag_us,
                       next_sample_timestamp_us,
                       corrected_next_timestamp_us,
                       timestamp_resync_count + 1);

                next_sample_timestamp_us = corrected_next_timestamp_us;
                timestamp_resync_count++;
            }
        }

        while (samples > 0) {
            uint32_t red = 0;
            uint32_t ir = 0;

            if (max30102_read_fifo_sample(&red, &ir) == ESP_OK) {
                if (!timestamp_initialized) {
                    next_sample_timestamp_us = esp_timer_get_time();
                    if (sample_seq == 0) {
                        acquisition_start_time_us = next_sample_timestamp_us;
                        last_stats_time_us = acquisition_start_time_us;
                    }
                    timestamp_initialized = true;
                    printf("# timestamp_sync event=%s sample_seq=%" PRIu64
                           " timestamp_us=%" PRId64 "\n",
                           (sample_seq == 0) ? "initial" : "resync_after_overflow",
                           sample_seq,
                           next_sample_timestamp_us);
                }

                // Finger detection: keep CSV clean; IR consistently below this threshold means no finger/poor contact.
                bool finger_present = (ir > MAX30102_FINGER_IR_THRESHOLD);
                (void)finger_present;

#if DEBUG_SIGNAL_QUALITY
                if (ir < ir_min) {
                    ir_min = ir;
                }
                if (ir > ir_max) {
                    ir_max = ir;
                }
                quality_sample_count++;
#endif

                int64_t timestamp_us = next_sample_timestamp_us;
                if (last_emitted_timestamp_us >= 0 &&
                    timestamp_us < (last_emitted_timestamp_us + MAX30102_SAMPLE_PERIOD_US)) {
                    timestamp_us = last_emitted_timestamp_us + MAX30102_SAMPLE_PERIOD_US;
                    next_sample_timestamp_us = timestamp_us;
                    timestamp_correction_count++;
                    printf("# warning event=timestamp_correction sample_seq=%" PRIu64
                           " corrected_timestamp_us=%" PRId64 " count=%" PRIu32 "\n",
                           sample_seq,
                           timestamp_us,
                           timestamp_correction_count);
                }

                int64_t timestamp_ms = timestamp_us / 1000;
                printf("%" PRIu64 ",%" PRId64 ",%" PRIu32 ",%" PRIu32 "\n", sample_seq, timestamp_ms, red, ir);
                last_emitted_timestamp_us = timestamp_us;
                next_sample_timestamp_us = timestamp_us + MAX30102_SAMPLE_PERIOD_US;
                sample_seq++;
            } else {
                printf("# warning event=fifo_read_failed i2c_errors=%" PRIu32 "\n",
                       max30102_get_i2c_error_count());
                break;
            }

            samples--;
        }

#if DEBUG_SIGNAL_QUALITY
        int64_t quality_now_ms = esp_timer_get_time() / 1000;
        if (quality_now_ms >= next_quality_status_ms) {
            uint32_t ir_range = (quality_sample_count > 0) ? (ir_max - ir_min) : 0;
            const char *status = "usable";

            if (quality_sample_count == 0) {
                status = "no_samples";
            } else if (ir_max >= (MAX30102_ADC_MAX_VALUE - SIGNAL_QUALITY_SATURATION_MARGIN)) {
                status = "saturated";
            } else if (ir_max < MAX30102_FINGER_IR_THRESHOLD) {
                status = "low";
            }

            printf("# signal_quality,timestamp_ms=%" PRId64 ",ir_min=%" PRIu32
                   ",ir_max=%" PRIu32 ",ir_range=%" PRIu32 ",samples=%" PRIu32
                   ",status=%s\n",
                   quality_now_ms, ir_min, ir_max, ir_range, quality_sample_count, status);

            ir_min = MAX30102_ADC_MAX_VALUE;
            ir_max = 0;
            quality_sample_count = 0;
            next_quality_status_ms = quality_now_ms + SIGNAL_QUALITY_STATUS_PERIOD_MS;
        }
#endif

        int64_t stats_now_us = esp_timer_get_time();
        if ((stats_now_us - last_stats_time_us) >= (ACQUISITION_STATS_PERIOD_MS * 1000)) {
            int64_t elapsed_us = stats_now_us - last_stats_time_us;
            uint64_t samples_since_last_status = sample_seq - last_stats_sample_seq;
            uint64_t rate_tenths_hz =
                (elapsed_us > 0) ? ((samples_since_last_status * 10000000ULL) / (uint64_t)elapsed_us) : 0;
            int64_t total_elapsed_us = stats_now_us - acquisition_start_time_us;
            uint64_t effective_rate_tenths_hz =
                (acquisition_start_time_us > 0 && total_elapsed_us > 0)
                    ? ((sample_seq * 10000000ULL) / (uint64_t)total_elapsed_us)
                    : 0;

            printf("# stats samples=%" PRIu64 " captured_samples=%" PRIu64
                   " rate_hz=%" PRIu64 ".%" PRIu64
                   " effective_rate_hz=%" PRIu64 ".%" PRIu64
                   " fifo_avail=%u ovf=%" PRIu32 " i2c_errors=%" PRIu32
                   " timestamp_resyncs=%" PRIu32 " timestamp_corrections=%" PRIu32
                   " overflow_recoveries=%" PRIu32 "\n",
                   sample_seq,
                   sample_seq,
                   rate_tenths_hz / 10,
                   rate_tenths_hz % 10,
                   effective_rate_tenths_hz / 10,
                   effective_rate_tenths_hz % 10,
                   (unsigned)latest_fifo_available,
                   overflow_count_total,
                   max30102_get_i2c_error_count(),
                   timestamp_resync_count,
                   timestamp_correction_count,
                   overflow_recovery_count);

            last_stats_sample_seq = sample_seq;
            last_stats_time_us = stats_now_us;
        }

        // FIFO reads: poll faster than the 32-sample FIFO fills at 100 Hz to avoid dropped samples.
        vTaskDelay(pdMS_TO_TICKS(MAX30102_POLL_DELAY_MS));
    }
}
