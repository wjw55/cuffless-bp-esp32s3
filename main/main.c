#include <inttypes.h>
#include <stdio.h>

#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "adxl345.h"
#include "live_hr.h"
#include "max30102.h"

#define I2C_MASTER_PORT I2C_NUM_0
#define I2C_MASTER_SDA_IO 8
#define I2C_MASTER_SCL_IO 9
#define I2C_MASTER_FREQ_HZ 100000

#define DEBUG_SIGNAL_QUALITY 0
#define SENSOR_POLL_DELAY_MS 10
#define ACQUISITION_STATS_PERIOD_MS 5000
#define SIGNAL_QUALITY_STATUS_PERIOD_MS 1000
#define SIGNAL_QUALITY_SATURATION_MARGIN 1000
#define TIMESTAMP_LAG_WARNING_STEP_US (5 * MAX30102_SAMPLE_PERIOD_US)
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

    bool imu_enabled = adxl345_is_connected();
    if (imu_enabled) {
        esp_err_t imu_init_result = adxl345_init();
        if (imu_init_result != ESP_OK) {
            printf("# warning event=imu_init_failed error=%s\n", esp_err_to_name(imu_init_result));
            imu_enabled = false;
        }
    } else {
        printf("# warning event=imu_not_detected address=0x%02X action=ppg_only\n", ADXL345_I2C_ADDRESS);
    }

    printf("sample_seq,timestamp_ms,red,ir\n");
    if (imu_enabled) {
        printf("imu,imu_seq,timestamp_ms,x_raw,y_raw,z_raw\n");
    }

    uint64_t sample_seq = 0;
    uint64_t last_stats_sample_seq = 0;
    uint32_t overflow_count_total = 0;
    uint32_t overflow_recovery_count = 0;
    uint32_t timestamp_resync_count = 0;
    uint32_t timestamp_correction_count = 0;
    uint32_t timestamp_lag_warning_count = 0;
    uint8_t latest_fifo_available = 0;
    bool timestamp_initialized = false;
    int64_t acquisition_start_time_us = 0;
    int64_t next_sample_timestamp_us = 0;
    int64_t last_emitted_timestamp_us = -1;
    int64_t last_stats_time_us = esp_timer_get_time();
    int64_t last_fifo_pointer_warning_time_us = 0;
    int64_t last_overflow_counter_warning_time_us = 0;
    int64_t next_timestamp_lag_warning_us = TIMESTAMP_LAG_WARNING_STEP_US;
    live_hr_state_t live_hr_state;
    live_hr_init(&live_hr_state);

    uint64_t imu_sample_seq = 0;
    uint64_t last_stats_imu_sample_seq = 0;
    uint32_t imu_fifo_overflow_count = 0;
    uint32_t imu_timestamp_resync_count = 0;
    uint32_t imu_timestamp_correction_count = 0;
    uint32_t imu_clock_adjustment_count = 0;
    int64_t imu_clock_adjustment_total_us = 0;
    uint8_t latest_imu_fifo_entries = 0;
    bool imu_timestamp_initialized = false;
    int64_t next_imu_sample_timestamp_us = 0;
    int64_t last_emitted_imu_timestamp_us = -1;

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
                next_timestamp_lag_warning_us = TIMESTAMP_LAG_WARNING_STEP_US;

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

            if (lag_us > next_timestamp_lag_warning_us) {
                timestamp_lag_warning_count++;
                printf("# warning event=timestamp_lag sample_seq=%" PRIu64
                       " lag_us=%" PRId64 " expected_latest_us=%" PRId64
                       " read_time_us=%" PRId64 " next_sample_us=%" PRId64
                       " count=%" PRIu32 "\n",
                       sample_seq,
                       lag_us,
                       expected_latest_timestamp_us,
                       read_time_us,
                       next_sample_timestamp_us,
                       timestamp_lag_warning_count);

                while (next_timestamp_lag_warning_us <= lag_us) {
                    next_timestamp_lag_warning_us += TIMESTAMP_LAG_WARNING_STEP_US;
                }
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
                    next_timestamp_lag_warning_us = TIMESTAMP_LAG_WARNING_STEP_US;
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
                live_hr_report_t hr_report = {0};
                if (live_hr_process_sample(&live_hr_state, timestamp_ms, ir, finger_present, &hr_report)) {
                    if (hr_report.status == LIVE_HR_STABLE) {
                        uint32_t bpm_tenths = (uint32_t)((hr_report.bpm * 10.0f) + 0.5f);
                        printf("# hr timestamp_ms=%" PRId64 " bpm=%" PRIu32 ".%" PRIu32
                               " status=%s beats=%u\n",
                               hr_report.timestamp_ms,
                               bpm_tenths / 10,
                               bpm_tenths % 10,
                               live_hr_status_name(hr_report.status),
                               (unsigned)hr_report.beats);
                    } else {
                        printf("# hr timestamp_ms=%" PRId64 " bpm=na status=%s beats=%u\n",
                               hr_report.timestamp_ms,
                               live_hr_status_name(hr_report.status),
                               (unsigned)hr_report.beats);
                    }
                }
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

        if (imu_enabled) {
            uint8_t imu_entries = 0;
            esp_err_t imu_entries_result = adxl345_get_fifo_entries(&imu_entries);
            if (imu_entries_result != ESP_OK) {
                printf("# warning event=imu_fifo_status_failed error=%s i2c_errors=%" PRIu32 "\n",
                       esp_err_to_name(imu_entries_result),
                       adxl345_get_i2c_error_count());
            } else if (imu_entries >= 32) {
                // A full stream FIFO means older motion samples may already have been overwritten.
                imu_fifo_overflow_count++;
                imu_timestamp_resync_count++;
                imu_timestamp_initialized = false;
                latest_imu_fifo_entries = 0;
                esp_err_t reset_result = adxl345_reset_fifo();
                printf("# warning event=imu_fifo_overflow count=%" PRIu32
                       " sample_seq=%" PRIu64 " action=%s\n",
                       imu_fifo_overflow_count,
                       imu_sample_seq,
                       (reset_result == ESP_OK) ? "fifo_reset" : "fifo_reset_failed");
            } else {
                latest_imu_fifo_entries = imu_entries;
                if (imu_entries > 0) {
                    int64_t read_time_us = esp_timer_get_time();
                    int64_t observed_oldest_timestamp_us =
                        read_time_us - ((int64_t)(imu_entries - 1) * ADXL345_SAMPLE_PERIOD_US);
                    if (!imu_timestamp_initialized) {
                        next_imu_sample_timestamp_us = observed_oldest_timestamp_us;
                        imu_timestamp_initialized = true;
                        printf("# imu_timestamp_sync event=%s sample_seq=%" PRIu64
                               " timestamp_us=%" PRId64 "\n",
                               (imu_sample_seq == 0) ? "initial" : "resync_after_overflow",
                               imu_sample_seq,
                               next_imu_sample_timestamp_us);
                    } else {
                        // Track the ADXL345's real sample clock against esp_timer. Its nominal
                        // 100 Hz oscillator can differ enough to drift by seconds over a trial.
                        int64_t phase_error_us = observed_oldest_timestamp_us - next_imu_sample_timestamp_us;
                        int64_t adjustment_us = phase_error_us / 8;
                        if (adjustment_us > 1000) {
                            adjustment_us = 1000;
                        } else if (adjustment_us < -1000) {
                            adjustment_us = -1000;
                        }
                        if (adjustment_us != 0) {
                            next_imu_sample_timestamp_us += adjustment_us;
                            imu_clock_adjustment_count++;
                            imu_clock_adjustment_total_us += adjustment_us;
                        }
                    }
                }

                while (imu_entries > 0) {
                    int16_t x = 0;
                    int16_t y = 0;
                    int16_t z = 0;
                    esp_err_t imu_read_result = adxl345_read_fifo_sample(&x, &y, &z);
                    if (imu_read_result != ESP_OK) {
                        printf("# warning event=imu_fifo_read_failed error=%s i2c_errors=%" PRIu32 "\n",
                               esp_err_to_name(imu_read_result),
                               adxl345_get_i2c_error_count());
                        break;
                    }

                    int64_t timestamp_us = next_imu_sample_timestamp_us;
                    if (last_emitted_imu_timestamp_us >= 0 && timestamp_us <= last_emitted_imu_timestamp_us) {
                        timestamp_us = last_emitted_imu_timestamp_us + 1;
                        next_imu_sample_timestamp_us = timestamp_us;
                        imu_timestamp_correction_count++;
                    }

                    printf("imu,%" PRIu64 ",%" PRId64 ",%d,%d,%d\n",
                           imu_sample_seq,
                           timestamp_us / 1000,
                           (int)x,
                           (int)y,
                           (int)z);
                    last_emitted_imu_timestamp_us = timestamp_us;
                    next_imu_sample_timestamp_us = timestamp_us + ADXL345_SAMPLE_PERIOD_US;
                    imu_sample_seq++;
                    imu_entries--;
                }
            }
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
                   " timestamp_lag_warnings=%" PRIu32
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
                   timestamp_lag_warning_count,
                   overflow_recovery_count);

            if (imu_enabled) {
                uint64_t imu_samples_since_last_status = imu_sample_seq - last_stats_imu_sample_seq;
                uint64_t imu_rate_tenths_hz =
                    (elapsed_us > 0) ? ((imu_samples_since_last_status * 10000000ULL) / (uint64_t)elapsed_us) : 0;
                uint64_t imu_effective_rate_tenths_hz =
                    (acquisition_start_time_us > 0 && total_elapsed_us > 0)
                        ? ((imu_sample_seq * 10000000ULL) / (uint64_t)total_elapsed_us)
                        : 0;
                printf("# imu_stats samples=%" PRIu64
                       " rate_hz=%" PRIu64 ".%" PRIu64
                       " effective_rate_hz=%" PRIu64 ".%" PRIu64
                       " fifo_entries=%u fifo_overflows=%" PRIu32
                       " i2c_errors=%" PRIu32
                       " timestamp_resyncs=%" PRIu32
                       " timestamp_corrections=%" PRIu32
                       " clock_adjustments=%" PRIu32
                       " clock_adjustment_us=%" PRId64 "\n",
                       imu_sample_seq,
                       imu_rate_tenths_hz / 10,
                       imu_rate_tenths_hz % 10,
                       imu_effective_rate_tenths_hz / 10,
                       imu_effective_rate_tenths_hz % 10,
                       (unsigned)latest_imu_fifo_entries,
                       imu_fifo_overflow_count,
                       adxl345_get_i2c_error_count(),
                       imu_timestamp_resync_count,
                       imu_timestamp_correction_count,
                       imu_clock_adjustment_count,
                       imu_clock_adjustment_total_us);
                last_stats_imu_sample_seq = imu_sample_seq;
            }

            last_stats_sample_seq = sample_seq;
            last_stats_time_us = stats_now_us;
        }

        // FIFO reads: poll faster than the 32-sample FIFO fills at 100 Hz to avoid dropped samples.
        vTaskDelay(pdMS_TO_TICKS(SENSOR_POLL_DELAY_MS));
    }
}
