#include <inttypes.h>
#include <stdio.h>

#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"
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
#define SIGNAL_QUALITY_STATUS_PERIOD_MS 1000
#define SIGNAL_QUALITY_SATURATION_MARGIN 1000

#if DEBUG_SIGNAL_QUALITY
static const char *TAG = "app";
#endif

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

void app_main(void)
{
    ESP_ERROR_CHECK(i2c_master_init());

    if (!max30102_is_connected()) {
        return;
    }

    ESP_ERROR_CHECK(max30102_init());

    printf("timestamp_ms,red,ir\n");

#if DEBUG_SIGNAL_QUALITY
    uint32_t ir_min = MAX30102_ADC_MAX_VALUE;
    uint32_t ir_max = 0;
    uint32_t quality_sample_count = 0;
    int64_t next_quality_status_ms = (esp_timer_get_time() / 1000) + SIGNAL_QUALITY_STATUS_PERIOD_MS;
#endif

    while (true) {
        uint8_t samples = max30102_available_samples();
        int64_t latest_sample_time_ms = esp_timer_get_time() / 1000;
        int64_t first_sample_time_ms =
            latest_sample_time_ms - ((int64_t)(samples > 0 ? samples - 1 : 0) * MAX30102_SAMPLE_PERIOD_MS);
        uint8_t sample_index = 0;

        while (samples > 0) {
            uint32_t red = 0;
            uint32_t ir = 0;

            if (max30102_read_fifo_sample(&red, &ir) == ESP_OK) {
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

                // Sampling rate: estimate per-sample timestamps at 100 Hz when draining FIFO batches.
                int64_t timestamp_ms =
                    first_sample_time_ms + ((int64_t)sample_index * MAX30102_SAMPLE_PERIOD_MS);
                printf("%" PRId64 ",%" PRIu32 ",%" PRIu32 "\n", timestamp_ms, red, ir);
            } else {
#if DEBUG_SIGNAL_QUALITY
                ESP_LOGW(TAG, "Failed to read MAX30102 FIFO sample");
#endif
                break;
            }

            samples--;
            sample_index++;
        }

#if DEBUG_SIGNAL_QUALITY
        int64_t now_ms = esp_timer_get_time() / 1000;
        if (now_ms >= next_quality_status_ms) {
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
                   now_ms, ir_min, ir_max, ir_range, quality_sample_count, status);

            ir_min = MAX30102_ADC_MAX_VALUE;
            ir_max = 0;
            quality_sample_count = 0;
            next_quality_status_ms = now_ms + SIGNAL_QUALITY_STATUS_PERIOD_MS;
        }
#endif

        // FIFO reads: poll faster than the 32-sample FIFO fills at 100 Hz to avoid dropped samples.
        vTaskDelay(pdMS_TO_TICKS(MAX30102_POLL_DELAY_MS));
    }
}
