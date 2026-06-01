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

#define MAX30102_SAMPLE_DELAY_MS 100
#define FINGER_IR_THRESHOLD 50000

static const char *TAG = "app";

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

    printf("timestamp_ms,red,ir,finger\n");

    while (true) {
        uint8_t samples = max30102_available_samples();

        while (samples > 0) {
            uint32_t red = 0;
            uint32_t ir = 0;

            if (max30102_read_fifo_sample(&red, &ir) == ESP_OK) {
                int finger = (ir > FINGER_IR_THRESHOLD) ? 1 : 0;
                int64_t timestamp_ms = esp_timer_get_time() / 1000;

                printf("%" PRId64 ",%" PRIu32 ",%" PRIu32 ",%d\n", timestamp_ms, red, ir, finger);
            } else {
                ESP_LOGW(TAG, "Failed to read MAX30102 FIFO sample");
                break;
            }

            samples--;
        }

        vTaskDelay(pdMS_TO_TICKS(MAX30102_SAMPLE_DELAY_MS));
    }
}
