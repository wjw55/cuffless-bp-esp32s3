#include "adxl345.h"

#include "driver/i2c.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define ADXL345_I2C_PORT I2C_NUM_0
#define ADXL345_I2C_TIMEOUT_MS 50

#define ADXL345_REG_DEVID 0x00
#define ADXL345_REG_BW_RATE 0x2C
#define ADXL345_REG_POWER_CTL 0x2D
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_DATAX0 0x32
#define ADXL345_REG_FIFO_CTL 0x38
#define ADXL345_REG_FIFO_STATUS 0x39

#define ADXL345_EXPECTED_DEVID 0xE5
#define ADXL345_RATE_100_HZ 0x0A
#define ADXL345_MEASURE_MODE 0x08
#define ADXL345_FULL_RES_4G 0x09
#define ADXL345_FIFO_BYPASS 0x00
#define ADXL345_FIFO_STREAM 0x80

static const char *TAG = "adxl345";
static uint32_t i2c_error_count = 0;

static esp_err_t track_i2c_result(esp_err_t result)
{
    if (result != ESP_OK) {
        i2c_error_count++;
    }
    return result;
}

static esp_err_t read_register(uint8_t reg, uint8_t *value)
{
    return track_i2c_result(i2c_master_write_read_device(
        ADXL345_I2C_PORT,
        ADXL345_I2C_ADDRESS,
        &reg,
        1,
        value,
        1,
        pdMS_TO_TICKS(ADXL345_I2C_TIMEOUT_MS)));
}

static esp_err_t write_register(uint8_t reg, uint8_t value)
{
    uint8_t data[] = {reg, value};
    return track_i2c_result(i2c_master_write_to_device(
        ADXL345_I2C_PORT,
        ADXL345_I2C_ADDRESS,
        data,
        sizeof(data),
        pdMS_TO_TICKS(ADXL345_I2C_TIMEOUT_MS)));
}

bool adxl345_is_connected(void)
{
    uint8_t device_id = 0;
    if (read_register(ADXL345_REG_DEVID, &device_id) != ESP_OK) {
        ESP_LOGW(TAG, "ADXL345 not found at 0x%02X", ADXL345_I2C_ADDRESS);
        return false;
    }
    if (device_id != ADXL345_EXPECTED_DEVID) {
        ESP_LOGE(TAG, "Unexpected device ID 0x%02X (expected 0x%02X)", device_id, ADXL345_EXPECTED_DEVID);
        return false;
    }

    ESP_LOGI(TAG, "ADXL345 device ID verified at 0x%02X", ADXL345_I2C_ADDRESS);
    return true;
}

esp_err_t adxl345_reset_fifo(void)
{
    ESP_RETURN_ON_ERROR(write_register(ADXL345_REG_FIFO_CTL, ADXL345_FIFO_BYPASS), TAG, "FIFO bypass failed");
    return write_register(ADXL345_REG_FIFO_CTL, ADXL345_FIFO_STREAM);
}

esp_err_t adxl345_init(void)
{
    // Configure in standby, then enable measurement only after range/rate/FIFO are ready.
    ESP_RETURN_ON_ERROR(write_register(ADXL345_REG_POWER_CTL, 0x00), TAG, "Standby failed");
    ESP_RETURN_ON_ERROR(write_register(ADXL345_REG_DATA_FORMAT, ADXL345_FULL_RES_4G), TAG, "Data format failed");
    ESP_RETURN_ON_ERROR(write_register(ADXL345_REG_BW_RATE, ADXL345_RATE_100_HZ), TAG, "Output rate failed");
    ESP_RETURN_ON_ERROR(adxl345_reset_fifo(), TAG, "FIFO setup failed");
    ESP_RETURN_ON_ERROR(write_register(ADXL345_REG_POWER_CTL, ADXL345_MEASURE_MODE), TAG, "Measurement mode failed");
    vTaskDelay(pdMS_TO_TICKS(20));
    return ESP_OK;
}

esp_err_t adxl345_get_fifo_entries(uint8_t *entries)
{
    uint8_t status = 0;
    esp_err_t result = read_register(ADXL345_REG_FIFO_STATUS, &status);
    if (result == ESP_OK) {
        *entries = status & 0x3F;
    }
    return result;
}

esp_err_t adxl345_read_fifo_sample(int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t reg = ADXL345_REG_DATAX0;
    uint8_t data[6] = {0};
    esp_err_t result = track_i2c_result(i2c_master_write_read_device(
        ADXL345_I2C_PORT,
        ADXL345_I2C_ADDRESS,
        &reg,
        1,
        data,
        sizeof(data),
        pdMS_TO_TICKS(ADXL345_I2C_TIMEOUT_MS)));

    if (result != ESP_OK) {
        return result;
    }

    *x = (int16_t)(((uint16_t)data[1] << 8) | data[0]);
    *y = (int16_t)(((uint16_t)data[3] << 8) | data[2]);
    *z = (int16_t)(((uint16_t)data[5] << 8) | data[4]);
    return ESP_OK;
}

uint32_t adxl345_get_i2c_error_count(void)
{
    return i2c_error_count;
}
