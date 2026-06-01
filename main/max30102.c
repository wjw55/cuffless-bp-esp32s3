#include "max30102.h"

#include "driver/i2c.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MAX30102_I2C_PORT I2C_NUM_0
#define MAX30102_ADDR 0x57
#define MAX30102_I2C_TIMEOUT_MS 50

#define MAX30102_REG_INTR_STATUS_1 0x00
#define MAX30102_REG_FIFO_WR_PTR 0x04
#define MAX30102_REG_OVF_COUNTER 0x05
#define MAX30102_REG_FIFO_RD_PTR 0x06
#define MAX30102_REG_FIFO_DATA 0x07
#define MAX30102_REG_FIFO_CONFIG 0x08
#define MAX30102_REG_MODE_CONFIG 0x09
#define MAX30102_REG_SPO2_CONFIG 0x0A
#define MAX30102_REG_LED1_PA 0x0C
#define MAX30102_REG_LED2_PA 0x0D

#define MAX30102_MODE_RESET 0x40
#define MAX30102_MODE_SPO2 0x03

static const char *TAG = "max30102";

static esp_err_t max30102_probe_address(void)
{
    i2c_cmd_handle_t command = i2c_cmd_link_create();

    i2c_master_start(command);
    i2c_master_write_byte(command, (MAX30102_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_stop(command);

    esp_err_t result = i2c_master_cmd_begin(
        MAX30102_I2C_PORT,
        command,
        pdMS_TO_TICKS(MAX30102_I2C_TIMEOUT_MS));

    i2c_cmd_link_delete(command);

    return result;
}

static esp_err_t max30102_write_register(uint8_t reg, uint8_t value)
{
    uint8_t data[] = {reg, value};

    return i2c_master_write_to_device(
        MAX30102_I2C_PORT,
        MAX30102_ADDR,
        data,
        sizeof(data),
        pdMS_TO_TICKS(MAX30102_I2C_TIMEOUT_MS));
}

static esp_err_t max30102_read_register(uint8_t reg, uint8_t *value)
{
    return i2c_master_write_read_device(
        MAX30102_I2C_PORT,
        MAX30102_ADDR,
        &reg,
        1,
        value,
        1,
        pdMS_TO_TICKS(MAX30102_I2C_TIMEOUT_MS));
}

bool max30102_is_connected(void)
{
    if (max30102_probe_address() != ESP_OK) {
        ESP_LOGW(TAG, "MAX30102 not found at 0x57");
        return false;
    }

    return true;
}

esp_err_t max30102_init(void)
{
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_MODE_CONFIG, MAX30102_MODE_RESET), TAG, "Reset failed");
    vTaskDelay(pdMS_TO_TICKS(100));

    // Start from an empty FIFO before streaming samples.
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_FIFO_WR_PTR, 0x00), TAG, "Clear FIFO write pointer failed");
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_OVF_COUNTER, 0x00), TAG, "Clear FIFO overflow counter failed");
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_FIFO_RD_PTR, 0x00), TAG, "Clear FIFO read pointer failed");

    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_FIFO_CONFIG, 0x40), TAG, "FIFO config failed");
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_SPO2_CONFIG, 0x27), TAG, "SpO2 config failed");
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_LED1_PA, 0x24), TAG, "RED LED config failed");
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_LED2_PA, 0x24), TAG, "IR LED config failed");
    ESP_RETURN_ON_ERROR(max30102_write_register(MAX30102_REG_MODE_CONFIG, MAX30102_MODE_SPO2), TAG, "SpO2 mode failed");

    uint8_t interrupt_status = 0;
    (void)max30102_read_register(MAX30102_REG_INTR_STATUS_1, &interrupt_status);

    return ESP_OK;
}

uint8_t max30102_available_samples(void)
{
    uint8_t write_pointer = 0;
    uint8_t read_pointer = 0;

    if (max30102_read_register(MAX30102_REG_FIFO_WR_PTR, &write_pointer) != ESP_OK ||
        max30102_read_register(MAX30102_REG_FIFO_RD_PTR, &read_pointer) != ESP_OK) {
        return 0;
    }

    write_pointer &= 0x1F;
    read_pointer &= 0x1F;

    if (write_pointer >= read_pointer) {
        return write_pointer - read_pointer;
    }

    return 32 + write_pointer - read_pointer;
}

esp_err_t max30102_read_fifo_sample(uint32_t *red, uint32_t *ir)
{
    uint8_t reg = MAX30102_REG_FIFO_DATA;
    uint8_t data[6] = {0};
    esp_err_t result = i2c_master_write_read_device(
        MAX30102_I2C_PORT,
        MAX30102_ADDR,
        &reg,
        1,
        data,
        sizeof(data),
        pdMS_TO_TICKS(MAX30102_I2C_TIMEOUT_MS));

    if (result != ESP_OK) {
        return result;
    }

    *red = (((uint32_t)data[0] << 16) | ((uint32_t)data[1] << 8) | data[2]) & 0x3FFFF;
    *ir = (((uint32_t)data[3] << 16) | ((uint32_t)data[4] << 8) | data[5]) & 0x3FFFF;

    return ESP_OK;
}
