idf_component_register(SRCS "telemetry.c" "motion_status.c" "main.c" "max30102.c" "adxl345.c" "live_hr.c" "ppg_clock.c"
                    INCLUDE_DIRS "."
                    REQUIRES bt driver esp_driver_usb_serial_jtag esp_timer nvs_flash)
