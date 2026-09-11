idf_component_register(SRCS "motion_status.c" "main.c" "max30102.c" "adxl345.c" "live_hr.c" "ppg_clock.c"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_timer)
