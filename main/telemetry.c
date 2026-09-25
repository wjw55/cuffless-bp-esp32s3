#include "telemetry.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_check.h"
#include "esp_mac.h"
#include "driver/usb_serial_jtag.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "host/ble_att.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define TELEMETRY_STATUS_STREAM_BYTES 16384
#define TELEMETRY_SAMPLE_QUEUE_LENGTH 256
#define TELEMETRY_LINE_BYTES 384
#define TELEMETRY_USB_TX_BUFFER_BYTES 8192
#define TELEMETRY_USB_RX_BUFFER_BYTES 256
#define TELEMETRY_PREFERRED_MTU 247
#define TELEMETRY_MAX_NOTIFY_BYTES (TELEMETRY_PREFERRED_MTU - 3)
#define TELEMETRY_TX_DELAY_SMALL_MTU_MS 7
#define TELEMETRY_TX_DELAY_LARGE_MTU_MS 30
#define TELEMETRY_STATUS_AFTER_RAW_PACKETS 6
#define TELEMETRY_DEVICE_NAME_BYTES 24

#define BLE_FRAME_STATUS 0x80
#define BLE_FRAME_PPG_ABSOLUTE 0x81
#define BLE_FRAME_IMU_ABSOLUTE 0x82
#define BLE_FRAME_PPG_BATCH_BASE 0x90
#define BLE_FRAME_IMU_BATCH_BASE 0xa0
#define BLE_FRAME_PPG_SELF_CONTAINED_BASE 0xb0
#define BLE_FRAME_IMU_SELF_CONTAINED_BASE 0xc0
#define BLE_FRAME_STATUS_FRAGMENT 0xd0
#define BLE_FRAME_BATCH_MAX 15

typedef struct {
    uint64_t sequence;
    int64_t timestamp_ms;
    uint32_t red;
    uint32_t ir;
} ppg_sample_t;

typedef struct {
    uint64_t sequence;
    int64_t timestamp_ms;
    int16_t x;
    int16_t y;
    int16_t z;
} imu_sample_t;

typedef enum {
    PENDING_NONE = 0,
    PENDING_STATUS,
    PENDING_PPG,
    PENDING_IMU,
} pending_kind_t;

typedef struct {
    uint8_t bytes[TELEMETRY_MAX_NOTIFY_BYTES];
    size_t length;
    pending_kind_t kind;
} pending_packet_t;

static const ble_uuid128_t telemetry_service_uuid = BLE_UUID128_INIT(
    0x82, 0x71, 0x60, 0x5f, 0x4e, 0x3d, 0x21, 0x9a,
    0x7d, 0x4b, 0x5a, 0x6f, 0x01, 0x00, 0x5c, 0x9f);
static const ble_uuid128_t telemetry_tx_uuid = BLE_UUID128_INIT(
    0x82, 0x71, 0x60, 0x5f, 0x4e, 0x3d, 0x21, 0x9a,
    0x7d, 0x4b, 0x5a, 0x6f, 0x02, 0x00, 0x5c, 0x9f);

static StreamBufferHandle_t telemetry_status_stream;
static QueueHandle_t telemetry_ppg_queue;
static QueueHandle_t telemetry_imu_queue;
static uint16_t telemetry_tx_handle;
static uint16_t telemetry_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint16_t telemetry_mtu = BLE_ATT_MTU_DFLT;
static uint8_t telemetry_addr_type;
static volatile bool telemetry_connected;
static volatile bool telemetry_subscribed;
static portMUX_TYPE telemetry_lock = portMUX_INITIALIZER_UNLOCKED;
static uint32_t telemetry_dropped_records;
static uint32_t telemetry_notifications;
static uint32_t telemetry_notify_errors;
static uint32_t telemetry_connect_events;
static int telemetry_last_connect_status;
static char telemetry_device_name[TELEMETRY_DEVICE_NAME_BYTES];
static bool telemetry_usb_driver_ready;

static void telemetry_advertise(void);

static void usb_write_nonblocking(const char *data, size_t length)
{
    if (!telemetry_usb_driver_ready || length == 0) {
        return;
    }
    (void)usb_serial_jtag_write_bytes(data, length, 0);
}

static void put_u16_le(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
}

static void put_u32_le(uint8_t *destination, uint32_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
    destination[2] = (uint8_t)(value >> 16);
    destination[3] = (uint8_t)(value >> 24);
}

static void put_bits(
    uint8_t *destination,
    size_t *bit_offset,
    uint32_t value,
    unsigned bit_count)
{
    for (unsigned bit = 0; bit < bit_count; bit++) {
        if ((value & (1U << bit)) != 0U) {
            size_t output_bit = *bit_offset + bit;
            destination[output_bit / 8] |= (uint8_t)(1U << (output_bit % 8));
        }
    }
    *bit_offset += bit_count;
}

static bool ppg_batch_compatible(const ppg_sample_t *previous, const ppg_sample_t *sample)
{
    int64_t delta_ms = sample->timestamp_ms - previous->timestamp_ms;
    return sample->sequence == previous->sequence + 1 &&
           delta_ms >= 8 && delta_ms <= 15 &&
           sample->red <= 0x3ffffU && sample->ir <= 0x3ffffU;
}

static bool imu_batch_compatible(const imu_sample_t *previous, const imu_sample_t *sample)
{
    int64_t delta_ms = sample->timestamp_ms - previous->timestamp_ms;
    return sample->sequence == previous->sequence + 1 &&
           delta_ms >= 8 && delta_ms <= 15 &&
           sample->x >= -1024 && sample->x <= 1023 &&
           sample->y >= -1024 && sample->y <= 1023 &&
           sample->z >= -1024 && sample->z <= 1023;
}

static void build_ppg_absolute(const ppg_sample_t *sample, pending_packet_t *packet)
{
    memset(packet, 0, sizeof(*packet));
    packet->bytes[0] = BLE_FRAME_PPG_ABSOLUTE;
    put_u32_le(&packet->bytes[1], (uint32_t)sample->sequence);
    put_u32_le(&packet->bytes[5], (uint32_t)sample->timestamp_ms);
    put_u32_le(&packet->bytes[9], sample->red);
    put_u32_le(&packet->bytes[13], sample->ir);
    packet->length = 17;
    packet->kind = PENDING_PPG;
}

static void build_imu_absolute(const imu_sample_t *sample, pending_packet_t *packet)
{
    memset(packet, 0, sizeof(*packet));
    packet->bytes[0] = BLE_FRAME_IMU_ABSOLUTE;
    put_u32_le(&packet->bytes[1], (uint32_t)sample->sequence);
    put_u32_le(&packet->bytes[5], (uint32_t)sample->timestamp_ms);
    put_u16_le(&packet->bytes[9], (uint16_t)sample->x);
    put_u16_le(&packet->bytes[11], (uint16_t)sample->y);
    put_u16_le(&packet->bytes[13], (uint16_t)sample->z);
    packet->length = 15;
    packet->kind = PENDING_IMU;
}

static bool build_ppg_packet(
    bool *held_valid,
    ppg_sample_t *held_sample,
    size_t payload_limit,
    pending_packet_t *packet)
{
    ppg_sample_t samples[BLE_FRAME_BATCH_MAX];
    size_t max_count = payload_limit * 8U >= 108U
        ? 1U + ((payload_limit * 8U) - 108U) / 39U
        : 0U;
    if (max_count > BLE_FRAME_BATCH_MAX) {
        max_count = BLE_FRAME_BATCH_MAX;
    }
    size_t count = 0;
    ppg_sample_t sample;
    if (*held_valid) {
        sample = *held_sample;
        *held_valid = false;
    } else if (xQueueReceive(telemetry_ppg_queue, &sample, 0) != pdTRUE) {
        return false;
    }

    if (sample.red > 0x3ffffU || sample.ir > 0x3ffffU || max_count == 0) {
        build_ppg_absolute(&sample, packet);
        return true;
    }

    samples[count++] = sample;
    while (count < max_count &&
           xQueueReceive(telemetry_ppg_queue, &sample, 0) == pdTRUE) {
        if (!ppg_batch_compatible(&samples[count - 1], &sample)) {
            *held_sample = sample;
            *held_valid = true;
            break;
        }
        samples[count++] = sample;
    }

    memset(packet, 0, sizeof(*packet));
    packet->bytes[0] = (uint8_t)(BLE_FRAME_PPG_SELF_CONTAINED_BASE | count);
    put_u32_le(&packet->bytes[1], (uint32_t)samples[0].sequence);
    put_u32_le(&packet->bytes[5], (uint32_t)samples[0].timestamp_ms);
    size_t bit_offset = 72;
    put_bits(packet->bytes, &bit_offset, samples[0].red, 18);
    put_bits(packet->bytes, &bit_offset, samples[0].ir, 18);
    for (size_t index = 1; index < count; index++) {
        uint32_t delta_code = (uint32_t)(
            samples[index].timestamp_ms - samples[index - 1].timestamp_ms - 8);
        put_bits(packet->bytes, &bit_offset, samples[index].red, 18);
        put_bits(packet->bytes, &bit_offset, samples[index].ir, 18);
        put_bits(packet->bytes, &bit_offset, delta_code, 3);
    }
    packet->length = (bit_offset + 7) / 8;
    packet->kind = PENDING_PPG;
    return true;
}

static bool build_imu_packet(
    bool *held_valid,
    imu_sample_t *held_sample,
    size_t payload_limit,
    pending_packet_t *packet)
{
    imu_sample_t samples[BLE_FRAME_BATCH_MAX];
    size_t max_count = payload_limit * 8U >= 105U
        ? 1U + ((payload_limit * 8U) - 105U) / 36U
        : 0U;
    if (max_count > BLE_FRAME_BATCH_MAX) {
        max_count = BLE_FRAME_BATCH_MAX;
    }
    size_t count = 0;
    imu_sample_t sample;
    if (*held_valid) {
        sample = *held_sample;
        *held_valid = false;
    } else if (xQueueReceive(telemetry_imu_queue, &sample, 0) != pdTRUE) {
        return false;
    }

    if (sample.x < -1024 || sample.x > 1023 ||
        sample.y < -1024 || sample.y > 1023 ||
        sample.z < -1024 || sample.z > 1023 || max_count == 0) {
        build_imu_absolute(&sample, packet);
        return true;
    }

    samples[count++] = sample;
    while (count < max_count &&
           xQueueReceive(telemetry_imu_queue, &sample, 0) == pdTRUE) {
        if (!imu_batch_compatible(&samples[count - 1], &sample)) {
            *held_sample = sample;
            *held_valid = true;
            break;
        }
        samples[count++] = sample;
    }

    memset(packet, 0, sizeof(*packet));
    packet->bytes[0] = (uint8_t)(BLE_FRAME_IMU_SELF_CONTAINED_BASE | count);
    put_u32_le(&packet->bytes[1], (uint32_t)samples[0].sequence);
    put_u32_le(&packet->bytes[5], (uint32_t)samples[0].timestamp_ms);
    size_t bit_offset = 72;
    put_bits(packet->bytes, &bit_offset, (uint16_t)samples[0].x & 0x7ffU, 11);
    put_bits(packet->bytes, &bit_offset, (uint16_t)samples[0].y & 0x7ffU, 11);
    put_bits(packet->bytes, &bit_offset, (uint16_t)samples[0].z & 0x7ffU, 11);
    for (size_t index = 1; index < count; index++) {
        uint32_t delta_code = (uint32_t)(
            samples[index].timestamp_ms - samples[index - 1].timestamp_ms - 8);
        put_bits(packet->bytes, &bit_offset, (uint16_t)samples[index].x & 0x7ffU, 11);
        put_bits(packet->bytes, &bit_offset, (uint16_t)samples[index].y & 0x7ffU, 11);
        put_bits(packet->bytes, &bit_offset, (uint16_t)samples[index].z & 0x7ffU, 11);
        put_bits(packet->bytes, &bit_offset, delta_code, 3);
    }
    packet->length = (bit_offset + 7) / 8;
    packet->kind = PENDING_IMU;
    return true;
}

static bool build_status_packet(
    size_t payload_limit,
    uint16_t *fragment_sequence,
    pending_packet_t *packet)
{
    if (payload_limit <= 3) {
        return false;
    }
    memset(packet, 0, sizeof(*packet));
    size_t count = xStreamBufferReceive(
        telemetry_status_stream,
        &packet->bytes[3],
        payload_limit - 3,
        0);
    if (count == 0) {
        return false;
    }
    packet->bytes[0] = BLE_FRAME_STATUS_FRAGMENT;
    put_u16_le(&packet->bytes[1], *fragment_sequence);
    (*fragment_sequence)++;
    packet->length = count + 3;
    packet->kind = PENDING_STATUS;
    return true;
}

static int telemetry_gatt_access(
    uint16_t conn_handle,
    uint16_t attr_handle,
    struct ble_gatt_access_ctxt *ctxt,
    void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)ctxt;
    (void)arg;
    return 0;
}

static const struct ble_gatt_svc_def telemetry_services[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &telemetry_service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = &telemetry_tx_uuid.u,
                .access_cb = telemetry_gatt_access,
                .val_handle = &telemetry_tx_handle,
                .flags = BLE_GATT_CHR_F_NOTIFY,
            },
            {0},
        },
    },
    {0},
};

static int telemetry_gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        portENTER_CRITICAL(&telemetry_lock);
        telemetry_connect_events++;
        telemetry_last_connect_status = event->connect.status;
        portEXIT_CRITICAL(&telemetry_lock);
        if (event->connect.status == 0) {
            struct ble_gap_upd_params connection_params = {
                .itvl_min = 6,
                .itvl_max = 12,
                .latency = 0,
                .supervision_timeout = 400,
                .min_ce_len = 0,
                .max_ce_len = 0,
            };
            portENTER_CRITICAL(&telemetry_lock);
            telemetry_conn_handle = event->connect.conn_handle;
            telemetry_connected = true;
            telemetry_subscribed = false;
            telemetry_mtu = BLE_ATT_MTU_DFLT;
            portEXIT_CRITICAL(&telemetry_lock);
            (void)ble_gap_update_params(event->connect.conn_handle, &connection_params);
        } else {
            telemetry_advertise();
        }
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        portENTER_CRITICAL(&telemetry_lock);
        telemetry_connected = false;
        telemetry_subscribed = false;
        telemetry_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        telemetry_mtu = BLE_ATT_MTU_DFLT;
        portEXIT_CRITICAL(&telemetry_lock);
        telemetry_advertise();
        return 0;

    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == telemetry_tx_handle) {
            portENTER_CRITICAL(&telemetry_lock);
            telemetry_subscribed = event->subscribe.cur_notify != 0;
            portEXIT_CRITICAL(&telemetry_lock);
        }
        return 0;

    case BLE_GAP_EVENT_MTU:
        portENTER_CRITICAL(&telemetry_lock);
        telemetry_mtu = event->mtu.value;
        portEXIT_CRITICAL(&telemetry_lock);
        return 0;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        telemetry_advertise();
        return 0;

    default:
        return 0;
    }
}

static void telemetry_advertise(void)
{
    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = (ble_uuid128_t *)&telemetry_service_uuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;
    if (ble_gap_adv_set_fields(&fields) != 0) {
        return;
    }

    struct ble_hs_adv_fields response = {0};
    response.name = (uint8_t *)telemetry_device_name;
    response.name_len = strlen(telemetry_device_name);
    response.name_is_complete = 1;
    if (ble_gap_adv_rsp_set_fields(&response) != 0) {
        return;
    }

    struct ble_gap_adv_params params = {0};
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    (void)ble_gap_adv_start(
        telemetry_addr_type,
        NULL,
        BLE_HS_FOREVER,
        &params,
        telemetry_gap_event,
        NULL);
}

static void telemetry_on_reset(int reason)
{
    (void)reason;
    portENTER_CRITICAL(&telemetry_lock);
    telemetry_connected = false;
    telemetry_subscribed = false;
    telemetry_conn_handle = BLE_HS_CONN_HANDLE_NONE;
    portEXIT_CRITICAL(&telemetry_lock);
}

static void telemetry_on_sync(void)
{
    if (ble_hs_util_ensure_addr(0) != 0) {
        return;
    }
    if (ble_hs_id_infer_auto(0, &telemetry_addr_type) != 0) {
        return;
    }
    telemetry_advertise();
}

static void telemetry_host_task(void *parameter)
{
    (void)parameter;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void discard_queued_telemetry(void)
{
    uint8_t status_bytes[TELEMETRY_MAX_NOTIFY_BYTES];
    ppg_sample_t ppg_sample;
    imu_sample_t imu_sample;
    while (xStreamBufferReceive(
               telemetry_status_stream,
               status_bytes,
               sizeof(status_bytes),
               0) > 0) {
    }
    while (xQueueReceive(telemetry_ppg_queue, &ppg_sample, 0) == pdTRUE) {
    }
    while (xQueueReceive(telemetry_imu_queue, &imu_sample, 0) == pdTRUE) {
    }
}

static void telemetry_tx_task(void *parameter)
{
    (void)parameter;
    pending_packet_t pending = {0};
    ppg_sample_t held_ppg = {0};
    imu_sample_t held_imu = {0};
    bool held_ppg_valid = false;
    bool held_imu_valid = false;
    bool previously_subscribed = false;
    bool prefer_ppg = true;
    unsigned raw_packets_since_status = 0;
    uint16_t status_fragment_sequence = 0;

    while (true) {
        bool subscribed;
        uint16_t conn_handle;
        uint16_t mtu;
        portENTER_CRITICAL(&telemetry_lock);
        subscribed = telemetry_connected && telemetry_subscribed;
        conn_handle = telemetry_conn_handle;
        mtu = telemetry_mtu;
        portEXIT_CRITICAL(&telemetry_lock);

        if (!subscribed) {
            pending.length = 0;
            held_ppg_valid = false;
            held_imu_valid = false;
            status_fragment_sequence = 0;
            if (previously_subscribed) {
                discard_queued_telemetry();
            }
            previously_subscribed = false;
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        previously_subscribed = true;
        vTaskDelay(pdMS_TO_TICKS(
            mtu > BLE_ATT_MTU_DFLT
                ? TELEMETRY_TX_DELAY_LARGE_MTU_MS
                : TELEMETRY_TX_DELAY_SMALL_MTU_MS));

        size_t payload_limit = mtu > 3 ? (size_t)(mtu - 3) : 20;
        if (payload_limit > sizeof(pending.bytes)) {
            payload_limit = sizeof(pending.bytes);
        }

        if (pending.length == 0) {
            bool built = false;
            bool status_due = raw_packets_since_status >= TELEMETRY_STATUS_AFTER_RAW_PACKETS;
            if (status_due) {
                built = build_status_packet(
                    payload_limit, &status_fragment_sequence, &pending);
                if (built) {
                    raw_packets_since_status = 0;
                }
            }
            if (!built) {
                if (prefer_ppg) {
                    built = build_ppg_packet(
                        &held_ppg_valid, &held_ppg, payload_limit, &pending);
                    if (!built) {
                        built = build_imu_packet(
                            &held_imu_valid, &held_imu, payload_limit, &pending);
                    }
                } else {
                    built = build_imu_packet(
                        &held_imu_valid, &held_imu, payload_limit, &pending);
                    if (!built) {
                        built = build_ppg_packet(
                            &held_ppg_valid, &held_ppg, payload_limit, &pending);
                    }
                }
                if (built) {
                    prefer_ppg = pending.kind != PENDING_PPG;
                    raw_packets_since_status++;
                }
            }
            if (!built) {
                (void)build_status_packet(
                    payload_limit, &status_fragment_sequence, &pending);
            }
        }
        if (pending.length == 0 || pending.length > payload_limit) {
            continue;
        }

        struct os_mbuf *buffer = ble_hs_mbuf_from_flat(pending.bytes, pending.length);
        int result = buffer == NULL
            ? BLE_HS_ENOMEM
            : ble_gatts_notify_custom(conn_handle, telemetry_tx_handle, buffer);
        portENTER_CRITICAL(&telemetry_lock);
        if (result == 0) {
            telemetry_notifications++;
            pending.length = 0;
            pending.kind = PENDING_NONE;
        } else {
            telemetry_notify_errors++;
        }
        portEXIT_CRITICAL(&telemetry_lock);
    }
}

static bool telemetry_is_subscribed(void)
{
    bool subscribed;
    portENTER_CRITICAL(&telemetry_lock);
    subscribed = telemetry_connected && telemetry_subscribed;
    portEXIT_CRITICAL(&telemetry_lock);
    return subscribed;
}

static void count_dropped_record(void)
{
    portENTER_CRITICAL(&telemetry_lock);
    telemetry_dropped_records++;
    portEXIT_CRITICAL(&telemetry_lock);
}

esp_err_t telemetry_init(void)
{
    if (usb_serial_jtag_is_driver_installed()) {
        telemetry_usb_driver_ready = true;
    } else {
        usb_serial_jtag_driver_config_t usb_config = {
            .tx_buffer_size = TELEMETRY_USB_TX_BUFFER_BYTES,
            .rx_buffer_size = TELEMETRY_USB_RX_BUFFER_BYTES,
        };
        telemetry_usb_driver_ready =
            usb_serial_jtag_driver_install(&usb_config) == ESP_OK;
    }

    telemetry_status_stream = xStreamBufferCreate(TELEMETRY_STATUS_STREAM_BYTES, 1);
    telemetry_ppg_queue = xQueueCreate(TELEMETRY_SAMPLE_QUEUE_LENGTH, sizeof(ppg_sample_t));
    telemetry_imu_queue = xQueueCreate(TELEMETRY_SAMPLE_QUEUE_LENGTH, sizeof(imu_sample_t));
    if (telemetry_status_stream == NULL || telemetry_ppg_queue == NULL || telemetry_imu_queue == NULL) {
        return ESP_ERR_NO_MEM;
    }

    uint8_t mac[6] = {0};
    ESP_RETURN_ON_ERROR(esp_read_mac(mac, ESP_MAC_BT), "telemetry", "read BLE MAC");
    snprintf(
        telemetry_device_name,
        sizeof(telemetry_device_name),
        "PPG-LOGGER-%02X%02X%02X",
        mac[3],
        mac[4],
        mac[5]);

    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES || result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_RETURN_ON_ERROR(nvs_flash_erase(), "telemetry", "erase NVS");
        result = nvs_flash_init();
    }
    ESP_RETURN_ON_ERROR(result, "telemetry", "initialize NVS");
    ESP_RETURN_ON_ERROR(nimble_port_init(), "telemetry", "initialize NimBLE");

    int rc = ble_att_set_preferred_mtu(TELEMETRY_PREFERRED_MTU);
    if (rc != 0) {
        return ESP_FAIL;
    }
    ble_svc_gap_init();
    ble_svc_gatt_init();
    rc = ble_gatts_count_cfg(telemetry_services);
    if (rc == 0) {
        rc = ble_gatts_add_svcs(telemetry_services);
    }
    if (rc != 0 || ble_svc_gap_device_name_set(telemetry_device_name) != 0) {
        return ESP_FAIL;
    }

    ble_hs_cfg.reset_cb = telemetry_on_reset;
    ble_hs_cfg.sync_cb = telemetry_on_sync;
    nimble_port_freertos_init(telemetry_host_task);
    if (xTaskCreatePinnedToCore(
            telemetry_tx_task,
            "ble_telemetry",
            4096,
            NULL,
            tskIDLE_PRIORITY + 1,
            NULL,
            1) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

int telemetry_printf(const char *format, ...)
{
    char line[TELEMETRY_LINE_BYTES];
    va_list arguments;
    va_start(arguments, format);
    int required = vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    if (required < 0) {
        return required;
    }

    size_t length = (size_t)required;
    if (length >= sizeof(line)) {
        length = sizeof(line) - 1;
    }
    usb_write_nonblocking(line, length);

    if (telemetry_is_subscribed()) {
        size_t available = xStreamBufferSpacesAvailable(telemetry_status_stream);
        if (available < length ||
            xStreamBufferSend(telemetry_status_stream, line, length, 0) != length) {
            count_dropped_record();
        }
    }
    return required;
}

void telemetry_ppg_sample(uint64_t sequence, int64_t timestamp_ms, uint32_t red, uint32_t ir)
{
    char line[96];
    int length = snprintf(
        line,
        sizeof(line),
        "%" PRIu64 ",%" PRId64 ",%" PRIu32 ",%" PRIu32 "\n",
        sequence,
        timestamp_ms,
        red,
        ir);
    if (length > 0) {
        usb_write_nonblocking(line, (size_t)length);
    }
    if (!telemetry_is_subscribed()) {
        return;
    }
    ppg_sample_t sample = {
        .sequence = sequence,
        .timestamp_ms = timestamp_ms,
        .red = red,
        .ir = ir,
    };
    if (xQueueSend(telemetry_ppg_queue, &sample, 0) != pdTRUE) {
        count_dropped_record();
    }
}

void telemetry_imu_sample(
    uint64_t sequence,
    int64_t timestamp_ms,
    int16_t x,
    int16_t y,
    int16_t z)
{
    char line[96];
    int length = snprintf(
        line,
        sizeof(line),
        "imu,%" PRIu64 ",%" PRId64 ",%d,%d,%d\n",
        sequence,
        timestamp_ms,
        (int)x,
        (int)y,
        (int)z);
    if (length > 0) {
        usb_write_nonblocking(line, (size_t)length);
    }
    if (!telemetry_is_subscribed()) {
        return;
    }
    imu_sample_t sample = {
        .sequence = sequence,
        .timestamp_ms = timestamp_ms,
        .x = x,
        .y = y,
        .z = z,
    };
    if (xQueueSend(telemetry_imu_queue, &sample, 0) != pdTRUE) {
        count_dropped_record();
    }
}

void telemetry_print_ble_stats(void)
{
    bool connected;
    bool subscribed;
    uint16_t mtu;
    uint32_t dropped_records;
    uint32_t notifications;
    uint32_t notify_errors;
    uint32_t connect_events;
    int last_connect_status;
    portENTER_CRITICAL(&telemetry_lock);
    connected = telemetry_connected;
    subscribed = telemetry_subscribed;
    mtu = telemetry_mtu;
    dropped_records = telemetry_dropped_records;
    notifications = telemetry_notifications;
    notify_errors = telemetry_notify_errors;
    connect_events = telemetry_connect_events;
    last_connect_status = telemetry_last_connect_status;
    portEXIT_CRITICAL(&telemetry_lock);
    size_t queued_status = xStreamBufferBytesAvailable(telemetry_status_stream);
    size_t queued_ppg = uxQueueMessagesWaiting(telemetry_ppg_queue);
    size_t queued_imu = uxQueueMessagesWaiting(telemetry_imu_queue);
    telemetry_printf(
        "# ble_stats connected=%s subscribed=%s mtu=%u queued_bytes=%u "
        "queued_ppg=%u queued_imu=%u dropped_records=%" PRIu32
        " notifications=%" PRIu32 " notify_errors=%" PRIu32
        " connect_events=%" PRIu32 " last_connect_status=%d\n",
        connected ? "true" : "false",
        subscribed ? "true" : "false",
        mtu,
        (unsigned int)queued_status,
        (unsigned int)queued_ppg,
        (unsigned int)queued_imu,
        dropped_records,
        notifications,
        notify_errors,
        connect_events,
        last_connect_status);
}
