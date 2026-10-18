#include "settings.h"
#include "stream_core.h"
#include "audio_frontend.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cJSON.h"
#include "driver/i2s_pdm.h"
#include "driver/gpio.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_websocket_client.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "nvs_flash.h"

static const char *TAG = "aria_mic";
enum { WIFI_READY = BIT0, WS_READY = BIT1, STREAMING = BIT2, AUDIO_FAULT = BIT3,
       VAD_ACTIVE = BIT4, TX_ACTIVITY = BIT5, CONNECTING = BIT6 };
static EventGroupHandle_t events;
static QueueHandle_t audio_queue;
static QueueHandle_t diagnostics_queue;
static i2s_chan_handle_t microphone;
static aria_text_buffer status_buffer; /* Used only by the WebSocket task. */
static aria_frontend frontend; /* Bounded pre-roll belongs in static RAM. */
static char client_id[64]; /* Same name in DHCP and the Aria handshake. */

typedef struct {
    int64_t captured_us;
    uint32_t sequence;
    int16_t samples[ARIA_SAMPLES_PER_CHUNK];
} audio_packet;

typedef struct {
    uint32_t frames, read_errors, samples, nonzero;
    float rms, peak, dc, ac_rms;
} capture_diagnostics;

/* This task owns the LED and serial diagnostics. Neither can stall capture. */
static void status_task(void *arg)
{
    ESP_ERROR_CHECK(gpio_reset_pin(ARIA_LED_GPIO));
    ESP_ERROR_CHECK(gpio_set_level(ARIA_LED_GPIO, ARIA_LED_ACTIVE_LOW ? 1 : 0));
    ESP_ERROR_CHECK(gpio_set_direction(ARIA_LED_GPIO, GPIO_MODE_OUTPUT));
    int64_t last_tx_ms = -1000;
    int64_t vad_flash_until_ms = 0;
    bool was_vad_active = false;
    for (;;) {
        /* Atomically consume the transmit hint; never clear connection/fault bits. */
        EventBits_t bits = xEventGroupClearBits(events, TX_ACTIVITY);
        int64_t now_ms = esp_timer_get_time() / 1000;
        if (bits & TX_ACTIVITY) last_tx_ms = now_ms;
        bool vad_active = (bits & VAD_ACTIVE) != 0;
        if (vad_active && !was_vad_active) vad_flash_until_ms = now_ms + 250;
        was_vad_active = vad_active;
        bool on;
        if (bits & CONNECTING) on = true;
        else if (!(bits & WIFI_READY))
            on = now_ms % ARIA_DISCONNECTED_LED_PERIOD_MS < 100;
        else if (!(bits & STREAMING)) {
            /* Two 100 ms on/off beats, then an extra 40 ms pulse; no long pause. */
            int phase = now_ms % (2 * ARIA_DISCONNECTED_LED_PERIOD_MS + 100);
            on = phase < 100 || (phase >= 200 && phase < 300) ||
                 (phase >= 400 && phase < 440);
        } else if (bits & AUDIO_FAULT) on = now_ms % 200 < 100;
        else if (now_ms < vad_flash_until_ms) on = true;
        else if (now_ms - last_tx_ms < 120) on = now_ms % 100 < 50;
        else if (bits & VAD_ACTIVE) on = true;
        else on = now_ms % 2000 < 40;
        gpio_set_level(ARIA_LED_GPIO, ARIA_LED_ACTIVE_LOW ? !on : on);

        capture_diagnostics meter;
        if (xQueueReceive(diagnostics_queue, &meter, 0) == pdTRUE) {
            ESP_LOGI(TAG, "capture_frames=%" PRIu32 " read_errors=%" PRIu32
                     " raw_rms=%.5f raw_peak=%.5f dc=%.5f ac_rms=%.5f"
                     " nonzero=%" PRIu32 "/%" PRIu32,
                     meter.frames, meter.read_errors, meter.rms, meter.peak,
                     meter.dc, meter.ac_rms, meter.nonzero, meter.samples);
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

_Static_assert(ARIA_SAMPLE_RATE == 16000 && ARIA_CHUNK_MS == 20,
               "Changing capture format requires updating the wire contract");
_Static_assert(sizeof(((audio_packet *)0)->samples) == 640, "20 ms PCM16 packet");
_Static_assert(sizeof(ARIA_HOSTNAME_PREFIX) > 1 && sizeof(ARIA_HOSTNAME_PREFIX) <= 20,
               "Hostname prefix must leave room for a hyphen and twelve MAC digits");

static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupClearBits(events, CONNECTING);
        xEventGroupSetBits(events, WIFI_READY);
        const ip_event_got_ip_t *ip = data;
        wifi_ap_record_t ap = {0};
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK)
            ESP_LOGI(TAG, "Wi-Fi ready ip=" IPSTR " rssi=%d channel=%u",
                     IP2STR(&ip->ip_info.ip), ap.rssi, ap.primary);
        else ESP_LOGI(TAG, "Wi-Fi ready ip=" IPSTR, IP2STR(&ip->ip_info.ip));
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(events, WIFI_READY | STREAMING | CONNECTING);
        const wifi_event_sta_disconnected_t *disconnected = data;
        ESP_LOGW(TAG, "Wi-Fi disconnected reason=%u; audio will not be replayed",
                 disconnected->reason);
    }
}

static void websocket_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (id == WEBSOCKET_EVENT_CONNECTED) {
        xEventGroupSetBits(events, WS_READY);
    } else if (id == WEBSOCKET_EVENT_DISCONNECTED || id == WEBSOCKET_EVENT_ERROR ||
               id == WEBSOCKET_EVENT_CLOSED) {
        xEventGroupClearBits(events, WS_READY | STREAMING | CONNECTING);
    } else if (id == WEBSOCKET_EVENT_DATA) {
        const esp_websocket_event_data_t *event = data;
        const char *text = aria_text_push(&status_buffer, event->op_code, event->fin,
                                         event->payload_len, event->payload_offset,
                                         event->data_ptr, event->data_len);
        if (!text) return;
        cJSON *root = cJSON_Parse(text);
        const cJSON *type = cJSON_GetObjectItemCaseSensitive(root, "type");
        const cJSON *status = cJSON_GetObjectItemCaseSensitive(root, "status");
        if (cJSON_IsString(type) && strcmp(type->valuestring, "request_status") == 0 &&
            cJSON_IsString(status)) {
            /* Status is diagnostic only. Completed is NOT playback completion.
             * Never retry requests or resume/mute capture from these messages. */
            ESP_LOGI(TAG, "Apollo request status: %.32s", status->valuestring);
        }
        cJSON_Delete(root);
    }
}

static void capture_task(void *arg)
{
    audio_packet packet = {0};
    uint32_t sequence = 0;
    capture_diagnostics meter = {0};
    uint64_t energy = 0;
    int64_t sum = 0;
    int32_t peak = 0;
    int64_t last_report = esp_timer_get_time();
    for (;;) {
        size_t received = 0;
        /* I2S continuously drains even without Wi-Fi; network cannot block DMA. */
        esp_err_t result = i2s_channel_read(microphone, packet.samples,
                                           sizeof(packet.samples), &received, 100);
        packet.sequence = sequence++;
        packet.captured_us = esp_timer_get_time();
        if (result != ESP_OK || received != sizeof(packet.samples)) {
            ++meter.read_errors;
            xEventGroupSetBits(events, AUDIO_FAULT);
            vTaskDelay(pdMS_TO_TICKS(20));
        } else {
            ++meter.frames;
            for (size_t i = 0; i < ARIA_SAMPLES_PER_CHUNK; ++i) {
                int32_t sample = packet.samples[i];
                int32_t magnitude = sample < 0 ? -sample : sample;
                energy += (uint64_t)((int64_t)sample * sample);
                sum += sample;
                if (magnitude > peak) peak = magnitude;
                if (sample) ++meter.nonzero;
            }
            meter.samples += ARIA_SAMPLES_PER_CHUNK;
            if (xEventGroupGetBits(events) & STREAMING) {
                if (xQueueSend(audio_queue, &packet, 0) != pdTRUE) {
                    xEventGroupSetBits(events, AUDIO_FAULT);
                    xEventGroupClearBits(events, STREAMING);
                }
            }
        }
        if (esp_timer_get_time() - last_report >= 1000000) {
            meter.rms = meter.samples ? sqrtf((float)energy / meter.samples) / 32768.f : 0;
            meter.peak = peak / 32768.f;
            double mean = meter.samples ? (double)sum / meter.samples : 0;
            double variance = meter.samples ? (double)energy / meter.samples - mean * mean : 0;
            meter.dc = mean / 32768.0;
            meter.ac_rms = sqrt(variance > 0 ? variance : 0) / 32768.0;
            xQueueOverwrite(diagnostics_queue, &meter);
            meter.samples = meter.nonzero = 0;
            energy = 0;
            sum = 0;
            peak = 0;
            last_report = esp_timer_get_time();
        }
    }
}

static bool capture_overflow(i2s_chan_handle_t channel, i2s_event_data_t *data, void *arg)
{
    BaseType_t woken = pdFALSE;
    xEventGroupSetBitsFromISR(events, AUDIO_FAULT, &woken);
    return woken == pdTRUE;
}

static void init_microphone(void)
{
    i2s_chan_config_t channel = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    channel.dma_desc_num = 4;
    channel.dma_frame_num = ARIA_SAMPLES_PER_CHUNK;
    ESP_ERROR_CHECK(i2s_new_channel(&channel, NULL, &microphone));
    i2s_pdm_rx_config_t pdm = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(ARIA_SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.clk = ARIA_MIC_CLK_GPIO, .din = ARIA_MIC_DATA_GPIO},
    };
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(microphone, &pdm));
    i2s_event_callbacks_t callbacks = {.on_recv_q_ovf = capture_overflow};
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(microphone, &callbacks, NULL));
    ESP_ERROR_CHECK(i2s_channel_enable(microphone));
}

static bool settings_valid(void)
{
    struct in_addr address;
    size_t ssid_length = strlen(ARIA_WIFI_SSID);
    size_t password_length = strlen(ARIA_WIFI_PASSWORD);
    /* Numeric RFC1918 address: no public endpoint, DNS, NTP, or discovery. */
    return ssid_length > 0 && ssid_length <= 32 && password_length <= 63 &&
           (password_length == 0 || password_length >= 8) &&
           ARIA_APOLLO_PORT > 0 && ARIA_APOLLO_PORT <= 65535 &&
           inet_pton(AF_INET, ARIA_APOLLO_IPV4, &address) == 1 &&
           aria_private_ipv4(ntohl(address.s_addr));
}

static void init_wifi(void)
{
    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES || result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        /* Do not erase existing board data automatically. */
        ESP_LOGE(TAG, "NVS requires maintenance before Wi-Fi can start");
    }
    ESP_ERROR_CHECK(result);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *station = esp_netif_create_default_wifi_sta();
    assert(station != NULL);
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED,
                                              wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                              wifi_event, NULL));
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    wifi_config_t wifi = {0};
    memcpy(wifi.sta.ssid, ARIA_WIFI_SSID, strlen(ARIA_WIFI_SSID));
    memcpy(wifi.sta.password, ARIA_WIFI_PASSWORD, strlen(ARIA_WIFI_PASSWORD));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    uint8_t mac[6];
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_STA, mac));
    snprintf(client_id, sizeof(client_id), "%s-%02x%02x%02x%02x%02x%02x", ARIA_HOSTNAME_PREFIX,
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    ESP_ERROR_CHECK(esp_netif_set_hostname(station, client_id));
    ESP_LOGI(TAG, "Client ID / DHCP hostname: %s", client_id);
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
}

typedef struct { esp_websocket_client_handle_t client; int64_t send_us; } transport_context;

static bool send_audio_event(void *context, aria_audio_event event, const int16_t *pcm, size_t count)
{
    transport_context *transport = context;
    int64_t began = esp_timer_get_time();
    int written, expected;
    if (event == ARIA_TURN_START) xEventGroupSetBits(events, VAD_ACTIVE);
    else if (event == ARIA_TURN_END) xEventGroupClearBits(events, VAD_ACTIVE);
    if (event == ARIA_AUDIO) {
        expected = count * sizeof(int16_t);
        written = esp_websocket_client_send_bin(transport->client, (const char *)pcm, expected,
                                                pdMS_TO_TICKS(ARIA_SEND_TIMEOUT_MS));
    } else {
        const char *message = event == ARIA_TURN_START ? "{\"type\":\"utterance_start\"}" :
                              "{\"type\":\"utterance_end\"}";
        expected = strlen(message);
        written = esp_websocket_client_send_text(transport->client, message, expected,
                                                 pdMS_TO_TICKS(ARIA_SEND_TIMEOUT_MS));
        ESP_LOGI(TAG, "turn_event=%d", event);
    }
    int64_t elapsed_us = esp_timer_get_time() - began;
    transport->send_us += elapsed_us;
    if (written != expected) {
        wifi_ap_record_t ap = {0};
        bool associated = esp_wifi_sta_get_ap_info(&ap) == ESP_OK;
        ESP_LOGW(TAG, "Send failure event=%d written=%d/%d elapsed_us=%" PRId64
                 " rssi=%d associated=%d", event, written, expected, elapsed_us,
                 ap.rssi, associated);
    }
    if (event == ARIA_AUDIO && written == expected) xEventGroupSetBits(events, TX_ACTIVITY);
    return written == expected;
}

static bool run_session(void)
{
    xEventGroupClearBits(events, WS_READY | STREAMING | AUDIO_FAULT | VAD_ACTIVE | TX_ACTIVITY);
    status_buffer = (aria_text_buffer){0};
    esp_websocket_client_config_t config = {
        .host = ARIA_APOLLO_IPV4,
        .port = ARIA_APOLLO_PORT,
        .path = "/ws/asr",
        .transport = WEBSOCKET_TRANSPORT_OVER_TCP,
        .disable_auto_reconnect = true,
        .network_timeout_ms = ARIA_CONNECT_TIMEOUT_MS,
        .buffer_size = ARIA_STATUS_CAPACITY,
        .task_stack = 6144,
        .ping_interval_sec = 10,
        .pingpong_timeout_sec = 10,
    };
    esp_websocket_client_handle_t client = esp_websocket_client_init(&config);
    if (!client) {
        ESP_LOGE(TAG, "Cannot allocate WebSocket client");
        return false;
    }
    bool client_started = false;
    bool frontend_started = false;
    transport_context transport = {.client = client};
    if (esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY,
                                      websocket_event, NULL) != ESP_OK) goto cleanup;
    if (esp_websocket_client_start(client) != ESP_OK) goto cleanup;
    client_started = true;
    EventBits_t bits = xEventGroupWaitBits(events, WS_READY, pdFALSE, pdTRUE,
                                         pdMS_TO_TICKS(ARIA_CONNECT_TIMEOUT_MS));
    if (!(bits & WS_READY) || !(bits & WIFI_READY)) goto cleanup;
    char start_message[256];
    int length = (int)aria_identified_start(start_message, sizeof(start_message), client_id);
    if (!length) { ESP_LOGE(TAG, "Invalid client name"); goto cleanup; }
    if (esp_websocket_client_send_text(client, start_message, length,
                                      pdMS_TO_TICKS(ARIA_SEND_TIMEOUT_MS)) != length) goto cleanup;
    aria_frontend_config audio_config = {
        .vad_enabled = ARIA_CLIENT_VAD, .agc_enabled = ARIA_CLIENT_AGC,
        .vad_mode = ARIA_CLIENT_VAD_MODE, .start_frames = ARIA_CLIENT_VAD_START_FRAMES,
        .silence_ms = ARIA_CLIENT_VAD_SILENCE_MS, .preroll_frames = ARIA_CLIENT_VAD_PRE_MS / 20,
        .min_rms = ARIA_CLIENT_VAD_MIN_RMS, .snr_db = ARIA_CLIENT_VAD_SNR_DB,
        .noise_alpha = ARIA_CLIENT_VAD_NOISE_ALPHA, .target_rms = ARIA_CLIENT_TARGET_RMS,
        .max_gain = ARIA_CLIENT_MAX_GAIN, .fixed_gain = ARIA_CLIENT_FIXED_GAIN,
    };
    if (!aria_frontend_init(&frontend, &audio_config, send_audio_event, &transport)) {
        ESP_LOGE(TAG, "Invalid audio configuration or VAD allocation failed");
        goto cleanup;
    }
    frontend_started = true;
    xQueueReset(audio_queue);
    xEventGroupClearBits(events, CONNECTING);
    xEventGroupSetBits(events, STREAMING);
    ESP_LOGI(TAG, "Streaming 16 kHz mono PCM16; local VAD=%d AGC=%d; existing Apollo protocol",
             ARIA_CLIENT_VAD, ARIA_CLIENT_AGC);
    aria_stream_guard guard = {0};
    audio_packet packet;
    int64_t last_stats = esp_timer_get_time(), peak_processing_us = 0;
    for (;;) {
        bits = xEventGroupGetBits(events);
        if ((bits & (WIFI_READY | WS_READY | STREAMING)) !=
            (WIFI_READY | WS_READY | STREAMING) || (bits & AUDIO_FAULT)) {
            ESP_LOGW(TAG, "Session interrupted wifi=%d ws=%d capture_fault=%d",
                     !!(bits & WIFI_READY), !!(bits & WS_READY), !!(bits & AUDIO_FAULT));
            break;
        }
        if (xQueueReceive(audio_queue, &packet, pdMS_TO_TICKS(100)) != pdTRUE) continue;
        if (!aria_stream_accept(&guard, packet.sequence, packet.captured_us,
                                esp_timer_get_time(), ARIA_MAX_AUDIO_AGE_MS)) {
            ESP_LOGW(TAG, "Audio gap or stale packet; discarding this session");
            break;
        }
        int64_t now = esp_timer_get_time();
        int64_t before_send = transport.send_us;
        if (!aria_frontend_push(&frontend, packet.samples)) {
            ESP_LOGW(TAG, "Frontend/send failed; ending session");
            break;
        }
        int64_t elapsed = esp_timer_get_time() - now - (transport.send_us - before_send);
        if (elapsed > peak_processing_us) peak_processing_us = elapsed;
        if (now - last_stats >= 1000000) {
            ESP_LOGI(TAG, "frames=%" PRIu32 " sent=%" PRIu32 " turns=%" PRIu32
                     " rms=%.5f gain=%.2f noise=%.5f frontend_peak_us=%" PRId64
                     " free_heap=%" PRIu32 " stack_free=%u",
                     frontend.frames, frontend.sent_frames, frontend.turns,
                     frontend.rms, frontend.gain, frontend.noise_rms, peak_processing_us,
                     esp_get_free_heap_size(), (unsigned)uxTaskGetStackHighWaterMark(NULL));
            last_stats = now;
        }
    }
cleanup:
    xEventGroupClearBits(events, STREAMING | VAD_ACTIVE | TX_ACTIVITY | CONNECTING);
    /* Never replay old audio or invent an utterance_end after transport loss.
     * Existing Apollo finalizes open client-VAD audio on disconnect; this client
     * cannot cancel that partial request through the existing protocol. */
    if (client_started) esp_websocket_client_stop(client);
    esp_websocket_client_destroy(client);
    if (frontend_started) aria_frontend_destroy(&frontend);
    xQueueReset(audio_queue);
    /* Initialized frontend means the start frame succeeded and streaming began. */
    return frontend_started;
}

void app_main(void)
{
    if (!settings_valid()) {
        ESP_LOGE(TAG, "Set Wi-Fi credentials and Apollo's private IPv4 in main/settings.local.h");
        return;
    }
    events = xEventGroupCreate();
    audio_queue = xQueueCreate(ARIA_AUDIO_QUEUE_DEPTH, sizeof(audio_packet));
    diagnostics_queue = xQueueCreate(1, sizeof(capture_diagnostics));
    assert(events && audio_queue && diagnostics_queue);
    BaseType_t status_created = xTaskCreate(status_task, "aria_status", 3072, NULL, 2, NULL);
    assert(status_created == pdPASS);
    init_wifi();
    init_microphone();
    BaseType_t created = xTaskCreate(capture_task, "aria_capture", 4096, NULL, 6, NULL);
    assert(created == pdPASS);
    unsigned wifi_retry_ms = 0, apollo_retry_ms = 0;
    for (;;) {
        if (!(xEventGroupGetBits(events) & WIFI_READY)) {
            unsigned delay_ms = aria_retry_wait_ms(&wifi_retry_ms, false);
            xEventGroupSetBits(events, CONNECTING);
            if (esp_wifi_connect() != ESP_OK) xEventGroupClearBits(events, CONNECTING);
            ESP_LOGI(TAG, "Wi-Fi reconnect wait: %u ms", delay_ms);
            xEventGroupWaitBits(events, WIFI_READY, pdFALSE, pdTRUE,
                                pdMS_TO_TICKS(delay_ms));
            xEventGroupClearBits(events, CONNECTING);
        } else {
            wifi_retry_ms = 0;
            xEventGroupSetBits(events, CONNECTING);
            bool connected = run_session();
            xEventGroupClearBits(events, CONNECTING);
            unsigned delay_ms = aria_retry_wait_ms(&apollo_retry_ms, connected);
            ESP_LOGW(TAG, "Session ended; Apollo retry in %u ms", delay_ms);
            vTaskDelay(pdMS_TO_TICKS(delay_ms));
        }
    }
}
