#pragma once

/* All settings are compile-time. Optionally keep private values in the ignored
 * settings.local.h beside this file. No provisioning UI or runtime config yet. */
#if __has_include("settings.local.h")
#include "settings.local.h"
#endif
#ifndef ARIA_WIFI_SSID
#define ARIA_WIFI_SSID ""
#endif
#ifndef ARIA_WIFI_PASSWORD
#define ARIA_WIFI_PASSWORD ""
#endif
#ifndef ARIA_APOLLO_IPV4
#define ARIA_APOLLO_IPV4 ""
#endif
#ifndef ARIA_APOLLO_PORT
#define ARIA_APOLLO_PORT 8000
#endif
#ifndef ARIA_HOSTNAME_PREFIX
#define ARIA_HOSTNAME_PREFIX "aria-wisp"
#endif

/* Seeed XIAO ESP32-S3 Sense expansion microphone (PDM, not standard I2S). */
#define ARIA_MIC_CLK_GPIO 42
#define ARIA_MIC_DATA_GPIO 41
#define ARIA_LED_GPIO 21
#define ARIA_LED_ACTIVE_LOW 1
#define ARIA_SAMPLE_RATE 16000
#define ARIA_CHUNK_MS 20
#define ARIA_SAMPLES_PER_CHUNK (ARIA_SAMPLE_RATE * ARIA_CHUNK_MS / 1000)
#define ARIA_AUDIO_QUEUE_DEPTH 8
#define ARIA_MAX_AUDIO_AGE_MS 200
#define ARIA_SEND_TIMEOUT_MS 100
#define ARIA_RETRY_INITIAL_MS 3000
#define ARIA_RETRY_MAX_MS 60000
#define ARIA_DISCONNECTED_LED_PERIOD_MS 200
#define ARIA_CONNECT_TIMEOUT_MS 5000

/* Compile-time microphone processing, matching the Mac client's feature set. */
#define ARIA_CLIENT_VAD 1
#define ARIA_CLIENT_VAD_MODE 2
#define ARIA_CLIENT_VAD_START_FRAMES 3
#define ARIA_CLIENT_VAD_SILENCE_MS 800
#define ARIA_CLIENT_VAD_PRE_MS 200
#define ARIA_CLIENT_VAD_MIN_RMS 0.0f
#define ARIA_CLIENT_VAD_SNR_DB 0.0f
#define ARIA_CLIENT_VAD_NOISE_ALPHA 0.05f
#define ARIA_CLIENT_AGC 1
#define ARIA_CLIENT_TARGET_RMS 0.05f
#define ARIA_CLIENT_MAX_GAIN 8.0f
#define ARIA_CLIENT_FIXED_GAIN 0.0f /* 0 selects automatic/no gain; >0 fixes it. */
