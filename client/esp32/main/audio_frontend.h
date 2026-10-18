#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ARIA_FRAME_SAMPLES 320
#define ARIA_PREROLL_CAPACITY 25
typedef enum { ARIA_TURN_START, ARIA_AUDIO, ARIA_TURN_END } aria_audio_event;
typedef bool (*aria_audio_emit)(void *, aria_audio_event, const int16_t *, size_t);
typedef struct {
    bool vad_enabled, agc_enabled;
    int vad_mode, start_frames, silence_ms, preroll_frames;
    float min_rms, snr_db, noise_alpha, target_rms, max_gain, fixed_gain;
} aria_frontend_config;
typedef struct {
    aria_frontend_config cfg;
    void *detector;
    aria_audio_emit emit;
    void *context;
    bool active, failed;
    int speech_run, silence_ms, pre_count, pre_next;
    float gain, rms, noise_rms, snr_multiplier;
    uint32_t frames, speech_frames, sent_frames, turns;
    int16_t preroll[ARIA_PREROLL_CAPACITY][ARIA_FRAME_SAMPLES];
} aria_frontend;

bool aria_frontend_init(aria_frontend *, const aria_frontend_config *, aria_audio_emit, void *);
void aria_frontend_destroy(aria_frontend *);
bool aria_frontend_push(aria_frontend *, const int16_t *);
/* Separate classifier result from segmentation for deterministic boundary tests. */
bool aria_frontend_push_classified(aria_frontend *, const int16_t *, bool speech);
