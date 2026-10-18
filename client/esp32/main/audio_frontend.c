#include "audio_frontend.h"
#include <math.h>
#include <string.h>
#include "webrtc/common_audio/vad/include/webrtc_vad.h"

static float minimum(float a, float b) { return a < b ? a : b; }
static float maximum(float a, float b) { return a > b ? a : b; }

static bool output(aria_frontend *f, aria_audio_event event, const int16_t *pcm)
{
    if (!f->emit(f->context, event, pcm, pcm ? ARIA_FRAME_SAMPLES : 0)) {
        f->failed = true;
        return false;
    }
    if (event == ARIA_AUDIO) ++f->sent_frames;
    return true;
}

bool aria_frontend_init(aria_frontend *f, const aria_frontend_config *c,
                        aria_audio_emit emit, void *context)
{
    if (!c || !emit || c->vad_mode < 0 || c->vad_mode > 3 ||
        c->start_frames < 1 || c->start_frames > 10 ||
        c->silence_ms < 200 || c->silence_ms > 3000 ||
        c->preroll_frames < 0 || c->preroll_frames > ARIA_PREROLL_CAPACITY ||
        !isfinite(c->target_rms) || c->target_rms <= 0 || c->target_rms > 1 ||
        !isfinite(c->max_gain) || c->max_gain < 1 || c->max_gain > 50 ||
        !isfinite(c->fixed_gain) || c->fixed_gain < 0 || c->fixed_gain > 50 ||
        !isfinite(c->min_rms) || c->min_rms < 0 || c->min_rms > 1 ||
        !isfinite(c->snr_db) || c->snr_db < 0 || c->snr_db > 60 ||
        !isfinite(c->noise_alpha) || c->noise_alpha <= 0 || c->noise_alpha > 1) return false;
    *f = (aria_frontend){.cfg = *c, .emit = emit, .context = context,
                         .gain = c->fixed_gain > 0 ? c->fixed_gain : 1.f,
                         .snr_multiplier = powf(10.f, c->snr_db / 20.f)};
    VadInst *detector = NULL;
    if (WebRtcVad_Create(&detector) != 0) return false;
    f->detector = detector;
    if (WebRtcVad_Init(detector) != 0 || WebRtcVad_set_mode(detector, c->vad_mode) != 0) {
        aria_frontend_destroy(f);
        return false;
    }
    return true;
}

void aria_frontend_destroy(aria_frontend *f)
{
    if (f->detector) WebRtcVad_Free(f->detector);
    f->detector = NULL;
}

bool aria_frontend_push(aria_frontend *f, const int16_t *pcm)
{
    int speech = WebRtcVad_Process(f->detector, 16000, pcm, ARIA_FRAME_SAMPLES);
    if (speech < 0) { f->failed = true; return false; }
    return aria_frontend_push_classified(f, pcm, speech != 0);
}

bool aria_frontend_push_classified(aria_frontend *f, const int16_t *pcm, bool speech)
{
    if (f->failed) return false;
    ++f->frames;
    float energy = 0;
    int32_t peak = 0;
    for (int i = 0; i < ARIA_FRAME_SAMPLES; ++i) {
        float sample = pcm[i] / 32768.f;
        energy += sample * sample;
        int32_t magnitude = pcm[i] < 0 ? -(int32_t)pcm[i] : pcm[i];
        if (magnitude > peak) peak = magnitude;
    }
    f->rms = sqrtf(energy / ARIA_FRAME_SAMPLES);
    /* Learn background only from detector-negative idle frames. Classify raw
     * audio: AGC must never lift background noise across the speech gate. */
    if (!f->active && !speech) {
        f->noise_rms = f->noise_rms > 0
            ? f->noise_rms + f->cfg.noise_alpha * (f->rms - f->noise_rms) : f->rms;
    }
    speech = speech && f->rms >= f->cfg.min_rms &&
             (f->cfg.snr_db == 0 || f->rms >= maximum(f->noise_rms, 1.e-6f) * f->snr_multiplier);
    if (speech) ++f->speech_frames;
    float next_gain = f->gain;
    if (f->cfg.fixed_gain > 0) next_gain = f->cfg.fixed_gain;
    else if (f->cfg.agc_enabled && speech && f->rms > 1.e-6f) {
        float desired = minimum(f->cfg.max_gain, maximum(1.f, f->cfg.target_rms / f->rms));
        float alpha = desired > f->gain ? .25f : .08f;
        next_gain += alpha * (desired - f->gain);
    }
    float peak_gain = peak ? 32113.f / peak : 50.f;
    /* Peak limiting can attenuate loud input below unity. Apply the ceiling to
     * the WHOLE frame, including the start of a rising/falling gain ramp. */
    float start_gain = minimum(f->gain, peak_gain);
    next_gain = minimum(next_gain, peak_gain);
    int16_t processed[ARIA_FRAME_SAMPLES];
    for (int i = 0; i < ARIA_FRAME_SAMPLES; ++i) {
        float gain = start_gain + (next_gain - start_gain) * (i + 1) / ARIA_FRAME_SAMPLES;
        float sample = pcm[i] * gain;
        processed[i] = (int16_t)maximum(-32113.f, minimum(32113.f, sample));
    }
    f->gain = next_gain;
    if (!f->cfg.vad_enabled) return output(f, ARIA_AUDIO, processed);

    if (!f->active) {
        int capacity = f->cfg.preroll_frames;
        if (capacity > 0) {
            memcpy(f->preroll[f->pre_next], processed, sizeof(processed));
            f->pre_next = (f->pre_next + 1) % capacity;
            if (f->pre_count < capacity) ++f->pre_count;
        }
        f->speech_run = speech ? f->speech_run + 1 : 0;
        if (f->speech_run < f->cfg.start_frames) return true;
        f->active = true;
        f->speech_run = f->silence_ms = 0;
        ++f->turns;
        if (!output(f, ARIA_TURN_START, NULL)) return false;
        if (!capacity) return output(f, ARIA_AUDIO, processed);
        int first = (f->pre_next + capacity - f->pre_count) % capacity;
        for (int i = 0; i < f->pre_count; ++i)
            if (!output(f, ARIA_AUDIO, f->preroll[(first + i) % capacity])) return false;
        f->pre_count = f->pre_next = 0;
        return true;
    }
    if (!output(f, ARIA_AUDIO, processed)) return false;
    f->silence_ms = speech ? 0 : f->silence_ms + 20;
    if (f->silence_ms >= f->cfg.silence_ms) {
        if (!output(f, ARIA_TURN_END, NULL)) return false;
        f->active = false;
        f->silence_ms = 0;
    }
    return true;
}
