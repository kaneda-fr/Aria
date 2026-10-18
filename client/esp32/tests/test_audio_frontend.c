#include "audio_frontend.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct { int starts, ends, frames; int16_t first, last; bool fail; } trace;
static bool emit(void *context, aria_audio_event event, const int16_t *pcm, size_t count)
{
    trace *t = context;
    if (t->fail) return false;
    if (event == ARIA_TURN_START) ++t->starts;
    if (event == ARIA_TURN_END) ++t->ends;
    if (event == ARIA_AUDIO) {
        assert(count == 320);
        if (!t->frames) t->first = pcm[0];
        t->last = pcm[319];
        ++t->frames;
    }
    return true;
}
static aria_frontend frontend;
static aria_frontend_config config = {
    .vad_enabled = true, .agc_enabled = false, .vad_mode = 2,
    .start_frames = 3, .silence_ms = 800, .preroll_frames = 10,
    .noise_alpha = .05f, .target_rms = .05f, .max_gain = 8.f,
};
static void frame(int amplitude, bool speech)
{
    int16_t pcm[320];
    for (int i = 0; i < 320; ++i) pcm[i] = amplitude;
    assert(aria_frontend_push_classified(&frontend, pcm, speech));
}
int main(int argc, char **argv)
{
    trace t = {0};
    assert(aria_frontend_init(&frontend, &config, emit, &t));
    for (int i = 0; i < 12; ++i) frame(i + 1, false);
    assert(t.starts == 0 && t.frames == 0);
    frame(101, true); frame(102, true);
    assert(t.starts == 0);
    frame(103, true);
    assert(t.starts == 1 && t.frames == 10 && t.first == 6 && t.last == 103);
    for (int i = 0; i < 39; ++i) frame(0, false);
    assert(t.ends == 0);
    frame(0, false);
    assert(t.ends == 1);
    frame(0, false);
    assert(t.ends == 1);
    frame(100, true); frame(100, true); frame(100, true);
    for (int i = 0; i < 22; ++i) frame(0, false);
    frame(100, true);
    assert(t.starts == 2 && t.ends == 1); /* 440 ms pause stays in turn. */
    aria_frontend_destroy(&frontend);

    config.agc_enabled = true;
    t = (trace){0};
    assert(aria_frontend_init(&frontend, &config, emit, &t));
    for (int i = 0; i < 100; ++i) frame(10, false);
    assert(frontend.gain == 1.f);
    for (int i = 0; i < 30; ++i) frame(100, true);
    assert(frontend.gain > 4.f && frontend.gain <= 8.f && t.last > 400);
    float gain = frontend.gain;
    for (int i = 0; i < 20; ++i) frame(10, false);
    assert(frontend.gain == gain); /* Noise never drives gain upward. */
    frame(32767, true);
    assert(t.last <= 32113 && t.last > 0);
    frame(-32768, true);
    assert(t.last >= -32113 && t.last < 0);
    for (int amplitude = -32768; amplitude <= 32767; amplitude += 127) {
        frame(amplitude, true);
        assert(isfinite(frontend.gain) && frontend.gain > 0 && frontend.gain <= 8.f);
        assert(t.last >= -32113 && t.last <= 32113);
    }
    t.fail = true;
    int16_t zero[320] = {0};
    assert(!aria_frontend_push_classified(&frontend, zero, true));
    assert(frontend.failed);
    aria_frontend_destroy(&frontend);

    config.min_rms = .01f;
    t = (trace){0};
    assert(aria_frontend_init(&frontend, &config, emit, &t));
    for (int i = 0; i < 50; ++i) frame(100, true);
    assert(!t.starts && frontend.gain == 1.f); /* Gate BEFORE gain. */
    aria_frontend_destroy(&frontend);
    config.min_rms = 0;
    config.snr_db = 8.f;
    assert(aria_frontend_init(&frontend, &config, emit, &t));
    for (int i = 0; i < 50; ++i) frame(100, false);
    for (int i = 0; i < 20; ++i) frame(110, true);
    assert(!t.starts);
    aria_frontend_destroy(&frontend);

    config.snr_db = 0;
    config.preroll_frames = 0;
    config.fixed_gain = 2;
    config.vad_enabled = false;
    t = (trace){0};
    assert(aria_frontend_init(&frontend, &config, emit, &t));
    frame(100, false);
    assert(t.frames == 1 && t.last == 200 && t.starts == 0);
    for (int i = 0; i < 50; ++i) assert(aria_frontend_push(&frontend, zero));
    aria_frontend_destroy(&frontend);
    config.silence_ms = 199;
    assert(!aria_frontend_init(&frontend, &config, emit, &t));
    const int endpoints[] = {199, 200, 201, 2999, 3000, 3001};
    for (size_t i = 0; i < sizeof(endpoints) / sizeof(*endpoints); ++i) {
        config.silence_ms = endpoints[i];
        bool expected = endpoints[i] >= 200 && endpoints[i] <= 3000;
        assert(aria_frontend_init(&frontend, &config, emit, &t) == expected);
        if (expected) aria_frontend_destroy(&frontend);
    }
    if (argc == 2) {
        config.silence_ms = 800;
        config.preroll_frames = 10;
        config.fixed_gain = 0;
        config.vad_enabled = true;
        t = (trace){0};
        assert(aria_frontend_init(&frontend, &config, emit, &t));
        FILE *fixture = fopen(argv[1], "rb");
        assert(fixture);
        int16_t input[160], pcm[320];
        while (fread(input, sizeof(int16_t), 160, fixture) == 160) {
            for (int i = 0; i < 160; ++i) pcm[2*i] = pcm[2*i+1] = input[i];
            assert(aria_frontend_push(&frontend, pcm));
        }
        fclose(fixture);
        for (int i = 0; i < 100; ++i) assert(aria_frontend_push(&frontend, zero));
        assert(t.starts > 0 && t.ends == t.starts && t.frames > 0);
        aria_frontend_destroy(&frontend);
    }
    puts("audio frontend checks passed");
}
