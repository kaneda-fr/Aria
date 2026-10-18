#include "stream_core.h"
#include "settings.h"
#include <string.h>
#include <stdio.h>

_Static_assert(ARIA_RETRY_INITIAL_MS > 0 && ARIA_RETRY_INITIAL_MS <= ARIA_RETRY_MAX_MS,
               "Invalid reconnect backoff bounds");

unsigned aria_retry_wait_ms(unsigned *next_ms, bool reset)
{
    unsigned delay = reset || !*next_ms ? ARIA_RETRY_INITIAL_MS : *next_ms;
    if (delay > ARIA_RETRY_MAX_MS) delay = ARIA_RETRY_MAX_MS;
    *next_ms = delay > ARIA_RETRY_MAX_MS / 2 ? ARIA_RETRY_MAX_MS : delay * 2;
    return delay;
}

const char aria_start_message[] =
    "{\"type\":\"start\",\"sample_rate\":16000,\"channels\":1,"
    "\"format\":\"pcm_s16le\",\"chunk_ms\":20,"
#if ARIA_CLIENT_VAD
    "\"client_vad\":true}"
#else
    "\"client_vad\":false}"
#endif
    ;

size_t aria_identified_start(char *output, size_t capacity, const char *client_id)
{
    if (!client_id || !output || !capacity) return 0;
    size_t length = strlen(client_id);
    if (!length || length > 64) return 0;
    for (size_t i = 0; i < length; ++i) {
        char c = client_id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              (i > 0 && (c == '-' || c == '_' || c == '.')))) return 0;
    }
    int count = snprintf(output, capacity, "%.*s,\"client_id\":\"%s\"}",
                         (int)strlen(aria_start_message) - 1, aria_start_message, client_id);
    return count > 0 && (size_t)count < capacity ? (size_t)count : 0;
}

bool aria_stream_accept(aria_stream_guard *guard, uint32_t sequence,
                        int64_t captured_us, int64_t now_us, int max_age_ms)
{
    if (captured_us > now_us || now_us - captured_us > (int64_t)max_age_ms * 1000 ||
        (guard->started && sequence != guard->next_sequence)) {
        return false;
    }
    guard->started = true;
    guard->next_sequence = sequence + 1;
    return true;
}

bool aria_private_ipv4(uint32_t address)
{
    return (address >> 24) == 10 || (address >> 20) == 0xac1 ||
           (address >> 16) == 0xc0a8;
}

const char *aria_text_push(aria_text_buffer *buffer, unsigned opcode, bool fin,
                           int payload_length, int offset, const char *data, int length)
{
    /* Control frames may interrupt fragmented messages. */
    if (opcode >= 8) return NULL;
    if (opcode == 1 && offset == 0) {
        *buffer = (aria_text_buffer){.active = true};
    } else if (opcode != 0 && opcode != 1) {
        buffer->active = false;
        return NULL;
    }
    if (!buffer->active) return NULL;
    if (payload_length < 0 || offset < 0 || length < 0 ||
        (length > 0 && data == NULL)) goto invalid;
    if (offset == 0) {
        if (buffer->frame_offset != 0) goto invalid;
        buffer->frame_length = (size_t)payload_length;
    }
    if ((size_t)payload_length != buffer->frame_length ||
        (size_t)offset != buffer->frame_offset || offset > payload_length ||
        length > payload_length - offset ||
        (size_t)length >= sizeof(buffer->text) - buffer->used) goto invalid;
    if (length) memcpy(buffer->text + buffer->used, data, (size_t)length);
    buffer->used += (size_t)length;
    buffer->frame_offset += (size_t)length;
    if (buffer->frame_offset == buffer->frame_length) {
        buffer->frame_offset = 0;
        if (fin) {
            buffer->text[buffer->used] = '\0';
            buffer->active = false;
            return buffer->text;
        }
    }
    return NULL;
invalid:
    buffer->active = false;
    return NULL;
}
