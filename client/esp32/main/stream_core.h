#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Wire format is deliberately fixed to the server's supported contract. */
extern const char aria_start_message[];
/* Stable client name; returns 0 for invalid names or insufficient space. */
size_t aria_identified_start(char *output, size_t capacity, const char *client_id);

typedef struct {
    bool started;
    uint32_t next_sequence;
} aria_stream_guard;

/* A false result requires a NEW connection, never splicing audio across a gap. */
bool aria_stream_accept(aria_stream_guard *guard, uint32_t sequence,
                        int64_t captured_us, int64_t now_us, int max_age_ms);
bool aria_private_ipv4(uint32_t host_order_address);

/* Independent caller-owned backoff state. Reset after a successful connection.
 * Returns the wait for this failure, then advances the next wait up to the cap. */
unsigned aria_retry_wait_ms(unsigned *next_ms, bool reset);

#define ARIA_STATUS_CAPACITY 1024
typedef struct {
    char text[ARIA_STATUS_CAPACITY];
    size_t used;
    size_t frame_offset;
    size_t frame_length;
    bool active;
} aria_text_buffer;

/* Handles both WebSocket continuation frames and IDF buffer-sized events.
 * Returns a NUL-terminated message only on its final event. Other opcodes,
 * malformed sequences, and oversized text are ignored without allocation. */
const char *aria_text_push(aria_text_buffer *buffer, unsigned opcode, bool fin,
                           int payload_length, int offset, const char *data, int length);
