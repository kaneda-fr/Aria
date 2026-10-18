#include "stream_core.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    unsigned retry = 0;
    const unsigned expected_waits[] = {3000, 6000, 12000, 24000, 48000, 60000, 60000};
    for (size_t i = 0; i < sizeof(expected_waits) / sizeof(expected_waits[0]); ++i)
        assert(aria_retry_wait_ms(&retry, false) == expected_waits[i]);
    for (int i = 0; i < 1000; ++i) assert(aria_retry_wait_ms(&retry, false) == 60000);
    assert(aria_retry_wait_ms(&retry, true) == 3000);
    assert(aria_retry_wait_ms(&retry, false) == 6000);
    unsigned independent_retry = 0;
    assert(aria_retry_wait_ms(&independent_retry, false) == 3000);
    assert(aria_retry_wait_ms(&retry, false) == 12000);
    retry = UINT32_MAX;
    assert(aria_retry_wait_ms(&retry, false) == 60000);
    char named_start[256];
    assert(aria_identified_start(named_start, sizeof(named_start), "aria-wisp-001122334455") > 0);
    assert(!aria_identified_start(named_start, 4, "valid-name"));
    assert(!aria_identified_start(named_start, sizeof(named_start), "bad\"name"));
    assert(!aria_identified_start(named_start, sizeof(named_start), ""));
    if (argc == 2 && strcmp(argv[1], "--start") == 0) {
        assert(aria_identified_start(named_start, sizeof(named_start), "aria-wisp-001122334455") > 0);
        puts(named_start);
        return 0;
    }
    aria_stream_guard guard = {0};
    assert(aria_stream_accept(&guard, 42, 0, 200000, 200));
    assert(aria_stream_accept(&guard, 43, 210000, 220000, 200));
    assert(!aria_stream_accept(&guard, 45, 230000, 240000, 200));
    assert(!aria_stream_accept(&guard, 43, 230000, 240000, 200));
    assert(!aria_stream_accept(&guard, 44, 230000, 430001, 200));
    assert(!aria_stream_accept(&guard, 44, 250000, 240000, 200));
    guard = (aria_stream_guard){0};
    assert(aria_stream_accept(&guard, UINT32_MAX, 0, 0, 200));
    assert(aria_stream_accept(&guard, 0, 20000, 20000, 200));

    assert(aria_private_ipv4(0x0a000001));
    assert(aria_private_ipv4(0xac100001));
    assert(aria_private_ipv4(0xac1ffffe));
    assert(aria_private_ipv4(0xc0a80101));
    assert(!aria_private_ipv4(0xac0fffff));
    assert(!aria_private_ipv4(0xac200001));
    assert(!aria_private_ipv4(0x08080808));
    assert(!aria_private_ipv4(0x7f000001));

    aria_text_buffer buffer = {0};
    /* One text frame split by the IDF receive buffer. */
    assert(!aria_text_push(&buffer, 1, true, 5, 0, "he", 2));
    const char *message = aria_text_push(&buffer, 1, true, 5, 2, "llo", 3);
    assert(message && strcmp(message, "hello") == 0);
    /* WebSocket fragmentation plus an intervening ping. */
    assert(!aria_text_push(&buffer, 1, false, 2, 0, "he", 2));
    assert(!aria_text_push(&buffer, 9, true, 1, 0, "!", 1));
    assert(!aria_text_push(&buffer, 0, true, 3, 0, "l", 1));
    message = aria_text_push(&buffer, 0, true, 3, 1, "lo", 2);
    assert(message && strcmp(message, "hello") == 0);
    /* No orphan continuation, reordered event, or changed frame length. */
    assert(!aria_text_push(&buffer, 0, true, 1, 0, "x", 1));
    assert(!aria_text_push(&buffer, 1, true, 5, 0, "he", 2));
    assert(!aria_text_push(&buffer, 1, true, 5, 3, "lo", 2));
    assert(!aria_text_push(&buffer, 1, true, 5, 0, "he", 2));
    assert(!aria_text_push(&buffer, 1, true, 6, 2, "llo!", 4));
    assert(!aria_text_push(&buffer, 1, true, -1, 0, "", 0));
    assert(!aria_text_push(&buffer, 1, true, 1, 0, NULL, 1));
    /* Exactly cap-1 fits; cap bytes and fragmented overflow are rejected. */
    char large[ARIA_STATUS_CAPACITY];
    memset(large, 'x', sizeof(large));
    message = aria_text_push(&buffer, 1, true, sizeof(large) - 1, 0,
                             large, sizeof(large) - 1);
    assert(message && strlen(message) == sizeof(large) - 1);
    assert(!aria_text_push(&buffer, 1, true, sizeof(large), 0, large, sizeof(large)));
    assert(!aria_text_push(&buffer, 1, false, sizeof(large) - 1, 0,
                           large, sizeof(large) - 1));
    assert(!aria_text_push(&buffer, 0, true, 1, 0, "x", 1));
    message = aria_text_push(&buffer, 1, true, 2, 0, "ok", 2);
    assert(message && strcmp(message, "ok") == 0);
    puts("stream core checks passed");
    return 0;
}
