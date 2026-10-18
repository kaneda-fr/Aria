# Vendored WebRTC VAD

Source: [wiseman/py-webrtcvad 2.0.10](https://github.com/wiseman/py-webrtcvad/tree/2.0.10),
commit `ba13ceea9dd3f3a068e869b2e94413205bd87f7d`. Only `cbits/webrtc/` and its
license are included; no Python wrapper, model, or runtime download is used.
The downloaded GitHub source archive SHA-256 was
`ff6f6417ba0a2e8c5dce63fe11605eafa270ec66d6d22ce9b98d412704af6044`.

Local portability change: `typedefs.h` identifies Xtensa as little-endian 32-bit,
using the generic C implementation. ESP-IDF supplies pthread-once initialization.
Host tests instrument upstream with AddressSanitizer; signed fixed-point arithmetic
uses `-fwrapv`, with UndefinedBehaviorSanitizer applied to Aria's own code separately.
Preserve the bundled license and upstream copyright notices when redistributing.
