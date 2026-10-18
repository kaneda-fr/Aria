# Client mute and confirmation — 2026-10-18

Maintainer decision: mute applies to the **current client**, not all Aria clients.
It pauses Apollo's processing of that client's incoming audio. The client continues
operating, capturing and transmitting; its connection and activity remain observable.
The monitor offers direct mute/unmute with five-minute automatic expiry. A spoken
request must first ask for confirmation, using a default or explicitly supplied
duration. Device tools remain log-only during testing. Mac capture/playback and LAN
microphone ingress remain off; use text and saved/synthetic PCM on Apollo.

## Behavior

- Say “Aria, mute yourself” or “Aria, mets-toi en sourdine” to request five minutes.
  Requests may specify seconds, minutes, hours, or supported combinations, for
  example “be quiet for ten minutes” / “tais-toi pendant dix minutes”.
- Aria names the current client and duration and asks for yes/no confirmation.
  Nothing mutes until an affirmative reply from the same session and attributed
  speaker. Recognition is attribution, not authentication; unknown speakers can
  confirm within the same client session.
- A pending confirmation expires after 30 seconds, is single-use, and is cleared
  by another request, cancellation, goodbye, conversation expiry/eviction or a manual
  UI change. Reconnect keeps the same client/session key; it does not renew the timer.
- Durations must resolve completely to 1–86,400 whole seconds. Ambiguous/unsupported
  durations ask for a new request; never silently substitute the default or truncate.
  The implementation is an explicit FR/EN imperative/number grammar, not unrestricted
  natural-language interpretation. Arbitrary synonyms, dates and compound French
  number forms are not claimed as supported. These boundaries need held-out evaluation.
- Muting discards incoming server audio and unfinished utterances. It does not turn
  off hardware capture, cancel already accepted requests, suppress their replies,
  or modify another client's state. A confirmation acknowledgment can still play.
  Unmute uses the monitor or timer; a muted audio stream cannot hear a spoken unmute.
- State is bounded and in-memory. A server restart clears mute; monotonic deadlines
  survive WebSocket reconnect and do not depend on browser clocks or an open page.

The shared processor handles the command before model/device tools. Confirmation
answers use the original request language for TTS. `ARIA.Session.MuteCommand` records
request/client IDs, status and duration; normal exact request/reply logs remain.
`app/llm/client_mute.py` owns parsing/pending state; `app/client_sessions.py` owns the
mute deadline; `/ws/asr` rejects stale audio; the monitor proxies only this control.

## Verification

Apollo: **287 targeted tests pass**, including cancellation, timeout, wrong speaker/client,
invalid duration, already-muted truthful reply, same-origin monitor guard, strict boolean
payloads, reconnect, monotonic expiry and mute during ASR. Local endpoint/monitor tests
run without opening any audio device; seven context tests and source coverage checks pass.

Initial run `20261018T132742Z-mute-confirmed-3e73e1` exposed French ASR's
“Je le confirme” variant; it fell through to the model. A confirmation regression
was added. Final immutable app run `20261018T133802Z-mute-verified-bbb07c` passes
all four saved voice steps (request and confirmation in each language): first audio
ready after saved PCM ends is EN **1.74/1.40 s**, FR **1.78/1.43 s**. This is generated
WAV readiness, not physical playback latency. No model/device tools are needed by
these commands. Raw transcripts and nearest-speaker scores remain in private logs.

A real five-minute mute expired at **300.6 s** (one-second observer polling), without
browser intervention. Short FR/EN durations, cancellation and reconnect were checked
on Apollo. A saved utterance sent while muted produced zero admitted/transcribed
requests. The other client stayed unmuted. Browser mute/unmute, per-client filtering
and request-correlated scores passed with synthetic names; no browser console errors.
Physical Mac/ESP32 reconnect tests remain deferred under the no-hardware-audio instruction.

The 32-case FR/EN conversation/control/lookup repeat has no functional flags, and
four queue cases pass with 3.81–4.22 s submitted-to-audio latency (median 4.01 s).
A final post-restart queue repeat also passes: 3.86–4.80 s, median 4.32 s. That
variation is retained rather than claiming exact latency equivalence. The maintained
queue harness now activates Aria before testing follow-ups.
[Broader lookup/science regressions](freeform-lookup-2026-10-18.md) remain separate;
this feature pass does not establish general conversation quality.

Network/lifecycle/source audits pass: initial run 62 accepted/62 terminal requests
plus four overload rejections, 31 observed loopback destinations, no external/DNS
attempts; final voice replay plus queue repeat 28/28 requests, four overload rejections,
and 18 loopback destinations. Runtime app
hashes match their archived manifests. The final runtime remains network-isolated
behind a Unix socket; only the monitor's LAN port is exposed. The monitor and host OS
are outside the traced inference process tree.
