# ARIA tuning guide (noise, VAD, ASR)

This doc is a practical checklist for reducing false triggers and noisy transcripts like "Yeah", "Oh", or "Okay" when nobody is speaking.

ARIA can run VAD either on the client (recommended for bandwidth + better control) or on the server.

## Recommended baseline (client VAD)

Use client-side VAD and keep AGC off unless the mic is extremely quiet:

```zsh
ARIA_CLIENT_VAD=1 \
ARIA_CLIENT_VAD_AGGR=3 \
ARIA_CLIENT_VAD_START_FRAMES=5 \
ARIA_CLIENT_VAD_SNR_DB=10 \
ARIA_CLIENT_VAD_SILENCE_MS=350 \
ARIA_CLIENT_DEBUG=1 \
python -m client.macos.mic_stream 192.168.100.100:8000
```

Notes:
- `ARIA_CLIENT_VAD_START_FRAMES` adds start hysteresis (reduces click/pop triggers).
- `ARIA_CLIENT_VAD_SNR_DB` compares current frame energy to an adaptive noise floor.
- When `ARIA_CLIENT_VAD=1`, the client defaults `ARIA_CLIENT_AGC=0` to avoid boosting noise.

If it still triggers in a noisy environment:
- Increase `ARIA_CLIENT_VAD_SNR_DB` (e.g. 12–16)
- Increase `ARIA_CLIENT_VAD_START_FRAMES` (e.g. 6–8)
- Optionally set `ARIA_CLIENT_VAD_MIN_RMS=0.01`

If it misses quiet speech:
- Decrease `ARIA_CLIENT_VAD_SNR_DB` (e.g. 6–8)
- Decrease `ARIA_CLIENT_VAD_START_FRAMES` (e.g. 3–4)
- If speech is truly too quiet, consider enabling AGC explicitly: `ARIA_CLIENT_AGC=1`

## Server VAD knobs (only when client VAD is off)

```zsh
ARIA_VAD_AGGR=3 \
ARIA_VAD_START_FRAMES=4 \
ARIA_VAD_SNR_DB=8 \
ARIA_VAD_SILENCE_MS=700 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Avoiding Parakeet noise transcripts (ASR gating)

Even with good VAD, some noise-like utterances can slip through. ARIA now gates ASR to skip Parakeet when an utterance is too short/quiet or has too little speech.

Key env vars:
- `ARIA_ASR_MIN_SECONDS` (default `0.30`)
- `ARIA_ASR_MIN_RMS` (default `0.004`)
- `ARIA_ASR_MIN_SPEECH_RATIO` (default `0.18`)
- `ARIA_ASR_DROP_FILLERS` (default `1`) drops single-word fillers when the utterance looks like noise

Suggested stricter preset if you still see "Yeah/Oh/Okay" from noise:

```zsh
ARIA_ASR_MIN_SPEECH_RATIO=0.30 \
ARIA_ASR_MIN_RMS=0.006
```

## What to look at in logs

For each final utterance, ARIA logs these fields:
- `seconds`: audio duration of the utterance
- `rms`: RMS energy (0..1)
- `speech_ratio`: fraction of 20ms frames that WebRTC VAD considers speech
- `asr_ran`: whether Parakeet ran for that utterance

If you share one problematic `ARIA.Utterance.Final` log line, it’s usually enough to recommend exact threshold values.
