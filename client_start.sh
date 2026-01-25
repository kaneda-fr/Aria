#!/usr/bin/env bash
set -euo pipefail

# Client-side VAD tuned to reduce false triggers in quiet/noisy rooms.
# Adjust thresholds if you miss quiet speech.
ARIA_CLIENT_VAD=1 \
ARIA_CLIENT_DEBUG=1 \
ARIA_CLIENT_AGC=0 \
ARIA_CLIENT_VAD_AGGR=3 \
ARIA_CLIENT_VAD_START_FRAMES=3 \
ARIA_CLIENT_VAD_SNR_DB=8 \
ARIA_CLIENT_VAD_MIN_RMS=0.005 \
ARIA_CLIENT_VAD_SILENCE_MS=400 \
python client/macos/mic_stream.py ws://192.168.100.100:8000/ws/asr
