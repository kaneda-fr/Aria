#!/bin/bash
# Generate English and French TTS samples for Aria using aria_tts_sonos_record.py

set -e

python aria_tts_sonos_record.py \
  --voices en=voices/en_US-amy-low.onnx,fr=voices/fr_FR-siwis-low.onnx \
  --sonos-ip 192.168.100.222 \
  --out samples \
  --normalize \
  --texts 'en=Hello, my name is ARIA.;fr=Bonjour, je m appelle ARIA.'