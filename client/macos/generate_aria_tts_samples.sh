#!/bin/bash
# Generate English and French TTS samples for Aria using aria_tts_sonos_record.py

set -e

SONOS_IP="${1:-${ARIA_SONOS_IP:-}}"
if [[ -z "$SONOS_IP" ]]; then
  echo "Usage: ARIA_SONOS_IP=<sonos_ip> $0 [sonos_ip]" >&2
  exit 2
fi

python aria_tts_sonos_record.py \
  --voices en=voices/en_US-amy-low.onnx,fr=voices/fr_FR-siwis-low.onnx \
  --sonos-ip "$SONOS_IP" \
  --out samples \
  --normalize \
  --texts 'en=Hello, my name is ARIA.;fr=Bonjour, je m appelle ARIA.'
