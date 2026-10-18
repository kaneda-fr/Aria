"""Capture-only factual transfer checks. Meaning requires manual review.

Criteria are evaluation data and never sent to the model. No remote judge,
microphone, playback, or real action is used.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from scripts.e2e.conversation_bench import measure, summarize
from scripts.e2e.quality_probe import fixtures

CRITERIA = {
    'sky': 'Air scatters shorter/blue wavelengths more strongly; not selective absorption or Earth emission.',
    'sunset': 'Longer atmospheric path removes more blue from direct sunlight, leaving red/orange dominant.',
    'ice': 'Ice is less dense than liquid water.',
    'seasons': 'Axial tilt changes sunlight angle/day length during the orbit; not distance or daily rotation alone.',
    'moon': 'Reflects sunlight; does not generate the visible light itself.',
    'vacuum': 'Sound needs a material medium and cannot propagate in vacuum.',
    'plant': 'Light, water and carbon dioxide; do not claim light alone is sufficient.',
    'uncertain': 'Admit the bicycle colour is unknown; no invented personal fact.',
    'rain': 'Water vapour condenses into droplets that become heavy enough to fall.',
    'metal': 'Metal conducts heat from the hand faster; equal room temperature does not mean metal is colder.',
    'magnet': 'Opposite magnetic poles attract, like poles repel.',
    'day': 'Earth rotates; the side facing the Sun has daylight.',
}
TRANSFER = [
    ('fr-rain', 'Comment se forme la pluie ? Une phrase simple.'),
    ('en-rain', 'How does rain form? One simple sentence.'),
    ('fr-metal', 'Pourquoi le métal paraît-il plus froid que le bois dans la même pièce ? Une phrase simple.'),
    ('en-metal', 'Why does metal feel colder than wood in the same room? One simple sentence.'),
    ('fr-magnet', 'Deux pôles nord d’aimants s’attirent-ils ? Une phrase simple.'),
    ('en-magnet', 'Do two north poles of magnets attract each other? One simple sentence.'),
    ('fr-day', 'Pourquoi fait-il jour puis nuit ? Une phrase simple.'),
    ('en-day', 'Why do we have day and night? One simple sentence.'),
]


async def run(server_run, output):
    socket = str(server_run / 'aria.sock')
    prefix = f'science-{time.time_ns()}'
    rows = []
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://localhost', timeout=300, trust_env=False) as client:
        cfg = (await client.get('/_live/config')).json()
        assert cfg['tools_mode'] == 'log_only' and cfg['warmup_ready']
        # Language groups keep the sky/sunset context adjacent. Other topics use
        # independent sessions. No reference answer enters model context.
        cases = fixtures + TRANSFER
        for cid, text in sorted(cases, key=lambda item: (item[0][:2], cases.index(item))):
            lang, topic = cid.split('-', 1)
            case = dict(id=cid, language=lang, kind='science', text=text,
                        session=cid.replace('sunset', 'sky'), answer=None,
                        expected_commands=[], allow_search=False)
            row = await measure(client, socket, case, prefix)
            row['review_criterion'] = CRITERIA[topic]
            rows.append(row)
            output.write_text(json.dumps(dict(config=cfg, cases=rows,
                                             summary=summarize(rows)), ensure_ascii=False, indent=2))
            print(json.dumps(dict(id=cid, reply=row['result'].get('response'),
                                  flags=row['flags'], latency=row['latency']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.server_run, args.output))
