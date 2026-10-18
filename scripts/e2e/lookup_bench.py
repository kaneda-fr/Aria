"""Private Apollo free-form lookup + existing conversation regression evidence."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import uuid

import httpx

from scripts.e2e.conversation_bench import build_cases, measure


async def run(args):
    socket = str(args.server_run / 'aria.sock')
    rows = []
    prefix = 'lookup-' + uuid.uuid4().hex
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://localhost', timeout=180, trust_env=False) as client:
        cfg = (await client.get('/_live/config')).json()
        assert cfg['tools_mode'] == 'log_only' and cfg['warmup_ready']
        snapshot = (await client.get('/api/inventory/snapshot')).json()
        cases = build_cases(snapshot)
        on = next(c for c in cases if c['id'] == 'en-on')['expected_commands'][0]['cmd_id']
        lamp = next(d for d in snapshot['devices'].values() if any(c['cmd_id'] == on for c in d['commands']))
        name, room = lamp['name'], lamp['room']
        texts = {
            'fr': [f'cherche l’appareil nommé « {name} » dans la pièce « {room} » et indique quelles commandes il propose.',
                   f'Quelles possibilités offre « {name} », située dans « {room} » ?',
                   f'Peux-tu retrouver « {name} » dans « {room} » et m’indiquer ses fonctions ?'],
            'en': [f'find the device named "{name}" in room "{room}" and tell me which commands it offers.',
                   f'What functions does "{name}" in "{room}" offer?',
                   f'Look up "{name}" in "{room}" and describe its capabilities.'],
        }
        for language, prompts in texts.items():
            for i, text in enumerate(prompts):
                cases.append(dict(id=f'{language}-lookup-{i}', session=f'{language}-lookup-{i}',
                                  language=language, kind='lookup', text=text, answer=None,
                                  expected_commands=[], allow_search=True))
        seen = set()
        for case in cases:
            if case['session'] not in seen:
                case['text'] = 'Aria, ' + case['text']; seen.add(case['session'])
            row = await measure(client, socket, case, prefix)
            if case['kind'] == 'lookup':
                calls = [e for e in row['events'] if e['event'] == 'Live.Tool']
                if not any(e['name'] == 'find_devices' and any(c['cmd_id'] == on for c in
                           e['result'].get('candidates', [])) for e in calls):
                    row['flags'].append('missing_grounded_lookup')
                row['expected_device'] = lamp
            rows.append(row)
            args.output.write_text(json.dumps({'config': cfg, 'cases': rows}, ensure_ascii=False, indent=2))
            print(case['id'], row['flags'], row['result'].get('response'), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
