"""Held-out inventory wording/state checks on Apollo; never opens audio hardware."""
import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx

from scripts.e2e.conversation_bench import build_cases, measure


def cases_for(snapshot):
    controls = build_cases(snapshot)
    def device(case_id):
        cmd = next(c for c in controls if c['id'] == case_id)['expected_commands'][0]['cmd_id']
        return next(d for d in snapshot['devices'].values()
                    if any(c['cmd_id'] == cmd for c in d['commands']))
    lamp, shutter = device('en-on'), device('en-shutter')
    absent = 'AbsentFixtureZYX900'
    assert all(absent not in d['name'] for d in snapshot['devices'].values())
    rows = []
    for lang in ['fr', 'en']:
        l, lr, s, sr = lamp['name'], lamp['room'], shutter['name'], shutter['room']
        prompts = ([
            (f'Dis-moi ce que permet « {l} » dans « {lr} ».', lamp, 'capability'),
            (f'Quelles options ai-je pour « {s} » dans « {sr} » ?', shutter, 'capability'),
            (f'La lampe « {l} » dans « {lr} » est-elle actuellement allumée ?', lamp, 'state'),
            (f'Retrouve « {absent} » dans « {lr} » et liste ses commandes.', None, 'missing'),
        ] if lang == 'fr' else [
            (f'Tell me what I can do with "{l}" in "{lr}".', lamp, 'capability'),
            (f'What options are available for "{s}" in "{sr}"?', shutter, 'capability'),
            (f'Is "{l}" in "{lr}" switched on right now?', lamp, 'state'),
            (f'Locate "{absent}" in "{lr}" and list its controls.', None, 'missing'),
        ])
        for i, (text, target, kind) in enumerate(prompts):
            rows.append(dict(id=f'{lang}-transfer-{i}', session=f'{lang}-transfer-{i}',
                             language=lang, kind=kind, text='Aria, '+text, answer=None,
                             expected_commands=[], allow_search=True, target=target))
    return rows


def assess(row):
    """Evidence checks supplement (not replace) manual semantic review."""
    case, flags = row['case'], row['flags']
    tools = [e for e in row['events'] if e['event'] == 'Live.Tool']
    searches = [e for e in tools if e['name'] == 'find_devices']
    if not searches:
        flags.append('missing_search')
    if any(e['name'] != 'find_devices' for e in tools):
        flags.append('unexpected_action_attempt')
    candidates = [c for e in searches for c in e['result'].get('candidates', [])]
    target = case.get('target')
    if target and not any(c.get('device_name') == target['name'] and
                          c.get('room') == target['room'] for c in candidates):
        flags.append('missing_target')
    if not target and candidates:
        flags.append('false_match')
    reply = row['result'].get('response', '')
    if any(str(c['cmd_id']) in reply for c in candidates if c.get('cmd_id')):
        flags.append('spoken_command_id')
    if case['kind'] == 'state' and not any(term in reply.lower() for term in
        ['inconnu', 'unknown', 'cannot', "can't", 'ne peux pas', 'pas accès']):
        flags.append('state_uncertainty_missing')


async def run(args):
    socket = str(args.server_run / 'aria.sock')
    prefix = 'transfer-' + uuid.uuid4().hex
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://localhost', timeout=180, trust_env=False) as client:
        cfg = (await client.get('/_live/config')).json()
        assert cfg['tools_mode'] == 'log_only' and cfg['warmup_ready']
        cases = cases_for((await client.get('/api/inventory/snapshot')).json())
        if args.fixtures:
            for fixture in json.loads((args.fixtures / 'manifest.json').read_text()):
                base = next(c for c in cases if c['id'] == fixture['case_id'])
                cases.append({**base, 'id': base['id']+'-voice', 'wav': fixture['path'],
                              'text': fixture['text']})
        rows = []
        for case in cases:
            row = await measure(client, socket, case, prefix)
            assess(row)
            rows.append(row)
            args.output.write_text(json.dumps(dict(config=cfg, cases=rows), ensure_ascii=False, indent=2))
            print(case['id'], row['flags'], row['result'].get('response'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path)
    asyncio.run(run(parser.parse_args()))
