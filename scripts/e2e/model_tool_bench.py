"""Read-only inventory questions through the real model loop and capture sink.

The existing conversation report supplies known command queries/IDs privately.
This supplements deterministic control tests; it never authorizes an action.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from scripts.e2e.conversation_bench import measure


async def run(server_run, reference, output):
    cases = {row['case']['id']: row['case'] for row in json.loads(reference.read_text())['cases']}
    rows = []
    socket = str(server_run / 'aria.sock')
    prefix = f'model-tools-{time.time_ns()}'
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://localhost', timeout=300, trust_env=False) as client:
        cfg = (await client.get('/_live/config')).json()
        assert cfg['tools_mode'] == 'log_only' and cfg['warmup_ready']
        for language in ['fr', 'en']:
            known = cases[language + '-on']
            prompts = [
                ('known', ('Quels appareils correspondent à cette demande : « ' + known['text'] + ' » ? Recherche uniquement, sans exécuter de commande.')
                 if language == 'fr' else ('Which devices match this request: "' + known['text'] + '"? Only look them up; do not execute any command.')),
                ('missing', 'Recherche les appareils nommés Fixture Introuvable, sans exécuter de commande.'
                 if language == 'fr' else 'Look up devices named Fixture Missing, without executing any command.'),
            ]
            for kind, text in prompts:
                case = dict(id=language + '-' + kind, language=language, kind='model_lookup',
                            text=text, session=language + '-' + kind, answer=None,
                            expected_commands=[], allow_search=True)
                row = await measure(client, socket, case, prefix)
                tools = [event for event in row['events'] if event['event'] == 'Live.Tool']
                searches = [event for event in tools if event['name'] == 'find_devices']
                if not searches:
                    row['flags'].append('lookup_not_requested')
                if any(event['name'] != 'find_devices' for event in tools):
                    row['flags'].append('unexpected_tool_attempt')
                candidates = [candidate for event in searches for candidate in event['result'].get('candidates', [])]
                if kind == 'known' and known['expected_commands'][0]['cmd_id'] not in {item['cmd_id'] for item in candidates}:
                    row['flags'].append('expected_candidate_missing')
                if kind == 'missing' and candidates:
                    row['flags'].append('invented_candidate')
                rows.append(row)
                output.write_text(json.dumps(dict(config=cfg, cases=rows), ensure_ascii=False, indent=2))
                print(json.dumps(dict(id=case['id'], flags=row['flags'],
                                      reply=row['result'].get('response'), latency=row['latency']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.server_run, args.reference, args.output))
