"""Private live-session transcript instrumentation; no production imports of this module."""
from contextvars import ContextVar
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
import time
from urllib.parse import urlsplit
from fastapi import HTTPException
from app.speech_output.playback import PlaybackBroker
from app.speech_output.speech_output import _wav_duration_seconds

import uvicorn

from scripts.e2e.server import app, record, RUN, SOCKET, CaptureSink, speech
import app.main as runtime
import app.llm.tools as tooling
from scripts.e2e.live_tools import install_live_tools, LogOnlyActions
from scripts.e2e.live_warmup import warmup
from app.llm.history_prefill import PREFILL

REQUEST = ContextVar('live_request', default=None)
PLAYBACK = PlaybackBroker()
original_process = runtime._process_request
original_tool = tooling.execute_tool
original_lifespan = app.router.lifespan_context


def log_turn(event, **fields):
    entry = {'event': event, 'timestamp': datetime.now(timezone.utc).isoformat(), **fields}
    with (RUN / 'conversation.jsonl').open('a', encoding='utf-8') as output:
        output.write(json.dumps(entry, ensure_ascii=False) + '\n')
    record(event, **{k: v for k, v in entry.items() if k != 'event'})


async def observe_model(response):
    if response.request.url.path != '/api/chat':
        return
    await response.aread()
    body = response.json()
    request = json.loads(response.request.content)
    log_turn('Live.Prefill' if PREFILL.get() else 'Live.Model', request_id=REQUEST.get(),
             tools_offered=bool(request.get('tools')),
             tool_calls=body.get('message', {}).get('tool_calls', []),
             history_messages=max(0, len(request['messages']) - 2),
             done_reason=body.get('done_reason'),
             **{key: body.get(key) for key in ('eval_count', 'eval_duration',
                  'prompt_eval_count', 'prompt_eval_cached_count', 'prompt_eval_duration',
                  'load_duration', 'total_duration')})


async def logged_process(app, job):
    token = REQUEST.set(job.request_id)
    started = time.monotonic()
    spk = getattr(app.state, 'spk_recognizer', None)
    enabled = bool(spk and spk.enabled)
    ready = bool(spk and spk._ready)
    log_turn('Live.Request', request_id=job.request_id, session_id=job.session_id,
             source=job.source, text=job.text, speaker_user=job.speaker_user,
             speaker_status=('not_evaluated' if job.source != 'voice' else
                             'disabled' if not enabled else 'not_ready' if not ready else
                             'attributed' if job.speaker_user != 'unknown' else 'unknown'))
    try:
        hooks = app.state.llm_client.event_hooks['response']
        if observe_model not in hooks:
            hooks.append(observe_model)
        result = await original_process(app, job)
        log_turn('Live.Reply', request_id=job.request_id, session_id=job.session_id,
                 text=result.get('response', ''), success=result.get('success'),
                 error=result.get('error'), speech_scheduled=result.get('spoken'),
                 processing_ms=round(1000 * (time.monotonic() - started)))
        return result
    except BaseException as exc:
        log_turn('Live.Error', request_id=job.request_id, error=repr(exc))
        raise
    finally:
        REQUEST.reset(token)


class LiveSink(CaptureSink):
    async def play_url(self, url):
        await super().play_url(url)
        filename = urlsplit(url).path.rsplit('/', 1)[-1]
        ticket = (PLAYBACK.begin(_wav_duration_seconds(RUN / 'tts' / 'sonos' / filename))
                  if PLAYBACK.player and PLAYBACK.deadline > PLAYBACK.clock() else None)
        log_turn('Live.Audio', request_id=REQUEST.get(), url=url,
                 playback_id=ticket[0] if ticket else None)
        return ticket[1] if ticket else None


async def logged_tool(**kwargs):
    result = await original_tool(**kwargs)
    log_turn('Live.Tool', request_id=REQUEST.get(), name=kwargs['tool_name'],
             arguments=kwargs['tool_args'], result=result)
    return result


@asynccontextmanager
async def live_lifespan(app):
    async with original_lifespan(app):
        install_live_tools(app, RUN, log_turn, REQUEST.get, '/data/aria/inventory.json')
        app.state.warmup_ready = False
        hooks = app.state.llm_client.event_hooks['response']
        if observe_model not in hooks:
            hooks.append(observe_model)
        token = REQUEST.set('startup-warmup')
        try:
            app.state.warmup_result = await warmup(
                app.state.llm_config, app.state.llm_client, runtime.SYSTEM_PROMPT,
                tooling.TOOLS, with_tools=os.environ.get('ARIA_E2E_WARMUP_TOOLS', '1') != '0')
            log_turn('Live.Warmup', **app.state.warmup_result)
            app.state.warmup_ready = True
        finally:
            REQUEST.reset(token)
        yield


runtime._process_request = logged_process
tooling.execute_tool = logged_tool
app.router.lifespan_context = live_lifespan
speech.SonosHttpSink = LiveSink


@app.get('/_live/config')
async def live_config():
    spk = getattr(app.state, 'spk_recognizer', None)
    return {'history_max_messages': runtime._llm_history_max_messages(),
            'speaker_enabled': bool(spk and spk.enabled),
            'speaker_ready': bool(spk and spk._ready),
            'speaker_profiles': len(spk._profiles) if spk else 0,
            'tools_mode': ('log_only' if isinstance(app.state.plugin_registry.get_plugin('actions'),
                                                  LogOnlyActions) else 'unavailable'),
            'conversation_logging': True,
            'warmup_ready': app.state.warmup_ready,
            'warmup': app.state.warmup_result}


@app.post('/_live/ready')
async def ready_speech():
    # Readiness must not issue a tool-free LLM call that replaces the warmed prefix.
    if not getattr(app.state, 'warmup_ready', False):
        from fastapi import HTTPException
        raise HTTPException(503, 'Warm-up incomplete')
    token = REQUEST.set('startup-ready')
    try:
        spoken = await runtime._speak_reply(app, 'Aria is ready.', 'en')
        return {'success': spoken, 'spoken': spoken, 'response': 'Aria is ready.'}
    finally:
        REQUEST.reset(token)


@app.post('/_live/player')
async def register_player():
    try:
        return {'player_id': PLAYBACK.register()}
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.put('/_live/player/{player}')
async def player_heartbeat(player: str):
    try:
        PLAYBACK.heartbeat(player)
        return {'success': True}
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.delete('/_live/player/{player}')
async def unregister_player(player: str):
    try:
        PLAYBACK.unregister(player)
        return {'success': True}
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post('/_live/player/{player}/clip/{clip}')
async def player_status(player: str, clip: str, body: dict):
    try:
        PLAYBACK.ack(player, clip, body.get('status'))
        log_turn('Live.Playback', playback_id=clip, status=body['status'])
        return {'success': True}
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get('/_live/playback')
async def playback_state():
    speech = getattr(app.state, 'speech_output', None)
    revision = getattr(speech, 'input_revision', 0)
    blocked = bool(getattr(speech, 'input_blocked', False))
    return {'input_blocked': blocked, 'input_revision': revision,
            'pending': len(PLAYBACK.pending),
            'player_connected': bool(PLAYBACK.player and PLAYBACK.deadline > PLAYBACK.clock())}


if __name__ == '__main__':
    uvicorn.run(app, uds=SOCKET, log_level='warning')
