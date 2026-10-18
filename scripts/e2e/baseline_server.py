"""Read-only inference instrumentation and synthetic fixtures for the baseline suite."""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess

import uvicorn
from fastapi import HTTPException
from fastapi.responses import FileResponse
from scripts.e2e.server import app, record, RUN, SOCKET


async def observe_ollama(response):
    if response.request.url.path != '/api/chat':
        return
    await response.aread()
    data = response.json()
    request = json.loads(response.request.content)
    record('Baseline.Model', message=data.get('message'),
           input=request['messages'][-1].get('content'),
           tools_offered=bool(request.get('tools')),
           **{k: data.get(k) for k in ('done_reason', 'eval_count', 'eval_duration',
               'prompt_eval_count', 'prompt_eval_duration', 'load_duration', 'total_duration')})


@app.post('/_e2e/baseline/start')
async def baseline_start(body: dict | None = None):
    hooks = app.state.llm_client.event_hooks['response']
    if observe_ollama not in hooks:
        hooks.append(observe_ollama)
    if body and body.get('observe_asr') is True and not getattr(app.state, 'e2e_asr_observed', False):
        original = app.state.asr.transcribe

        def observed_asr(pcm, *args, **kwargs):
            text = original(pcm, *args, **kwargs)
            record('Baseline.ASR', pcm_sha256=hashlib.sha256(bytes(pcm)).hexdigest(),
                   bytes=len(pcm), text=text)
            return text

        app.state.asr.transcribe = observed_asr
        app.state.e2e_asr_observed = True
    return {'instrumented': True, 'num_predict': app.state.llm_config.num_predict}


@app.post('/_e2e/baseline/fixture')
async def baseline_fixture(body: dict):
    language, text = body.get('language'), body.get('text')
    if language not in ('en', 'fr') or not isinstance(text, str) or not 0 < len(text) < 500:
        raise HTTPException(400)
    voice = 'en_US-amy-low' if language == 'en' else 'fr_FR-siwis-low'
    path = RUN / ('input-' + hashlib.sha256((language + text).encode()).hexdigest()[:16] + '.wav')
    if not path.exists():
        raw = await asyncio.to_thread(app.state.speech_output._engine.synthesize_to_wav,
            text, voice=f'/data/aria/voices/{voice}.onnx', rate=1.0)
        await asyncio.to_thread(subprocess.run, ['/usr/bin/ffmpeg', '-nostdin', '-v', 'error',
            '-y', '-i', str(raw), '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(path)],
            check=True, timeout=30)
    return FileResponse(path, media_type='audio/wav')


if __name__ == '__main__':
    uvicorn.run(app, uds=SOCKET, log_level='info')
