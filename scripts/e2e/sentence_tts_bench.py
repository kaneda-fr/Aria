"""Apollo-only paired speech-path probe. Captures WAVs; never plays real audio.

Run in an empty network namespace with prepared Piper/ffmpeg and dependencies.
Use separate processes/source roots for baseline and candidate. No LLM or tools.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import wave

CASES = [
    ('fr', 'fr', 'La porte est ouverte. La lumière est allumée.'),
    ('en', 'en', 'The door is open. The light is on.'),
    ('fr_en', 'fr', 'La porte du salon est maintenant ouverte. The light in the kitchen is on.'),
    ('en_fr', 'en', 'The living room door is now open. La lumière de la cuisine est allumée.'),
    ('short_fr_en', 'fr', 'Bonjour. The light in the kitchen is on.'),
    ('short_en_fr', 'en', 'Hello. La lumière de la cuisine est allumée.'),
]


async def run(args):
    import app.speech_output.speech_output as module
    from app.tts.tts_engine import TtsConfig
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.output.parent / (args.output.stem + '-cache')
    root.mkdir(exist_ok=True)
    os.environ.update(ARIA_SONOS_IP='127.0.0.1', ARIA_HTTP_BASE_URL='http://capture.invalid',
        ARIA_TTS_VOICE_MAP='en:/data/aria/voices/en_US-amy-low.onnx,fr:/data/aria/voices/fr_FR-siwis-low.onnx',
        ARIA_TTS_CHUNKING='0', ARIA_PIPER_BIN='/opt/piper-tts/piper', ARIA_TTS_LANG_DETECT='heuristic')
    observations = []
    state = {}
    class Capture:
        def __init__(self, **kwargs): pass
        async def set_volume(self, volume): pass
        async def stop_all(self): pass
        async def play_url(self, url):
            path=root/'sonos'/url.rsplit('/',1)[1]
            with wave.open(str(path)) as audio:
                assert audio.getnframes()>0
                assert audio.getframerate()==44100 and audio.getnchannels()==2
                duration=audio.getnframes()/audio.getframerate()
            now=time.perf_counter()
            clips=state['clips']
            clips.append({'start_s':now-state['start'],'duration_s':duration,
                'gap_ms':None if not clips else (now-state['previous_end'])*1000,
                'wav_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            done=asyncio.get_running_loop().create_future()
            def finish():
                state['previous_end']=time.perf_counter()
                if not done.done():done.set_result(None)
            asyncio.get_running_loop().call_later(duration if state['realtime'] else 0,finish)
            return done
    module.SonosHttpSink=Capture
    module.verify_sonos_reachable=lambda *a,**k:'capture-only'
    speech=module.SpeechOutput(TtsConfig(True,'piper','/data/aria/voices/en_US-amy-low.onnx',1.,root,60.))
    synth=speech._engine.synthesize_to_wav
    def tracked(text,**kwargs):
        start=time.perf_counter()
        result=synth(text,**kwargs)
        state['synth'].append({'text':text,'voice':Path(kwargs['voice']).name,
            'start_s':start-state['start'],'elapsed_ms':(time.perf_counter()-start)*1000})
        return result
    from types import SimpleNamespace
    speech._engine=SimpleNamespace(synthesize_to_wav=tracked)
    async def one(case,cache,realtime,iteration):
        name,hint,text=case
        if cache=='cold':
            for child in root.iterdir():
                if child.is_dir():shutil.rmtree(child)
        state.clear();state.update(start=time.perf_counter(),clips=[],synth=[],realtime=realtime)
        await speech.speak(text,language=hint)
        await speech._task
        assert await speech.audio_ready
        elapsed=time.perf_counter()-state['start']
        assert state['clips']
        observation={'case':name,'hint':hint,'text':text,'cache':cache,'realtime':realtime,'iteration':iteration,
            'first_audio_ms':state['clips'][0]['start_s']*1000,'completion_ms':elapsed*1000,
            'clips':state['clips'],'synth':state['synth']}
        observations.append(observation)
        args.output.write_text(json.dumps({'source':str(Path(module.__file__).resolve()),
            'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'results':observations},indent=2))
    # Same process baseline/candidate invocation order is alternated by the driver.
    for iteration in range(args.rounds):
        for case in CASES:
            await one(case,'cold',False,iteration)
            await one(case,'warm',False,iteration)
    if args.playback:
        for case in CASES[2:]:
            await one(case,'cold',True,0)
            await one(case,'warm',True,0)
    print(json.dumps({'output':str(args.output),'cases':len(observations)}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--playback',action='store_true',help='Simulate delivery time with timers, without audio devices.')
    args=parser.parse_args()
    asyncio.run(run(args))


if __name__=='__main__': main()
