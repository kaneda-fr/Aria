"""LAN audio ingress for the isolated, log-only development instance.

Only /ws/asr is forwarded. The monitoring page and mute controls use a separate port.
The active run is selected from the same local pointer as the log reader.
"""
import asyncio
import contextlib
import json
import os
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, WebSocket

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
connections = 0


def active_socket():
    metadata = json.loads(Path(os.environ.get('ARIA_MONITOR_ACTIVE', '/data/aria/monitor/active.json')).read_text())
    root = Path(os.environ.get('ARIA_MONITOR_RUN_ROOT', '/data/aria/e2e/runs')).resolve()
    run = Path(metadata['run_dir']).resolve()
    if run.parent != root or metadata.get('tools_mode') != 'log_only':
        raise ValueError('Expected an isolated log-only run')
    return str(run / 'aria.sock')


@app.websocket('/ws/asr')
async def forward(socket: WebSocket):
    global connections
    if connections >= 32:
        await socket.close(code=1013)
        return
    connections += 1
    tasks = []
    try:
        path = active_socket()
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=path),
                                    base_url='http://aria', timeout=2, trust_env=False) as client:
            config = (await client.get('/_live/config')).json()
            if config.get('tools_mode') != 'log_only' or not config.get('warmup_ready'):
                raise ValueError('Log-only instance is not ready')
        async with websockets.unix_connect(path, uri='ws://localhost/ws/asr',
                                           max_size=65536, max_queue=8, write_limit=65536) as upstream:
            await socket.accept()
            async def upload():
                while True:
                    message = await socket.receive()
                    if message['type'] == 'websocket.disconnect':
                        return
                    payload = message.get('bytes') if message.get('bytes') is not None else message.get('text')
                    if payload is not None:
                        await asyncio.wait_for(upstream.send(payload), timeout=2)
            async def download():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await asyncio.wait_for(socket.send_bytes(message), timeout=2)
                    else:
                        await asyncio.wait_for(socket.send_text(message), timeout=2)
            tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
    except (OSError, ValueError, KeyError, httpx.HTTPError, websockets.WebSocketException,
            asyncio.TimeoutError, RuntimeError):
        with contextlib.suppress(Exception):
            await socket.close(code=1013)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await socket.close()
        connections -= 1
