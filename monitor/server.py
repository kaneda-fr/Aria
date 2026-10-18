"""LAN log monitor with per-client input mute, separate from inference.

Run: ARIA_MONITOR_ACTIVE=/path/active.json python -m uvicorn monitor.server:app
The active file contains run_dir and tools_mode; only mute is proxied as a control.
"""
import asyncio
from collections import deque, OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import time
import threading
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Body, HTTPException
from pydantic import StrictBool
from monitor.ingress import active_socket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).with_name('static')
PREFIXES = ('Live.', 'ARIA.Session.', 'ARIA.Utterance.Started', 'ARIA.Request.', 'ARIA.LLM.Start', 'ARIA.LLM.Done', 'ARIA.LLM.Error',
            'ARIA.LLM.Empty', 'ARIA.Control.', 'ARIA.Conversation', 'ARIA.HistoryPrefill.',
            'ARIA.SPK', 'ARIA.ASR.', 'ARIA.Utterance.Final', 'ARIA.TTS.')


class Tail:
    def __init__(self, path):
        self.path = path
        self.position = None
        self.identity = None
        self.pending = b''

    def read(self):
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or self.position is None or stat.st_size < self.position:
                self.identity = identity
                self.position = max(0, stat.st_size - 262144)
                self.pending = b''
                skip = self.position > 0
            else:
                skip = False
            with self.path.open('rb') as source:
                source.seek(self.position)
                chunk = source.read(131072)
                self.position = source.tell()
            lines = (self.pending + chunk).split(b'\n')
            self.pending = lines.pop()[-65536:]
            if skip and lines:
                lines.pop(0)
            rows = []
            for line in lines:
                if len(line) > 65536:
                    continue
                try:
                    row = json.loads(line)
                    name = row.get('event', row.get('msg', ''))
                    if not isinstance(name, str) or not name.startswith(PREFIXES):
                        continue
                    stamp = row.get('time')
                    if not isinstance(stamp, (int, float)):
                        stamp = datetime.fromisoformat(row.get('timestamp', row.get('ts', '')).replace('Z', '+00:00')).timestamp()
                    rows.append(dict(event=name, time=stamp, data=row))
                except (ValueError, TypeError, AttributeError):
                    continue
            return rows
        except OSError:
            return []


class Feed:
    def __init__(self, active, root):
        self.active, self.root = Path(active), Path(root).resolve()
        self.rows = deque(maxlen=1000)
        self.request_clients = OrderedDict()
        self.seq = 0
        self.run = None
        self.tails = []
        self.status = {'source': 'waiting', 'run_id': None, 'tools_mode': 'unknown'}
        self.clients = 0
        self.client_snapshot = {'available': False, 'clients': []}
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            self._update()

    def _update(self):
        try:
            metadata = json.loads(self.active.read_text())
            run = Path(metadata['run_dir']).resolve()
            if run.parent != self.root:
                raise ValueError('Run must be a direct child of the configured run root')
            if run != self.run:
                self.run = run
                self.rows.clear()
                self.request_clients.clear()
                self.client_snapshot = {'available': False, 'clients': []}
                self.tails = [Tail(run / name) for name in ('conversation.jsonl', 'server.log')]
            self.status = {'source': 'reading' if (run / 'server.log').exists() else 'unavailable',
                           'run_id': run.name, 'tools_mode': metadata.get('tools_mode', 'unknown')}
            incoming = sorted((r for tail in self.tails for r in tail.read()), key=lambda r: r['time'])
            for row in incoming:
                data = row['data']
                session = data.get('session_id') or ''
                name = data.get('client_id') or (session[7:] if isinstance(session, str) and session.startswith('client:') else None)
                request = data.get('request_id')
                if request and name:
                    self.request_clients[request] = name
                    if len(self.request_clients) > 1024:
                        self.request_clients.popitem(last=False)
                row['client_id'] = name or self.request_clients.get(request)
                self.seq += 1
                self.rows.append({**row, 'seq': self.seq})
        except (OSError, ValueError, KeyError, TypeError):
            self.status = {**self.status, 'source': 'unavailable'}

    def packet(self, cursor, run_id):
        with self.lock:
            return self._packet(cursor, run_id)

    def _packet(self, cursor, run_id):
        unseen = [r for r in self.rows if r['seq'] > cursor]
        reset = (run_id != self.status['run_id'] or len(unseen) > 200 or
                 bool(self.rows and cursor < self.rows[0]['seq'] - 1))
        rows = list(self.rows)[-200:] if reset else unseen
        return {'type': 'snapshot' if reset else 'events', 'events': rows, 'cursor': self.seq,
                'status': self.status, 'clients': self.client_snapshot, 'server_time': time.time()}


@asynccontextmanager
async def lifespan(app):
    app.state.feed = Feed(os.environ.get('ARIA_MONITOR_ACTIVE', '/data/aria/monitor/active.json'),
                          os.environ.get('ARIA_MONITOR_RUN_ROOT', '/data/aria/e2e/runs'))
    async def follow():
        last_clients = 0
        while True:
            await asyncio.to_thread(app.state.feed.update)
            feed = app.state.feed
            if feed.run and time.monotonic() - last_clients >= 1:
                run = feed.run
                try:
                    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(run / 'aria.sock')),
                                                 base_url='http://aria', timeout=.5, trust_env=False) as client:
                        response = await client.get('/api/clients')
                        response.raise_for_status()
                        snapshot = {'available': True, 'clients': response.json()['clients']}
                except (httpx.HTTPError, ValueError, KeyError, OSError):
                    snapshot = {'available': False, 'clients': []}
                with feed.lock:
                    if feed.run == run:
                        feed.client_snapshot = snapshot
                last_clients = time.monotonic()
            await asyncio.sleep(.25)
    task = asyncio.create_task(follow())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/assets', StaticFiles(directory=STATIC), name='assets')


@app.middleware('http')
async def local_assets_only(request, call_next):
    response = await call_next(request)
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
async def index():
    return FileResponse(STATIC / 'index.html')


@app.post('/api/clients/{client_id}/mute')
async def mute_client(request: Request, client_id: str,
                      muted: StrictBool = Body(embed=True, strict=True), run_id: str = Body(embed=True)):
    # Like offline-list cleanup, require an explicit same-origin request;
    # no inference, arbitrary proxy, or device-action route is exposed.
    origin = request.headers.get('origin', '')
    if origin != f"{request.url.scheme}://{request.headers.get('host')}" or request.headers.get('x-aria-control') != 'mute':
        raise HTTPException(403, 'Same-origin control required')
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', client_id):
        raise HTTPException(422, 'Invalid client name')
    try:
        path = active_socket()
        if Path(path).parent.name != run_id:
            raise HTTPException(409, 'Active run changed; refresh client status')
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=path),
                                     base_url='http://aria', timeout=2, trust_env=False) as client:
            response = await client.post(f'/api/clients/{client_id}/mute', json={'muted': muted})
        if response.status_code == 404:
            raise HTTPException(404, 'Unknown client')
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        raise HTTPException(503, 'Aria client controls unavailable')


@app.post('/api/clients/clear-offline')
async def clear_offline_clients(request: Request, run_id: str = Body(embed=True)):
    origin = request.headers.get('origin', '')
    if origin != f"{request.url.scheme}://{request.headers.get('host')}" or request.headers.get('x-aria-control') != 'clear-offline':
        raise HTTPException(403, 'Same-origin control required')
    try:
        path = active_socket()
        if Path(path).parent.name != run_id:
            raise HTTPException(409, 'Active run changed; refresh client status')
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=path),
                                     base_url='http://aria', timeout=2, trust_env=False) as client:
            response = await client.post('/api/clients/clear-offline')
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        raise HTTPException(503, 'Aria client controls unavailable')


@app.websocket('/ws/events')
async def events(socket: WebSocket):
    origin = socket.headers.get('origin')
    if origin and origin.split('://', 1)[-1] != socket.headers.get('host'):
        await socket.close(code=1008)
        return
    feed = socket.app.state.feed
    if feed.clients >= 16:
        await socket.close(code=1013)
        return
    await socket.accept()
    feed.clients += 1
    cursor, run_id = -1, None
    try:
        while True:
            packet = feed.packet(cursor, run_id)
            await asyncio.wait_for(socket.send_json(packet), timeout=3)
            cursor, run_id = packet['cursor'], packet['status']['run_id']
            # Consume and discard browser messages; this channel has no commands.
            try:
                message = await asyncio.wait_for(socket.receive(), timeout=.5)
                if message['type'] == 'websocket.disconnect':
                    break
                await asyncio.sleep(.05)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        pass
    finally:
        feed.clients -= 1
