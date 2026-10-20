"""Owned-LAN MQTT reads through a private Unix socket; no action endpoint."""
import argparse
import asyncio
import ipaddress
import json
import math
import os
import threading
import time
from pathlib import Path

from .normalizer import JeedomNormalizer
from .provider import InventoryProvider


class JeedomReadBridge:
    def __init__(self, cache, *, host, port=1883, root='jeedom'):
        if not ipaddress.ip_address(host).is_private:
            raise ValueError('Jeedom broker must be an owned private-LAN IP')
        if not root or '/' in root or '+' in root or '#' in root:
            raise ValueError('Invalid MQTT root')
        self.host, self.port, self.root = host, port, root
        self.provider = InventoryProvider('inventory', {'cache_path': str(cache), 'sources': {}})
        self.provider.start()
        self.seed_last_update = json.loads(Path(cache).read_text()).get('last_update')
        self.connected = self.subscribed = False
        self.client = None
        self.lock = threading.RLock()
        self.read_lock = threading.Lock()
        self.pending = {}
        self.live_discovery_count = self.event_count = self.read_count = 0
        self.last_discovery_at = None
        self.generation = time.time_ns()

    def start(self):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(client_id=f'aria-read-{os.getpid()}')
        def connected(client, userdata, flags, rc):
            self.connected = rc == 0
            self.subscribed = False
            if self.connected:
                client.subscribe([(f'{self.root}/discovery/eqLogic/#', 1),
                                  (f'{self.root}/cmd/event/#', 1),
                                  (f'{self.root}/cmd/value/#', 1)])
        def subscribed(client, userdata, mid, qos):
            self.subscribed = all(int(q) < 128 for q in qos)
        def disconnected(client, userdata, rc):
            self.connected = self.subscribed = False
        self.client.on_connect = connected
        self.client.on_subscribe = subscribed
        self.client.on_disconnect = disconnected
        self.client.on_message = lambda c, u, m: self.receive(m.topic, m.payload, bool(m.retain))
        self.client.connect(self.host, self.port, 30)
        self.client.loop_start()

    def stop(self):
        if self.client:
            self.client.disconnect()
            self.client.loop_stop()
        self.save_catalog()

    def receive(self, topic, payload, retained=False):
        if len(payload) > 1024 * 1024:
            return
        try:
            text = payload.decode('utf-8')
            prefix = f'{self.root}/discovery/eqLogic/'
            with self.lock:
                if topic.startswith(prefix):
                    raw = json.loads(text)
                    if not isinstance(raw, dict) or str(raw.get('id')) != topic[len(prefix):]:
                        return
                    raw['isEnable'] = str(raw.get('isEnable', '0'))
                    if raw['isEnable'] == '0':
                        self.provider.indexer.remove_device(str(raw['id']))
                        self.live_discovery_count += 1
                        self.last_discovery_at = time.time()
                        return
                    device = JeedomNormalizer.normalize_device(raw, source='jeedom_mqtt')
                    if device:
                        self.provider.indexer.add_device(device)
                        self.live_discovery_count += 1
                        self.last_discovery_at = time.time()
                    return
                if topic.startswith(f'{self.root}/cmd/event/'):
                    self.event_count += 1
                    return  # Events prove connectivity, not a new measurement timestamp.
                if not topic.startswith(f'{self.root}/cmd/value/') or retained:
                    return
                cmd_id = topic.rsplit('/', 1)[-1]
                pending = self.pending.get(cmd_id)
                if not pending:
                    return
                command = self.provider.indexer.get_command(cmd_id)
                device = self.provider.indexer.get_device(command.device_id) if command else None
                if not command or command.type != 'info' or not device or not device.enabled:
                    return
                if command.subType in ('numeric', 'binary'):
                    value = float(text)
                    if not math.isfinite(value) or (command.subType == 'binary' and value not in (0, 1)):
                        return
                else:
                    if len(text) > 512 or not text.strip():
                        return
                    value = text.strip()
                pending['value'] = value
                pending['retrieved_at'] = time.time()
                pending['event'].set()
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
            return

    def catalog(self, version=None):
        with self.lock:
            metadata = {'connected': self.connected, 'subscribed': self.subscribed,
                        'seed_last_update': self.seed_last_update,
                        'live_discovery_count': self.live_discovery_count,
                        'event_count': self.event_count, 'read_count': self.read_count,
                        'last_discovery_at': self.last_discovery_at, 'catalog_complete': False,
                        'catalog_version': f'{self.generation}-{self.live_discovery_count}'}
            if version == metadata['catalog_version']:
                return {'unchanged': True, **metadata}
            return {**self.provider.get_snapshot(), **metadata}

    def save_catalog(self):
        """Debounced, atomic definitions-only persistence; readings are never cached."""
        with self.lock:
            if getattr(self, 'persisted_discovery_count', 0) == self.live_discovery_count:
                return
            data = {'devices': list(self.provider.get_snapshot()['devices'].values()),
                    'last_update': self.last_discovery_at or self.seed_last_update,
                    'catalog_complete': False}
            path = self.provider.cache_path
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False))
            temporary.replace(path)
            self.persisted_discovery_count = self.live_discovery_count

    def read(self, ids, timeout=.8):
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
                or any(not isinstance(i, str) or not i.isdecimal() for i in ids)):
            return {'success': False, 'status': 'invalid_read'}
        if not self.read_lock.acquire(timeout=.1):
            return {'success': False, 'status': 'busy'}
        try:
            with self.lock:
                if not self.connected or not self.subscribed:
                    return {'success': False, 'status': 'disconnected'}
                commands = []
                for cmd_id in dict.fromkeys(ids):
                    cmd = self.provider.indexer.get_command(cmd_id)
                    device = self.provider.indexer.get_device(cmd.device_id) if cmd else None
                    if not cmd or cmd.type != 'info' or not device or not device.enabled:
                        return {'success': False, 'status': 'invalid_read'}
                    commands.append((cmd, device))
                self.pending = {cmd.cmd_id: {'event': threading.Event()} for cmd, _ in commands}
                for cmd, _ in commands:
                    # This is the only application publish in this service.
                    self.client.publish(f'{self.root}/cmd/get/{cmd.cmd_id}', '', qos=1, retain=False)
                    self.read_count += 1
            deadline = time.monotonic() + min(max(timeout, .01), 2)
            for item in list(self.pending.values()):
                item['event'].wait(max(0, deadline - time.monotonic()))
            with self.lock:
                readings = []
                for cmd, device in commands:
                    p = self.pending[cmd.cmd_id]
                    row = {'cmd_id': cmd.cmd_id, 'device_id': device.eq_id,
                           'device_name': device.name, 'name': cmd.name,
                           'capability': cmd.capability, 'unit': cmd.unit,
                           'source': device.source, 'observed_at': None,
                           'status': 'ok' if 'value' in p else 'unavailable'}
                    if 'value' in p:
                        row.update(value=p['value'], retrieved_at=p['retrieved_at'])
                    readings.append(row)
                return {'success': any(r['status'] == 'ok' for r in readings),
                        'status': 'read', 'readings': readings}
        finally:
            with self.lock:
                self.pending = {}
            self.read_lock.release()


def create_app(bridge):
    from contextlib import asynccontextmanager, suppress
    from fastapi import FastAPI, Request, HTTPException
    @asynccontextmanager
    async def lifespan(app):
        bridge.start()
        async def persist():
            import logging
            while True:
                await asyncio.sleep(2)
                try:
                    await asyncio.to_thread(bridge.save_catalog)
                except OSError:
                    logging.getLogger(__name__).exception('Could not persist Jeedom catalog')
        flush = asyncio.create_task(persist())
        try:
            yield
        finally:
            flush.cancel()
            with suppress(asyncio.CancelledError):
                await flush
            bridge.stop()
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    @app.get('/catalog')
    async def catalog(request: Request):
        return bridge.catalog(request.query_params.get('version'))
    @app.post('/read')
    async def read(request: Request):
        body = await request.body()
        if len(body) > 1024:
            raise HTTPException(413)
        try:
            ids = json.loads(body).get('cmd_ids')
        except (ValueError, AttributeError):
            raise HTTPException(400)
        return await asyncio.to_thread(bridge.read, ids)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=1883)
    parser.add_argument('--root', default='jeedom')
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    import uvicorn
    bridge = JeedomReadBridge(args.cache, host=args.host, port=args.port, root=args.root)
    uvicorn.run(create_app(bridge), uds=args.socket, log_level='info')


if __name__ == '__main__':
    main()
