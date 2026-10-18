"""Stable client names bind connections to bounded, in-memory conversations.

Names are caller assertions, not authentication. No audio or request is replayed.
"""
import re
import time
from collections import OrderedDict


def validate_client_id(payload):
    if 'client_id' not in payload:
        return None
    name = payload['client_id']
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', name):
        raise ValueError('client_id must be a unique lowercase name of 1..64 characters')
    return name


class ClientSessions:
    def __init__(self, limit=256):
        self.limit = limit
        self.active = {}
        self.known = OrderedDict()
        self.requests = {}

    def claim(self, name, connection_id):
        if name in self.active:
            raise ValueError('client_id_in_use')
        if len(self.active) >= self.limit:
            raise ValueError('client_limit_reached')
        if name not in self.known and len(self.known) >= self.limit:
            evict = next((n for n in self.known if n not in self.active and not self.is_muted(n) and
                          not any(r[0] == n for r in self.requests.values())), None)
            if evict is None:
                raise ValueError('client_limit_reached')
            self.known.pop(evict)
        self.active[name] = connection_id
        self.known[name] = {**self.known.get(name, {}), 'client_id': name, 'session_id': 'client:' + name,
                            'last_audio': 0., 'last_seen': time.time(), '_hidden': False}
        self.known.move_to_end(name)
        # Separate the stable conversation key from the per-socket connection ID.
        return 'client:' + name

    def release(self, name, connection_id):
        if self.active.get(name) == connection_id:
            self.active.pop(name)
            self.known[name]['last_seen'] = time.time()

    def audio(self, name, connection_id):
        if name is not None and self.active.get(name) == connection_id:
            self.known[name].update(last_audio=time.monotonic(), last_seen=time.time())

    def request(self, event):
        name = (event.get('session_id') or '').removeprefix('client:')
        if name not in self.known or event.get('session_id') != 'client:' + name:
            return
        rid, status = event['request_id'], event['status']
        if status in {'accepted', 'running'}:
            self.requests[rid] = (name, status)
        elif status in {'completed', 'failed', 'cancelled', 'expired', 'rejected'}:
            self.requests.pop(rid, None)

    def is_muted(self, name):
        return self.known.get(name, {}).get('_mute_deadline', 0) > time.monotonic()

    def mute_revision(self, name):
        return self.known.get(name, {}).get('_mute_revision', 0)

    def set_muted(self, name, muted, *, seconds=300):
        if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 86400:
            raise ValueError('Mute duration must be 1..86400 seconds')
        row = self.known[name]  # Unknown names cannot create arbitrary registry entries.
        if self.is_muted(name) != muted:
            row['_mute_deadline'] = time.monotonic() + seconds if muted else 0
            row['_mute_revision'] = self.mute_revision(name) + 1
        return self.mute_status(name)

    def mute_status(self, name):
        remaining = max(0., self.known[name].get('_mute_deadline', 0) - time.monotonic())
        return {'muted': remaining > 0, 'mute_remaining_s': remaining}

    def clear_offline(self):
        """Hide disconnected rows until reconnect, preserving mute/session state."""
        cleared = []
        for name, row in self.known.items():
            if name not in self.active and not row.get('_hidden'):
                row['_hidden'] = True
                cleared.append(name)
        return cleared

    def snapshot(self):
        now = time.monotonic()
        return [{**{k: v for k, v in row.items() if k != 'last_audio' and not k.startswith('_')},
                 **self.mute_status(name),
                 'connected': name in self.active,
                 'transmitting': name in self.active and now - row['last_audio'] < 1,
                 'processing': any(n == name and s == 'running' for n, s in self.requests.values()),
                 'queued': any(n == name and s == 'accepted' for n, s in self.requests.values())}
                for name, row in self.known.items() if not row.get('_hidden')]
