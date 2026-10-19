"""Bounded, optional acknowledgment for a LAN output player."""
import asyncio
import math
import time
from uuid import uuid4


class PlaybackBroker:
    """One player lease and one clip; no historical audio replay or retry."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.player = None
        self.deadline = 0.
        self.pending = {}

    def register(self):
        if self.player and self.deadline > self.clock():
            raise ValueError('player_already_connected')
        if self.player:
            self.unregister(self.player)
        self.player = uuid4().hex
        self.deadline = self.clock() + 15
        return self.player

    def heartbeat(self, player):
        if player != self.player or self.deadline <= self.clock():
            raise ValueError('player_lease_expired')
        self.deadline = self.clock() + 15

    def unregister(self, player):
        if player != self.player:
            raise ValueError('unknown_player')
        self.player, self.deadline = None, 0.
        for row in self.pending.values():
            if not row['future'].done():
                row['future'].set_exception(RuntimeError('player_disconnected'))

    def begin(self, duration):
        if not self.player or self.deadline <= self.clock():
            return None
        if self.pending:
            raise ValueError('playback_already_pending')
        if not math.isfinite(duration) or not 0 < duration <= 300:
            raise ValueError('invalid_audio_duration')
        clip = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        row = {'future': future, 'player': self.player, 'started': False}
        self.pending[clip] = row
        async def wait():
            finish_deadline = self.clock() + duration + 15
            try:
                while not future.done():
                    now = self.clock()
                    if self.player != row['player'] or self.deadline <= now:
                        raise RuntimeError('player_lease_expired')
                    remaining = min(finish_deadline, self.deadline) - now
                    if remaining <= 0:
                        raise asyncio.TimeoutError('playback_timeout')
                    # Do not cancel the completion future at each lease boundary:
                    # a heartbeat may have renewed the lease while this waited.
                    await asyncio.wait({future}, timeout=remaining)
                await future
            finally:
                self.pending.pop(clip, None)
        task = asyncio.create_task(wait())
        def release(_):
            # Also runs when cancellation happens before wait() starts.
            self.pending.pop(clip, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()  # Consume a failure even if wait() never ran.
        task.add_done_callback(release)
        return clip, task

    def ack(self, player, clip, status):
        self.heartbeat(player)
        row = self.pending.get(clip)
        if not row or row['player'] != player or row['future'].done():
            raise ValueError('unknown_playback')
        if status == 'started':
            row['started'] = True
        elif status == 'finished' and row['started']:
            row['future'].set_result(None)
        elif status == 'failed':
            row['future'].set_exception(RuntimeError('player_failed'))
        else:
            raise ValueError('invalid_playback_transition')
