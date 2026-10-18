"""Bounded, single-worker admission shared by text and finalized voice requests."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Awaitable, Callable
from uuid import uuid4


@dataclass
class AssistantRequest:
    text: str
    session_id: str | None
    speaker_user: str
    speak_response: bool
    use_tools: bool
    source: str
    notify: Callable[[dict], Awaitable[None]] | None = None
    conversation_text: str | None = None  # Internal normalized request; text stays verbatim.
    conversation_started_at: float | None = None  # Server-observed voice onset; not queue age.
    request_id: str = field(default_factory=lambda: uuid4().hex)
    queued_at: float = field(default_factory=time.monotonic)
    status: str = 'queued'
    result: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())

    def status_message(self, status: str | None = None, error: str | None = None) -> dict:
        return {'type': 'request_status', 'request_id': self.request_id,
                'session_id': self.session_id, 'status': status or self.status, 'error': error}


class RequestQueue:
    """One running request plus bounded FIFO pending work; never evict accepted work.

    Expiry is checked before execution. Cancelling a caller cancels queued work only;
    a running action is allowed to finish and is never automatically retried.
    """

    def __init__(self, capacity: int = 1, max_wait_s: float = 30,
                 on_event: Callable[[dict], None] | None = None):
        if not 1 <= capacity <= 16 or not 0 < max_wait_s <= 300:
            raise ValueError('Request queue requires capacity 1..16 and wait limit (0, 300] seconds')
        self.pending: asyncio.Queue[AssistantRequest] = asyncio.Queue(maxsize=capacity)
        self.max_wait_s = max_wait_s
        self.on_event = on_event or (lambda _: None)
        self.closing = False

    def event(self, request: AssistantRequest, status: str, error: str | None = None) -> None:
        self.on_event({**request.status_message(status, error), 'source': request.source,
                       'queue_pending': self.pending.qsize()})

    def submit(self, **kwargs: Any) -> AssistantRequest:
        request = AssistantRequest(**kwargs)
        if self.closing:
            self.finish(request, 'rejected', error='shutdown')
        else:
            try:
                self.pending.put_nowait(request)
                self.event(request, 'accepted')
            except asyncio.QueueFull:
                self.finish(request, 'rejected', error='busy')
        return request

    def finish(self, request: AssistantRequest, status: str, result: dict | None = None,
               error: str | None = None) -> None:
        if request.result.done():
            return
        request.status = status
        response = {'success': False, 'text': request.text, 'response': '',
                    'spoken': False, 'tool_calls_made': 0, 'error': error, **(result or {}),
                    'request_id': request.request_id, 'status': status}
        request.result.set_result(response)
        self.event(request, status, response.get('error'))

    def cancel_pending(self, request: AssistantRequest) -> None:
        if request.status == 'queued':
            self.finish(request, 'cancelled', error='caller_cancelled')

    async def notify(self, request: AssistantRequest) -> None:
        if request.notify is not None:
            try:
                async with asyncio.timeout(.2):
                    await request.notify(request.status_message(
                        error=request.result.result().get('error')))
            except Exception:
                self.event(request, 'notification_unavailable')

    async def run(self, processor: Callable[[AssistantRequest], Awaitable[dict]]) -> None:
        try:
            while not self.closing and not asyncio.current_task().cancelling():
                request = await self.pending.get()
                try:
                    if request.result.done():
                        continue
                    if time.monotonic() - request.queued_at > self.max_wait_s:
                        self.finish(request, 'expired', error='queue_expired')
                        continue
                    request.status = 'running'
                    self.event(request, 'running')
                    result = await processor(request)
                    self.finish(request, 'completed' if result.get('success') else 'failed', result)
                except asyncio.CancelledError:
                    self.finish(request, 'cancelled', error='execution_interrupted')
                    raise
                except Exception as exc:
                    self.finish(request, 'failed', error=str(exc))
                finally:
                    self.pending.task_done()
                    await self.notify(request)
        finally:
            self.closing = True
            while not self.pending.empty():
                request = self.pending.get_nowait()
                self.finish(request, 'cancelled', error='shutdown')
                self.pending.task_done()
                await self.notify(request)
