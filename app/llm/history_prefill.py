"""Opportunistic cache work, never a conversation turn or a tool dispatcher."""
import asyncio
import contextlib
import copy
from contextvars import ContextVar
from dataclasses import replace
import logging
import os
import time

from app.llm.ollama_client import OllamaConfig, generate_llm_response

PREFILL = ContextVar('history_prefill', default=False)
log = logging.getLogger('ARIA.history_prefill')


async def cancel_prefill(state):
    """New admitted work and shutdown take priority over speculative cache work."""
    task = getattr(state, 'history_prefill_task', None)
    state.history_prefill_task = None
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def schedule_prefill(state, *, session_id, messages, tools, audio_ready, request_id):
    """At most one four-second prefill, only after all speech chunks are ready.

    Preserve the exact next-turn prefix (including tools and bounded history).
    The empty user message is cache scaffolding only; no result is persisted.
    Do not wake a cold model or compete with pending work/TTS synthesis.
    """
    if (os.environ.get('ARIA_HISTORY_PREFILL', '1') == '0'
            or not isinstance(state.llm_config, OllamaConfig)
            or not isinstance(audio_ready, asyncio.Future)
            or not session_id
            or getattr(state, 'llm_recent_session', None) != session_id
            or time.monotonic() - getattr(state, 'llm_recent_at', 0) > 60
            or not state.llm_queue.pending.empty()):
        return
    prior = getattr(state, 'history_prefill_task', None)
    if prior is not None and not prior.done():
        return
    snapshot = copy.deepcopy(messages) + [{'role': 'user', 'content': ''}]
    config = replace(state.llm_config, num_predict=1, stream=False)
    tool_snapshot = copy.deepcopy(tools)

    async def run():
        started = None
        outcome = 'skipped'
        token = PREFILL.set(True)
        try:
            ready = await asyncio.wait_for(asyncio.shield(audio_ready), timeout=30)
            if not ready or not state.llm_queue.pending.empty():
                return
            started = time.monotonic()
            log.info('ARIA.HistoryPrefill.Start', extra={'fields': {'request_id': request_id}})
            # Closing the HTTP request on cancellation also cancels Ollama work.
            # Target measurements must include any remaining backend batch time.
            await asyncio.wait_for(generate_llm_response(
                text='', messages=snapshot, config=config,
                client=state.llm_client, tools=tool_snapshot), timeout=4)
            outcome = 'completed'
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        except TimeoutError:
            outcome = 'timeout'
        except Exception:
            outcome = 'failed'
        finally:
            PREFILL.reset(token)
            log.info('ARIA.HistoryPrefill.Done', extra={'fields': {
                'request_id': request_id, 'outcome': outcome,
                'seconds': time.monotonic() - started if started is not None else 0}})

    state.history_prefill_task = asyncio.create_task(run())
