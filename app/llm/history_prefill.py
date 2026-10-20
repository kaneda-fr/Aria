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


def schedule_prefill(state, *, session_id, messages, tools, audio_ready, request_id, rollover=False, server_read=False):
    """Prepare exact history after audio; read turns never consume generated output.

    Preserve the exact next-turn prefix (including tools and bounded history).
    The empty user message is cache scaffolding only; no result is persisted.
    Do not wake a cold model or compete with pending work/TTS synthesis.
    """
    if (os.environ.get('ARIA_HISTORY_PREFILL', '1') == '0'
            or not isinstance(state.llm_config, OllamaConfig)
            or not isinstance(audio_ready, asyncio.Future)
            or not session_id
            or (not server_read and (
                getattr(state, 'llm_recent_session', None) != session_id
                or time.monotonic() - getattr(state, 'llm_recent_at', 0) > 60))
            or (server_read and getattr(state, 'llm_client', None) is None)
            or not state.llm_queue.pending.empty()):
        return
    prior = getattr(state, 'history_prefill_task', None)
    if prior is not None and not prior.done():
        return
    snapshot = copy.deepcopy(messages) + [{'role': 'user', 'content': ''}]
    config = replace(state.llm_config, num_predict=1, stream=False)
    tool_snapshot = copy.deepcopy(tools)
    budget = 20 if rollover or server_read else 4
    reason = 'server_read' if server_read else 'history_eviction' if rollover else 'local_answer'
    stages = [snapshot]
    if rollover or server_read:
        # Grow the exact prefix a few retained messages at a time. A single large
        # Ollama prompt can keep running after HTTP cancellation; smaller calls
        # bound that residual work without changing the next real request.
        history_count = max(0, len(snapshot) - 2)
        stride = 2 * max(1, (history_count + 11) // 12)
        ends = [*range(1 + stride, len(snapshot) - 1, stride), len(snapshot) - 1]
        stages = [snapshot[:end] + snapshot[-1:] for end in ends]

    async def run():
        started = None
        outcome = 'skipped'
        completed_steps = 0
        token = PREFILL.set(True)
        try:
            ready = await asyncio.wait_for(asyncio.shield(audio_ready), timeout=30)
            if not ready or not state.llm_queue.pending.empty():
                return
            if server_read:
                # A brief grace period avoids starting inference for an immediate
                # follow-up. /api/ps checks residency without loading a model.
                await asyncio.sleep(0.15)
                response = await state.llm_client.get(
                    config.url.rstrip('/') + '/api/ps', timeout=1.0)
                response.raise_for_status()
                canonical = lambda name: name if ':' in name.rsplit('/', 1)[-1] else name + ':latest'
                resident = any(canonical(str(m.get('name') or m.get('model') or '')) ==
                               canonical(config.model) for m in response.json().get('models', []))
                if not resident or not state.llm_queue.pending.empty():
                    return
            started = time.monotonic()
            log.info('ARIA.HistoryPrefill.Start', extra={'fields': {
                'request_id': request_id, 'reason': reason, 'budget_s': budget,
                'steps_planned': len(stages)}})
            # Cancellation closes the current request. Target timings must still
            # include the backend's remaining work for that individual stage.
            async with asyncio.timeout(budget):
                for stage in stages:
                    if not state.llm_queue.pending.empty():
                        outcome = 'superseded'
                        return
                    await generate_llm_response(
                        text='', messages=stage, config=config,
                        client=state.llm_client, tools=tool_snapshot)
                    completed_steps += 1
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
                'request_id': request_id, 'outcome': outcome, 'reason': reason,
                'steps_completed': completed_steps,
                'seconds': time.monotonic() - started if started is not None else 0}})

    state.history_prefill_task = asyncio.create_task(run())
