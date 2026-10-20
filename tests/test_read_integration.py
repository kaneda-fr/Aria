"""Clarification, deterministic reads and speculative cache share one request path."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from test_read_clarification import reader
from test_shared_requests import server


@pytest.mark.asyncio
@pytest.mark.parametrize('question,choice,followup', [
    ('Quel est l’état dans Atelier ?', 'La lampe.', 'Et son état ?'),
    ('What is the state in Atelier?', 'The lamp.', 'What is its state?'),
])
async def test_full_clarification_then_reference_preserves_history_and_prefill(
        server, reader, monkeypatch, question, choice, followup):
    import app.main as runtime
    import app.llm.history_prefill as prefill
    client, state = server
    provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    model = AsyncMock(side_effect=AssertionError('No foreground model for reads'))
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    cache = Mock()
    monkeypatch.setattr(prefill, 'schedule_prefill', cache)
    state.speech_output.audio_ready = asyncio.get_running_loop().create_future()
    state.speech_output.audio_ready.set_result(True)
    replies = []
    for text in (question, choice, followup):
        result = (await client.post('/api/llm/process', json={
            'text': text, 'session_id': 'integration', 'speak_response': True,
        })).raise_for_status().json()
        assert result['success']
        replies.append(result['response'])
    assert 'Lampe Atelier' in replies[0] and 'Switch Atelier' in replies[0]
    assert reads == [['101', '102'], ['101', '102']]
    assert all('Lampe Atelier' in reply for reply in replies[1:])
    model.assert_not_awaited()
    assert len(cache.call_args_list) == 3
    assert all(call.kwargs['server_read'] for call in cache.call_args_list)
    assert state.llm_history['integration'][-1]['content'] == replies[-1]
    assert cache.call_args.kwargs['messages'][1:] == list(state.llm_history['integration'])
    await provider.close_read_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    'Quel est l’état de la lumière Switch Atelier ?',
    'What is the state of the light Switch Atelier?',
])
async def test_direct_read_cannot_bypass_native_type_conflict(server, reader, monkeypatch, text):
    import app.main as runtime
    client, state = server
    provider, reads = reader
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    model = AsyncMock(side_effect=AssertionError('No model needed to reject a conflict'))
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    result = (await client.post('/api/llm/process', json={
        'text': text, 'session_id': 'conflict', 'speak_response': False,
    })).raise_for_status().json()
    assert result['lookup_status'] == 'not_found'
    assert not reads
    model.assert_not_awaited()
    await provider.close_read_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize('text,language,fragment', [
    ('Status of Lampe Canapé', 'en', 'Jeedom reports:'),
    ('State of Lampe Canapé', 'en', 'Jeedom reports:'),
    ('What is the state of Lampe Canapé?', 'en', 'Jeedom reports:'),
    ('Statut de Lampe Canapé', 'fr', 'Jeedom indique :'),
    ('État de Lampe Canapé', 'fr', 'Jeedom indique :'),
    ('Quel est l’état de Lampe Canapé ?', 'fr', 'Jeedom indique :'),
])
async def test_read_syntax_controls_reply_and_voice_despite_accented_name(
        server, reader, monkeypatch, text, language, fragment):
    from dataclasses import replace
    import app.main as runtime
    client, state = server
    provider, reads = reader
    device = provider.indexer.get_device('1')
    provider.indexer.add_device(replace(device, name='Lampe Canapé'))
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    model = AsyncMock(side_effect=AssertionError('Read must not generate a reply'))
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    try:
        reply = (await client.post('/api/llm/process', json={
            'text': text, 'session_id': 'read-language', 'speak_response': True,
        })).raise_for_status().json()
        assert fragment in reply['response']
        assert 'Lampe Canapé' in reply['response']
        assert reads == [['101', '102']]
        assert state.speech_output.speak.await_args.kwargs['language'] == language
        assert state.llm_history['read-language'][-1]['content'] == reply['response']
        model.assert_not_awaited()
    finally:
        await provider.close_read_connection()
