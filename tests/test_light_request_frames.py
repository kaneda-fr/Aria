"""Conversational read framing is reusable; fuzzy repair never selects a device."""
import pytest

from test_shared_requests import server
from test_light_questions import lighting, lights_server, ask
from app.plugins.inventory.light_queries import resolve_light_query


@pytest.mark.asyncio
@pytest.mark.parametrize('text,language', [
    ('Donne-moi le statut des lumières.', 'fr'),
    ('Donnez-nous l’état des lumières.', 'fr'),
    ('Dis-moi quelles lumières sont allumées.', 'fr'),
    ('Indique-moi le statut de mon éclairage.', 'fr'),
    ('Peux-tu me donner le statut des lumières ?', 'fr'),
    ('Pourriez-vous m’indiquer l’état des lumières ?', 'fr'),
    ('Montre-moi l’état des lumières.', 'fr'),
    ('Give me the status of my lights.', 'en'),
    ('Show me which lights are on.', 'en'),
    ('Tell me the state of the lights.', 'en'),
    ('Could you now give me the status of my lights?', 'en'),
    ('I would like to know the status of my lights.', 'en'),
])
async def test_read_framing_preserves_state_intent(lights_server, text, language):
    result = await ask(lights_server, text)
    assert result['success']
    assert lights_server[3] == [['101', '102', '301', '401', '701', '801']]
    assert lights_server[1].speech_output.speak.await_args.kwargs['language'] == language
    lights_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    'donde moi le statut des lumières.',
    'done moi le statut des lumières.',
    'donen moi le statut des lumières.',
    'donde moi la statue des lumières.',
])
async def test_spoken_read_frame_repair_keeps_raw_history(lights_server, text):
    await ask(lights_server, text, voice=True)
    assert lights_server[3] == [['101', '102', '301', '401', '701', '801']]
    assert lights_server[1].llm_history['lights'][-2]['content'] == '[speaker=person] ' + text
    lights_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    'donde moi le statut des lumières.',  # HTTP spelling is not an ASR correction.
    'Cache-moi le statut des lumières.',
    'Donne-moi pas le statut des lumières.',
    'Give me no light states.',
    'Do not give me the status of my lights.',
    'Donne-moi le statut des lumières dans la pièce introuvable.',
    'Give me the status of lights in the unknown room.',
])
async def test_read_frames_do_not_hide_negation_or_unknown_scope(lights_server, text):
    await ask(lights_server, text)
    assert not lights_server[3]


@pytest.mark.parametrize('name,text', [
    ('Donde Moi', 'Donde Moi le statut de la lumière'),
    ('Give Me Light', 'Give Me Light status'),
])
def test_declared_names_outrank_request_frame_recovery(lighting, name, text):
    provider, calls, _ = lighting
    snapshot = provider.indexer.get_snapshot()
    snapshot.devices['3'].name = name
    query = resolve_light_query(text, snapshot, spoken=True)
    if query is None:
        from app.plugins.inventory.read_requests import resolve_read_request
        native = resolve_read_request(text, snapshot)
        assert native is not None and native.device_id == '3'
    else:
        assert {eq for eq, _, _ in query.targets} == {'3'}
        assert not query.speech_recovered
    assert not calls

@pytest.mark.parametrize('text', [
    'Statut du volet du bureau.',
    'Give me the status of shutters in the office.',
    'Show me the state of the kitchen shutters.',
])
def test_other_explicit_device_types_keep_their_own_handler(lighting, text):
    provider, calls, _ = lighting
    assert resolve_light_query(text, provider.indexer.get_snapshot()) is None
    assert not calls

@pytest.mark.asyncio
@pytest.mark.parametrize('text,language', [
    ('Montre-moi l’état de la lumière bureau.', 'fr'),
    ('Pourriez-vous m’indiquer le statut de la lumière bureau ?', 'fr'),
    ('Show me the state of the office light.', 'en'),
    ('Could you now give me the status of the office light?', 'en'),
])
async def test_framing_a_single_device_cannot_delegate_a_different_intent(lights_server, text, language):
    await ask(lights_server, text)
    assert lights_server[3] == [['101', '102']]
    assert lights_server[1].speech_output.speak.await_args.kwargs['language'] == language
    lights_server[5].assert_not_called()
