"""Held-out lighting formulations and selection/observation safety properties."""
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from test_light_questions import lighting, lights_server, ask
from test_shared_requests import server
from app.plugins.inventory.light_queries import resolve_light_query


@pytest.mark.asyncio
@pytest.mark.parametrize('text,expected',[
    ('How many lights are on?',4), ('Combien de lumières sont éteintes ?',1),
    ('Combien de lampes sont allumées ?',4), ('How many lights are off?',1),
])
async def test_count_devices_without_counting_each_channel_as_a_light(lights_server,text,expected):
    result=await ask(lights_server,text)
    assert str(expected) in result['response'],result
    assert len(lights_server[3])==1 and len(lights_server[3][0])==6


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['Are all my lights off?','Toutes mes lumières sont-elles éteintes ?'])
async def test_mixed_channels_cannot_be_reported_as_everything_off(lights_server,text):
    result=await ask(lights_server,text)
    assert result['response'].startswith(('No.','Non.'))
    assert 'part' in result['response'] or 'some of the lights' in result['response']


@pytest.mark.asyncio
async def test_all_unavailable_is_not_an_assertion_that_no_lights_are_on(lights_server):
    lights_server[4].clear()
    result=await ask(lights_server,'Are any lights still on?')
    assert not result['response'].startswith(('Yes.','No.'))
    assert 'unknown' in result['response'] or 'do not know' in result['response']


@pytest.mark.asyncio
async def test_one_known_on_channel_proves_presence_despite_other_missing_values(lights_server):
    lights_server[4].clear();lights_server[4]['101']=1
    result=await ask(lights_server,'Are any lights still on?')
    assert result['response'].startswith('Yes.'),result
    assert 'unknown' in result['response'] or 'do not know' in result['response']


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
    'Ne lis pas l’état des lumières.', 'Do not read the light states.',
    'Which lights were on yesterday?', 'Quel sera l’état des lampes demain ?',
    'Which lights are on except the kitchen light?',
])
async def test_negative_or_unsupported_scope_does_not_read(lights_server,text):
    result=await ask(lights_server,text)
    assert result['response'] and not lights_server[3]


@pytest.mark.parametrize('text',[
    'Turn on all the lights.', 'Allume les lumières.', 'Dim the couch lamp to 30 percent.',
    'State light as a wave phenomenon.', 'Why does light travel in a vacuum?',
    'Comment fonctionne la lumière ?', 'What is the state in Bureau?',
])
def test_action_science_and_untyped_room_requests_keep_their_existing_paths(lighting,text):
    assert resolve_light_query(text,lighting[0].indexer.get_snapshot()) is None


@pytest.mark.asyncio
async def test_current_scope_and_language_followup_remain_inventory_grounded(lights_server):
    await ask(lights_server,'List the office lights.')
    first=await ask(lights_server,'And their state?')
    second=await ask(lights_server,'Et dans la cuisine ?')
    assert lights_server[3]==[['101','102'],['401']]
    voices=[c.kwargs['language'] for c in lights_server[1].speech_output.speak.await_args_list]
    assert voices==['en','en','fr']


@pytest.mark.asyncio
async def test_duplicate_choice_by_ordinal_matches_only_the_offered_light(lights_server):
    await ask(lights_server,'Is the garden lamp on?')
    result=await ask(lights_server,'The second, please.')
    assert lights_server[3]==[['801']],result


def test_home_scope_is_not_a_device_name_match(lighting):
    provider=lighting[0];d=provider.indexer.get_device('3')
    provider.indexer.add_device(replace(d,eq_id='9',name='Home',commands=[]))
    query=resolve_light_query('List the lights at home.',provider.indexer.get_snapshot(),language='en')
    assert query is not None and len(query.targets)==6


@pytest.mark.asyncio
async def test_multiple_partial_names_cannot_select_one_room_silently(lights_server):
    result=await ask(lights_server,'Quel est l’état des lumières du bureau et de la cuisine ?')
    assert result['response'].rstrip().endswith('?') and not lights_server[3]


@pytest.mark.asyncio
async def test_explicit_unknown_channel_never_reads_other_channels(lights_server):
    result=await ask(lights_server,'Status of the office light channel 9.')
    assert not lights_server[3]
    assert 'ne connais pas' in result['response'].lower() or 'do not know' in result['response'].lower()


@pytest.mark.asyncio
async def test_group_room_followup_retains_plural_scope(lights_server):
    await ask(lights_server, 'Quel est l’état des lumières ?')
    result = await ask(lights_server, 'Et dans RDC ?')
    assert lights_server[3][-1] == ['101', '102', '401'], result


@pytest.mark.asyncio
async def test_ordinal_keeps_the_requested_channel(lights_server):
    provider = lights_server[2]
    for eq in ('7', '8'):
        device = provider.indexer.get_device(eq)
        first = replace(device.commands[0], name='Etat-1')
        second = replace(first, cmd_id=eq+'02', name='Etat-2')
        provider.indexer.add_device(replace(device, commands=[first, second]))
        lights_server[4][second.cmd_id] = 1
    await ask(lights_server, 'Is the garden lamp on, channel 2?')
    result = await ask(lights_server, 'The second, please.')
    assert lights_server[3] == [['802']], result


@pytest.mark.asyncio
async def test_spoken_homophone_context_survives_text_transport_followup(lights_server):
    await ask(lights_server, 'statue des lumières à la maison.', voice=True)
    result = await ask(lights_server, 'Et dans la cuisine ?')
    assert lights_server[3][-1] == ['401'], result


@pytest.mark.asyncio
async def test_off_question_includes_the_off_channel_of_a_mixed_light(lights_server):
    for cid in lights_server[4]:
        lights_server[4][cid] = 1
    lights_server[4]['101'] = 0
    result = await ask(lights_server, 'Are any lights off?')
    assert result['response'].startswith('Yes.'), result
    assert 'Bureau' in result['response'] and 'some of the lights are off' in result['response']
    assert 'channel' not in result['response']
