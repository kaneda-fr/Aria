"""Query operators do not implicitly select homonymous equipment."""
from dataclasses import replace
import pytest
from test_shared_requests import server
from test_light_questions import lighting, lights_server, ask
from test_cover_status import cover_server

@pytest.mark.asyncio
@pytest.mark.parametrize('name,text',[
    ('On','Which lights are on?'),('Off','Which lights are off?'),
    ('Allumées','Quelles lumières sont allumées ?'),('Éteintes','Quelles lumières sont éteintes ?'),
    ('On','List the lights that are on.'),('Status','What is the status of my lights?'),
    ('État','Quel est l’état des lumières ?'),('How Many','How many lights are on?'),
    ('Combien','Combien de lumières sont allumées ?'),('All','Are all my lights off?'),
    ('Any','Are any lights on?'),('Still','Which lights are still on?'),
    ('Count','Count the lights that are on.'),
])
async def test_generic_light_query_never_binds_an_operator_name(lights_server,name,text):
    provider=lights_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('3'),name=name))
    result=await ask(lights_server,text)
    assert result['success'] and lights_server[3]==[['101','102','301','401','701','801']]
    lights_server[5].assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
@pytest.mark.parametrize('known',[False,True])
async def test_light_state_named_room_is_literal(lights_server,language,known):
    provider=lights_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('3'),name='On'))
    if known:provider.indexer.add_device(replace(provider.indexer.get_device('4'),room='On'))
    text=('Quel est le statut des lumières dans la pièce On ?' if language=='fr' else
          'What is the status of the lights in the room On?')
    result=await ask(lights_server,text)
    assert result['success'] and lights_server[3]==([['401']] if known else [])
    lights_server[5].assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
async def test_light_operator_followup_preserves_scope(lights_server,language):
    provider=lights_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('3'),name='On' if language=='en' else 'Allumées'))
    await ask(lights_server,'Quel est le statut des lumières ?' if language=='fr' else
              'What is the status of my lights?')
    result=await ask(lights_server,'Et sont-elles allumées ?' if language=='fr' else 'And which are on?')
    assert result['success'] and lights_server[3]==[['101','102','301','401','701','801']]*2
    lights_server[5].assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
@pytest.mark.parametrize('name',['On','Off','How Many','All'])
async def test_explicit_light_name_is_not_a_count_or_predicate(lights_server,language,name):
    provider=lights_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('3'),name=name))
    text=('Quel est le statut de la lumière nommée '+name+' ?' if language=='fr' else
          'What is the status of the light named '+name+'?')
    result=await ask(lights_server,text)
    assert result['success'] and lights_server[3]==[['301']]
    assert name in result['response']
    lights_server[5].assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize('name,text',[
    ('All','Are all my shutters closed?'),('Combien','Combien de volets sont fermés ?'),
    ('Count','Count closed shutters'),
])
async def test_generic_cover_operator_names_do_not_limit_scope(cover_server,name,text):
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name=name))
    result=await ask(cover_server,text)
    assert result['success'] and cover_server[3]==[['901','902','903']]
    cover_server[5].assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize('name',['All','How Many'])
async def test_explicit_cover_name_does_not_enable_count_or_quantifier(cover_server,name):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name=name))
    text='What is the status of the shutter named '+name+'?'
    query=resolve_cover_query(text,provider.indexer.get_snapshot(),language='en')
    assert query and query.status=='ready' and query.wanted is None
    assert query.quantifier is None and not query.count
    result=await ask(cover_server,text)
    assert result['success'] and cover_server[3]==[['901']]
    cover_server[5].assert_not_called()
