"""Lighting questions use typed inventory and real read contracts, never action search."""
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

import app.main as runtime
from app.plugins.inventory.provider import InventoryProvider
from app.plugins.inventory.state_read import ReadOnlyInventory
from test_shared_requests import server


@pytest.fixture
def lighting(tmp_path):
    def device(eq,name,room,domain,commands,enabled=True):
        return dict(source='jeedom_mqtt',eq_id=eq,name=name,room=room,domain=domain,
                    enabled=enabled,visible=True,seen_at=0,tags=[],commands=[
                        dict(cmd_id=cid,device_id=eq,name=label,type=kind,subType=sub,
                             capability=cap,unit=None,tags=[],visible=True)
                        for cid,label,kind,sub,cap in commands])
    def state(cid,label='Etat',sub='binary',cap='light_state'):
        return (cid,label,'info',sub,cap)
    def action(cid,cap):return (cid,cap,'action','other',cap)
    devices=[
        device('1','Lumière Bureau','RDC','light',[state('101','Etat-1'),state('102','Etat-2'),action('111','light_on'),action('112','light_off')]),
        device('2','Switch Bureau','','generic',[state('201',cap='generic_info')]),
        device('3','Lampe Canapé','Séjour','light',[state('301'),action('311','light_on'),action('312','light_off')]),
        device('4','Plafonnier Cuisine','RDC','light',[state('401',sub='numeric'),action('411','light_slider')]),
        device('5','Lampe Hors Service','RDC','light',[state('501')],False),
        device('6','Capteur Cuisine','RDC','weather',[state('601',cap='temperature')]),
        device('7','Lampe Jardin','Terrasse Nord','light',[state('701')]),
        device('8','Lampe Jardin','Terrasse Sud','light',[state('801')]),
    ]
    path=tmp_path/'lights.json';path.write_text(json.dumps({'devices':devices}))
    provider=ReadOnlyInventory('inventory',{'cache_path':str(path),'sources':{}})
    InventoryProvider.start(provider)
    provider.apply_catalog({**provider.get_snapshot(),'connected':True,'subscribed':True})
    calls=[];values={'101':0,'102':1,'201':1,'301':1,'401':25,'501':1,'601':18,'701':0,'801':1}
    def handler(request):
        if request.url.path=='/catalog':return httpx.Response(200,json={'unchanged':True,'connected':True,'subscribed':True,'catalog_version':'fixture'})
        assert request.url.path=='/read'
        ids=json.loads(request.content)['cmd_ids'];assert 0<len(ids)<=8
        assert all(provider.indexer.get_command(cid).type=='info' for cid in ids)
        calls.append(ids)
        rows=[]
        for cid in ids:
            cmd=provider.indexer.get_command(cid);dev=provider.indexer.get_device(cmd.device_id)
            row=dict(cmd_id=cid,device_id=dev.eq_id,device_name=dev.name,name=cmd.name,
                     status='ok' if cid in values else 'unavailable',observed_at=None,retrieved_at=100)
            if cid in values:row['value']=values[cid]
            rows.append(row)
        return httpx.Response(200,json={'success':True,'status':'read','readings':rows})
    provider.read_client=httpx.AsyncClient(transport=httpx.MockTransport(handler),base_url='http://fixture')
    return provider,calls,values


@pytest_asyncio.fixture
async def lights_server(server,lighting,monkeypatch):
    client,state=server;provider,calls,values=lighting
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES','6')
    state.llm_config=state.llm_client=None
    state.plugin_registry=SimpleNamespace(get_plugins_by_type=lambda _:[provider],get_plugin=lambda _:None)
    model=AsyncMock(side_effect=AssertionError('lighting question reached generation'))
    monkeypatch.setattr(runtime,'generate_llm_response',model)
    yield client,state,provider,calls,values,model
    await provider.close_read_connection()


async def ask(env,text,session='lights',speaker='person',voice=False):
    if voice:
        return await runtime._enqueue_final_llm(runtime.app,text,session,speaker).result
    response=await env[0].post('/api/llm/process',json={'text':text,'session_id':session,'speaker_user':speaker,'speak_response':True})
    assert response.status_code==200,response.text
    return response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
    'Quel est le statut des lumières à la maison ?', 'État de mes lampes.',
    'Mes lumières sont-elles allumées ?', 'Est-ce qu’il reste des lumières allumées ?',
    'Quelles lumières sont éteintes ?', 'Y a-t-il des lumières encore allumées ?',
    'What is the status of my lights?', 'Read all the light states in the house.',
    'Are any lights still on?', 'Which lights are off?', 'Are all my lights off?',
])
async def test_household_questions_read_only_enabled_light_states(lights_server,text):
    result=await ask(lights_server,text)
    assert result['success'],result
    assert lights_server[3]==[['101','102','301','401','701','801']]
    assert 'Switch Bureau' not in result['response'] and 'Capteur Cuisine' not in result['response']
    assert 'Hors Service' not in result['response']
    lights_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
    'Liste des lumières à la maison.', 'Quelles sont mes lampes ?',
    'Montre les éclairages disponibles.', 'Combien de lumières ai-je ?',
    'List my lights.', 'What lights do I have?', 'How many lights are available?',
])
async def test_light_enumeration_needs_no_state_read(lights_server,text):
    result=await ask(lights_server,text)
    assert result['success'] and lights_server[3]==[]
    assert '5' in result['response'],result
    assert 'Bureau' in result['response'] or '5' in result['response']
    assert 'Switch' not in result['response']


@pytest.mark.asyncio
@pytest.mark.parametrize('text,wanted',[
    ('Liste les lumières allumées.', 'on'),
    ('Liste les lumières éteintes.', 'off'),
    ('List the lights that are on.', 'on'),
    ('List the lights that are off.', 'off'),
    ('Liste le statut des lumières.', None),
    ('List the state of my lights.', None),
])
async def test_list_framing_with_state_predicate_reads_and_filters(lights_server,text,wanted):
    from app.plugins.inventory.light_queries import resolve_light_query
    provider=lights_server[2]
    query=resolve_light_query(text,provider.indexer.get_snapshot())
    assert query is not None and query.kind=='light_state'
    assert query.wanted==wanted
    result=await ask(lights_server,text)
    assert result['success'],result
    assert lights_server[3]==[['101','102','301','401','701','801']]
    assert 'available' not in result['response'] and 'équipements disponibles' not in result['response']
    if wanted=='off':assert 'Lampe Canapé' not in result['response']
    else:assert 'Lampe Canapé' in result['response']
    lights_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
    'List the lights that are on in the nonexistent room.',
    'List the lights in That that are on.',
    'Liste les lumières éteintes dans la pièce inconnue.',
])
async def test_state_lists_preserve_unknown_location_qualifiers(lights_server,text):
    result=await ask(lights_server,text)
    assert result['response'] and not lights_server[3]
    lights_server[5].assert_not_called()


@pytest.mark.parametrize('name,text,predicate',[
    ('On','List the lights named On.','off'),
    ('Off','List the lights named Off.','on'),
    ('Allumée','Liste la lumière Allumée.','eteinte'),
    ('Éteinte','Liste la lumière Éteinte.','allumee'),
])
def test_list_state_words_in_equipment_names_are_not_predicates(name,text,predicate):
    from app.plugins.inventory.light_queries import resolve_light_query
    device=SimpleNamespace(eq_id='named',name=name,room='',domain='light',enabled=True,
        commands=[SimpleNamespace(cmd_id='named-state',device_id='named',type='info',capability='light_state',name='Etat')])
    snapshot=SimpleNamespace(devices={'named':device})
    query=resolve_light_query(text,snapshot)
    assert query is not None and query.kind=='light_list' and query.status=='ready'
    state_text=text.rstrip('.')+(' that are ' if name in ('On','Off') else ' qui est ')+predicate+'.'
    query=resolve_light_query(state_text,snapshot)
    assert query is not None and query.kind=='light_state' and query.status=='ready'
    assert query.wanted==('on' if predicate in ('on','allumee') else 'off')


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
    'Quel est le statut des lumières du bureau ?', 'Les lumières du bureau sont-elles allumées ?',
    'Le bureau est allumé.', 'Is the office light on?', 'Status of lights in RDC.',
])
async def test_inventory_location_and_channel_values(lights_server,text):
    result=await ask(lights_server,text)
    expected=['101','102','401'] if 'RDC' in text else ['101','102']
    assert lights_server[3]==[expected],result
    assert 'une partie' in result['response'] or 'some of the lights' in result['response']
    assert not any(w in result['response'] for w in ('canal','channel','circuit'))


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['statue des lumières à la maison.','Statue lumière bureau.'])
async def test_spoken_state_homophone_keeps_raw_history(lights_server,text):
    result=await ask(lights_server,text,voice=True)
    assert result['success'] and lights_server[3]
    assert lights_server[1].llm_history['lights'][-2]['content']=='[speaker=person] '+text


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['La lumière du garage est-elle allumée ?','Which lights are on in the nonexistent room?'])
async def test_unknown_location_never_becomes_household_read(lights_server,text):
    result=await ask(lights_server,text)
    assert result['response'] and not lights_server[3]


@pytest.mark.asyncio
async def test_duplicate_name_requires_room_before_read(lights_server):
    first=await ask(lights_server,'La lampe jardin est-elle allumée ?')
    assert first['response'].rstrip().endswith('?') and not lights_server[3]
    second=await ask(lights_server,'Celle de Terrasse Sud.')
    assert lights_server[3]==[['801']],second
    assert 'Jardin' in second['response']


@pytest.mark.asyncio
async def test_list_then_scoped_followup_then_state_reference(lights_server):
    await ask(lights_server,'Liste des lumières à la maison.')
    await ask(lights_server,'Et leur état ?')
    assert lights_server[3]==[['101','102','301','401','701','801']]
    await ask(lights_server,'Et dans la cuisine ?')
    assert lights_server[3][-1]==['401']


@pytest.mark.asyncio
async def test_synonym_and_explicit_channel(lights_server):
    await ask(lights_server,'Is the couch lamp on?')
    await ask(lights_server,'Quel est l’état de la lumière bureau canal 2 ?')
    assert lights_server[3]==[['301'],['102']]


@pytest.mark.asyncio
async def test_missing_and_malformed_values_remain_unknown(lights_server):
    lights_server[4].pop('301');lights_server[4]['401']='not a binary state'
    result=await ask(lights_server,'Quel est l’état des lumières ?')
    assert 'ne connais pas' in result['response'].lower(),result
    assert 'Canapé' in result['response'] and 'Cuisine' in result['response']


@pytest.mark.asyncio
async def test_followup_cannot_inherit_other_speaker_or_session(lights_server):
    await ask(lights_server,'Liste des lumières du bureau.')
    lights_server[1].llm_config=SimpleNamespace();lights_server[1].llm_client=object()
    lights_server[5].side_effect=None;lights_server[5].return_value={'content':'Please specify the device.','tool_calls':[]}
    await ask(lights_server,'Et leur état ?',speaker='different')
    await ask(lights_server,'Et leur état ?',session='different')
    assert not lights_server[3]


@pytest.mark.asyncio
async def test_read_budget_clarifies_before_any_io(lights_server):
    provider=lights_server[2]
    from dataclasses import replace
    device=provider.indexer.get_device('1')
    # Nine enabled state channels cannot be hidden behind an eight-ID truncation.
    for i in range(10,18):
        cmd=replace(device.commands[0],cmd_id=str(900+i),device_id=str(i),name='Etat')
        other=replace(device,eq_id=str(i),name='Lampe '+str(i),room='Zone '+str(i),commands=[cmd])
        provider.indexer.add_device(other)
    result=await ask(lights_server,'Quel est l’état de toutes les lumières ?')
    assert result['response'].rstrip().endswith('?') and not lights_server[3]
