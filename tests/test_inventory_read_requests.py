"""Server-owned intent/entity resolution: paraphrases, collisions, and no actions."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.plugins.inventory.models import DeviceRecord, CommandRecord, InventorySnapshot
from app.plugins.inventory.read_requests import resolve_read_request, answer_read_request
from app.plugins.inventory.state_read import ReadOnlyInventory


def device(id, name, room, domain='light', enabled=True):
    commands = [CommandRecord(id+'s', id, 'info', 'binary', domain+'_state', None, None, set(), None, 'Etat'),
                CommandRecord(id+'a', id, 'action', 'other', domain+'_on', None, None, set(), {'topic':'never'}, 'On')]
    return DeviceRecord('test', id, name, room, domain, enabled, True, commands, set(), 0)


@pytest.fixture
def snapshot():
    devices = [device('1','Lampe Canapé','Salon'), device('2','Lampe Bureau','Bureau'),
               device('3','Volet Est','Chambre','flap'), device('4','Lampe Canapé','Cuisine'),
               device('5','Lampe Retirée','Salon',enabled=False)]
    return InventorySnapshot({d.eq_id:d for d in devices},{c.cmd_id:c for d in devices for c in d.commands},0)


@pytest.mark.parametrize('text,kind,id', [
    ('La lampe canapé dans le salon est-elle actuellement allumée ?', 'state','1'),
    ('Is the couch lamp in the living room switched on right now?', 'state','1'),
    ('Quel est le statut de la lampe divan dans le séjour ?', 'state','1'),
    ('Read the state of the sofa lamp in the sitting room.', 'state','1'),
    ('Dis-moi ce que permet la lampe canapé dans le salon.', 'capabilities','1'),
    ('What can the couch lamp in the living room do?', 'capabilities','1'),
    ('Quelles fonctionnalités propose la lampe canapé dans le salon ?', 'capabilities','1'),
    ('Show the available controls for the lamp in the office.', 'capabilities','2'),
    ('Quelles commandes pour le volet est dans la chambre à coucher ?', 'capabilities','3'),
    ('What capabilities does the Volet Est in the bedroom have?', 'capabilities','3'),
])
def test_paraphrases(snapshot,text,kind,id):
    plan=resolve_read_request(text,snapshot)
    assert plan is not None and (plan.kind,plan.status,plan.device_id)==(kind,'ready',id),plan


@pytest.mark.parametrize('frame', [
    'Tell me what I can do with {target}.',
    'Show me what we can do with {target}.',
    'What can you do with {target}?',
    'Dis-moi ce que je peux faire avec {target}.',
    'Que pouvons nous faire avec {target} ?',
    'Montre ce que vous pouvez faire avec {target}.',
])
@pytest.mark.parametrize('target,status', [
    ('Lampe Canapé dans Salon', 'ready'),
    ('Lampe Canapé rouge dans Salon', 'not_found'),
])
def test_personal_capability_grammar_preserves_device_qualifiers(snapshot, frame, target, status):
    plan = resolve_read_request(frame.format(target=target), snapshot)
    assert plan is not None and (plan.kind, plan.status) == ('capabilities', status)
    assert plan.device_id == ('1' if status == 'ready' else None)
    assert plan.language == ('en' if frame.startswith(('Tell', 'Show', 'What')) else 'fr')


@pytest.mark.parametrize('text', [
    'Quel est l’état de la lampe canapé ?', 'What are the lamp capabilities?',
    'Quel est l’état de la lampe canapé dans le garage ?',
    'What is the state of Lampe Bureau in the kitchen?',
    'Quel est l’état de Lampe Retirée ?',
    'What is the state of Lampe Bureau and Volet Est?',
    'Quel est l’état des lampes sauf celle du salon ?',
    'What is the state of Lampe Bureau tomorrow?',
    'What is the state and temperature of Lampe Bureau?',
    'Quel est l’état de Lampe Bureau canal 1 et canal 2 ?',
])
def test_ambiguous_missing_qualified_never_reads(snapshot,text):
    plan=resolve_read_request(text,snapshot)
    assert plan is not None and plan.status!='ready',plan
    provider=SimpleNamespace(read_resolved_state=AsyncMock())
    asyncio.run(answer_read_request(plan,provider,'en'))
    provider.read_resolved_state.assert_not_called()


@pytest.mark.parametrize('text', [
    'Pourquoi le ciel est-il bleu ?', 'What is photosynthesis?',
    'Quelle est la température du Soleil ?', 'Why does a lamp produce light?',
    'Allume la lampe canapé dans le salon', 'Turn off Lampe Bureau',
    'Do not turn on Lampe Bureau', 'La lampe canapé est allumée.',
])
def test_conversation_and_actions_are_not_read_intents(snapshot,text):
    assert resolve_read_request(text,snapshot) is None


def test_reference_is_explicit_and_scoped(snapshot):
    assert resolve_read_request('What is its state?',snapshot).status=='no_reference'
    assert resolve_read_request('What is its state?',snapshot,'2').device_id=='2'
    assert resolve_read_request('Et son état ?',snapshot,'2').device_id=='2'
    assert resolve_read_request('What is the state of the lamp?',snapshot,'2').status!='ready'


def test_native_name_wins_alias_collision_but_unknown_alias_does_not_choose(snapshot):
    other=device('6','Lampe Sofa','Salon')
    snapshot=replace(snapshot,devices={**snapshot.devices,'6':other})
    assert resolve_read_request('État de Lampe Sofa dans le salon ?',snapshot) is not None
    assert resolve_read_request('Quel est l’état de Lampe Sofa dans le salon ?',snapshot).device_id=='6'
    assert resolve_read_request('What is the state of the divan lamp in the living room?',snapshot).status=='ambiguous'


def test_capabilities_use_catalog_without_reading_state_or_actions(snapshot):
    plan=resolve_read_request('Quelles commandes pour Lampe Bureau ?',snapshot)
    provider=SimpleNamespace(indexer=SimpleNamespace(get_snapshot=lambda:snapshot),read_resolved_state=AsyncMock())
    text,result=asyncio.run(answer_read_request(plan,provider,'fr'))
    assert 'allumer' in text and 'État actuel inconnu' in text
    assert [r['cmd_id'] for r in result['candidates']]==['2a']
    provider.read_resolved_state.assert_not_called()


def test_negative_read_is_not_performed(snapshot):
    assert resolve_read_request('Ne lis pas l’état de Lampe Bureau',snapshot).status=='no_read'
    assert resolve_read_request('Do not read the state of Lampe Bureau',snapshot).status=='no_read'


@pytest.mark.asyncio
async def test_resolved_reader_revalidates_id_and_never_posts_actions(snapshot):
    provider=ReadOnlyInventory('test',{})
    provider._started=True
    provider.indexer.rebuild(list(snapshot.devices.values()))
    provider.refresh_live_inventory=AsyncMock()
    calls=[]
    def response(request):
        import json
        body=json.loads(request.content);calls.append(body)
        return httpx.Response(200,json={'success':True,'readings':[{'cmd_id':i,'device_name':'Lampe Canapé','status':'ok','value':1} for i in body['cmd_ids']]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(response),base_url='http://fixture') as client:
        provider.read_client=client
        result=await provider.read_resolved_state('1','state')
        assert result['success'] and calls==[{'cmd_ids':['1s']}]
        assert not (await provider.read_resolved_state('5','state'))['success']
        assert not (await provider.read_resolved_state('missing','state'))['success']
        assert calls==[{'cmd_ids':['1s']}]


@pytest.mark.asyncio
async def test_catalog_identity_changed_before_read_fails_closed(snapshot):
    provider=ReadOnlyInventory('test',{});provider._started=True
    provider.indexer.rebuild(list(snapshot.devices.values()))
    async def refresh(**kw):
        replacement=replace(snapshot.devices['1'],room='New Room')
        provider.indexer.rebuild([replacement])
    provider.refresh_live_inventory=refresh
    provider.read_client=SimpleNamespace(post=AsyncMock())
    result=await provider.read_resolved_state('1','state',expected_identity=('Lampe Canapé','Salon'))
    assert result['status']=='not_found'
    provider.read_client.post.assert_not_called()


@pytest.mark.parametrize('text',[
    'Quel est l’état de la lampe canapé dans le salon et dans la cuisine ?',
    'What is the state of Lampe Bureau except its second channel?',
    'Read the state of Lampe Bureau channel 0',
    'Quelles capacités pour Lampe Bureau canal 1 puis allume la ?',
])
def test_constraints_are_not_discarded(snapshot,text):
    plan=resolve_read_request(text,snapshot)
    assert plan is not None and plan.status!='ready'


def test_room_word_inside_reordered_device_name_is_not_a_room_filter(snapshot):
    desk=replace(snapshot.devices['2'],room='Cuisine')
    snapshot=replace(snapshot,devices={**snapshot.devices,'2':desk})
    plan=resolve_read_request('What is the state of the office lamp?',snapshot)
    assert plan.status=='ready' and plan.device_id=='2'
    assert resolve_read_request('What is the state of the office lamp in the office?',snapshot).status=='not_found'


@pytest.mark.asyncio
async def test_channel_limits_apply_after_selecting_channel(snapshot):
    provider=ReadOnlyInventory('test',{});provider._started=True
    d=snapshot.devices['1']
    d=replace(d,commands=[replace(d.commands[0],cmd_id='s'+str(i),name='Etat '+str(i)) for i in range(1,17)])
    provider.indexer.rebuild([d]);provider.refresh_live_inventory=AsyncMock()
    seen=[]
    def response(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'success':False,'status':'unavailable','readings':[]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(response),base_url='http://fixture') as client:
        provider.read_client=client
        await provider.read_resolved_state('1','state',12)
        assert seen==[{'cmd_ids':['s12']}]
