"""Speech-like FR/EN cover queries select info IDs, never actions or guesses."""
import json
from dataclasses import replace

import pytest
import pytest_asyncio

from test_light_questions import lighting, lights_server, ask
from test_shared_requests import server
from test_status_facts import fixture


@pytest_asyncio.fixture
async def cover_server(lights_server,tmp_path,monkeypatch):
    env=lights_server;provider=env[2];mappings={}
    from app.plugins.inventory.status_facts import command_identity
    for eq,cid,name,room,value in [('90','901','Volet Bureau','RDC',0),
            ('91','902','Volet Cuisine','RDC',50),('92','903','Volet Jardin','Terrasse',100)]:
        snapshot,_,_=fixture('cover',(value,),unit='%');d=snapshot.devices['1'];c=d.commands[0]
        d.eq_id=eq;d.name=name;d.room=room;c.cmd_id=cid;c.device_id=eq
        provider.indexer.add_device(d);env[4][cid]=value
        mappings[d.source+':'+cid]={'identity':command_identity(d,c),'closed_value':0,'open_value':100}
    path=tmp_path/'mapping.json';path.write_text(json.dumps({'schema':'aria.status-map/1','mappings':mappings}))
    monkeypatch.setenv('ARIA_STATUS_MAP_FILE',str(path))
    yield env


@pytest.mark.asyncio
@pytest.mark.parametrize('text,lang',[
    ('Quel est le statut des volets ?','fr'),('Les volets sont-ils ouverts ou fermés ?','fr'),
    ('Donne-moi la position de mes volets.','fr'),('Est-ce que tous les stores sont fermés ?','fr'),
    ('What is the status of my shutters?','en'),('Are the blinds open or closed?','en'),
    ('Show me the positions of the shutters.','en'),('Are all my shutters closed?','en')])
async def test_cover_group_reads_all_typed_states_in_one_batch(cover_server,text,lang):
    r=await ask(cover_server,text)
    assert r['success'],r
    assert cover_server[3]==[['901','902','903']]
    reply=r['response'].lower()
    assert ('à moitié ouvert' in reply if lang=='fr' else 'half open' in reply)
    assert ('fermé' in reply if lang=='fr' else 'closed' in reply)
    assert not any(w in reply for w in ('canal','channel','circuit','flap_state'))
    cover_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('question,choice',[
    ('Quel volet est ouvert dans le RDC ?','Le deuxième.'),
    ('Which shutter is open in RDC?','The second one.')])
async def test_cover_clarification_resumes_only_the_offered_choice(cover_server,question,choice):
    first=await ask(cover_server,question)
    assert first['success'] and not cover_server[3]
    assert 'Volet Bureau' in first['response'] and 'Volet Cuisine' in first['response']
    follow=await ask(cover_server,choice)
    assert follow['success'] and cover_server[3]==[['902']]


@pytest.mark.asyncio
async def test_unknown_cover_mapping_is_not_spoken_as_closed(cover_server,monkeypatch):
    monkeypatch.delenv('ARIA_STATUS_MAP_FILE')
    r=await ask(cover_server,'Quel est l’état du volet Bureau ?')
    assert cover_server[3]==[['901']]
    assert 'ne peux pas déterminer' in r['response']
    assert 'fermé' not in r['response']


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
@pytest.mark.parametrize('name',['Open','Closed','Ouvert','Fermé'])
async def test_cover_name_state_words_do_not_filter_status(cover_server,tmp_path,monkeypatch,language,name):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    from app.plugins.inventory.status_facts import command_identity
    provider=cover_server[2]
    device=replace(provider.indexer.get_device('90'),name='Volet '+name)
    provider.indexer.add_device(device)
    command=device.commands[0]
    path=tmp_path/'named-cover-map.json'
    path.write_text(json.dumps({'schema':'aria.status-map/1','mappings':{device.source+':'+command.cmd_id:{
        'identity':command_identity(device,command),'closed_value':0,'open_value':100}}}))
    monkeypatch.setenv('ARIA_STATUS_MAP_FILE',str(path))
    text=('Quel est le statut du volet '+name+' ?' if language=='fr' else
          'What is the status of the shutter '+name+'?')
    query=resolve_cover_query(text,provider.indexer.get_snapshot(),language=language)
    assert query is not None and query.status=='ready' and query.wanted is None
    result=await ask(cover_server,text)
    assert result['success'] and cover_server[3]==[['901']]
    assert device.name in result['response']
    assert ('fermé' if language=='fr' else 'closed') in result['response']
    cover_server[5].assert_not_called()
    opened=name in ('Open','Ouvert')
    state=('fermé' if opened else 'ouvert') if language=='fr' else ('closed' if opened else 'open')
    explicit=('Le volet '+name+' est-il '+state+' ?' if language=='fr' else
              'Is the shutter '+name+' '+state+'?')
    query=resolve_cover_query(explicit,provider.indexer.get_snapshot(),language=language)
    assert query is not None and query.status=='ready'
    assert query.wanted==('closed' if opened else 'open')


@pytest.mark.asyncio
@pytest.mark.parametrize('name,text,wanted',[
    ('Open','Which shutters are open?','open'),
    ('Closed','Which shutters are closed?','closed'),
    ('Open','Are any shutters open?','open'),
    ('Closed','Are all my shutters closed?','closed'),
    ('Open','Which shutters are still open?','open'),
    ('Closed','Which shutters are currently closed?','closed'),
    ('Open','What is the status of my open shutters?','open'),
    ('Closed','What is the status of my closed shutters?','closed'),
    ('Ouverts','Quels volets sont ouverts ?','open'),
    ('Fermés','Quels volets sont fermés ?','closed'),
    ('Ouverts','Les volets sont-ils encore ouverts ?','open'),
    ('Fermés','Quel est le statut des volets fermés ?','closed'),
    ('Ouverts','Quels volets sont déjà ouverts ?','open'),
    ('Status','What is the status of my shutters?',None),
    ('État','Quel est l’état des volets ?',None),
])
async def test_generic_cover_predicates_do_not_select_homonymous_equipment(cover_server,name,text,wanted):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name=name))
    query=resolve_cover_query(text,provider.indexer.get_snapshot())
    assert query is not None and query.status=='ready' and query.wanted==wanted
    assert {t[0] for t in query.targets}=={'90','91','92'}
    if wanted=='closed':cover_server[4]['902']=0
    result=await ask(cover_server,text)
    assert result['success'] and cover_server[3]==[['901','902','903']]
    if wanted=='open':assert 'Volet Jardin' in result['response']
    if wanted=='closed':assert 'Volet Cuisine' in result['response']
    cover_server[5].assert_not_called()


@pytest.mark.parametrize('name,text,wanted',[
    ('Open','Is the shutter Open open?','open'),
    ('Closed','Is the shutter Closed closed?','closed'),
    ('Ouvert','Le volet Ouvert est-il ouvert ?','open'),
    ('Fermé','Le volet Fermé est-il fermé ?','closed'),
    ('Open','Which shutter named Open is closed?','closed'),
    ('Open','Is the shutter called Open closed?','closed'),
    ('Fermé','Quel volet appelé Fermé est ouvert ?','open'),
])
@pytest.mark.asyncio
async def test_explicit_cover_names_keep_the_actual_predicate(cover_server,name,text,wanted):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name=name))
    query=resolve_cover_query(text,provider.indexer.get_snapshot())
    assert query is not None and query.status=='ready'
    assert [t[0] for t in query.targets]==['90'] and query.wanted==wanted
    assert not cover_server[3]


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
@pytest.mark.parametrize('known',[False,True])
@pytest.mark.parametrize('room_name',['Open','Living Open'])
async def test_literal_state_room_does_not_select_equipment_with_that_name(cover_server,language,known,room_name):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name='Open'))
    if known:provider.indexer.add_device(replace(provider.indexer.get_device('91'),room=room_name))
    text=('Quel est le statut des volets dans la pièce '+room_name+' ?' if language=='fr' else
          'What is the status of my shutters in the room '+room_name+'?')
    query=resolve_cover_query(text,provider.indexer.get_snapshot(),language=language)
    assert query is not None and query.language==language
    if known:
        assert query.status=='ready' and query.wanted is None
        assert [t[0] for t in query.targets]==['91']
    else:assert query.status!='ready'
    result=await ask(cover_server,text)
    assert result['success'] and cover_server[3]==([['902']] if known else [])
    cover_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
async def test_unknown_multiword_state_room_is_never_broadened(cover_server,language):
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name='Open'))
    provider.indexer.add_device(replace(provider.indexer.get_device('91'),room='Living Closed'))
    text=('Quels volets sont fermés dans la pièce Living Open ?' if language=='fr' else
          'Which shutters are closed in the room Living Open?')
    result=await ask(cover_server,text)
    assert result['success'] and not cover_server[3]
    cover_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
async def test_state_named_room_followup_expands_scope(cover_server,language):
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name='Open'))
    for eq in ('91','92'):provider.indexer.add_device(replace(provider.indexer.get_device(eq),room='Open'))
    await ask(cover_server,'Quel est le statut du volet Open ?' if language=='fr' else
              'What is the status of the shutter Open?')
    result=await ask(cover_server,'Et dans la pièce Open ?' if language=='fr' else
                     'And in the room Open?')
    assert result['success'] and cover_server[3]==[['901'],['902','903']]
    cover_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
async def test_generic_state_ellipsis_keeps_all_offered_covers(cover_server,language):
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('90'),name='Open' if language=='en' else 'Ouverts'))
    await ask(cover_server,'Quel est le statut des volets ?' if language=='fr' else
              'What is the status of my shutters?')
    result=await ask(cover_server,'Et sont-ils ouverts ?' if language=='fr' else
                     'And which are open?')
    assert result['success'] and cover_server[3]==[['901','902','903'],['901','902','903']]
    cover_server[5].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
async def test_state_and_room_interpretations_require_clarification(cover_server,language):
    provider=cover_server[2]
    provider.indexer.add_device(replace(provider.indexer.get_device('91'),room='Open'))
    provider.indexer.add_device(replace(provider.indexer.get_device('92'),room='Open Closed'))
    text=('Est-ce que les volets dans la pièce Open Closed ?' if language=='fr' else
          'Are the shutters in the room Open Closed?')
    result=await ask(cover_server,text)
    assert result['success'] and not cover_server[3]
    assert ('pièce' if language=='fr' else 'room') in result['response']
    cover_server[5].assert_not_called()


def test_actions_explanations_negatives_and_other_domain_are_not_cover_reads(cover_server=None):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    s,_,_=fixture('cover',(0,),unit='%')
    for text in ['Ouvre les volets','Close all blinds','Why do shutters exist?',
                 'Les lumières sont-elles allumées ?','Ne lis pas les volets']:
        q=resolve_cover_query(text,s)
        assert q is None or q.status!='ready'


@pytest.mark.asyncio
async def test_tool_read_of_one_light_does_not_include_unrequested_properties(lights_server):
    result=await lights_server[2].read_state({'query':'Lumière Bureau channel 2','metric':'state'})
    assert lights_server[3]==[['102']]
    report=result['status_report']
    assert report['coverage']['expected_channels']==1
    assert report['facts'][0]['unknown_channels']==0


@pytest.mark.asyncio
@pytest.mark.parametrize('lang',['fr','en'])
async def test_cover_location_followups_survive_three_exchange_history(cover_server,lang):
    first='Quel est le statut des volets ?' if lang=='fr' else 'What is the status of my shutters?'
    await ask(cover_server,first)
    for _ in range(5):
        reply=await ask(cover_server,'Et dans le RDC ?' if lang=='fr' else 'And in RDC?')
        assert reply['success'],reply
        assert cover_server[3][-1]==['901','902']
    assert len(cover_server[3])==6


@pytest.mark.asyncio
async def test_location_followup_can_expand_from_one_shutter_to_a_room(cover_server):
    await ask(cover_server,'Quel est le statut des volets ?')
    await ask(cover_server,'Et dans Terrasse ?')
    assert cover_server[3][-1]==['903']
    reply=await ask(cover_server,'Et dans RDC ?')
    assert reply['success'] and cover_server[3][-1]==['901','902']


@pytest.mark.asyncio
async def test_light_followups_survive_three_exchange_history(lights_server):
    await ask(lights_server,'Quel est le statut des lumières ?')
    for _ in range(5):
        reply=await ask(lights_server,'Et dans RDC ?')
        assert reply['success'] and lights_server[3][-1]==['101','102','401']
    assert len(lights_server[3])==6


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['Which shutters are still open?',
    'Are my blinds currently closed?','Are my shutters already closed?'])
async def test_state_adverbs_are_not_unknown_equipment_names(cover_server,text):
    reply=await ask(cover_server,text)
    assert reply['success'] and cover_server[3]==[['901','902','903']]


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['What is the status of the shutters in room Still?',
                                'Are the shutters in room Still open?'])
async def test_state_adverb_word_remains_a_literal_room_qualifier(cover_server,text):
    cover_server[2].indexer.get_device('92').room='Still'
    reply=await ask(cover_server,text)
    assert reply['success'] and cover_server[3]==[['903']]
