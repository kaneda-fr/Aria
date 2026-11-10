"""Review reproductions: scope cannot disappear and metadata cannot drift."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_status_facts import fixture
from app.plugins.inventory.status_facts import project_status,render_status


@pytest.mark.parametrize('missing',['device','property'])
def test_all_requires_every_selected_device_and_property(missing):
    s,ids,r=fixture(values=(0,));d=s.devices['1']
    if missing=='property':s.devices['2']=replace(d,eq_id='2',name='Other',commands=[])
    report=project_status(s,ids+(('2','Other',d.room),),r,'light',mappings={})
    assert not render_status(report,'en',wanted='off',quantifier='all').startswith('Yes.')


@pytest.mark.parametrize('unit',['W','Wh','°C','m³'])
def test_light_state_with_incompatible_unit_is_unknown(unit):
    s,ids,r=fixture(values=(20,),unit=unit);s.devices['1'].commands[0].subType='numeric'
    report=project_status(s,ids,r,'light',mappings={})
    assert report['facts'][0]['value']=='unknown'


@pytest.mark.parametrize('domain,cap',[('opening','flap_up'),('opening',None),('generic','flap_down')])
def test_covers_without_state_stay_in_query_scope(domain,cap):
    from app.plugins.inventory.cover_queries import resolve_cover_query
    s,ids,r=fixture('cover',(0,),unit='%');d=s.devices['1'];c=d.commands[0]
    s.devices['2']=replace(d,eq_id='2',name='Other',domain=domain,
                           commands=[replace(c,cmd_id='201',device_id='2',type='action',capability=cap)])
    q=resolve_cover_query('Are all my shutters closed?',s)
    assert q.status=='ready' and len(q.targets)==2


@pytest.mark.asyncio
@pytest.mark.parametrize('change',['insert','rename','remove','duplicate'])
async def test_ordinal_reasks_if_displayed_choices_change(change):
    from app.plugins.inventory.cover_queries import resolve_cover_query,answer_cover_query
    s,ids,r=fixture('cover',(0,),unit='%');a=s.devices['1'];c=a.commands[0]
    s.devices['2']=replace(a,eq_id='2',name='Volet Cuisine',commands=[replace(c,cmd_id='201',device_id='2')])
    if change=='duplicate':s.devices['2'].name=a.name
    question='Quel volet est ouvert ?';q=resolve_cover_query(question,s)
    reply,_=await answer_cover_query(q,None)
    if change=='insert':s.devices['0']=replace(a,eq_id='0',name='Volet Jardin',commands=[replace(c,cmd_id='301',device_id='0')])
    elif change=='rename':s.devices['2'].name='Volet Renommé'
    elif change=='remove':del s.devices['2']
    follow=resolve_cover_query('Le deuxième.',s,[{'role':'user','content':question},{'role':'assistant','content':reply}])
    assert follow is None or follow.status!='ready'


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value',[('subType','binary'),('unit','W'),('name','Changed'),('source','new_source')])
async def test_light_read_refresh_cannot_reuse_previous_definition(field,value):
    from app.plugins.inventory.light_queries import LightQuery,answer_light_query
    s,ids,r=fixture(values=(25,));old=s.devices['1'];old.commands[0].subType='numeric'
    changed=replace(old,commands=[replace(old.commands[0])])
    setattr(changed if field=='source' else changed.commands[0],field,value)
    state={'snapshot':s}
    async def read(*_):
        state['snapshot']=SimpleNamespace(devices={'1':changed})
        return r
    provider=SimpleNamespace(indexer=SimpleNamespace(get_snapshot=lambda:state['snapshot']),read_light_states=read)
    text,result=await answer_light_query(LightQuery('light_state','ready',ids),provider)
    assert result['status_report']['facts'][0]['value']=='unknown'


@pytest.mark.asyncio
async def test_read_definition_is_frozen_before_provider_can_mutate_it_in_place():
    from app.plugins.inventory.light_queries import LightQuery,answer_light_query
    s,ids,r=fixture(values=(25,));s.devices['1'].commands[0].subType='numeric'
    async def read(*_):
        s.devices['1'].commands[0].name='Different meaning'
        return r
    provider=SimpleNamespace(indexer=SimpleNamespace(get_snapshot=lambda:s),read_light_states=read)
    _,result=await answer_light_query(LightQuery('light_state','ready',ids),provider)
    assert result['status_report']['facts'][0]['value']=='unknown'
