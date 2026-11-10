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


@pytest.mark.parametrize('domain', ['light','cover'])
def test_observation_requires_explicit_device_identity(domain):
    from app.plugins.inventory.status_facts import command_identity
    s,ids,r=fixture(domain,(0,),unit='%');d=s.devices['1'];c=d.commands[0]
    mappings={} if domain=='light' else {d.source+':'+c.cmd_id:{'identity':command_identity(d,c),'closed_value':0,'open_value':100}}
    r['readings'][0].pop('device_id')
    result=project_status(s,ids,r,domain,mappings=mappings)
    assert result['facts'][0]['value']=='unknown'
    assert result['coverage']['interpreted_channels']==0


@pytest.mark.parametrize('lang',['fr','en'])
def test_incomplete_catalog_cannot_prove_all_or_exact_count_or_negative_any(lang):
    s,ids,r=fixture(values=(1,));report=project_status(s,ids,r,'light',mappings={})
    yes='Oui.' if lang=='fr' else 'Yes.';no='Non.' if lang=='fr' else 'No.'
    assert not render_status(report,lang,wanted='on',quantifier='all').startswith(yes)
    assert ('au moins' if lang=='fr' else 'at least') in render_status(report,lang,wanted='on',count=True)
    assert render_status(report,lang,wanted='on',quantifier='any').startswith(yes)
    assert render_status(report,lang,wanted='off',quantifier='all').startswith(no)
    r['readings'][0]['value']=0
    report=project_status(s,ids,r,'light',mappings={})
    assert not render_status(report,lang,wanted='on',quantifier='any').startswith(no)
    complete=project_status(s,ids,r,'light',mappings={},catalog_complete=True)
    assert render_status(complete,lang,wanted='off',quantifier='all').startswith(yes)


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
@pytest.mark.parametrize('change',['insert','remove','rename','duplicate'])
async def test_light_ordinal_reasks_after_displayed_choices_change(language,change):
    from app.plugins.inventory.light_queries import resolve_light_query,answer_light_query
    s,_,_=fixture(values=(1,));base=s.devices['1']
    a=replace(base,eq_id='10',name='DeskAlpha',room='Bureau')
    b=replace(base,eq_id='20',name='DeskBeta',room='Bureau')
    s.devices={'10':a,'20':b}
    question='Quelle lumière est allumée dans Bureau ?' if language=='fr' else 'Which light is on in Bureau?'
    first=resolve_light_query(question,s,language=language);assert first.status=='ambiguous'
    reply,_=await answer_light_query(first,SimpleNamespace())
    if change=='insert':s.devices['05']=replace(base,eq_id='05',name='DeskGamma',room='Bureau')
    elif change=='remove':s.devices.pop('10')
    elif change=='rename':s.devices['20']=replace(b,name='DeskRenamed')
    else:s.devices['20']=replace(b,name=a.name)
    history=[{'role':'user','content':question},{'role':'assistant','content':reply}]
    choice='Le deuxième.' if language=='fr' else 'The second one.'
    follow=resolve_light_query(choice,s,history,language=language)
    assert follow is not None and follow.status=='ambiguous'


@pytest.mark.asyncio
@pytest.mark.parametrize('method',['tool','resolved'])
@pytest.mark.parametrize('mutation',['replace','in_place'])
async def test_legacy_status_read_revalidates_frozen_definitions(tmp_path,method,mutation):
    import httpx
    from app.plugins.inventory.state_read import ReadOnlyInventory,render_state
    from app.plugins.inventory.provider import InventoryProvider
    provider=ReadOnlyInventory('inventory',{'cache_path':str(tmp_path/'empty.json'),'sources':{}})
    InventoryProvider.start(provider)
    s,_,_=fixture(values=(1,),unit='%');d=s.devices['1'];d.name='DeskLight';d.commands[0].subType='numeric';provider.indexer.add_device(d)
    async def refresh(force=False):return None
    provider.refresh_live_inventory=refresh
    async def handle(request):
        assert request.url.path=='/read'
        current=provider.indexer.get_device('1')
        if mutation=='replace':provider.indexer.add_device(replace(current,commands=[replace(current.commands[0],unit='W')]))
        else:current.commands[0].unit='W'
        return httpx.Response(200,json={'success':True,'status':'read','readings':[{'cmd_id':'101','device_id':'1','status':'ok','value':1,'observed_at':None}]})
    provider.read_client=httpx.AsyncClient(transport=httpx.MockTransport(handle),base_url='http://fixture')
    try:
        result=(await provider.read_state({'query':'DeskLight','metric':'state','request_text':'What is the state of DeskLight?'}) if method=='tool' else await provider.read_resolved_state('1'))
        assert result['status_report']['facts'][0]['value']=='unknown'
        assert result['readings'][0]['status']=='unavailable'
        assert 'definition_changed' in result['status_report']['limitations']
        assert 'I do not know' in render_state(result,'en')
    finally:await provider.read_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('language',['fr','en'])
async def test_light_ordinal_uses_only_unchanged_displayed_choices(language):
    from app.plugins.inventory.light_queries import resolve_light_query,answer_light_query
    s,_,_=fixture(values=(1,));base=s.devices['1']
    s.devices={'10':replace(base,eq_id='10',name='DeskAlpha',room='Bureau'),
               '20':replace(base,eq_id='20',name='DeskBeta',room='Bureau')}
    question='Quelle lumière est allumée dans Bureau ?' if language=='fr' else 'Which light is on in Bureau?'
    first=resolve_light_query(question,s,language=language)
    reply,_=await answer_light_query(first,SimpleNamespace())
    history=[{'role':'user','content':question},{'role':'assistant','content':reply}]
    follow=resolve_light_query('Le deuxième.' if language=='fr' else 'The second one.',s,history,language=language)
    assert follow.status=='ready' and follow.targets==(('20','DeskBeta','Bureau'),)
    other=resolve_light_query('And in room One?',s,history,language=language)
    assert other is None or other.status!='ready'


@pytest.mark.asyncio
@pytest.mark.parametrize('method',['tool','resolved'])
async def test_status_read_does_not_interpret_when_post_read_catalog_check_fails(tmp_path,method):
    import httpx
    from app.plugins.inventory.state_read import ReadOnlyInventory
    from app.plugins.inventory.provider import InventoryProvider
    provider=ReadOnlyInventory('inventory',{'cache_path':str(tmp_path/'empty.json'),'sources':{}})
    InventoryProvider.start(provider)
    s,_,_=fixture(values=(1,),unit='%');d=s.devices['1'];d.name='DeskLight';d.commands[0].subType='numeric';provider.indexer.add_device(d)
    checks=[];provider.bridge_status={'connected':True}
    async def refresh(force=False):
        checks.append(force)
        if len(checks)>1:provider.bridge_status['connected']=False
    provider.refresh_live_inventory=refresh
    provider.read_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
        'success':True,'status':'read','readings':[{'cmd_id':'101','device_id':'1','status':'ok','value':1,'observed_at':None}]})),base_url='http://fixture')
    try:
        result=(await provider.read_state({'query':'DeskLight','metric':'state','request_text':'What is the state of DeskLight?'}) if method=='tool' else await provider.read_resolved_state('1'))
        assert len(checks)==2
        assert result['status_report']['facts'][0]['value']=='unknown'
        assert not result['success'] and result['readings'][0]['error']=='catalog_unavailable'
    finally:await provider.read_client.aclose()
