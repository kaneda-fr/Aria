"""Typed observations must support every ordinary-language assertion."""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.plugins.inventory.models import CommandRecord, DeviceRecord


def fixture(domain='light', values=(0, 1), unit=None):
    cap = 'light_state' if domain == 'light' else 'flap_state'
    commands = [CommandRecord(str(101+i), '1', 'info', 'binary' if domain == 'light' else 'numeric',
                 cap, unit, None, set(), None, name=f'Etat-{i+1}') for i in range(len(values))]
    device = DeviceRecord('jeedom_mqtt', '1', 'Éclairage Bureau' if domain == 'light' else 'Volet Bureau',
                          'Bureau', domain, True, True, commands, set(), 0)
    snapshot = SimpleNamespace(devices={'1': device})
    rows = [dict(cmd_id=c.cmd_id, device_id='1', status='ok', value=v, observed_at=None,
                 retrieved_at=100) for c,v in zip(commands,values)]
    return snapshot, (('1', device.name, device.room),), {'success':True,'readings':rows}


def project(*args, **kwargs):
    from app.plugins.inventory.status_facts import project_status
    return project_status(*args, **kwargs)


def test_mixed_light_is_a_fact_not_technical_spoken_channels():
    from app.plugins.inventory.status_facts import render_status
    s,ids,read=fixture()
    report=project(s,ids,read,'light',mappings={})
    assert report['facts'][0]['value']=='partly_on'
    assert report['coverage']['expected_channels']==2
    assert report['facts'][0]['freshness']=='unknown'
    assert 'measurement_time_unknown' in report['limitations']
    for lang in ('fr','en'):
        text=render_status(report,lang).lower()
        assert not any(t in text for t in ('canal','channel','circuit','relais','binary','partly_on'))
        assert ('une partie' in text if lang=='fr' else 'some of' in text)


@pytest.mark.parametrize('value',[None, '', 'NaN', float('nan'), float('inf'), -1, 101, 'banana', True])
def test_invalid_light_readings_never_become_off(value):
    s,ids,r=fixture(values=(value,));s.devices['1'].commands[0].subType='numeric'
    report=project(s,ids,r,'light',mappings={})
    assert report['facts'][0]['value']=='unknown'
    assert report['coverage']['interpreted_channels']==0


def test_missing_channel_keeps_positive_evidence_but_prevents_all_claim():
    from app.plugins.inventory.status_facts import render_status
    s,ids,r=fixture(values=(1,0));r['readings'].pop()
    report=project(s,ids,r,'light',mappings={})
    assert report['facts'][0]['unknown_channels']==1
    assert render_status(report,'en',wanted='on',quantifier='any').startswith('Yes.')
    assert not render_status(report,'en',wanted='on',quantifier='all').startswith('Yes.')
    assert 'unknown' in report['facts'][0]['channel_states']


@pytest.mark.parametrize('closed,opened,value,state',[(0,100,0,'closed'),(0,100,100,'open'),
    (100,0,0,'open'),(100,0,100,'closed'),(0,99,49.5,'partly_open')])
def test_cover_mapping_is_explicit_and_identity_bound(closed,opened,value,state):
    from app.plugins.inventory.status_facts import command_identity
    s,ids,r=fixture('cover',(value,),unit='%');d=s.devices['1'];c=d.commands[0]
    maps={d.source+':'+c.cmd_id:{'identity':command_identity(d,c),'closed_value':closed,'open_value':opened}}
    report=project(s,ids,r,'cover',mappings=maps)
    assert report['facts'][0]['value']==state
    d.room='Other'
    invalid=project(s,(('1',d.name,d.room),),r,'cover',mappings=maps)
    assert invalid['facts'][0]['value']=='unknown'
    assert 'mapping_identity_changed' in invalid['limitations']


def test_unmapped_cover_does_not_treat_zero_as_closed():
    s,ids,r=fixture('cover',(0,),unit='%')
    report=project(s,ids,r,'cover',mappings={})
    assert report['facts'][0]['value']=='unknown'
    assert 'mapping_missing' in report['limitations']


def test_duplicate_or_foreign_observation_cannot_satisfy_a_property():
    s,ids,r=fixture(values=(1,));r['readings']*=2
    assert project(s,ids,r,'light',mappings={})['facts'][0]['value']=='unknown'
    r['readings']=r['readings'][:1];r['readings'][0]['device_id']='other'
    assert project(s,ids,r,'light',mappings={})['facts'][0]['value']=='unknown'


def test_catalogue_and_observation_coverage_are_distinct():
    s,ids,r=fixture(values=(0,))
    report=project(s,ids,r,'light',mappings={})
    assert report['scope']['catalog_complete'] is False
    assert report['coverage']['interpreted_channels']==1
    assert report['actions_performed']==[]
    assert report['observations'][0]['retrieved_at']==100
    assert report['facts'][0]['evidence_refs']


def test_shared_single_device_read_uses_same_natural_language():
    from app.plugins.inventory.state_read import render_state
    s,ids,r=fixture();r['status_report']=project(s,ids,r,'light',mappings={})
    text=render_state(r,'fr').lower()
    assert 'une partie' in text and 'canal' not in text and '0' not in text


def test_tool_context_uses_facts_instead_of_raw_values():
    from app.llm.tools import format_tool_result_for_llm
    s,ids,r=fixture();r['status_report']=project(s,ids,r,'light',mappings={})
    text=format_tool_result_for_llm(r,language='fr')
    assert 'aria.status/1' in text
    assert 'readings' not in text and 'observations' not in text
    assert 'partly_on' in text


def test_missing_selected_device_does_not_prove_absence():
    from app.plugins.inventory.status_facts import render_status
    s,ids,r=fixture(values=(0,));s.devices={}
    report=project(s,ids,r,'light',mappings={})
    assert not render_status(report,'en',wanted='on',quantifier='any').startswith('No.')


@pytest.mark.parametrize('observed,now,ttl,expected',[(90,100,20,'fresh'),(10,100,20,'stale'),
    (None,100,20,'unknown'),(101,100,20,'unknown'),(90,100,None,'unknown')])
def test_retrieval_does_not_reset_observation_age(observed,now,ttl,expected):
    from app.plugins.inventory.status_facts import command_identity
    s,ids,r=fixture(values=(1,));r['readings'][0]['observed_at']=observed
    d=s.devices['1'];c=d.commands[0]
    maps={d.source+':'+c.cmd_id:{'identity':command_identity(d,c),'max_age_s':ttl}}
    report=project(s,ids,r,'light',mappings=maps,now=now)
    assert report['facts'][0]['freshness']==expected


def test_light_brightness_is_not_a_cover_position():
    s,ids,r=fixture(values=(25,),unit='%');s.devices['1'].commands[0].subType='numeric'
    fact=project(s,ids,r,'light',mappings={})['facts'][0]
    assert fact['open_percent'] is None
    assert fact['level_percent']==25


@pytest.mark.parametrize('lang,noun',[('fr','lumière'),('en','light')])
def test_count_is_spoken_in_everyday_language(lang,noun):
    from app.plugins.inventory.status_facts import render_status
    s,ids,r=fixture(values=(1,));report=project(s,ids,r,'light',mappings={})
    text=render_status(report,lang,wanted='on',count=True)
    assert '1 '+noun in text and '(s)' not in text


def test_zero_known_witnesses_and_missing_values_do_not_give_a_useless_lower_bound():
    from app.plugins.inventory.status_facts import render_status
    s,ids,r=fixture(values=(None,));report=project(s,ids,r,'light',mappings={})
    text=render_status(report,'fr',wanted='on',count=True)
    assert 'ne sais pas combien' in text and '0 lumière' not in text


@pytest.mark.parametrize('closed,value,state', [(0,0,'closed'), (100,100,'closed'), (0,50,'unknown'), (0,99,'unknown')])
def test_closed_only_local_calibration_never_invents_open_position(tmp_path, monkeypatch, closed, value, state):
    import json
    from app.plugins.inventory.status_facts import command_identity, render_status
    s,ids,r=fixture('cover',(value,),unit='%');d=s.devices['1'];c=d.commands[0]
    path=tmp_path/'status-map.json'
    path.write_text(json.dumps({'schema':'aria.status-map/1','mappings':{
        d.source+':'+c.cmd_id:{'identity':command_identity(d,c),
        'calibration':'closed_only','closed_value':closed}}}))
    monkeypatch.setenv('ARIA_STATUS_MAP_FILE',str(path))
    report=project(s,ids,r,'cover')
    assert report['facts'][0]['value']==state
    assert report['facts'][0]['open_percent']==(0 if state=='closed' else None)
    for lang in ('fr','en'):
        text=render_status(report,lang)
        assert ('fermé' if lang=='fr' else 'closed') in text if state=='closed' else ('déterminer' if lang=='fr' else 'determine') in text
    r['readings'][0]['status']='unavailable'
    assert project(s,ids,r,'cover')['facts'][0]['value']=='unknown'
    r['readings'][0]['status']='ok';d.room='Changed'
    assert project(s,(('1',d.name,d.room),),r,'cover')['facts'][0]['value']=='unknown'


def test_partial_calibration_does_not_enable_implicit_or_conflicting_mappings():
    from app.plugins.inventory.status_facts import command_identity
    s,ids,r=fixture('cover',(0,),unit='%');d=s.devices['1'];c=d.commands[0]
    mapping={'identity':command_identity(d,c),'closed_value':0}
    maps={d.source+':'+c.cmd_id:mapping}
    assert project(s,ids,r,'cover',mappings=maps)['facts'][0]['value']=='unknown'
    mapping.update(calibration='closed_only',open_value=99)
    assert project(s,ids,r,'cover',mappings=maps)['facts'][0]['value']=='unknown'
