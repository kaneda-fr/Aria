"""Bounded, model-free interpretation. Observations never authorize an action."""
from collections import Counter
from functools import lru_cache
import json
import math
import os
import time


def command_identity(device, command):
    """Bindings expire when identity or interpretation metadata changes."""
    return dict(source=device.source, device_id=device.eq_id, device_name=device.name,
                room=device.room, command_id=command.cmd_id, command_name=command.name,
                subtype=command.subType, unit=command.unit,
                range=list(command.range) if command.range else None,
                capability=command.capability)


@lru_cache(maxsize=4)
def _load_file(path):
    try:
        with open(path, encoding='utf-8') as stream:
            data=json.load(stream)
        if data.get('schema')=='aria.status-map/1' and isinstance(data.get('mappings'),dict):
            return data['mappings']
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def load_mappings():
    path=os.getenv('ARIA_STATUS_MAP_FILE','')
    return _load_file(path) if path else {}

def recover_status_scope(devices, reply):
    """Recover only labels explicitly present in a server-style state report."""
    import re
    prefix=r'^(?:(?:Oui|Non|Yes|No)\.\s*)?(?:Lumières|Lights|Volets|Shutters)\s*:\s*'
    if not re.match(prefix,reply):
        return []
    text=re.sub(prefix,'',reply,count=1)
    matches=[(m.start(),m.end(),d) for d in devices
             for m in re.finditer(r'(?:^|;\s*)'+re.escape(d.name)+r'\s*:',text)]
    matches=[(s,e,d) for s,e,d in matches if not any(
        s>=a and e<=b and (s,e)!=(a,b) for a,b,_ in matches)]
    seen=set();result=[]
    for _,_,device in sorted(matches,key=lambda x:x[0]):
        if device.eq_id not in seen:
            result.append(device);seen.add(device.eq_id)
    return result


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float,str)) or value=='':
        return None
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError,OverflowError):
        return None


def _interpret(device, command, row, domain, mappings):
    identity=command_identity(device,command)
    mapping=mappings.get(device.source+':'+command.cmd_id)
    if mapping is not None and (not isinstance(mapping,dict) or mapping.get('identity')!=identity):
        return 'unknown',None,'mapping_identity_changed'
    if row.get('status')!='ok':
        return 'unknown',None,'definition_changed' if row.get('error')=='definition_changed' else 'reading_unavailable'
    value=number(row.get('value'))
    if value is None:
        return 'unknown',None,'invalid_value'
    if domain=='light':
        if command.unit not in (None,'','%'):
            return 'unknown',None,'incompatible_unit'
        if command.subType=='binary' and value in (0,1):
            return ('on' if value else 'off'),None,None
        if command.subType=='numeric' and 0<=value<=100:
            if command.range and not command.range[0]<=value<=command.range[1]:
                return 'unknown',None,'invalid_value'
            return ('on' if value else 'off'),value if command.unit=='%' else None,None
        return 'unknown',None,'invalid_value'
    if mapping is None:
        return 'unknown',None,'mapping_missing'
    closed=number(mapping.get('closed_value'));opened=number(mapping.get('open_value'))
    if mapping.get('calibration')=='closed_only':
        if closed is None or mapping.get('open_value') is not None:
            return 'unknown',None,'invalid_mapping'
        # One witnessed endpoint does not establish the other endpoint or a scale.
        return ('closed',0,None) if value==closed else ('unknown',None,'mapping_incomplete')
    if closed is None or opened is None or closed==opened:
        return 'unknown',None,'invalid_mapping'
    if not min(closed,opened)<=value<=max(closed,opened):
        return 'unknown',None,'invalid_value'
    position=100*(value-closed)/(opened-closed)
    return ('closed' if position==0 else 'open' if position==100 else 'partly_open'),position,None


def project_status(snapshot, identities, result, domain, mappings=None, *, channel=None,
                   catalog_complete=False, catalog_version=None, now=None):
    """Pure evidence projection for a selected scope; no I/O or generated metadata."""
    from .selection import matches_channel
    mappings=load_mappings() if mappings is None else mappings
    now=time.time() if now is None else now
    cap='light_state' if domain=='light' else 'flap_state'
    rows=result.get('readings',[])
    frequencies=Counter(r.get('cmd_id') for r in rows if isinstance(r,dict))
    by_id={r.get('cmd_id'):r for r in rows if isinstance(r,dict)}
    facts=[];observations=[];limitations=set();expected=read_count=interpreted=0
    for eq,name,room in identities:
        d=snapshot.devices.get(eq)
        if not d or not d.enabled or (d.name,d.room)!=(name,room):
            limitations.add('identity_changed')
            continue
        commands=[c for c in d.commands if c.type=='info' and c.capability==cap and
                  (channel is None or matches_channel(c,channel))]
        states=[];positions=[];refs=[];unknown=0;freshness=[]
        for c in commands:
            expected+=1;row=by_id.get(c.cmd_id,{})
            valid=(frequencies[c.cmd_id]==1 and row.get('device_id',eq)==eq)
            if not valid:row={}
            if valid and row.get('status')=='ok':read_count+=1
            state,position,reason=_interpret(d,c,row,domain,mappings)
            if reason:limitations.add(reason)
            if state=='unknown':unknown+=1
            else:interpreted+=1
            observed=number(row.get('observed_at'))
            if observed is None or observed>now:
                observed=None;limitations.add('measurement_time_unknown')
            mapping=mappings.get(d.source+':'+c.cmd_id,{})
            max_age=number(mapping.get('max_age_s')) if isinstance(mapping,dict) else None
            age=max(0,now-observed) if observed is not None else None
            fresh=('unknown' if age is None or max_age is None or max_age<=0 else
                   'stale' if age>max_age else 'fresh')
            freshness.append(fresh)
            if fresh=='stale':limitations.add('stale_observation')
            if fresh!='fresh' and state!='unknown':limitations.add('freshness_unverified')
            ref=d.source+':'+c.cmd_id
            refs.append(ref);states.append(state);positions.append(position)
            observations.append(dict(ref=ref,definition=command_identity(d,c),
                value=row.get('value'),status=row.get('status','unavailable'),
                observed_at=observed,retrieved_at=row.get('retrieved_at'),
                interpreted_state=state,open_percent=position if domain=='cover' else None,
                freshness=fresh,reason=reason))
        if not states:limitations.add('state_property_missing')
        known=set(states)-{'unknown'}
        aggregate=('partly_on' if {'on','off'}<=known else 'on' if 'on' in known else
                   'off' if states and not unknown else 'unknown') if domain=='light' else (
                   next(iter(known)) if len(known)==1 and not unknown else
                   'mixed' if len(known)>1 else 'unknown')
        facts.append(dict(entity=d.source+':'+eq,device_id=eq,label=name,room=room,
            property='light.aggregate' if domain=='light' else 'cover.position',value=aggregate,
            channel_states=states,on_channels=states.count('on'),off_channels=states.count('off'),
            unknown_channels=unknown,open_percent=positions[0] if domain=='cover' and len(positions)==1 else None,
            level_percent=positions[0] if domain=='light' and len(positions)==1 else None,
            freshness='fresh' if freshness and all(x=='fresh' for x in freshness) else
                      'stale' if 'stale' in freshness else 'unknown',
            observed_at=min((o['observed_at'] for o in observations if o['ref'] in refs and
                             o['observed_at'] is not None),default=None),
            evidence_refs=refs,rule=domain+'.aggregate/1'))
    if not catalog_complete:limitations.add('catalog_completeness_unknown')
    return dict(schema='aria.status/1',scope=dict(domain=domain,catalog_complete=bool(catalog_complete)),
                catalog_version=catalog_version,evaluated_at=now,
                coverage=dict(expected_channels=expected,read_channels=read_count,
                              interpreted_channels=interpreted,selected_devices=len(identities)),
                facts=facts,observations=observations,limitations=sorted(limitations),actions_performed=[])


def render_status(report, language='fr', *, wanted=None, count=False, quantifier=None):
    fr=language=='fr';domain=report['scope']['domain'];light=domain=='light'
    prefix=('Lumières : ' if fr else 'Lights: ') if light else ('Volets : ' if fr else 'Shutters: ')
    facts=report['facts'];details=[]
    if not facts:
        return prefix+('je ne connais pas leur état.' if fr else 'I do not know their state.')
    selected=[]
    for f in facts:
        state=f['value'];states=f['channel_states']
        if wanted is None or (wanted=='on' and 'on' in states) or (
                wanted=='off' and (state=='off' if count else 'off' in states)) or state==wanted:
            selected.append(f)
        if light:
            description=({'on':'allumée','off':'éteinte','partly_on':'une partie des lumières est allumée',
                          'unknown':'je ne connais pas son état'} if fr else
                         {'on':'on','off':'off','partly_on':'some of the lights are on',
                          'unknown':'I do not know its state'})[state]
        else:
            p=f['open_percent']
            description=({'open':'ouvert','closed':'fermé','partly_open':
                          'à moitié ouvert' if p==50 else 'partiellement ouvert',
                          'mixed':'leurs positions sont différentes',
                          'unknown':'je ne peux pas déterminer sa position'} if fr else
                         {'open':'open','closed':'closed','partly_open':
                          'half open' if p==50 else 'partly open','mixed':'their positions differ',
                          'unknown':'I cannot determine its position'})[state]
        if light and state=='partly_on' and wanted=='off':
            description=('une partie des lumières est éteinte' if fr else 'some of the lights are off')
        if f['unknown_channels'] and state!='unknown':
            description+=('; je ne connais pas le reste' if fr else '; the rest is unknown')
        # Keep unknown properties visible even when filtering known states.
        if quantifier or f in selected or f['unknown_channels'] or state=='unknown':
            details.append(f['label']+' : '+description)
    uncertain=(not facts or len(facts)!=report['coverage']['selected_devices'] or
               any(f['unknown_channels'] or f['value']=='unknown' for f in facts))
    if count:
        n=len(selected)
        noun=(('lumière' if light else 'volet') if fr else ('light' if light else 'shutter'))+('s' if n!=1 else '')
        suffix=({'on':'allumée','off':'éteinte','open':'ouvert','closed':'fermé'} if fr else
                {'on':'on','off':'off','open':'open','closed':'closed'}).get(wanted,'')
        if fr and n!=1 and suffix:suffix+='s'
        text=prefix+('au moins ' if fr and uncertain else 'at least ' if uncertain else '')+f'{n} {noun} {suffix}'.strip()+'.'
        if n==0 and uncertain:
            text=prefix+('je ne sais pas combien correspondent à cette demande.' if fr else
                         'I do not know how many match this request.')
    else:
        text=prefix+'; '.join(details)+'.' if details else prefix+(
            'aucune ne correspond parmi les éclairages vérifiés.' if fr and light else
            'aucun ne correspond parmi les volets vérifiés.' if fr else 'none match among those checked.')
    if quantifier and wanted:
        states=[v for f in facts for v in f['channel_states']]
        yes=(wanted in states) if quantifier=='any' else bool(states) and not uncertain and all(v==wanted for v in states)
        no=(not uncertain and not yes) or (quantifier=='all' and any(v not in (wanted,'unknown') for v in states))
        text=(('Oui. ' if fr else 'Yes. ') if yes else ('Non. ' if fr else 'No. ') if no else '')+text
    if not report['scope']['catalog_complete']:
        text+=(' Cela concerne les équipements répertoriés.' if fr else ' This covers the listed devices.')
    if any(f['freshness']!='fresh' for f in facts):
        text+=(' Je ne sais pas si ces informations sont à jour.' if fr else ' I cannot confirm how up to date this is.')
    return text
