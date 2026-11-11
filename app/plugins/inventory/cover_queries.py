"""Inventory-grounded cover questions. No execution capability is accessible."""
import re

from .light_queries import LightQuery, ORDINAL, READ_FRAME
from .read_requests import (PREFIX, FRAME, NO_READ, PERIOD, COMPLEX, CAPABILITY,
                            _form, _device_forms, _spans, _without, read_syntax_language)
from .selection import DOMAINS, normalize
from .query_roles import status_query_scope


def is_cover(device):
    return device.enabled and (device.domain in ('cover','opening') or any(
        (c.capability or '').startswith('flap_') for c in device.commands))





def resolve_cover_query(text, snapshot, history=(), *, language='fr', _depth=0):
    from app.llm.control import resolve as control, PREFIX as CONTROL_PREFIX
    if snapshot is None or control(text,None) is not None:return None
    raw=normalize(re.sub(r'^\[speaker=[^\]]+\]\s*','',text))
    for _ in range(5):
        trimmed=CONTROL_PREFIX.sub('',PREFIX.sub('',raw,count=1),count=1)
        if trimmed==raw:break
        raw=trimmed
    if re.match(r'^(?:why|pourquoi|comment|explain|explique|how (?:do|does))\b',raw):return None
    if CAPABILITY.search(raw):return None
    raw=READ_FRAME.sub('',raw,count=1)
    devices={k:d for k,d in snapshot.devices.items() if is_cover(d)}
    words=set(raw.split());has_type=bool(words & DOMAINS['flap'])
    context=None
    if len(history)>=2 and _depth<3:
        previous,reply=history[-2:]
        if previous.get('role')=='user' and reply.get('role')=='assistant' and re.match(
                r'^(?:(?:Oui|Non|Yes|No)\. )?(?:Volets :|Shutters:|Quel volet|Which shutter)',reply.get('content','')):
            context=resolve_cover_query(previous.get('content',''),snapshot,history[:-2],
                                        language=language,_depth=_depth+1)
            if context is None:
                from .status_facts import recover_status_scope
                recovered=recover_status_scope(devices.values(),reply.get('content',''))
                if recovered:
                    unique=len({d.name for d in recovered})==len(recovered)
                    context=LightQuery('cover_state','ready' if unique else 'ambiguous',
                        tuple((d.eq_id,d.name,d.room) for d in recovered),
                        'en' if 'Shutters:' in reply['content'] else 'fr',match='cover_inventory')
    key,mentions,ignored,room,ambiguous_scope,operator_text=status_query_scope(raw,devices)
    state_text=_without(key,[(s,e) for s,e,_ in mentions]+ignored)
    state_words=set(state_text.split())
    state=bool(re.search(r'\b(?:etat|etats|statut|status|state|states|position|positions|ouverts?|ouvertes?|fermes?|fermees?|open|closed)\b',state_text))
    if not (has_type or mentions or context) or not (state or context or ambiguous_scope):return None
    language=read_syntax_language(state_text) or ('en' if state_words & {'open','closed','my','our','their','and','second'} else
                                         context.language if context else language)
    def plan(status,targets=(),**kw):
        return LightQuery('cover_state',status,tuple((d.eq_id,d.name,d.room) for d in targets),language,
                          match='cover_inventory',**kw)
    if NO_READ.search(raw):return plan('no_read')
    if PERIOD.search(raw) or COMPLEX.search(raw) or words & {'not','pas','jamais','never'}:return plan('unsupported')
    if ambiguous_scope:return plan('ambiguous_location')
    opened=bool(state_words & {'open','ouvert','ouverte','ouverts','ouvertes'})
    closed=bool(state_words & {'closed','ferme','fermee','fermes','fermees'})
    wanted='open' if opened and not closed else 'closed' if closed and not opened else None
    count=bool(re.search(r'\b(?:combien|how many|nombre de|number of)\b|^count\b',operator_text))
    operator_words=set(operator_text.split())
    quantifier='all' if operator_words & {'tous','toutes','all'} else 'any' if operator_words & {'any','encore','reste','there'} else None
    channel_matches=list(re.finditer(r'\b(?:channel|canal|voie)\s+([1-9][0-9]*)\b',key))
    if len(channel_matches)>1:return plan('ambiguous')
    channel=int(channel_matches[0][1]) if channel_matches else None
    rest=_without(key,[(s,e) for s,e,_ in mentions]+ignored+[(m.start(),m.end()) for m in channel_matches])
    syntax=set(_form(' '.join(FRAME)).split())|set('flap etats states position positions mes mon ma nos notre my our your leur leurs ils elles il elle it its et and tous toutes all any encore reste y t a ai je have i do does maison home house chez ou or ouvert ouverte ouverts ouvertes ferme fermee fermes fermees open closed le la les one deuxieme second premier premiere first'.split())
    selectors=set(rest.split())-syntax-set(ORDINAL)-{'called'}
    if count:selectors-={'combien','how','many','nombre','number','count'}
    ordinal=next((ORDINAL[w] for w in words if w in ORDINAL),None)
    # Only adjacent choices/locations/state ellipses inherit scope. Other domains defer.
    if context and not has_type and not mentions:
        if selectors and not any(selectors<=set(_form(d.name+' '+d.room).split()) for d in devices.values()):return None
    if context and not has_type and not mentions and ordinal is not None:
        labels=[name+(f' ({room})' if room else '') for _,name,room in context.targets[:5]]
        offered=('Quel volet souhaitez-vous : ' if context.language=='fr' else 'Which shutter do you mean: ')+', '.join(labels)+' ?'
        if (context.status!='ambiguous' or offered!=history[-1].get('content') or
                len(labels)!=len(set(labels))):
            return plan('ambiguous',[devices[k] for k,_,_ in context.targets if k in devices])
    candidates=[devices[k] for k in sorted({k for _,_,k in mentions})] if mentions else list(devices.values())
    if context and not has_type and not mentions and ordinal is not None and context.status=='ambiguous' and not selectors:
        offered=context.targets[:5]
        if ordinal>=len(offered):return plan('ambiguous',[devices[k] for k,_,_ in offered if k in devices])
        candidates=[devices[offered[ordinal][0]]] if offered[ordinal][0] in devices else []
    elif selectors:
        candidates=[d for d in candidates if selectors<=set(_form(d.name+' '+d.room).split())]
    elif context and not has_type and not mentions and room is None:
        candidates=[devices[k] for k,_,_ in context.targets if k in devices]
    if room is not None:candidates=[d for d in candidates if _form(d.room)==room]
    candidates.sort(key=lambda d:d.eq_id)
    if not candidates:return plan('not_found')
    room_followup=bool(context and context.status=='ready' and not has_type and
                       (selectors or room is not None) and re.match(r'^(?:et|and)\b',raw) and
                       (room is not None or any(selectors<=set(_form(d.room).split()) for d in candidates)))
    plural=bool(words & {'volets','stores','shutters','blinds','tous','toutes','all','any'}) or count or room_followup
    if len(candidates)>1 and not plural and not (context and len(context.targets)>1 and context.status=='ready'):
        return plan('ambiguous',candidates)
    return plan('ready',candidates,channel=channel,wanted=wanted,count=count,quantifier=quantifier)


async def answer_cover_query(query, provider):
    fr=query.language=='fr'
    if query.status=='ambiguous':
        names=', '.join(name+(f' ({room})' if room else '') for _,name,room in query.targets[:5])
        return ('Quel volet souhaitez-vous : ' if fr else 'Which shutter do you mean: ')+names+' ?',{'success':False,'status':'ambiguous'}
    if query.status!='ready':
        return ('Quel volet ou quelle pièce souhaitez-vous vérifier ?' if fr else
                'Which shutter or room should I check?'),{'success':False,'status':query.status}
    reader=getattr(provider,'read_status_states',None)
    result=await reader(query.targets,'cover',query.channel) if reader else {'success':False,'status':'unavailable'}
    if result.get('status')=='too_many':
        return ('Dans quelle pièce souhaitez-vous vérifier les volets ?' if fr else
                'Which room should I check the shutters in?'),result
    from .status_facts import project_status, render_status
    snapshot=provider.indexer.get_snapshot()
    result['status_report']=project_status(snapshot,query.targets,result,'cover',channel=query.channel,
        catalog_complete=getattr(provider,'bridge_status',{}).get('catalog_complete',False))
    return render_status(result['status_report'],query.language,wanted=query.wanted,
                         count=query.count,quantifier=query.quantifier),result
