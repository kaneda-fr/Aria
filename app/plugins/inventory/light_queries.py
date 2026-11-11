"""Bounded, inventory-grounded lighting questions; no generation or action authority."""
from dataclasses import dataclass
import math
import re

from .selection import DOMAINS, normalize, contains, matches_channel
from .read_requests import FRAME, PREFIX, CAPABILITY, NO_READ, PERIOD, COMPLEX, _form, _device_forms, _spans, _without, read_syntax_language


MAX_STATE_IDS = 8
MAX_LIST = 24
LIST = re.compile(r'\b(?:list|liste|lister|enumere|enumerate|montre|show|disponibles?|available|inventory|inventaire)\b|\bwhat .+ (?:do i|have i) have\b')
COUNT = re.compile(r'\b(?:combien|how many|number of|nombre de)\b')
STATE = re.compile(r'\b(?:etat|statut|state|states|status|allumes?|allumee?s?|eteints?|eteintee?s?|on|off|lit|active|actives|actifs)\b')
ACTION = re.compile(r'^(?:allume|allumer|eteins|eteindre|regle|regler|dim|set|turn|switch)\b|\b(?:turn|switch) (?:on|off)\b')
EXTRA = set(_form('mes ma mon nos notre vos votre tes ta ton my our your their its elles ils leur leurs celles ceux celle celui lesquels lesquelles '
                  'tous toutes tout toute all any some still encore reste restes rester y t il a do does have ai je i we you '
                  'maison home house chez anywhere partout there disponibles disponible available etats states desquelles '
                  'liste lister enumere enumerate nombre number lights eclairages ampoule ampoules plafonnier plafonniers '
                  'qu on actives actif actifs active marche fonctionnement fonctionnent tell me about combien many').split())
SYNTAX = set(_form(' '.join(FRAME)).split()) | EXTRA
LIGHT_WORDS = DOMAINS['light'] | {'eclairages','ampoule','ampoules','plafonnier','plafonniers','luminaire','luminaires'}
PLURAL = {'lights','lamps','bulbs','lumieres','lampes','eclairages','ampoules','plafonniers','luminaires'}
ORDINAL = {'first':0,'premier':0,'premiere':0,'second':1,'seconde':1,'deuxieme':1,'third':2,'troisieme':2,'fourth':3,'quatrieme':3,'fifth':4,'cinquieme':4}


READ_FRAME = re.compile(
    r'^(?:pourriez vous\s+)?(?:just\s+|simply\s+|simplement\s+)?(?:'
    r'(?:donne[rz]?|montre[rz]?|indique[rz]?|affiche[rz]?|dis|dites|lis|lisez)(?:\s+(?:moi|nous))?'
    r'|(?:me|m|nous)\s+(?:donner|dire|lire|montrer|indiquer|afficher)'
    r'|(?:give|show|tell|get)(?:\s+(?:me|us))?'
    r'|i (?:would like|want|need) to (?:know|see)'
    r'|je (?:veux|voudrais|souhaite) (?:connaitre|savoir|voir))\b\s*')
STATE_LABEL = re.compile(r'^(?:(?:le|la|l)\s+)?statue\b')


def _one_edit(word, target):
    """One substitution, insertion/deletion or adjacent swap; bounded verb only."""
    if len(word) == len(target):
        return sum(a != b for a, b in zip(word, target)) <= 1 or any(
            word[:i]+word[i+1]+word[i]+word[i+2:] == target for i in range(len(word)-1))
    if abs(len(word)-len(target)) != 1:
        return False
    short, long = sorted((word, target), key=len)
    return any(long[:i]+long[i+1:] == short for i in range(len(long)))


def is_light(device):
    return device.enabled and (device.domain == 'light' or any(
        (c.capability or '').startswith('light_') for c in device.commands))


@dataclass(frozen=True)
class LightQuery:
    kind: str
    status: str
    targets: tuple = ()
    language: str = 'fr'
    channel: int | None = None
    wanted: str | None = None
    count: bool = False
    quantifier: str | None = None
    speech_recovered: bool = False
    metric: str = 'state'
    match: str = 'lighting_inventory'

    @property
    def device_id(self):
        return self.targets[0][0] if len(self.targets) == 1 else None


def resolve_light_query(text, snapshot, history=(), *, spoken=False, language='fr', _depth=0):
    """Recognize reusable question constructions and intersect all entity qualifiers.

    History is the shared worker's contiguous, same-speaker history. Context is
    reconstructed only from an adjacent question handled by this resolver and its
    marked answer. It never comes from generated tool arguments or a global session.
    """
    if snapshot is None:
        return None
    # The existing control grammar owns imperatives, including polite/adverb
    # prefixes, references and refusals. Classify with no inventory so this
    # check cannot resolve a device or create execution authorization.
    from app.llm.control import resolve as resolve_control, PREFIX as CONTROL_PREFIX
    if resolve_control(text, None) is not None:
        return None
    raw = normalize(re.sub(r'^\[speaker=[^\]]+\]\s*', '', text))
    for _ in range(5):
        trimmed=CONTROL_PREFIX.sub('',PREFIX.sub('',raw,count=1),count=1)
        if trimmed == raw:break
        raw=trimmed
    if (ACTION.search(raw) or re.match(r'^(?:why|pourquoi|comment|explain|explique|how (?:does|do))\b', raw)
            or re.match(r'^state\b.+\bas\b', raw)):
        return None
    devices={k:d for k,d in snapshot.devices.items() if is_light(d)}
    words=set(raw.split());has_type=bool(words & LIGHT_WORDS)
    if not has_type and any(words & aliases for domain, aliases in DOMAINS.items() if domain != 'light'):
        return None
    # Repair only the request verb, with an explicit state cue and a typed noun.
    # Never fuzzy-match an equipment/room or discard an unknown target qualifier.
    declared_prefix = any(d.enabled and normalize(d.name) and raw.startswith(normalize(d.name))
                          for d in snapshot.devices.values())
    pieces = raw.split()
    frame_recovered = bool(spoken and has_type and not declared_prefix and len(pieces)>2
                           and pieces[1]=='moi' and pieces[0] not in {'donne','donnez'}
                           and re.search(r'\b(?:statut|etat|statue)\b', raw)
                           and _one_edit(pieces[0], 'donne'))
    if frame_recovered:
        raw = 'donne ' + ' '.join(pieces[1:])
    frame = READ_FRAME.match(raw) if not declared_prefix else None
    framed = frame is not None
    if framed:
        raw = raw[frame.end():]
        words=set(raw.split());has_type=bool(words & LIGHT_WORDS)
    if ACTION.search(raw):
        return None
    # A declared object name outranks an ASR homophone correction.
    state_recovered=bool(spoken and STATE_LABEL.match(raw) and has_type and not any(
        d.enabled and contains(raw,normalize(d.name)) and STATE_LABEL.match(normalize(d.name))
        for d in snapshot.devices.values()))
    recovered = frame_recovered or state_recovered
    if state_recovered:
        raw=re.sub(r'\bstatue\b','statut',raw,count=1);words=set(raw.split())
    context=None
    if history and _depth < 3 and len(history)>=2:
        previous,answer=history[-2:]
        if previous.get('role')=='user' and answer.get('role')=='assistant' and re.match(r'^(?:(?:Oui|Non|Yes|No)\. )?(?:Lumières :|Lights:|Quelle lumière|Which light|Dans quelle pièce|Which room)',answer.get('content','')):
            context=resolve_light_query(previous.get('content',''),snapshot,history[:-2],spoken=True,language=language,_depth=_depth+1)
            if context is None:
                from .status_facts import recover_status_scope
                recovered_scope=recover_status_scope(devices.values(),answer.get('content',''))
                if recovered_scope:
                    unique=len({d.name for d in recovered_scope})==len(recovered_scope)
                    context=LightQuery('light_state','ready' if unique else 'ambiguous',
                        tuple((d.eq_id,d.name,d.room) for d in recovered_scope),
                        'en' if 'Lights:' in answer['content'] else 'fr')
    language=read_syntax_language(raw) or ('en' if re.match(r'^(?:my|our|your|can|could|and|the|those|their|them|in)\b',raw) else
                                         'fr' if re.match(r'^(?:mes|nos|vos|combien|y a|statue|et|celle|celui|leur|leurs|dans|meme)\b',raw) else
                                         context.language if context else language)
    short_choice=bool(context and any(raw in ('the '+w+' one',w+' one') for w in ORDINAL)
                      and not any(normalize(d.name)==raw for d in devices.values()))
    if short_choice:
        raw=raw[:-4];words=set(raw.split())
    key=_form(raw)
    if not re.search(r'\b(?:room|piece)\b', raw):
        key=re.sub(r'\b(?:a la maison|dans la maison|at home|in the house|around the house|chez moi)\b', ' ', key)
        key=' '.join(key.split())
    mentions=[(s,e,k) for k,d in devices.items() for form in _device_forms(d.name)
              if set(form.split())-{'light'} for s,e in _spans(key,form)]
    mentions=[(s,e,k) for s,e,k in mentions if not any(s>=a and e<=b and (s,e)!=(a,b) for a,b,_ in mentions)]
    if short_choice:mentions=[]
    nonlight=any(d.enabled and not is_light(d) and contains(key,_form(d.name))
                 for d in snapshot.devices.values())
    if nonlight and not has_type:return None
    count=bool(COUNT.search(raw))
    # Within list framing, selected equipment names are not state predicates.
    state_text=_without(key,[(s,e) for s,e,_ in mentions]) if LIST.search(raw) else raw
    state_words=set(state_text.split())
    kind=('light_capabilities' if CAPABILITY.search(raw) else
          'light_state' if STATE.search(state_text) else
          'light_list' if count or LIST.search(raw) or re.search(r'\b(?:quelles sont|quels sont|what lights|what lamps)\b',raw) or framed else None)
    ordinal=next((ORDINAL[w] for w in words if w in ORDINAL),None)
    if context and not has_type and not mentions:
        # A prior lighting answer is not evidence that every new question is
        # about lights. Keep only state/count ellipses, choices and locations.
        channel_reference = bool(re.search(r'\b(?:channel|canal|voie)\s+[1-9][0-9]*\b', key))
        reference_key = re.sub(r'\b(?:channel|canal|voie)\s+[1-9][0-9]*\b', '', key)
        remainder = set(reference_key.split()) - SYNTAX - set(ORDINAL)
        short_reference = not remainder and (kind is not None or ordinal is not None or channel_reference)
        inventory_reference = bool(remainder) and any(
            remainder <= set(_form(d.name+' '+d.room).split()) for d in devices.values())
        if not (short_reference or inventory_reference):
            context = None
    if kind is None and context:
        kind=context.kind;count=context.count
    if kind is None or not (has_type or mentions or context):
        # A short location+state phrase can identify a typed light from inventory.
        wanted=set(key.split())-SYNTAX
        binary_cue = bool(words & {'on','off','allume','allumee','allumes','allumees','eteint','eteinte','eteints','eteintes','lit'})
        if not (binary_cue and wanted and any(wanted<=set(_form(d.name+' '+d.room).split()) for d in devices.values())):
            return None
        kind='light_state'
    def plan(status,targets=(),**kw):
        return LightQuery(kind,status,tuple((d.eq_id,d.name,d.room) for d in targets),language,
                          speech_recovered=recovered,count=count,**kw)
    if NO_READ.search(raw) or (framed and words & {'no','none','aucun','aucune'}):return plan('no_read')
    if PERIOD.search(raw) or COMPLEX.search(raw):return plan('unsupported')
    if nonlight:return plan('not_found')
    # Only state questions use these predicates; no write path exists in this module.
    on=bool(state_words & {'on','allume','allumee','allumes','allumees','lit','active','actives','actifs','marche'})
    off=bool(state_words & {'off','eteint','eteinte','eteints','eteintes'})
    wanted='on' if on and not off else 'off' if off and not on else None
    if words & {'pas','not','jamais','never'} and not NO_READ.search(raw):return plan('unsupported')
    quantifier=('all' if words & {'all','toutes','tous','toute','tout','eclairage','lighting'} else
                'any' if words & {'any','reste','encore','still'} or 'y a' in raw or 'there' in words else None)
    channels=list(re.finditer(r'\b(?:channel|canal|voie)\s+([1-9][0-9]*)\b',key))
    if len(channels)>1:return plan('ambiguous')
    channel=int(channels[0].group(1)) if channels else None
    rest=_without(key,[(s,e) for s,e,_ in mentions]+[(m.start(),m.end()) for m in channels])
    selectors=set(rest.split())-SYNTAX-set(ORDINAL)
    # Relative-clause syntax, not a globally ignored location/name token.
    relative_target=(re.search(r'\blight that are\b',key) or
                     any(re.match(r'\s+that are\b',key[e:]) for _,e,_ in mentions))
    if relative_target:selectors.discard('that')
    if context and not has_type and not mentions and ordinal is not None:
        labels=[name+(f' ({room})' if room else '') for _,name,room in context.targets[:5]]
        offered=('Quelle lumière souhaitez-vous : ' if context.language=='fr' else 'Which light do you mean: ')+', '.join(labels)+'?'
        if (context.status!='ambiguous' or offered!=history[-1].get('content') or
                len(labels)!=len(set(labels))):
            return plan('ambiguous',[devices[i] for i,_,_ in context.targets if i in devices])
    candidates=[devices[k] for k in sorted({k for _,_,k in mentions})] if mentions else list(devices.values())
    if context and not has_type and not mentions and channel is None:
        channel=context.channel
    if ordinal is not None and context and context.status=='ambiguous' and not selectors:
        offered=context.targets[:5]
        if ordinal>=len(offered):return plan('ambiguous',[devices[i] for i,_,_ in offered if i in devices])
        chosen=offered[ordinal][0];candidates=[devices[chosen]] if chosen in devices else []
    elif selectors:
        candidates=[d for d in candidates if selectors<=set(_form(d.name+' '+d.room).split())]
    elif context and not has_type and not mentions:
        candidates=[devices[i] for i,_,_ in context.targets if i in devices]
        if wanted is None:wanted=context.wanted
        if channel is None:channel=context.channel
    candidates.sort(key=lambda d:d.eq_id)
    if not candidates:return plan('not_found',channel=channel,wanted=wanted,quantifier=quantifier)
    if len(candidates)==1 and mentions and not recovered and not count and not context:
        # Preserve the established single-device read/capability contract.
        from .read_requests import resolve_read_request
        existing = resolve_read_request(raw, snapshot)
        if (existing is not None and existing.status == 'ready' and
                (channel is None or any(matches_channel(c, channel) for c in candidates[0].commands if c.capability == 'light_state'))):
            return None
    plural=bool(words & (PLURAL|{'all','tous','toutes','eclairage','lighting','there'})) or kind=='light_list' or count
    if len(candidates)>1 and not plural and not (context and context.status=='ready' and len(context.targets)>1 and not has_type and not mentions):
        return plan('ambiguous',candidates,channel=channel,wanted=wanted,quantifier=quantifier)
    return plan('ready',candidates,channel=channel,wanted=wanted,quantifier=quantifier)


def light_state_commands(snapshot, identities, channel=None):
    """Revalidate names/rooms/types and collect all requested light state channels."""
    if not identities or len(identities)>MAX_STATE_IDS:
        return {'status':'too_many'}
    commands=[]
    for eq,name,room in identities:
        d=snapshot.devices.get(eq)
        if d is None or not is_light(d) or (d.name,d.room)!=(name,room):
            return {'status':'not_found'}
        commands.extend(c for c in d.commands if c.type=='info' and c.capability=='light_state'
                        and (channel is None or matches_channel(c,channel)))
    if len(commands)>MAX_STATE_IDS:return {'status':'too_many'}
    if len({c.cmd_id for c in commands})!=len(commands):return {'status':'unavailable'}
    return {'status':'ready','commands':commands}


def _binary(value, sub_type):
    if isinstance(value,str):
        text=normalize(value)
        if text in {'on','true','allume','allumee'}:return 'on'
        if text in {'off','false','eteint','eteinte'}:return 'off'
        try:value=float(value)
        except (ValueError,TypeError):return 'unknown'
    if not isinstance(value,(bool,int,float)) or not math.isfinite(value):return 'unknown'
    if sub_type=='binary' and value not in (0,1):return 'unknown'
    if not 0<=value<=100:return 'unknown'
    return 'on' if value>0 else 'off'


def render_light_states(query,snapshot,result):
    from .status_facts import project_status, render_status
    report=result.get('status_report') or project_status(snapshot,query.targets,result,'light',
                                                       channel=query.channel)
    result['status_report']=report
    return render_status(report,query.language,wanted=query.wanted,count=query.count,
                         quantifier=query.quantifier)


async def answer_light_query(query,provider):
    fr=query.language=='fr'
    question=lambda: ('Dans quelle pièce souhaitez-vous connaître l’état des lumières ?' if fr else 'Which room should I read the lights in?')
    if query.status=='ambiguous':
        labels=[name+(f' ({room})' if room else '') for _,name,room in query.targets[:5]]
        return (('Quelle lumière souhaitez-vous : ' if fr else 'Which light do you mean: ')+', '.join(labels)+'?' if labels else question()),{'status':'ambiguous','success':False}
    if query.status!='ready':
        text=(('Aucune lecture effectuée.' if fr else 'No state read performed.') if query.status=='no_read' else
              ('Cette demande de lumières nécessite une précision. Quelle lumière ou pièce souhaitez-vous ?' if fr else 'Please specify the light or room for this request.'))
        return text,{'status':query.status,'success':False}
    snapshot=provider.indexer.get_snapshot()
    if query.kind in ('light_list','light_capabilities'):
        n=len(query.targets)
        if query.count:return (f'Lumières : {n} équipements disponibles.' if fr else f'Lights: {n} devices available.'),{'status':'listed','success':True}
        if n>MAX_LIST:return (f'{n} lumières disponibles. Quelle pièce souhaitez-vous ?' if fr else f'{n} lights available. Which room do you mean?'),{'status':'too_many','success':False}
        labels=[]
        caps={'light_on':('allumer','turn on'),'light_off':('éteindre','turn off'),'light_slider':('régler la luminosité','dim')}
        for eq,name,room in query.targets:
            label=name+(f' ({room})' if room else '')
            if query.kind=='light_capabilities':
                d=snapshot.devices[eq];available=sorted({c.capability for c in d.commands if c.type=='action' and c.capability in caps})
                label+=': '+(', '.join(caps[c][0 if fr else 1] for c in available) or ('aucune commande répertoriée' if fr else 'no controls listed'))
            labels.append(label)
        return (f'Lumières : {n} équipements disponibles : ' if fr else f'Lights: {n} devices available: ')+'; '.join(labels)+'.',{'status':'listed','success':True}
    selection=light_state_commands(snapshot,query.targets,query.channel)
    if selection['status']=='too_many':return question(),{'status':'too_many','success':False}
    if selection['status']!='ready':return question(),{'status':selection['status'],'success':False}
    from .status_facts import command_identity
    expected={c.cmd_id:command_identity(snapshot.devices[c.device_id],c) for c in selection['commands']}
    reader=getattr(provider,'read_light_states',None)
    if reader is not None:
        result=await reader(query.targets,query.channel)
    else:
        # Optional providers implementing the existing single-device read contract.
        result={'success':False,'status':'unavailable','readings':[]}
        single=getattr(provider,'read_resolved_state',None)
        if single is not None:
            for eq,name,room in query.targets:
                part=await single(eq,'state',query.channel,expected_identity=(name,room))
                result['readings'].extend(part.get('readings',[]));result['success']|=bool(part.get('success'))
    if result.get('status')=='too_many':return question(),result
    allowed={c.cmd_id for c in selection['commands']}
    result={**result,'readings':[r for r in result.get('readings',[]) if r.get('cmd_id') in allowed]}
    current=provider.indexer.get_snapshot()
    actual={c.cmd_id:command_identity(d,c) for d in current.devices.values() for c in d.commands
            if c.cmd_id in allowed and c.type=='info'}
    result['readings']=[r if actual.get(r['cmd_id'])==expected.get(r['cmd_id']) else
                        {**r,'status':'unavailable','error':'definition_changed'} for r in result['readings']]
    from .status_facts import project_status
    result['status_report']=project_status(current,query.targets,result,'light',channel=query.channel,
        catalog_complete=getattr(provider,'bridge_status',{}).get('catalog_complete',False),
        catalog_version=getattr(provider,'bridge_status',{}).get('catalog_version'))
    return render_light_states(query,current,result),result
