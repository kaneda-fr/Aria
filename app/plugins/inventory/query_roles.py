"""Shared inventory-name and query-operator roles; no inference or I/O."""
import re
from .read_requests import _form, _device_forms, _spans, _without


def status_query_scope(raw, devices, *, domain='cover', key=None, listing=False):
    """Separate literal target/room names from functional query words."""
    key=_form(raw) if key is None else key
    opened={'open','ouvert','ouverte','ouverts','ouvertes'}
    closed={'closed','ferme','fermee','fermes','fermees'}
    type_word='flap'
    plural_words={'volets','stores','shutters','blinds','tous','toutes','all','any'}
    if domain=='light':
        type_word='light'
        opened={'on','allume','allumee','allumes','allumees','lit','active','actives','actifs','marche'}
        closed={'off','eteint','eteinte','eteints','eteintes'}
        plural_words={'lights','lamps','bulbs','lumieres','lampes','eclairages','ampoules','plafonniers','luminaires','tous','toutes','all','any'}
    elif domain!='cover':raise ValueError('Unsupported status-query domain')
    metrics={'etat','etats','statut','status','state','states','position','positions'}
    modifiers={'still','currently','already','encore','deja','actuellement'}
    operators={'all','any','tous','toutes','tout','toute','combien','how','many','nombre','number','count','reste','there'}
    functional=opened|closed|metrics|modifiers|operators
    state_pattern=r'(?:'+ '|'.join(sorted(opened|closed))+r')'
    named=r'\b(?:named|called|nomme|nommee|appele|appelee)\s*$'
    location=r'\b(?:in|at|dans|en|from|near|room|piece)\b\s*(.*)$'
    direct_location=r'\b(?:in|at|dans|en|from|near|room|piece)\s+(?:(?:the|le|la|l|named|called|nomme|nommee)\s+)*$'
    mentions=[(s,e,k) for k,d in devices.items() for form in _device_forms(d.name)
              if set(form.split())-{type_word} for s,e in _spans(key,form)]
    mentions=[m for m in mentions if not any(m[0]>=a and m[1]<=b and m[:2]!=(a,b)
                                            for a,b,_ in mentions)]
    protected=[(s,e) for s,e,_ in mentions if not (set(key[s:e].split())-{type_word})<=functional
               and not re.search(direct_location,key[:s])]
    room_forms={_form(d.room) for d in devices.values() if d.room}
    locations=[];ambiguous=False
    for room_form in room_forms:
        for start,end in _spans(key,room_form):
            prefix=re.search(direct_location,key[:start])
            if prefix:locations.append((prefix.start(),end,room_form))
    for token in re.finditer(r'\b(?:'+'|'.join(sorted(functional))+r')\b',key):
        if any(a<=token.start() and token.end()<=b for a,b in protected):continue
        prefix=re.search(location,key[:token.start()])
        if prefix is None or re.search(r'\b(?:is|are|est|sont)\b',prefix[1]):continue
        scope=re.sub(r'^(?:(?:the|le|la|l|room|piece|named|called|nomme|nommee)\b\s*)+', '',prefix[1]).strip()
        full=(scope+' '+token[0]).strip()
        if scope in room_forms and re.match(r'^(?:is|are|est|sont)\b',raw):
            ambiguous|=full in room_forms
            continue
        locations.append((prefix.start(),token.end(),full))
    room=None;ignored=[]
    if locations:
        start,end,room=max(locations,key=lambda x:x[1]);ignored.append((start,end))
    plural=bool(set(raw.split()) & plural_words)
    labels=list(re.finditer(r'\b(?:'+'|'.join(sorted(metrics))+r')\b',key))
    states=[m for m in re.finditer(r'\b'+state_pattern+r'\b',key)
            if not any(a<=m.start() and m.end()<=b for a,b in ignored)]
    literals=[]
    for start,end,eq in mentions:
        if any(start<b and a<end for a,b in ignored):continue
        core=set(key[start:end].split())-{type_word}
        if not core<=functional:
            literals.append((start,end,eq));continue
        explicit=re.search(named,key[:start])
        predicate=(re.search(r'\b(?:is|are|est|sont)(?:\s+(?:il|elle|ils|elles))?\s*$',key[:start]) or
                   (re.match(r'^(?:is|are|est|sont)\b',raw) and states and
                    start<=states[-1].start()<end))
        metric=not plural and any(m.end()<=start for m in labels)
        following=not plural and bool(core & (opened|closed)) and re.match(
            r'^\s*(?:(?:is|are|est|sont)(?:\s+(?:il|elle|ils|elles))?\s+)?'
            r'(?:(?:still|currently|already|encore|deja|actuellement)\s+)?'+state_pattern+r'\b',key[end:])
        if explicit or (not predicate and (metric or following or (listing and not plural))):
            literals.append((start,end,eq))
    operator_text=_without(key,[(s,e) for s,e,_ in literals]+ignored)
    for modifier in re.finditer(r'\b(?:'+'|'.join(sorted(modifiers))+r')\s+(?='+state_pattern+r'\b)',key):
        if any(modifier.start()<b and a<modifier.end() for a,b,_ in literals):continue
        if any(a<=modifier.start()<b for a,b in ignored):continue
        if re.search(r'\b(?:room|piece|in|at|dans|en|de|du|au|from|near|named|nomme)\s*$',key[:modifier.start()]):continue
        ignored.append((modifier.start(),modifier.end()))
    return key,literals,ignored,room,ambiguous,operator_text
