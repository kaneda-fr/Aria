"""Inventory-grounded FR/EN read requests, without inference or action authority.

Entity equivalences reuse the reviewed lexicon. Request patterns describe reusable
constructions, never individual devices or canned benchmark answers.
"""
from dataclasses import dataclass
import re

from .selection import normalize, entity_text, DOMAINS, contains
from .state_read import METRICS, render_state
from app.llm.lookup_reply import render_lookup_results


@dataclass(frozen=True)
class ReadRequest:
    kind: str
    status: str
    device_id: str | None = None
    metric: str | None = None
    channel: int | None = None
    candidates: tuple[str, ...] = ()
    match: str | None = None
    device_name: str | None = None
    room: str | None = None
    language: str | None = None


CAPABILITY = re.compile(
    r'\b(?:capabilities|capability|capacites|capacite|fonctionnalites|fonctionnalite|'
    r'functions|fonctions|possibilites|commandes|commands|controls)\b|'
    r'\b(?:que permet|ce que permet|what .+ (?:can do|supports)|what can .*do)\b|'
    r'\bque (?:(?:je|tu|nous|vous) (?:peux|pouvons|pouvez)|'
    r'(?:peux|pouvons|pouvez) (?:je|tu|nous|vous)) faire\b')
QUESTION = re.compile(r'^(?:et |and )?(?:quel|quelle|quels|quelles|what|which|how|combien|'
                      r'etat|statut|state|status|est|sont|is|are|lis|lire|read|donne|give|affiche|show|dis moi|tell me)\b')
BINARY = re.compile(r'\b(?:allumee?s?|eteinte?s?|ouverte?s?|fermee?s?|on|off|lit|closed|open)\b')
PREFIX = re.compile(r'^(?:bonjour aria|hello aria|hey aria|aria|bonjour|hello|please|'
                    r's il te plait|s il vous plait|peux tu|pouvez vous|pourrais tu|'
                    r'can you|could you|would you)\b\s*')
DEVICE_WORDS = set().union(*DOMAINS.values()) | {'device','devices','appareil','appareils','sensor','capteur','thermometre','thermometer'}
REFERENCE = {'it','its','elle','il','son','sa','ses','celui','celle','same','meme'}
FRAME = set(('the a an le la les l un une des du de d dans en in room piece named nomme '
             'nommee appele appelee au aux pour for of from at quelle quel quels quelles what '
             'which how is are est sont ce que qui permet peut faire can do '
             'i we you je tu nous vous peux pouvons pouvez '
             'does have propose supports support me moi dis tell donne give lis lire read affiche montre show '
             'list liste available disponibles disponible with avec please stp currently '
             'current actuel actuelle actual now actuellement maintenant right statut '
             'etat state status capabilities capability capacites capacite fonctionnalites '
             'fonctionnalite functions fonctions possibilites commandes commands controls '
             'it its elle il son sa ses celui celle same meme et and switched turned '
             'allume allumee allumes allumees eteint eteinte eteints eteintes ouvert '
             'ouverte ouverts ouvertes ferme fermee fermes fermees on off lit closed open '
             'temperature temperatures temp humidity humidite pressure pression wind vent '
             'rain pluie precipitation precipitations power puissance energy energie '
             'consommation consumption condition conditions position brightness luminosite '
             'not pas ne n').split()) | DEVICE_WORDS
PERIOD = re.compile(r'\b(?:demain|tomorrow|forecast|prevision|hier|yesterday|semaine|week|tonight|soir)\b')
COMPLEX = re.compile(r'\b(?:except|excluding|sauf|sans|unless|instead|plutot|if|si|when|quand|'
                     r'then|puis|compare|comparer|versus|before|after|avant|apres)\b')
NO_READ = re.compile(r'^(?:ne (?:lis|donne|montre|affiche) pas|ne pas (?:lire|afficher)|'
                     r'do not (?:read|show|tell)|don t (?:read|show|tell))\b')
TYPE_FORMS = {alias: domain for domain, aliases in DOMAINS.items() for alias in aliases}


def _form(text):
    return ' '.join(TYPE_FORMS.get(word, word) for word in entity_text(text).split())


def _device_forms(name):
    value = _form(name)
    words = value.split()
    forms = {value}
    # FR type-first and EN type-last noun phrases keep every name qualifier.
    if len(words) > 1 and words[0] in DOMAINS:
        forms.add(' '.join(words[1:] + words[:1]))
    return forms


def _spans(text, phrase):
    return [(m.start(), m.end()) for m in re.finditer(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text)] if phrase else []


def _without(text, spans):
    chars = list(text)
    for start, end in spans:
        chars[start:end] = ' ' * (end - start)
    return ' '.join(''.join(chars).split())


def resolve_read_request(text, snapshot, previous_device=None):
    raw = normalize(text)
    for _ in range(5):
        trimmed = PREFIX.sub('', raw, count=1)
        if trimmed == raw:
            break
        raw = trimmed
    if re.match(r'^(?:etat|statut|state|status)\b', raw):
        from .state_read import resolved_read_request
        if snapshot is None or resolved_read_request(snapshot, raw, []) is None:
            return None
    key, words = _form(raw), set(raw.split())
    devices = {k: d for k, d in snapshot.devices.items() if d.enabled} if snapshot else {}
    mentions = list({(s, e, k) for k, d in devices.items() for form in _device_forms(d.name)
                     for s, e in _spans(key, form)})
    mentions = [(s, e, k) for s, e, k in mentions
                if not any(s >= a and e <= b and (s, e) != (a, b) for a, b, _ in mentions)]
    rest = _without(key, [(s, e) for s, e, _ in mentions])
    known_rooms = {d.room for d in devices.values() if d.room}
    room_hits = [(s, e, room) for room in known_rooms for s, e in _spans(rest, _form(room))]
    room_hits = [(s, e, room) for s, e, room in room_hits
                 if not any(s >= a and e <= b and (s, e) != (a, b) for a, b, _ in room_hits)]
    if not (mentions or room_hits or words & (DEVICE_WORDS | REFERENCE)):
        return None
    # Inspect outside names: a device called "Power" does not select a power metric.
    semantic_words = set(rest.split())
    metrics = {kind for kind, aliases in METRICS.items() if semantic_words & aliases}
    if 'statut' in semantic_words:
        metrics.add('state')
    interrogative = (QUESTION.search(raw) or re.search(r'\b(?:est (?:elle|il)|sont (?:elles|ils))\b', raw)
                     or re.match(r'^(?:et|and) (?:son|sa|ses|its|the)\b', raw))
    kind = ('capabilities' if CAPABILITY.search(rest) else
            'state' if (interrogative and (metrics or BINARY.search(rest))) or NO_READ.search(raw) else None)
    if kind is None:
        return None
    # Inventory labels are not evidence of the user's language.
    syntax = rest
    if re.match(r'^(?:and )?(?:what|which|how|is|are|read|show|tell|give|list|do not|don t|its|state|status|capabilities|controls)\b', syntax):
        language = 'en'
    elif re.match(r'^(?:et )?(?:que|quel|quelle|quels|quelles|la|le|les|est|sont|lis|lire|affiche|montre|dis|donne|ne|son|sa|ses|etat|statut|capacites|commandes)\b', syntax):
        language = 'fr'
    else:
        language = None
    def result(status, **kw):
        return ReadRequest(kind, status, language=language, **kw)
    if NO_READ.search(raw):
        return result('no_read')
    if COMPLEX.search(rest):
        return result('unsupported')
    if PERIOD.search(rest):
        return result('unsupported_period')
    if snapshot is None:
        return result('unavailable')
    if len(metrics) > 1:
        return result('unsupported_metric')
    metric = next(iter(metrics), 'state') if kind == 'state' else None
    channels = list(re.finditer(r'\b(?:channel|canal|voie)\s+([1-9][0-9]*)\b', rest))
    if len(channels) > 1:
        return result('ambiguous')
    channel = int(channels[0].group(1)) if channels else None
    rest = _without(rest, [(m.start(), m.end()) for m in channels])
    room_spans = [(s, e) for _, _, room in room_hits for s, e in _spans(rest, _form(room))]
    residual = _without(rest, room_spans)
    from .state_read import READ_WORDS
    unknown = set(residual.split()) - set(_form(' '.join(FRAME | READ_WORDS)).split())
    if unknown and mentions:
        return result('not_found')
    rooms = {room for _, _, room in room_hits}
    if len(rooms) > 1:
        return result('ambiguous')
    if re.search(r'\b(?:dans|in|piece|room)\b', rest) and not rooms:
        return result('ambiguous')
    ids = {k for _, _, k in mentions}
    if len({(s, e) for s, e, _ in mentions}) > 1:
        return result('ambiguous', candidates=tuple(sorted(ids)))
    if ids:
        native = {k for k in ids if contains(raw, normalize(devices[k].name))}
        ids = native or ids
        match = 'name' if native else 'lexical'
    elif words & REFERENCE and not words & DEVICE_WORDS and not unknown:
        if previous_device not in devices:
            return result('no_reference')
        ids, match = {previous_device}, 'reference'
    else:
        ids, match = set(devices), 'type_room'
        types = {domain for domain, aliases in DOMAINS.items() if words & aliases}
        if len(types) > 1:
            return result('ambiguous')
        if types:
            domain = next(iter(types))
            ids = {k for k in ids if any((c.capability or '').startswith(domain + '_') for c in devices[k].commands)}
        elif not rooms and not unknown:
            return result('ambiguous')
        if unknown:
            ids = {k for k in ids if unknown <= set(_form(devices[k].name).split())}
    from .selection import domain_for
    domain = domain_for(rest)
    if domain:
        ids = {k for k in ids if devices[k].domain == domain or any(
            (c.capability or '').startswith(domain + '_') for c in devices[k].commands)}
    if rooms:
        room = next(iter(rooms))
        ids = {k for k in ids if devices[k].room == room}
    if not ids:
        return result('not_found')
    if len(ids) != 1:
        return result('ambiguous', candidates=tuple(sorted(ids)))
    device_id = next(iter(ids))
    device = devices[device_id]
    return result('ready', device_id=device_id, metric=metric, channel=channel, match=match,
                  device_name=device.name, room=device.room)


def render_read_problem(plan, language):
    fr = language == 'fr'
    if plan.status in ('ambiguous', 'no_reference', 'unsupported'):
        return ('Précise un seul appareil et sa pièce, ainsi que l’état ou les capacités souhaités.' if fr else
                'Please specify one device and its room, and the state or capabilities you want.')
    if plan.status == 'no_read':
        return ('D’accord, je ne consulte pas cet appareil.' if fr else 'Okay, I will not read that device.')
    if plan.kind == 'capabilities' and plan.status == 'unavailable':
        return render_lookup_results([{'success': False}], language=language)
    return render_state({'status': plan.status, 'success': False}, language)


async def answer_read_request(plan, provider, language):
    """Evidence-derived prose. No action provider or inference is accessible here."""
    if plan.status != 'ready':
        if (plan.kind == 'state' and plan.status == 'ambiguous' and plan.candidates
                and getattr(provider, 'indexer', None) is not None):
            snapshot = provider.indexer.get_snapshot()
            names = [snapshot.devices[k].name for k in plan.candidates
                     if k in snapshot.devices and snapshot.devices[k].enabled]
            result = {'success': False, 'status': 'ambiguous',
                      'devices': names[:5], 'total_devices': len(names)}
            return render_state(result, language), result
        return render_read_problem(plan, language), {'success': False, 'status': plan.status}
    if plan.kind == 'state':
        reader = getattr(provider, 'read_resolved_state', None)
        result = await reader(plan.device_id, plan.metric, plan.channel, expected_identity=(plan.device_name, plan.room)) if reader else {'success': False, 'status': 'unavailable'}
        return render_state(result, language), result
    snapshot = provider.indexer.get_snapshot()
    device = snapshot.devices.get(plan.device_id)
    if device is None or not device.enabled or (device.name, device.room) != (plan.device_name, plan.room):
        result = {'success': False, 'status': 'not_found'}
        return render_state(result, language), result
    commands = [c for c in device.commands if c.type == 'action' and c.capability]
    if plan.channel is not None:
        commands = [c for c in commands if normalize(c.name).split()[-1:] == [str(plan.channel)]]
    if not commands:
        return ((f'Aucune commande de contrôle répertoriée pour {device.name}. Aucune action effectuée.' if language == 'fr' else
                 f'No controls are listed for {device.name}. No action taken.'), {'success': True, 'status': 'no_controls'})
    candidates = [{'device_id': plan.device_id, 'device_name': device.name, 'room': device.room,
                   'cmd_id': c.cmd_id, 'capability': c.capability} for c in commands]
    result = {'success': True, 'status': 'ready', 'candidates': candidates, 'total_found': len(candidates)}
    return render_lookup_results([result], language=language), result
