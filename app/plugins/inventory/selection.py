"""Conservative, accent-insensitive inventory selection; no fuzzy fallback actions."""
import re
import unicodedata
from .lexical import lexical_form


def normalize(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(text).lower())
                   if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[a-z0-9]+', text))


DOMAINS = {
    'light': {'light', 'lights', 'lamp', 'lamps', 'bulb', 'bulbs', 'lumiere', 'lumieres', 'lampe', 'lampes', 'eclairage'},
    'flap': {'shutter', 'shutters', 'blind', 'blinds', 'volet', 'volets', 'store', 'stores'},
}
FILLER = set('the a an le la les l un une des du de d dans en in room piece named name nomme nommee appele appelee light lights lamp lamps bulb bulbs lumiere lumieres lampe lampes eclairage shutter shutters blind blinds volet volets store stores turn switch on off allume allumer eteins eteindre open close ouvre ouvrir ferme fermer set regle regler dim to a at percent pour cent please'.split())
GENERIC_LOCATIONS = frozenset({'salle', 'salles', 'piece', 'pieces', 'room', 'rooms'})


def domain_for(text):
    words = set(normalize(text).split())
    domains = [domain for domain, aliases in DOMAINS.items() if words & aliases]
    return domains[0] if len(domains) == 1 else None


def contains(text, phrase):
    return bool(phrase) and f' {phrase} ' in f' {text} '


# Local vocabulary equivalences, independent of any home's equipment names.
ENTITY_ALIASES = (
    ('salle a manger', 'salle manger'), ('dining room', 'salle manger'),
    ('living room', 'salon'), ('lounge', 'salon'), ('hallway', 'couloir'),
    ('corridor', 'couloir'), ('kitchen', 'cuisine'), ('bedroom', 'chambre'),
    ('office', 'bureau'), ('entrance', 'entree'), ('sofa', 'canape'), ('couch', 'canape'),
)


def entity_text(text):
    value = normalize(text)
    for alias, canonical in ENTITY_ALIASES:
        value = re.sub(r'\b' + alias + r'\b', canonical, value)
    return lexical_form(value)


def location_needs_clarification(snapshot, text, room=None, extra=()):
    """Generic location words do not identify a longer equipment or room name."""
    raw, key = normalize(text), entity_text(text)
    room_key = entity_text(room) if room else ''
    words = set(key.split())
    if not ((words | set(room_key.split())) & GENERIC_LOCATIONS):
        return False
    devices = [device for device in snapshot.devices.values() if device.enabled]
    rooms = {entity_text(device.room) for device in devices if device.room}
    if room:
        if room_key in rooms:
            return False
        room_words = set(room_key.split())
        if room_words & GENERIC_LOCATIONS and not (room_words - FILLER - GENERIC_LOCATIONS):
            return True
    if any(contains(raw, normalize(device.name)) or contains(key, entity_text(device.name))
           for device in devices):
        return False
    if any(contains(key, known) for known in rooms):
        return False
    return bool(words & GENERIC_LOCATIONS) and not (words - FILLER - set(extra) - GENERIC_LOCATIONS)


def split_channel(query):
    matches = list(re.finditer(r'\b(?:channel|canal|voie)\s+([1-9][0-9]*)\b', query))
    if len(matches) != 1:
        return query, None
    match = matches[0]
    return (query[:match.start()] + ' ' + query[match.end():]).strip(), int(match.group(1))


def matches_channel(command, channel):
    tokens = normalize(command.name).split()
    return bool(tokens and tokens[-1].isdigit() and int(tokens[-1]) == channel)

def select_commands(snapshot, text, context=None):
    """Return all strict matches, preserving native names before vocabulary aliases."""
    context = context or {}
    raw_query = normalize(text)
    query, channel = split_channel(entity_text(text))
    if not query and not context.get('device_id'):
        return []
    domain = context.get('domain') or domain_for(query)
    # A different device type can be a name qualifier, not disposable filler.
    other_types = set().union(*(aliases for kind, aliases in DOMAINS.items()
                                if domain and kind != domain))
    filler = FILLER - other_types
    devices = [(key, d) for key, d in snapshot.devices.items() if d.enabled]
    native = [(key, d) for key, d in devices if contains(raw_query, normalize(d.name))]
    if location_needs_clarification(snapshot, text, context.get('room')):
        return []
    named = native or [(key, d) for key, d in devices if contains(query, entity_text(d.name))]
    room = entity_text(context['room']) if context.get('room') else None
    if room:
        exact_room = [(key, d) for key, d in devices if normalize(d.room) == normalize(context['room'])]
        devices = exact_room or [(key, d) for key, d in devices if entity_text(d.room) == room]
    else:
        room_query = query
        # A room word embedded in a device name is not a separate room constraint.
        for _, d in named:
            room_query = (' ' + room_query + ' ').replace(' ' + entity_text(d.name) + ' ', ' ').strip()
        rooms = {entity_text(d.room) for _, d in devices if contains(room_query, entity_text(d.room))}
        if len(rooms) == 1:
            room = rooms.pop()
            devices = [(key, d) for key, d in devices if entity_text(d.room) == room]
        elif len(rooms) > 1:
            return []
    if context.get('device_id'):
        devices = [(key, d) for key, d in devices if key == context['device_id']]
    else:
        exact = [(key, d) for key, d in devices if contains(raw_query, normalize(d.name))]
        if native and not exact:
            return []
        exact = exact or [(key, d) for key, d in devices if contains(query, entity_text(d.name))]
        if exact:
            longest = max(len(entity_text(d.name)) for _, d in exact)
            devices = [(key, d) for key, d in exact if len(entity_text(d.name)) == longest]
            remainder = query
            for _, d in devices:
                remainder = remainder.replace(entity_text(d.name), ' ')
            if room:
                remainder = remainder.replace(room, ' ')
            if set(remainder.split()) - filler:
                return []
        else:
            remainder = query.replace(room, ' ') if room else query
            wanted = set(remainder.split()) - filler
            if wanted:
                devices = [(key, d) for key, d in devices if wanted <= set(entity_text(d.name).split())]
            elif not domain:
                return []
    found = {}
    for key, device in devices:
        for command in device.commands:
            if command.type != 'action' or not command.capability:
                continue
            if channel is not None and not matches_channel(command, channel):
                continue
            if domain and not command.capability.startswith(domain + '_'):
                continue
            if context.get('capability') and command.capability != context['capability']:
                continue
            found[command.cmd_id] = {'id': command.cmd_id, 'score': 1.0, 'data': {
                'cmd_id': command.cmd_id, 'device_id': key, 'device_name': device.name,
                'room': device.room, 'capability': command.capability,
                'type': command.type, 'subType': command.subType,
                **({'channel': channel} if channel is not None else {})}}
    return [found[key] for key in sorted(found)]
