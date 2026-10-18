"""Conservative, accent-insensitive inventory selection; no fuzzy fallback actions."""
import re
import unicodedata


def normalize(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(text).lower())
                   if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[a-z0-9]+', text))


DOMAINS = {
    'light': {'light', 'lights', 'lamp', 'lamps', 'bulb', 'bulbs', 'lumiere', 'lumieres', 'lampe', 'lampes', 'eclairage'},
    'flap': {'shutter', 'shutters', 'blind', 'blinds', 'volet', 'volets', 'store', 'stores'},
}
FILLER = set('the a an le la les l un une des du de d dans en in room piece named name nomme nommee appele appelee light lights lamp lamps bulb bulbs lumiere lumieres lampe lampes eclairage shutter shutters blind blinds volet volets store stores turn switch on off allume allumer eteins eteindre open close ouvre ouvrir ferme fermer set regle regler dim to a at percent pour cent please'.split())


def domain_for(text):
    words = set(normalize(text).split())
    domains = [domain for domain, aliases in DOMAINS.items() if words & aliases]
    return domains[0] if len(domains) == 1 else None


def contains(text, phrase):
    return bool(phrase) and f' {phrase} ' in f' {text} '


def select_commands(snapshot, text, context=None):
    """Return all strict matches so truncation cannot turn ambiguity into certainty."""
    context = context or {}
    query = normalize(text)
    if not query and not context.get('device_id'):
        return []
    room = normalize(context['room']) if context.get('room') else None
    domain = context.get('domain') or domain_for(query)
    devices = [(key, d) for key, d in snapshot.devices.items() if d.enabled]
    if room:
        devices = [(key, d) for key, d in devices if normalize(d.room) == room]
    else:
        rooms = {normalize(d.room) for _, d in devices if contains(query, normalize(d.room))}
        if len(rooms) == 1:
            room = rooms.pop()
            devices = [(key, d) for key, d in devices if normalize(d.room) == room]
        elif len(rooms) > 1:
            return []
    if context.get('device_id'):
        devices = [(key, d) for key, d in devices if key == context['device_id']]
    else:
        exact = [(key, d) for key, d in devices if contains(query, normalize(d.name))]
        if exact:
            # Do not let a shorter name embedded in a longer one create a false match.
            longest = max(len(normalize(d.name)) for _, d in exact)
            devices = [(key, d) for key, d in exact if len(normalize(d.name)) == longest]
            # Unknown selectors must not disappear just because a device name matches.
            remainder = query
            for _, d in devices:
                remainder = remainder.replace(normalize(d.name), ' ')
            if room:
                remainder = remainder.replace(room, ' ')
            if set(remainder.split()) - FILLER:
                return []
        else:
            remainder = query.replace(room, ' ') if room else query
            wanted = set(remainder.split()) - FILLER
            if wanted:
                devices = [(key, d) for key, d in devices
                           if wanted <= set(normalize(d.name).split())]
            elif not domain:
                return []
    found = {}
    for key, device in devices:
        for command in device.commands:
            if command.type != 'action' or not command.capability:
                continue
            if domain and not command.capability.startswith(domain + '_'):
                continue
            if context.get('capability') and command.capability != context['capability']:
                continue
            found[command.cmd_id] = {'id': command.cmd_id, 'score': 1.0, 'data': {
                'cmd_id': command.cmd_id, 'device_id': key, 'device_name': device.name,
                'room': device.room, 'capability': command.capability,
                'type': command.type, 'subType': command.subType}}
    return [found[key] for key in sorted(found)]
