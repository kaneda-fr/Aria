"""Inventory-grounded information selection and bilingual replies without generation."""
import re
import time

from .selection import normalize, contains, FILLER, entity_text, split_channel, matches_channel, location_needs_clarification, domain_for
from .provider import InventoryProvider
from .weather_conditions import PROVIDER_CONDITIONS


METRICS = {
    'temperature': {'temperature', 'temperatures', 'temp'},
    'humidity': {'humidity', 'humidite'},
    'pressure': {'pressure', 'pression'},
    'wind': {'wind', 'vent'},
    'rain': {'rain', 'pluie', 'precipitation', 'precipitations'},
    'power': {'power', 'puissance'},
    'energy': {'energy', 'energie', 'consommation', 'consumption'},
    'condition': {'condition', 'conditions'},
    'state': {'state', 'status', 'statut', 'etat', 'position', 'brightness', 'luminosite'},
}
LABELS = {
    'temperature': ('température', 'temperature'), 'humidity': ('humidité', 'humidity'),
    'pressure': ('pression', 'pressure'), 'wind': ('vent', 'wind'),
    'rain': ('pluie', 'rain'), 'power': ('puissance', 'power'),
    'energy': ('consommation', 'energy'), 'condition': ('conditions', 'conditions'),
    'state': ('état', 'state'),
}
# Domain enum labels, not question/answer templates. Unknown text is never translated by guesswork.
WEATHER_CONDITIONS = [
    ('clear sky', 'ciel dégagé'), ('sunny', 'ensoleillé'),
    ('few clouds', 'peu nuageux'), ('partly cloudy', 'partiellement nuageux'),
    ('scattered clouds', 'nuages épars'), ('cloudy', 'nuageux'),
    ('overcast clouds', 'couvert'), ('broken clouds', 'nuageux'),
    ('light rain', 'pluie faible'), ('moderate rain', 'pluie modérée'),
    ('heavy rain', 'forte pluie'), ('heavy intensity rain', 'pluie de forte intensité'),
    ('very heavy rain', 'très forte pluie'), ('extreme rain', 'pluie extrême'),
    ('rain', 'pluie'), ('rain showers', 'averses de pluie'),
    ('light shower rain', 'faibles averses de pluie'),
    ('shower rain', 'averses de pluie'), ('freezing rain', 'pluie verglaçante'),
    ('drizzle', 'bruine'), ('light drizzle', 'bruine légère'),
    ('thunderstorm', 'orage'), ('thunderstorm with rain', 'orage avec pluie'),
    ('mist', 'brume'), ('fog', 'brouillard'), ('haze', 'brume sèche'),
    ('snow', 'neige'), ('light snow', 'faible neige'), ('heavy snow', 'forte neige'),
    ('sleet', 'grésil'), ('hail', 'grêle'), ('windy', 'venteux'),
] + list(PROVIDER_CONDITIONS)
READ_WORDS = set('weather meteo read get show tell me current currently actual actuelle actuel '
                 'now reading readings value values what whats is are the of for in at a an '
                 'quel quelle quels quelles est sont donne lis lire affiche valeur valeurs '
                 'maintenant actuellement le la les de du des au aux une un il fait faitil '
                 'light lights lumiere lumieres shutter shutters volet volets please stp '
                 'can you could would peux tu appareil device devices sensor capteur'.split())
READ_WORDS |= set().union(*METRICS.values()) | FILLER
READ_WORDS |= {'indique', 'indiquent', 'indicates', 'indicate', 'report', 'reports',
               'measure', 'measures', 'measuring', 'mesure', 'mesures', 'mesurer',
               'consomme', 'consomment', 'using'}


def metric_for(command):
    tokens = set(normalize(command.name + ' ' + (command.capability or '')).split())
    if (command.capability or '').endswith('_state'):
        return 'state'
    for metric, words in METRICS.items():
        if tokens & words:
            return None if metric == 'condition' and command.subType != 'string' else metric
    return None

def current_info(command):
    return command.type == 'info' and not re.search(
        r'\+|\b\d+\s*h\b|\b(min|max|minimum|maximum|demain|tomorrow|day|jour|hour|heure)\b',
        command.name.lower())


def measurement_request(text):
    """Recognize explicit measurement questions when a model chooses catalog search."""
    key = normalize(text)
    words = set(key.split())
    if words & {'commands', 'command', 'controls', 'capability', 'capabilities', 'tools',
                'functions', 'available', 'support', 'supports', 'commandes', 'commande',
                'outils', 'fonctions', 'capacites', 'disponibles', 'list', 'liste', 'find', 'cherche'}:
        return False
    if words & {'not', 'pas', 'never', 'jamais'}:
        return False
    return bool(re.match(r'^(?:what|which|quel|quelle|quels|quelles|combien|lis|lire|read|donne|give|affiche|show|how)\b', key)
                and words & set().union(*METRICS.values()))


def history_source_choice(snapshot, original, history):
    """Resolve a reply only against sources in the last rendered clarification."""
    ordinal = re.fullmatch(
        r'(?:(?:the|la|le) )?(first|second|third|fourth|fifth|premier|premiere|deuxieme|seconde|troisieme|quatrieme|cinquieme)'
        r'(?: (?:source|one))?(?: (?:please|s il te plait|s il vous plait))?', original)
    prior = next((m.get('content', '') for m in reversed(history)
                  if m.get('role') == 'assistant'), '')
    prior_key = normalize(prior)
    clarification = prior.rstrip().endswith('?') and prior_key.startswith(
        ('quelle source souhaitez vous ', 'which source do you mean '))
    if not clarification:
        return {} if ordinal else None
    if re.match(r'^(?:turn|switch|allume|allumer|eteins|eteindre|ouvre|ouvrir|ferme|fermer|'
                r'open|close|stop|arrete|set|regle|dim)\b', original) and not any(
                    normalize(d.name) == re.sub(r' (?:please|s il te plait|s il vous plait)$', '', original)
                    for d in snapshot.devices.values() if d.enabled):
        return None
    if set(original.split()) & {'not', 'pas', 'never', 'jamais'}:
        return None
    matches = []
    for device in snapshot.devices.values():
        name = normalize(device.name)
        position = (' ' + prior_key + ' ').find(' ' + name + ' ')
        if device.enabled and position >= 0 and any(current_info(c) for c in device.commands):
            matches.append((position, -len(name), device))
    choices, end = [], -1
    for position, length, device in sorted(matches, key=lambda x: (x[0], x[1])):
        if position >= end:
            choices.append(device)
            end = position - length
    if not choices:
        return {} if ordinal else None
    if ordinal:
        ordinals = {'first': 0, 'premier': 0, 'premiere': 0, 'second': 1,
                    'deuxieme': 1, 'seconde': 1, 'third': 2, 'troisieme': 2,
                    'fourth': 3, 'quatrieme': 3, 'fifth': 4, 'cinquieme': 4}
        index = ordinals[ordinal.group(1)]
        labels = [normalize(label) for label in prior.split(':', 1)[-1].rstrip(' ?').split(', ')]
        selected = [d for d in choices if index < len(labels) and normalize(d.name) == labels[index]]
        if index < len(labels) and not selected:
            return {'status': 'not_found'}
    else:
        choice_text = re.sub(
            r'^(?:je parle de|je veux dire|je choisis|c est|i mean|i meant|i choose)\b', '', original).strip()
        key, _ = split_channel(entity_text(choice_text))
        native = [d for d in choices if contains(choice_text, normalize(d.name))]
        exact = native or [d for d in choices if contains(key, entity_text(d.name))]
        domain_text = choice_text
        for device in native:
            domain_text = (' ' + domain_text + ' ').replace(' ' + normalize(device.name) + ' ', ' ').strip()
        domain = domain_for(domain_text)
        selected = exact or choices
        if domain:
            selected = [d for d in selected if d.domain == domain or any(
                (c.capability or '').startswith(domain + '_') for c in d.commands)]
        if exact:
            longest = max(len(entity_text(d.name)) for d in exact)
            selected = [d for d in selected if len(entity_text(d.name)) == longest]
            for device in selected:
                key = (' ' + key + ' ').replace(' ' + entity_text(device.name) + ' ', ' ').strip()
        wanted = set(key.split()) - READ_WORDS - {'meme', 'juste', 'just', 'plutot'}
        if not wanted and not exact and not domain:
            return None
        selected = [d for d in selected if wanted <= set(entity_text(d.name).split())]
        if not selected:
            return None
    if len(selected) != 1:
        return {'status': 'ambiguous', 'devices': [d.name for d in (selected or choices)[:5]],
                'total_devices': len(selected or choices)}
    device = selected[0]
    # Duplicate names cannot be disambiguated by an ordinal/text label alone.
    if sum(normalize(d.name) == normalize(device.name) for d in snapshot.devices.values()
           if d.enabled) > 1:
        return {'status': 'ambiguous', 'devices': [device.name]}
    question = ''
    # Keep the closest explicit metric across consecutive source clarifications.
    for i in range(len(history) - 1, 0, -1):
        if history[i].get('role') != 'assistant':
            continue
        answer = history[i].get('content', '')
        if not normalize(answer).startswith(
                ('quelle source souhaitez vous ', 'which source do you mean ')):
            break
        if history[i - 1].get('role') != 'user':
            break
        question = normalize(history[i - 1].get('content', ''))
        for source in snapshot.devices.values():
            question = (' ' + question + ' ').replace(' ' + normalize(source.name) + ' ', ' ').strip()
        if set(question.split()) & set().union(*METRICS.values()):
            break
    current = original
    for source in snapshot.devices.values():
        current = (' ' + current + ' ').replace(' ' + normalize(source.name) + ' ', ' ').strip()
    if set(current.split()) & set().union(*METRICS.values()):
        question = current
    _, channel = split_channel(original)
    if channel is None:
        _, channel = split_channel(question)
    query = device.name + (' channel ' + str(channel) if channel is not None else '')
    return {'query': query, 'metric_text': question}


def resolved_read_request(snapshot, text, history):
    """Dispatch a grounded source choice or explicit state label without generation."""
    original = normalize(text)
    reference = history_source_choice(snapshot, original, history)
    if reference:
        metrics = sorted(k for k, words in METRICS.items()
                         if set(reference.get('metric_text', '').split()) & words)
        return {'query': reference.get('query', text),
                **({'metric': metrics[0] if len(metrics) == 1 else metrics} if metrics else {})}
    if re.match(r'^(?:etat|statut|status|state)\b', original):
        target_text, _ = split_channel(entity_text(original))
        wanted = set(target_text.split()) - READ_WORDS
        known_targets = [set(entity_text(d.name).split()) | set(entity_text(d.room or '').split())
                         for d in snapshot.devices.values() if d.enabled]
        # "State" can also be a general instruction or a science noun. Only
        # bypass generation when inventory evidence accounts for the target.
        if (wanted and any(wanted <= known for known in known_targets)) or (
                not wanted and domain_for(original)):
            return {'query': text, 'metric': 'state'}
    return None


def ground_read_source(snapshot, original, query, room, history=()):
    """Keep named sources/explicit rooms from the user, not generated arguments."""
    reference = history_source_choice(snapshot, original, history)
    if reference is not None:
        if not reference:
            return {"status": "not_found"}
        if reference.get("status"):
            return reference
        return {**reference, "room": None}
    location = re.search(r'\b(?:in(?: the)?(?: room)?|dans(?: la piece)?|de la piece)\s+(.+?)(?:\s+please)?$', original)
    explicit_room = re.search(r'\b(?:in(?: the)? room|dans(?: la)? piece|de la piece)\b', original)
    source_text = original[:location.start()] if location and explicit_room else original
    normalizer = normalize
    if not any(contains(source_text, normalize(d.name)) for d in snapshot.devices.values()):
        normalizer = entity_text
    if location_needs_clarification(snapshot, source_text,
                                    location.group(1) if location else None, READ_WORDS):
        return {'status': 'ambiguous', 'devices': [], 'clarification': 'location'}
    remaining = normalizer(source_text)
    named = []
    names = sorted({normalizer(d.name) for d in snapshot.devices.values()}, key=len, reverse=True)
    for name in names:
        # A field word alone is not a source mention.
        if not (set(name.split()) - READ_WORDS):
            explicit = re.search(r'\b(?:source|appareil|device|capteur|named|nomme|from|de|depuis)\s+' + re.escape(name) + r'\b', original)
            if not explicit and not re.search(r'\b(?:compare|comparez|comparer|versus)\b', original):
                continue
        if contains(remaining, name):
            named.extend(d for d in snapshot.devices.values() if normalizer(d.name) == name)
            remaining = (' ' + remaining + ' ').replace(' ' + name + ' ', ' ').strip()
    if len(named) > 1:
        return {'status': 'ambiguous', 'devices': [d.name for d in named[:5]], 'total_devices': len(named)}
    if room and not contains(entity_text(remaining), entity_text(room)):
        room = None
    if location:
        location_text = location.group(1)
        # A known room can precede a state predicate instead of ending the
        # sentence. Trim only recognized predicate grammar, never unknown room
        # qualifiers (including a lone French "est", which can mean east).
        predicates = {'allume', 'allumee', 'eteint', 'eteinte', 'switched', 'on', 'off'}
        grammar = READ_WORDS | predicates | {'elle', 'right'}
        rooms = sorted({normalize(d.room) for d in snapshot.devices.values() if d.room},
                       key=len, reverse=True)
        for known_room in rooms:
            if location_text.startswith(known_room + ' '):
                suffix = set(location_text[len(known_room):].split())
                if suffix & predicates and suffix <= grammar:
                    location_text = known_room
                    break
        words = set(entity_text(location_text).split()) - READ_WORDS
        sources = named or list(snapshot.devices.values())
        entity_location = words and any(words <= set(entity_text(d.name).split()) for d in sources)
        if explicit_room or not entity_location:
            room = location_text
        else:
            room = None
            if not named:
                query = ('weather ' if set(original.split()) & {'weather', 'meteo'} else '') + location_text
    if named:
        query = named[0].name
    elif original and set(entity_text(query).split()) - READ_WORDS - set(entity_text(original).split()):
        return {'status': 'not_found'}
    _, channel = split_channel(original)
    if channel is not None:
        query, _ = split_channel(entity_text(query))
        query += ' channel ' + str(channel)
    return {'query': query, 'room': room, 'metric_text': remaining,
            'domain': domain_for(remaining) if named else domain_for(original)}

def select_read_commands(snapshot, query, metric=None, room=None, *, domain=None):
    raw_key = normalize(query)
    key, channel = split_channel(entity_text(query))
    if not room and not any(contains(raw_key, normalize(d.name))
                            for d in snapshot.devices.values() if d.enabled):
        location = re.search(r'\b(?:room|piece)\s+(.+)$', key)
        if location:
            room = location.group(1)
            key = key[:location.start()].strip()
    if re.search(r'\b(demain|tomorrow|forecast|prevision|hier|yesterday|semaine|week|tonight|soir)\b', key):
        return {'status': 'unsupported_period'}
    if location_needs_clarification(snapshot, query, room, READ_WORDS):
        return {'status': 'ambiguous', 'devices': [], 'clarification': 'location'}
    raw_metrics = metric if isinstance(metric, (set, frozenset, tuple, list)) else [metric] if metric else []
    requested = set()
    for value in raw_metrics:
        word = normalize(value)
        kind = next((k for k, v in METRICS.items() if word == k or word in v), None)
        if kind is None:
            return {'status': 'unsupported_metric'}
        requested.add(kind)
    if not raw_metrics:
        requested = {k for k, v in METRICS.items() if set(key.split()) & v}
    if len(requested) > 8:
        return {'status': 'unsupported_metric'}
    weather = bool(set(key.split()) & {'meteo', 'weather'})
    native = [d for d in snapshot.devices.values() if d.enabled and
              contains(raw_key, normalize(d.name))]
    if native:
        longest_native = max(len(normalize(d.name)) for d in native)
        native_ids = {d.eq_id for d in native if len(normalize(d.name)) == longest_native}
    devices = [d for d in snapshot.devices.values() if d.enabled and
               any(c.type == 'info' for c in d.commands)]
    if domain:
        devices = [d for d in devices if d.domain == domain or any(
            (c.capability or '').startswith(domain + '_') for c in d.commands)]
    if room:
        devices = [d for d in devices if entity_text(d.room) == entity_text(room)]
    if weather:
        devices = [d for d in devices if any('weather' in t.lower() or 'meteofull' in t.lower()
                                           for t in d.tags) or
                   set(entity_text(d.name).split()) & {'meteo', 'weather'}]
    if native:
        devices = [d for d in devices if d.eq_id in native_ids]
        if not devices:
            return {'status': 'not_found'}
    exact = [d for d in devices if contains(raw_key, normalize(d.name))]
    exact = exact or [d for d in devices if contains(key, entity_text(d.name))]
    if exact:
        longest = max(len(entity_text(d.name)) for d in exact)
        devices = [d for d in exact if len(entity_text(d.name)) == longest]
        remainder = key
        for d in devices:
            remainder = remainder.replace(entity_text(d.name), ' ')
        if set(remainder.split()) - READ_WORDS:
            return {'status': 'not_found'}
    else:
        from .selection import domain_for
        domain = domain_for(key)
        if domain:
            devices = [d for d in devices if any((c.capability or '').startswith(domain + '_')
                                                for c in d.commands)]
        wanted = set(key.split()) - READ_WORDS
        if not wanted and not weather and not room:
            return {'status': 'not_found'}
        devices = [d for d in devices if wanted <= set(entity_text(d.name).split())]
    if not devices:
        return {'status': 'not_found'}
    if requested:
        devices = [d for d in devices if requested <=
                   {metric_for(c) for c in d.commands if current_info(c) and
                    (channel is None or matches_channel(c, channel))}]
        if not devices:
            return {'status': 'unsupported_metric'}
    if len(devices) != 1:
        return {'status': 'ambiguous', 'devices': [d.name for d in devices[:5]],
                'total_devices': len(devices)}
    device = devices[0]
    weather = weather or any('weather' in t.lower() or 'meteofull' in t.lower() for t in device.tags)
    commands = []
    for c in device.commands:
        if channel is not None and not matches_channel(c, channel):
            continue
        if c.type != 'info' or re.search(r'\+|\b\d+\s*h\b|\b(min|max|minimum|maximum|demain|tomorrow|day|jour|hour|heure)\b', c.name.lower()):
            continue
        kind = metric_for(c)
        if requested and kind not in requested:
            continue
        if not requested and kind not in ({'temperature', 'humidity', 'wind'} if weather else {'state'}):
            continue
        commands.append(c)
    # Jeedom's typed capability outranks a coincidental word in a technical label.
    preferred = {}
    for c in commands:
        kind = metric_for(c)
        if c.capability == kind or (kind == 'state' and (c.capability or '').endswith('_state')) or (
                kind == 'energy' and c.capability == 'consumption'):
            preferred.setdefault(kind, set()).add(c.cmd_id)
    commands = [c for c in commands if metric_for(c) not in preferred or
                c.cmd_id in preferred[metric_for(c)]]
    if not commands or len(commands) > 8:
        return {'status': 'unsupported_metric'}
    return {'status': 'ready', 'commands': commands}


def render_state(result, language='en'):
    fr = language == 'fr'
    status = result.get('status')
    if status == 'ambiguous':
        if not result.get('devices'):
            return ('De quel appareil ou de quelle pièce souhaitez-vous lire une valeur ?' if fr else
                    'Which device or room would you like me to read a value from?')
        names = ', '.join(result.get('devices', []))
        return ('Quelle source souhaitez-vous : ' if fr else 'Which source do you mean: ') + names + ' ?'
    if status == 'unsupported_period':
        return ('Je peux lire les valeurs actuelles de Jeedom, mais pas encore cette période.' if fr else
                'I can read Jeedom’s current values, but that time period is not supported yet.')
    if status == 'not_found':
        return ('Je ne trouve pas cet appareil dans l’inventaire disponible.' if fr else
                'I cannot find that device in the available inventory.')
    if status == 'unsupported_metric':
        return ('Quelle mesure de cet appareil souhaitez-vous lire ?' if fr else
                'Which measurement from that device would you like to read?')
    rows = result.get('readings', [])
    good = [r for r in rows if r.get('status') == 'ok']
    if not good:
        return ('Jeedom ne fournit pas de valeur disponible pour cette demande.' if fr else
                'Jeedom has no available value for this request.')
    parts = []
    untranslated = False
    for row in good:
        kind = row.get('metric', 'state')
        label = LABELS.get(kind, LABELS['state'])[0 if fr else 1]
        if sum(other.get('device_id', other.get('device_name')) == row.get('device_id', row.get('device_name'))
               and other.get('metric', 'state') == kind for other in good) > 1:
            suffix = normalize(row.get('name', '')).split()
            if suffix and suffix[-1].isdigit():
                label += (' canal ' if fr else ' channel ') + suffix[-1]
            elif row.get('name'):
                label += ' (' + row['name'] + ')'
        value = row['value']
        if kind == 'condition':
            pair = next((p for p in WEATHER_CONDITIONS
                         if normalize(value) in (normalize(p[0]), normalize(p[1]))), None)
            if pair is None:
                untranslated = True
                continue
            value = pair[1 if fr else 0]
        if isinstance(value, float):
            value = format(value, 'g')
        parts.append(f"{row['device_name']}, {label} : {value} {row.get('unit') or ''}".strip())
    unavailable_condition = ('La condition météo de Jeedom n’est pas reconnue.' if fr else
                             'Jeedom returned an unrecognized weather condition.')
    if not parts:
        return unavailable_condition
    reply = ('Jeedom indique : ' if fr else 'Jeedom reports: ') + '; '.join(parts) + '.'
    reply += (' Heure de mesure inconnue.' if fr else ' Measurement time is unknown.')
    if untranslated:
        reply += ' ' + unavailable_condition
    if len(good) < len(rows):
        reply += (' Certaines valeurs sont indisponibles.' if fr else ' Some values are unavailable.')
    return reply


class ReadOnlyInventory(InventoryProvider):
    """Use the LAN bridge from an otherwise loopback-only inference process."""
    def start(self):
        import httpx
        super().start()
        self.read_socket = self.config['read_socket']
        self.catalog_checked_at = 0
        self.bridge_status = {}
        with httpx.Client(transport=httpx.HTTPTransport(uds=self.read_socket),
                          base_url='http://localhost', trust_env=False, timeout=2) as client:
            response = client.get('/catalog')
            response.raise_for_status()
            self.apply_catalog(response.json())
        self.read_client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self.read_socket),
            base_url='http://localhost', trust_env=False, timeout=2)

    def apply_catalog(self, data):
        devices = [self._deserialize_device(d) for d in data['devices'].values()]
        self.indexer.rebuild(devices)
        self.bridge_status = {key: data.get(key) for key in (
            'connected', 'subscribed', 'seed_last_update', 'live_discovery_count',
            'last_discovery_at', 'catalog_complete', 'catalog_version')}
        self.catalog_checked_at = time.monotonic()

    async def refresh_live_inventory(self, force=False):
        if not force and time.monotonic() - self.catalog_checked_at < 2:
            return
        import httpx
        try:
            response = await self.read_client.get('/catalog', timeout=.5, params={
                'version': self.bridge_status.get('catalog_version', '')})
            response.raise_for_status()
            data = response.json()
            if data.get('unchanged'):
                self.bridge_status.update({k: v for k, v in data.items() if k != 'unchanged'})
                self.catalog_checked_at = time.monotonic()
            else:
                self.apply_catalog(data)
        except (httpx.HTTPError, ValueError, KeyError):
            self.bridge_status['connected'] = False
            self.catalog_checked_at = time.monotonic()

    async def read_state(self, args):
        import httpx
        original = normalize(args.get('request_text', ''))
        if re.search(r'\b(demain|tomorrow|forecast|prevision|hier|yesterday|semaine|week|tonight|soir)\b', original):
            return {'success': False, 'status': 'unsupported_period'}
        await self.refresh_live_inventory(force=True)
        snapshot = self.indexer.get_snapshot()
        source = ground_read_source(snapshot, original, str(args.get('query', '')), args.get('room'),
                                    args.get('request_history', ()))
        if source.get('status'):
            return {'success': False, **source}
        explicit_metrics = {k for k, words in METRICS.items() if set(source['metric_text'].split()) & words}
        metric = (next(iter(explicit_metrics)) if len(explicit_metrics) == 1 else
                  sorted(explicit_metrics) if explicit_metrics else args.get('metric'))
        selection = select_read_commands(snapshot, source['query'], metric, source['room'],
                                         domain=source.get('domain'))
        if selection['status'] != 'ready':
            return {'success': False, **selection}
        commands = selection['commands']
        try:
            response = await self.read_client.post('/read', json={'cmd_ids': [c.cmd_id for c in commands]})
            response.raise_for_status()
            result = response.json()
            result['requested_metric'] = metric
            result['metric_recovered'] = bool(explicit_metrics and normalize(metric or '') != normalize(args.get('metric') or ''))
        except (httpx.HTTPError, ValueError):
            return {'success': False, 'status': 'disconnected'}
        kinds = {c.cmd_id: metric_for(c) for c in commands}
        for row in result.get('readings', []):
            row['metric'] = kinds.get(row['cmd_id'])
        return result

    async def close_read_connection(self):
        await self.read_client.aclose()

    async def read_resolved_state(self, device_id, metric='state', channel=None, *, expected_identity=None):
        """Internal server plan: revalidate identity, then read info commands only.

        This method is not a model tool argument or a public execution endpoint.
        A room/name already resolved by the server must not be parsed a second time.
        """
        from dataclasses import replace
        import httpx
        if not self.is_started:
            return {'success': False, 'status': 'unavailable'}
        await self.refresh_live_inventory(force=True)
        snapshot = self.indexer.get_snapshot()
        device = snapshot.devices.get(device_id)
        if (device is None or not device.enabled or
                (expected_identity is not None and (device.name, device.room) != expected_identity)):
            return {'success': False, 'status': 'not_found'}
        if channel is not None:
            device = replace(device, commands=[c for c in device.commands
                             if normalize(c.name).split()[-1:] == [str(channel)]])
        scoped = replace(snapshot, devices={device_id: device})
        selection = select_read_commands(scoped, device.name, metric, device.room)
        if selection['status'] != 'ready':
            return {'success': False, **selection}
        commands = selection['commands']
        if channel is not None:
            commands = [c for c in commands if normalize(c.name).split()[-1:] == [str(channel)]]
        if not commands or len(commands) > 8 or any(c.type != 'info' for c in commands):
            return {'success': False, 'status': 'unsupported_metric'}
        try:
            response = await self.read_client.post('/read', json={'cmd_ids': [c.cmd_id for c in commands]})
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError):
            return {'success': False, 'status': 'disconnected'}
        kinds = {c.cmd_id: metric_for(c) for c in commands}
        result['readings'] = [dict(row, metric=kinds[row['cmd_id']])
                              for row in result.get('readings', []) if row.get('cmd_id') in kinds]
        result['requested_metric'] = metric
        return result
