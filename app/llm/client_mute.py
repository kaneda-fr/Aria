"""Local, confirmed client-input mute. No model or device tools authorize it.

Explicit FR/EN imperative grammar; arbitrary phrasing is not an intent classifier.
Confirmation is scoped to the same conversation and attributed speaker (not auth).
"""
from collections import OrderedDict
from dataclasses import dataclass
import re
import time
import unicodedata

DEFAULT_SECONDS = 300
MAX_SECONDS = 86400


def normalize(text):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                           if not unicodedata.combining(c)).replace('’', "'").split()).strip(' .!?')


EN = re.compile(r"^(?:(?:please|can you|could you|would you)\s+)*(?:mute(?: yourself)?|be quiet|go silent|stop (?:listening(?: to me)?|responding|talking))(?:\s+(.*))?$")
FR = re.compile(r"^(?:(?:s'il te plait|peux[- ]tu|pourrais[- ]tu|tu peux)\s+)*(?:mets[- ]toi en sourdine|met[- ]toi en sourdine|passe en mode silencieux|tais[- ]toi|te taire|te mettre en sourdine|fais silence|silence|arrete (?:de (?:parler|repondre)|d'ecouter|de m'ecouter)|ne m'ecoute plus)(?:\s+(.*))?$")
NUMBERS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
           'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
           'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
           'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
           'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
           'un': 1, 'une': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
           'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10, 'onze': 11, 'douze': 12,
           'treize': 13, 'quatorze': 14, 'quinze': 15, 'seize': 16, 'vingt': 20,
           'trente': 30, 'quarante': 40, 'cinquante': 50, 'soixante': 60}
UNITS = {'second': 1, 'seconds': 1, 'seconde': 1, 'secondes': 1,
         'minute': 60, 'minutes': 60, 'hour': 3600, 'hours': 3600, 'heure': 3600, 'heures': 3600}
YES = {'yes', 'yes please', 'yes i confirm', 'oui je confirme', 'confirm', 'confirmed', 'i confirm', 'go ahead', 'oui',
       'oui merci', 'oui confirme', 'confirme', 'je confirme', 'je le confirme', 'i confirm that', "d'accord", 'vas-y', 'ok', 'okay'}
NO = {'no', 'no thanks', 'cancel', 'never mind', 'nevermind', 'non', 'non merci',
      'annule', 'annuler', 'laisse tomber', 'pas maintenant'}


def duration(text):
    """Consume every token; unsupported/ambiguous units never silently default."""
    text = re.sub(r'^(?:for|pendant|pour)\s+', '', text)
    if not text:
        return DEFAULT_SECONDS
    text = text.replace('half an hour', '30 minutes').replace('a quarter of an hour', '15 minutes')
    text = re.sub(r"(?:une |une? )?demi[- ]heure", '30 minutes', text)
    text = text.replace("un quart d'heure", '15 minutes')
    text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)
    tokens = text.split()
    total, amount = 0., []
    for token in tokens:
        if token in UNITS:
            if not amount:
                return None
            words = [w for w in amount if w not in {'and', 'et'}]
            try:
                if len(words) == 1 and re.fullmatch(r'\d+(?:[.,]\d+)?', words[0]):
                    value = float(words[0].replace(',', '.'))
                elif len(words) == 1:
                    value = NUMBERS[words[0]]
                elif len(words) == 2 and NUMBERS[words[0]] >= 20 and 0 < NUMBERS[words[1]] < 10:
                    value = NUMBERS[words[0]] + NUMBERS[words[1]]
                else:
                    return None
            except (KeyError, ValueError):
                return None
            total += value * UNITS[token]
            amount = []
        elif token not in {'and', 'et'} or amount:
            amount.append(token)
    return int(total) if not amount and 1 <= total <= MAX_SECONDS and total.is_integer() else None


def intent(text):
    text = normalize(text)
    text = re.sub(r"(?:,? (?:please|s'il te plait|merci))$", '', text)
    for pattern, language in ((FR, 'fr'), (EN, 'en')):
        match = pattern.fullmatch(text)
        if match:
            return duration(match[1] or ''), language
    return None


def duration_label(seconds, language):
    for unit, size in (('hour' if language == 'en' else 'heure', 3600), ('minute', 60)):
        if seconds % size == 0:
            n = seconds // size
            return f'{n} {unit}{"s" if n != 1 else ""}'
    return f'{seconds} {"secondes" if language == "fr" else "seconds"}'


@dataclass
class Reply:
    text: str
    language: str
    status: str
    client_id: str | None = None
    seconds: int | None = None


class MuteCommands:
    def __init__(self, clock=time.monotonic, limit=256):
        self.clock, self.limit = clock, limit
        self.pending = OrderedDict()

    def forget_session(self, key):
        self.pending.pop(key, None)

    def handle(self, text, session, speaker, registry):
        now = self.clock()
        old = self.pending.pop(session, None)
        parsed = intent(text)
        if parsed is not None:
            seconds, language = parsed
            fr = language == 'fr'
            name = session[7:] if session and session.startswith('client:') else None
            if not registry or name not in registry.known:
                return Reply("Je ne peux pas identifier le client à mettre en sourdine." if fr else
                             "I can't identify which client to mute.", language, 'unavailable')
            if registry.is_muted(name):
                return Reply("Ce client est déjà en sourdine. Le minuteur actuel reste inchangé." if fr else
                             "This client is already muted. Its current timer is unchanged.", language, 'already_muted', name)
            if seconds is None:
                return Reply("Redemande la mise en sourdine avec une durée entre une seconde et 24 heures." if fr else
                             "Please ask to mute again with a duration between one second and 24 hours.", language, 'invalid_duration', name)
            self.pending[session] = (now + 30, speaker, name, seconds, language, registry.mute_revision(name))
            while len(self.pending) > self.limit:
                self.pending.popitem(last=False)
            label = duration_label(seconds, language)
            return Reply(f"Mettre ce client, {name}, en sourdine pendant {label} ? Confirme par oui ou non." if fr else
                         f"Mute this client, {name}, for {label}? Please confirm yes or no.", language, 'confirmation_required', name, seconds)
        if old is None:
            return None
        expires, owner, name, seconds, language, revision = old
        answer, fr = normalize(text), language == 'fr'
        if answer not in YES | NO:
            return None  # A different request abandons the old confirmation.
        if now >= expires or speaker != owner or registry is None or name not in registry.known or registry.mute_revision(name) != revision:
            return Reply("Cette confirmation n'est plus valide. Redemande la mise en sourdine." if fr else
                         "That confirmation is no longer valid. Please ask to mute again.", language, 'confirmation_invalid', name)
        if answer in NO:
            return Reply("Annulé. Je reste à l'écoute." if fr else "Cancelled. I'm still listening.", language, 'cancelled', name)
        registry.set_muted(name, True, seconds=seconds)
        label = duration_label(seconds, language)
        return Reply(f"Ce client est en sourdine pendant {label}. Je reprendrai ensuite automatiquement." if fr else
                     f"This client is muted for {label}. I'll resume automatically afterwards.", language, 'muted', name, seconds)
