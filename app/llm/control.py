"""Bounded explicit-command grounding and outcome-based speech, shared by voice/text.

This deliberately accepts a small imperative grammar. Unsupported/compound requests
cannot acquire execution authority from model output. No inferred scheduling or state.
"""
from collections import OrderedDict
from dataclasses import dataclass
import copy
import re
import time

from app.plugins.inventory.selection import normalize, domain_for, select_commands


@dataclass
class Authorization:
    cmd_id: str
    device_id: str
    capability: str
    value: object
    execution: dict
    device_name: str
    room: str
    consumed: bool = False


@dataclass
class Plan:
    status: str
    language: str = 'en'
    command: dict | None = None
    authorization: Authorization | None = None


class ControlMemory:
    """Short-lived references belong to a session AND attributed speaker."""
    def __init__(self, limit=128, ttl=120):
        self.items = OrderedDict()
        self.limit, self.ttl = limit, ttl

    def get(self, key):
        item = self.items.get(key)
        if item is None:
            return None
        if time.monotonic() - item[1] > self.ttl:
            self.items.pop(key, None)
            return None
        return item[0]

    def remember(self, key, device_id):
        if not key[0]:
            return
        self.items[key] = (device_id, time.monotonic())
        self.items.move_to_end(key)
        while len(self.items) > self.limit:
            self.items.popitem(last=False)

    def forget(self, key):
        self.items.pop(key, None)

    def forget_session(self, session_id):
        for key in list(self.items):
            if key[0] == session_id:
                self.items.pop(key, None)


PREFIX = re.compile(r'^(?:bonjour aria|hello aria|hey aria|aria|bonjour|hello|please|s il te plait|s il vous plait|peux tu|pouvez vous|pourrais tu|could you|can you|would you|maintenant|now)\b\s*')
ACTION = re.compile(r'^(turn on|switch on|turn off|switch off|allume|allumer|eteins|eteindre|open|ouvre|ouvrir|close|ferme|fermer|set|regle|regler|dim)\b\s*(.*)$')
VERBS = {'turn on':'on','switch on':'on','allume':'on','allumer':'on',
         'turn off':'off','switch off':'off','eteins':'off','eteindre':'off',
         'open':'up','ouvre':'up','ouvrir':'up','close':'down','ferme':'down','fermer':'down',
         'set':'slider','regle':'slider','regler':'slider','dim':'slider'}
UNSAFE = re.compile(r'\b(?:not|don t|never|ne|pas|jamais|if|si|unless|when|quand|after|before|later|tomorrow|demain|apres|avant|then|puis|and|et|or|ou|instead|plutot|except|sauf)\b')
REFERENCE = re.compile(r'^(?:it|that same light|the same light|that light|that same shutter|the same shutter|that shutter|cette meme lumiere|cette lumiere|cette meme lampe|cette lampe|ce meme volet|ce volet|le|la)$')


def resolve(text, inventory, previous_device=None):
    query = normalize(text)
    for _ in range(4):
        newer = PREFIX.sub('', query, count=1)
        if newer == query:
            break
        query = newer
    # Explicit negation is a no-action request, not a model decision to reverse.
    english_negative = re.fullmatch(r'(?:do not|don t|never)\s+(.+)', query)
    if english_negative and ACTION.fullmatch(english_negative.group(1)):
        return Plan('no_action', 'en')
    french_negative = re.fullmatch(
        r'(?:ne|n)\s+(allume|allumer|eteins|eteindre|ouvre|ouvrir|ferme|fermer|regle|regler)\s+(?:pas|jamais)\s+.+', query)
    french_infinitive = re.fullmatch(r'ne\s+(?:pas|jamais)\s+(.+)', query)
    if french_negative or (french_infinitive and ACTION.fullmatch(french_infinitive.group(1))):
        return Plan('no_action', 'fr')
    # Also support "turn the kitchen light off".
    query = re.sub(r'^(?:turn|switch) (.+) (on|off)$', r'turn \2 \1', query)
    match = ACTION.fullmatch(query)
    if not match:
        return None
    verb, target = match.groups()
    target = re.sub(r'\s+(?:please|s il te plait|s il vous plait)$', '', target)
    language = 'fr' if verb in {'allume','allumer','eteins','eteindre','ouvre','ouvrir','ferme','fermer','regle','regler'} else 'en'
    plan = Plan('unsupported', language)
    if (UNSAFE.search(target) or not target
            or re.search(r'\b(?:turn|switch|allume|allumer|eteins|eteindre|open|ouvre|ouvrir|close|ferme|fermer|set|regle|regler|dim)\b', target)):
        return plan
    if not inventory or not getattr(inventory, 'is_started', False):
        plan.status = 'unavailable'
        return plan
    action = VERBS[verb]
    value = None
    if action == 'slider':
        # Require an explicit percentage; reject missing, fractional, relative values.
        level = re.search(r'\s+(?:a|to|at)\s+(\d+)\s*(?:pour cent|percent)?$', target)
        if not level or re.search(r'[-+]\s*\d|\d[.,]\d', text):
            plan.status = 'value_required'
            return plan
        value = int(level.group(1))
        target = target[:level.start()].strip()
        if not 0 <= value <= 100:
            plan.status = 'invalid_value'
            return plan
    domain = domain_for(target)
    context = {}
    if REFERENCE.fullmatch(target):
        if not previous_device:
            plan.status = 'no_reference'
            return plan
        context['device_id'] = previous_device
        snapshot = inventory.indexer.get_snapshot()
        prior = snapshot.devices.get(previous_device)
        if prior is None:
            plan.status = 'not_found'
            return plan
        domain = domain or domain_for(prior.name) or ('flap' if action in ('up','down') else 'light')
    else:
        # Explicit location is a hard filter, including nonexistent rooms.
        location = re.search(r'\b(?:dans(?: la piece)?|in(?: the)?(?: room)?|de la piece)\s+(.+)$', target)
        if location:
            context['room'] = location.group(1)
            target = target[:location.start()].strip()
    if not domain:
        domain = 'flap' if action in ('up','down') else 'light'
    if (domain == 'light' and action in ('up','down')) or (domain == 'flap' and action in ('on','off')):
        return plan
    context.update(domain=domain, capability=domain+'_'+action)
    candidates = select_commands(inventory.indexer.get_snapshot(), target, context)
    if len(candidates) != 1:
        plan.status = 'ambiguous' if candidates else 'not_found'
        return plan
    data = candidates[0]['data']
    command = inventory.indexer.get_command(data['cmd_id'])
    if not command or not command.execution:
        plan.status = 'unavailable'
        return plan
    plan.status, plan.command = 'ready', data
    plan.authorization = Authorization(data['cmd_id'], data['device_id'], data['capability'], value,
                                       copy.deepcopy(command.execution), data['device_name'], data['room'])
    return plan


def render(plan, result=None):
    fr = plan.language == 'fr'
    if plan.status != 'ready':
        replies = {
            'no_action': ("D’accord, je n’envoie aucune commande.", 'Okay, I will send no command.'),
            'ambiguous': ('Quel appareil et quelle pièce souhaitez-vous contrôler ?', 'Which device and room do you mean?'),
            'no_reference': ('De quel appareil parlez-vous ?', 'Which device do you mean?'),
            'not_found': ('Je ne trouve pas cet appareil dans la pièce demandée.', 'I cannot find that device in the requested room.'),
            'value_required': ('Indiquez un pourcentage entier pour cet appareil.', 'Please give a whole-number percentage for that device.'),
            'invalid_value': ('Ce pourcentage est hors limites.', 'That percentage is out of range.'),
            'unavailable': ('Le contrôle de cet appareil est indisponible.', 'Control of that device is unavailable.'),
            'unsupported': ('Précisez un seul appareil et une seule action immédiate.', 'Please specify one device and one immediate action.'),
        }
        return replies.get(plan.status, replies['unsupported'])[0 if fr else 1]
    if not result or not result.get('success'):
        return 'La commande a été refusée ; aucun changement confirmé.' if fr else 'The command was rejected; no change is confirmed.'
    action = plan.command['capability'].split('_',1)[1]
    wording = {'on':('allumer','turn on'), 'off':('éteindre','turn off'),
               'up':('ouvrir','open'), 'down':('fermer','close'), 'slider':('régler','set')}
    verb = wording[action][0 if fr else 1]
    target = plan.command['device_name']
    if action == 'slider':
        target += (' à ' if fr else ' to ') + str(plan.authorization.value) + ' %'
    if result.get('dry_run'):
        return (f'Test : commande pour {verb} {target} enregistrée ; aucun appareil modifié.' if fr else
                f'Test: command to {verb} {target} recorded; no device changed.')
    # Publish acceptance is not an observation of the physical device.
    return (f'Commande pour {verb} {target} transmise ; changement non confirmé.' if fr else
            f'Command to {verb} {target} submitted; change not confirmed.')
