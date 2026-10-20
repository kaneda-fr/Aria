"""Small, explicit requests whose answers come from local runtime facts."""
from dataclasses import dataclass
from datetime import datetime
import os
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.plugins.inventory.selection import normalize


@dataclass(frozen=True)
class LocalAnswer:
    text: str
    language: str
    kind: str


def clock_answer(text: str, *, now: datetime | None = None) -> LocalAnswer | None:
    """Read the host clock, never a model's estimate; do not match mixed intents."""
    key = normalize(text)
    match = re.fullmatch(
        r"(?:aria |please |s il te plait )?"
        r"(?P<question>what time is it|quelle heure est il)"
        r"(?: now| maintenant)?"
        r"(?: (?P<format>answer in hh mm|reponds au format hh mm))?"
        r"(?: (?P<zone>zurich time|heure de zurich|in zurich|a zurich|utc))?",
        key,
    )
    if not match:
        return None
    language = 'fr' if match['question'].startswith('quelle') else 'en'
    zone = match['zone']
    zone_name = ('UTC' if zone == 'utc' else 'Europe/Zurich') if zone else os.environ.get('ARIA_TIMEZONE', '').strip()
    try:
        tz = ZoneInfo(zone_name) if zone_name else None
        instant = now if now is not None else datetime.now().astimezone()
        if instant.tzinfo is None:
            raise ValueError('Clock instant must have a timezone')
        value = instant.astimezone(tz).strftime('%H:%M')
    except (ZoneInfoNotFoundError, ValueError):
        return LocalAnswer(
            'Le fuseau horaire local est mal configuré.' if language == 'fr' else
            'The local time zone is not configured correctly.', language, 'clock_unavailable')
    reply = value if match['format'] else (f'Il est {value}.' if language == 'fr' else f'It is {value}.')
    return LocalAnswer(reply, language, 'clock')


def capability_answer(text: str, inventory=None, actions=None, *, tools_enabled=True) -> LocalAnswer | None:
    """Describe registered capabilities without asking the model to invoke them."""
    from app.llm.tools import available_tools
    key = normalize(text)
    technical_fr = re.fullmatch(
        r'(?:nomme|liste) (?:les|tes)(?: deux)? outils(?: de fonction)?(?: dont tu disposes| disponibles)?'
        r'|quels (?:sont tes outils|outils as tu|outils peux tu utiliser)', key)
    technical_en = re.fullmatch(
        r'(?:name|list) (?:the|your)(?: two)?(?: function)? tools(?: available to you| available)?'
        r'|what tools (?:do you have|can you use)', key)
    general_fr = re.fullmatch(r'(?:aria )?(?:que peux tu faire|quelles sont tes capacites)', key)
    general_en = re.fullmatch(r'(?:aria )?(?:what can you do|what are your capabilities)', key)
    if not any((technical_fr, technical_en, general_fr, general_en)):
        return None
    french = bool(technical_fr or general_fr)
    language = 'fr' if french else 'en'
    registered = {item['function']['name'] for item in available_tools(inventory)}
    search = tools_enabled and bool(getattr(inventory, 'is_started', False)) and 'find_devices' in registered
    control = search and bool(getattr(actions, 'is_started', False)) and 'execute_command' in registered
    reading = tools_enabled and bool(getattr(inventory, 'is_started', False)) and 'get_device_state' in registered
    names = (['find_devices'] if search else []) + (['execute_command'] if control else []) + (['get_device_state'] if reading else [])
    if technical_fr or technical_en:
        reply = (' et ' if french else ' and ').join(names) + '.' if names else (
            'Aucun outil de contrôle disponible pour cette demande.' if french else
            'No device tools are available for this request.')
    else:
        reply = ('Je peux converser en français et en anglais et donner l’heure locale.' if french else
                 'I can converse in French and English and tell you the local time.')
        if search:
            reply += (' Je peux rechercher vos appareils.' if french else ' I can find your devices.')
        if reading:
            reply += (' Je peux lire les états, les capteurs et la météo de Jeedom.' if french else
                      ' I can read device states, sensors and weather from Jeedom.')
        if control:
            reply += (' Je peux envoyer une commande précise après validation.' if french else
                      ' I can submit a specific command after validation.')
    if control and (general_fr or general_en) and getattr(actions, 'execution_mode', None) == 'log_only':
        reply += (' En mode test, les commandes sont seulement enregistrées ; aucun appareil n’est modifié.' if french else
                  ' In test mode, commands are only recorded; no device is changed.')
    return LocalAnswer(reply, language, 'capabilities')
