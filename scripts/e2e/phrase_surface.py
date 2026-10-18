"""Diagnostic coverage of natural paraphrases, using synthetic inventory only.

These are evaluation examples, never runtime phrase rules. A miss is reported,
not hidden by modifying the expected result or adding that sentence to the app.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from app.llm.control import resolve
from app.llm.local_answers import clock_answer, capability_answer
from app.speech_output.speech_output import _detect_lang_heuristic


CONTROL_CASES = [
    ('fr', 'Allume Atlas', 'light_on'),
    ('fr', 'Peux-tu allumer Atlas ?', 'light_on'),
    ('fr', 'Mets Atlas en marche', 'light_on'),
    ('fr', 'Active la lampe Atlas', 'light_on'),
    ('fr', 'Éclaire la cuisine avec Atlas', 'light_on'),
    ('fr', 'Éteins Atlas', 'light_off'),
    ('fr', 'Coupe Atlas', 'light_off'),
    ('fr', 'Désactive la lampe Atlas', 'light_off'),
    ('fr', 'Allume Atlas dans la cuisine', 'light_on'),
    ('fr', 'Règle Atlas à 30 %', 'light_slider'),
    ('fr', 'Règle Atlas à trente pour cent', 'light_slider'),
    ('fr', 'Baisse Atlas un peu', 'light_slider'),
    ('en', 'Turn on Atlas', 'light_on'),
    ('en', 'Could you switch Atlas on?', 'light_on'),
    ('en', 'Power up Atlas', 'light_on'),
    ('en', 'Activate the Atlas lamp', 'light_on'),
    ('en', 'Light up the kitchen with Atlas', 'light_on'),
    ('en', 'Turn Atlas off', 'light_off'),
    ('en', 'Shut off Atlas', 'light_off'),
    ('en', 'Deactivate the Atlas lamp', 'light_off'),
    ('en', 'Turn on Atlas in the kitchen', 'light_on'),
    ('en', 'Set Atlas to 30 percent', 'light_slider'),
    ('en', 'Set Atlas to thirty percent', 'light_slider'),
    ('en', 'Make Atlas a little dimmer', 'light_slider'),
]
UTILITY_CASES = [
    ('fr', 'Quelle heure est-il ?', 'clock'),
    ('fr', 'Tu as l’heure ?', 'clock'),
    ('fr', 'Donne-moi l’heure actuelle', 'clock'),
    ('fr', 'Peux-tu me dire quelle heure il est ?', 'clock'),
    ('en', 'What time is it?', 'clock'),
    ('en', 'What’s the time?', 'clock'),
    ('en', 'Tell me the current time', 'clock'),
    ('en', 'Could you tell me what time it is?', 'clock'),
    ('fr', 'Que peux-tu faire ?', 'capabilities'),
    ('fr', 'Comment peux-tu m’aider ?', 'capabilities'),
    ('fr', 'Tu sais faire quoi ?', 'capabilities'),
    ('fr', 'Présente-moi tes fonctions', 'capabilities'),
    ('en', 'What can you do?', 'capabilities'),
    ('en', 'How can you help me?', 'capabilities'),
    ('en', 'Tell me about your features', 'capabilities'),
    ('en', 'What are you able to do?', 'capabilities'),
]


def inventory(language):
    commands = [SimpleNamespace(cmd_id='fixture-'+action, type='action',
                                capability='light_'+action, subType='slider' if action == 'slider' else 'other',
                                execution={'topic': 'never-published', 'payload': action})
                for action in ('on', 'off', 'slider')]
    snapshot = SimpleNamespace(devices={'atlas': SimpleNamespace(
        enabled=True, name='Atlas', room='cuisine' if language == 'fr' else 'kitchen', commands=commands)})
    indexer = SimpleNamespace(get_snapshot=lambda: snapshot,
                              get_command=lambda cid: next((c for c in commands if c.cmd_id == cid), None))
    return SimpleNamespace(is_started=True, indexer=indexer)


def audit():
    rows = []
    for language, text, expected in CONTROL_CASES:
        plan = resolve(text, inventory(language))
        rows.append(dict(kind='control', language=language, text=text, expected=expected,
                         route=plan.status if plan else 'model_fallback_without_authorization',
                         granted_capability=plan.authorization.capability if plan and plan.authorization else None,
                         effective_language=plan.language if plan else _detect_lang_heuristic(text)))
    for language, text, expected in UTILITY_CASES:
        answer = clock_answer(text) or capability_answer(text, inventory(language), SimpleNamespace(is_started=True))
        rows.append(dict(kind='utility', language=language, text=text, expected=expected,
                         route=answer.kind if answer else 'ungrounded_model_fallback',
                         effective_language=answer.language if answer else _detect_lang_heuristic(text)))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = audit()
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps({'control_authorized': sum(bool(x.get('granted_capability')) for x in rows),
                      'control_cases': len(CONTROL_CASES),
                      'utility_grounded': sum(x['kind'] == 'utility' and x['route'] == x['expected'] for x in rows),
                      'utility_cases': len(UTILITY_CASES)}, indent=2))
