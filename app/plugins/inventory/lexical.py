"""Reviewed local WordNet/WOLF terms; no model or runtime network dependency."""
import json
import re
from importlib.resources import files


_DATA = json.loads(files(__package__).joinpath('entity_lexicon.json').read_text(encoding='utf-8'))
if _DATA['schema'] != 1:
    raise ValueError('Unsupported entity lexicon schema')
_ALIASES = _DATA['aliases']
_PATTERN = re.compile(r'\b(?:' + '|'.join(
    re.escape(term) for term in sorted(_ALIASES, key=lambda term: (-len(term), term))
) + r')\b')


def lexical_form(normalized):
    """Replace whole phrases once; senses and collisions were reviewed offline."""
    return _PATTERN.sub(lambda match: _ALIASES[match.group()], normalized)
