"""
ARIA Self-Learning Router

This package implements an embedding-based router that classifies user utterances
as CHAT (conversation) or CONTROL (device action), learning from verified outcomes.

Uses per-language embedding models:
- English: bge-small-en-v1.5
- French: multilingual-e5-small
"""

from .router import SelfLearningRouter, RouterDecision, get_router, DEFAULT_MODEL_EN, DEFAULT_MODEL_FR
from .seed_data import get_seed_data, get_seed_stats, get_seed_data_by_lang
from .lang_detect import detect_lang, detect_lang_heuristic, detect_lang_langid, LangCode

__all__ = [
    "SelfLearningRouter",
    "RouterDecision",
    "get_router",
    "get_seed_data",
    "get_seed_data_by_lang",
    "get_seed_stats",
    "detect_lang",
    "detect_lang_heuristic",
    "detect_lang_langid",
    "LangCode",
    "DEFAULT_MODEL_EN",
    "DEFAULT_MODEL_FR",
]
