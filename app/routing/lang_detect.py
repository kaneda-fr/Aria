"""
Language detection for routing classification.

Detects language (fr/en) in transcribed text before embedding-based classification.
"""

import re
from typing import Literal

LangCode = Literal["fr", "en"]


def detect_lang_heuristic(text: str) -> LangCode:
    """
    Detect language using heuristic marker-based scoring.

    Args:
        text: Input text to analyze

    Returns:
        "fr" for French, "en" for English
    """
    text = (text or "").lower()
    if not text:
        return "en"

    # French accented characters are strong signal
    if re.search(r"[àâäçéèêëîïôöùûüÿœæ]", text):
        return "fr"

    fr_markers = {
        "le", "la", "les", "un", "une", "des", "du", "de",
        "est", "sont", "être", "avoir",
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "avec", "pour", "sur", "dans", "mais", "donc",
        "bonjour", "merci", "salut",
        "mon", "ton", "nom", "comment",
        "quel", "quelle", "quels", "quelles", "qui", "quoi",
        "s'il", "ça", "allume", "éteins", "mets", "baisse", "monte",
        "lumière", "lumières", "lampe", "lampes",
        "salon", "cuisine", "chambre", "bureau", "salle",
    }

    en_markers = {
        "the", "a", "an", "is", "are", "be", "have",
        "i", "you", "he", "she", "we", "they",
        "with", "for", "on", "in", "but", "so",
        "turn", "switch", "set", "dim", "light", "lights",
        "living", "room", "bedroom", "kitchen", "office",
        "please", "can", "could", "would",
    }

    tokens = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœæ']+", text)
    fr_score = sum(1 for t in tokens if t in fr_markers)
    en_score = sum(1 for t in tokens if t in en_markers)

    return "fr" if fr_score > en_score else "en"


def detect_lang_langid(text: str) -> LangCode:
    """
    Detect language using langid library with heuristic fallback.

    Args:
        text: Input text to analyze

    Returns:
        "fr" for French, "en" for English
    """
    try:
        import langid
    except ImportError:
        return detect_lang_heuristic(text)

    try:
        lang, _score = langid.classify(text or "")
    except Exception:
        return detect_lang_heuristic(text)

    return "fr" if lang == "fr" else "en"


def detect_lang(text: str, method: str = "heuristic") -> LangCode:
    """
    Detect language using specified method.

    Args:
        text: Input text to analyze
        method: Detection method ("heuristic" or "langid")

    Returns:
        "fr" for French, "en" for English
    """
    if method == "langid":
        return detect_lang_langid(text)
    return detect_lang_heuristic(text)
