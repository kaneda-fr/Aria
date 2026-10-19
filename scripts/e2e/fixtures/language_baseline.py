"""Frozen evaluation baseline from the 2026-10-19 Apollo source snapshot.

Evaluation-only: do not import this into the application.
Full original source SHA256: 11b1a3274d7dbd10910863955b1b46c0ccfcc3f64a82bfc832caf06b05572ebd
"""
import re

def _detect_lang_heuristic(text: str) -> str:
    text = (text or "").lower()
    if not text:
        return "en"
    if re.search(r"[àâäçéèêëîïôöùûüÿœæ]", text):
        return "fr"

    fr_markers = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "est",
        "sont",
        "être",
        "avoir",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "avec",
        "pour",
        "sur",
        "dans",
        "mais",
        "donc",
        "bonjour",
        "merci",
        "salut",
        "mon",
        "ton",
        "nom",
        "comment",
        "combien",
        "pourquoi",
        "quand",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "qui",
        "quoi",
        "s'il",
        "ça",
    }
    en_markers = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "be",
        "have",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "with",
        "for",
        "on",
        "in",
        "but",
        "so",
    }

    tokens = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœæ']+", text)
    fr_score = sum(1 for t in tokens if t in fr_markers)
    en_score = sum(1 for t in tokens if t in en_markers)
    return "fr" if fr_score > en_score else "en"

def speech_language(text: str, requested: str) -> str:
    """Use clear prose evidence; neutral numbers/names keep the request language."""
    words = set(re.findall(r"[a-zàâçéèêîôùûü]+", (text or '').lower()))
    if words and words <= {'oui', 'non', 'bonjour', 'bonsoir', 'merci', 'salut', 'désolé'}:
        return 'fr'
    if words and words <= {'yes', 'no', 'hello', 'thanks', 'sorry', 'goodbye'}:
        return 'en'
    fr = len(words & {'je', 'tu', 'il', 'elle', 'nous', 'vous', 'le', 'la', 'les',
                      'une', 'des', 'du', 'est', 'sont', 'ne', 'pas', 'votre',
                      'pour', 'dans', 'quel', 'quelle', 'peux', 'pouvez'})
    en = len(words & {'i', 'you', 'he', 'she', 'we', 'they', 'the', 'a', 'an',
                      'is', 'are', 'not', 'your', 'could', 'would', 'please',
                      'with', 'for', 'in', 'what', 'which', 'can'})
    if en >= 2 and en - fr >= 2:
        return 'en'
    if fr >= 2 and fr - en >= 2:
        return 'fr'
    return requested
