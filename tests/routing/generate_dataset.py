#!/usr/bin/env python3
"""
Generate synthetic dataset for chat vs control intent classification.

Creates balanced examples of control commands and chat queries in French and English.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


# Control command templates
CONTROL_TEMPLATES_FR = [
    "{action} {article} {device} {location}",
    "{action} {article} {device}",
    "{action} {device} {location}",
    "{action} {device}",
    "met {article} {device} à {value}",
    "met {device} à {value}",
    "{device} {location} {action}",
    "peux-tu {action} {article} {device} {location}",
    "s'il te plaît {action} {article} {device}",
]

CONTROL_TEMPLATES_EN = [
    "{action} {article} {device} {location}",
    "{action} {article} {device}",
    "{action} {device} in {location}",
    "{action} {device}",
    "set {device} to {value}",
    "set {article} {device} to {value}",
    "{device} {location} {action}",
    "can you {action} {article} {device} {location}",
    "please {action} {article} {device}",
]

# Vocabulary
DEVICES_FR = ["lumière", "lumières", "lampe", "volet", "volets", "store", "chauffage", "thermostat"]
DEVICES_EN = ["light", "lights", "lamp", "shutter", "shutters", "blind", "heating", "thermostat"]

ACTIONS_FR = [
    "allume", "éteins", "ouvre", "ferme", "baisse", "monte", "augmente", "diminue"
]
ACTIONS_EN = [
    "turn on", "turn off", "switch on", "switch off", "open", "close", "raise", "lower", "increase", "decrease"
]

ARTICLES_FR = ["la", "le", "les", ""]
ARTICLES_EN = ["the", ""]

LOCATIONS_FR = ["salon", "chambre", "cuisine", "salle de bain", "bureau", "entrée"]
LOCATIONS_EN = ["living room", "bedroom", "kitchen", "bathroom", "office", "hallway"]

VALUES = ["50%", "50", "100%", "0%", "maximum", "minimum", "moitié"]

# Edge case templates - IMPLICIT COMMANDS (context-dependent, should be control)
IMPLICIT_CONTROL_FR = [
    "il fait noir",
    "il fait froid",
    "il fait chaud",
    "je ne vois rien",
    "j'ai froid",
    "j'ai chaud",
    "c'est sombre",
    "c'est trop clair",
]

IMPLICIT_CONTROL_EN = [
    "it's dark",
    "it's cold",
    "it's hot",
    "I can't see",
    "I'm cold",
    "I'm hot",
    "it's too bright",
    "too dark",
]

# Edge case templates - PARTIAL/MINIMAL COMMANDS (ambiguous, should be control)
MINIMAL_CONTROL_FR = [
    "lumière",
    "lumières",
    "volet",
    "volets",
    "chauffage",
    "salon",
    "chambre",
    "cuisine",
]

MINIMAL_CONTROL_EN = [
    "lights",
    "light",
    "shutter",
    "shutters",
    "heating",
    "bedroom",
    "kitchen",
    "living room",
]

# Edge case templates - COMMANDS WITH CONTEXT (should be control)
CONTEXTUAL_CONTROL_FR = [
    "peux-tu m'aider avec {article} {device}",
    "j'ai besoin de {article} {device}",
    "{article} {device} s'il te plaît",
    "{device} maintenant",
    "{device} {location} s'il te plaît",
]

CONTEXTUAL_CONTROL_EN = [
    "can you help me with {article} {device}",
    "I need {article} {device}",
    "{article} {device} please",
    "{device} now",
    "{device} in {location} please",
]

# Edge case templates - STATUS QUERIES (asking about state, should be chat)
STATUS_QUERY_FR = [
    "les {device} sont allumées ?",
    "est-ce que {article} {device} est allumée ?",
    "{article} {device} {location} est ouverte ?",
    "quelle est la température ?",
    "montre-moi {article} {device}",
    "liste les {device}",
]

STATUS_QUERY_EN = [
    "are {article} {device} on?",
    "is {article} {device} on?",
    "is {article} {device} in {location} open?",
    "what's the temperature?",
    "show me {article} {device}",
    "list {article} {device}",
]

# Edge case templates - DEVICE INFO QUERIES (should be chat)
INFO_QUERY_FR = [
    "quelles {device} j'ai ?",
    "combien de {device} il y a ?",
    "où sont les {device} ?",
    "comment contrôler les {device} ?",
    "parle-moi des {device}",
    "qu'est-ce qu'une {device} ?",
]

INFO_QUERY_EN = [
    "what {device} do I have?",
    "how many {device} are there?",
    "where are {article} {device}?",
    "how do I control {article} {device}?",
    "tell me about {article} {device}",
    "what devices are in {location}?",
]

# Chat query templates
CHAT_TEMPLATES_FR = [
    "quelle heure est-il ?",
    "quel jour sommes-nous ?",
    "raconte-moi une blague",
    "comment vas-tu ?",
    "qui es-tu ?",
    "qu'est-ce que tu peux faire ?",
    "quel temps fait-il ?",
    "aide-moi",
    "merci",
    "bonjour",
    "bonsoir",
    "bonne nuit",
    "à demain",
    "comment ça marche ?",
    "explique-moi",
    "pourquoi ?",
    "c'est quoi ça ?",
    "tu peux répéter ?",
    "je ne comprends pas",
    "qu'est-ce que je peux te demander ?",
]

CHAT_TEMPLATES_EN = [
    "what time is it?",
    "what day is it?",
    "tell me a joke",
    "how are you?",
    "who are you?",
    "what can you do?",
    "what's the weather like?",
    "help me",
    "thank you",
    "hello",
    "good evening",
    "good night",
    "see you tomorrow",
    "how does this work?",
    "explain",
    "why?",
    "what is that?",
    "can you repeat?",
    "I don't understand",
    "what can I ask you?",
]


def generate_control_example(lang: str = "fr", edge_case_prob: float = 0.3) -> Dict[str, str]:
    """Generate a single control command example."""
    # Decide if this should be an edge case
    edge_type = None
    if random.random() < edge_case_prob:
        edge_type = random.choice(["implicit", "minimal", "contextual"])

    if lang == "fr":
        if edge_type == "implicit":
            text = random.choice(IMPLICIT_CONTROL_FR)
        elif edge_type == "minimal":
            text = random.choice(MINIMAL_CONTROL_FR)
        elif edge_type == "contextual":
            template = random.choice(CONTEXTUAL_CONTROL_FR)
            device = random.choice(DEVICES_FR)
            article = random.choice(ARTICLES_FR)
            location = random.choice(LOCATIONS_FR)
            text = template.format(device=device, article=article, location=location)
        else:
            # Standard control command
            template = random.choice(CONTROL_TEMPLATES_FR)
            action = random.choice(ACTIONS_FR)
            device = random.choice(DEVICES_FR)
            article = random.choice(ARTICLES_FR)
            location = random.choice(LOCATIONS_FR)
            value = random.choice(VALUES)
            text = template.format(
                action=action,
                device=device,
                article=article,
                location=location,
                value=value
            )
    else:
        if edge_type == "implicit":
            text = random.choice(IMPLICIT_CONTROL_EN)
        elif edge_type == "minimal":
            text = random.choice(MINIMAL_CONTROL_EN)
        elif edge_type == "contextual":
            template = random.choice(CONTEXTUAL_CONTROL_EN)
            device = random.choice(DEVICES_EN)
            article = random.choice(ARTICLES_EN)
            location = random.choice(LOCATIONS_EN)
            text = template.format(device=device, article=article, location=location)
        else:
            # Standard control command
            template = random.choice(CONTROL_TEMPLATES_EN)
            action = random.choice(ACTIONS_EN)
            device = random.choice(DEVICES_EN)
            article = random.choice(ARTICLES_EN)
            location = random.choice(LOCATIONS_EN)
            value = random.choice(VALUES)
            text = template.format(
                action=action,
                device=device,
                article=article,
                location=location,
                value=value
            )

    # Remove double spaces
    text = text.strip()
    while "  " in text:
        text = text.replace("  ", " ")

    return {"text": text, "label": "control", "lang": lang}


def generate_chat_example(lang: str = "fr", edge_case_prob: float = 0.3) -> Dict[str, str]:
    """Generate a single chat query example."""
    # Decide if this should be an edge case
    edge_type = None
    if random.random() < edge_case_prob:
        edge_type = random.choice(["status", "info"])

    if lang == "fr":
        if edge_type == "status":
            template = random.choice(STATUS_QUERY_FR)
            device = random.choice(DEVICES_FR)
            article = random.choice(ARTICLES_FR)
            location = random.choice(LOCATIONS_FR)
            text = template.format(device=device, article=article, location=location)
        elif edge_type == "info":
            template = random.choice(INFO_QUERY_FR)
            device = random.choice(DEVICES_FR)
            article = random.choice(ARTICLES_FR)
            location = random.choice(LOCATIONS_FR)
            text = template.format(device=device, article=article, location=location)
        else:
            text = random.choice(CHAT_TEMPLATES_FR)
    else:
        if edge_type == "status":
            template = random.choice(STATUS_QUERY_EN)
            device = random.choice(DEVICES_EN)
            article = random.choice(ARTICLES_EN)
            location = random.choice(LOCATIONS_EN)
            text = template.format(device=device, article=article, location=location)
        elif edge_type == "info":
            template = random.choice(INFO_QUERY_EN)
            device = random.choice(DEVICES_EN)
            article = random.choice(ARTICLES_EN)
            location = random.choice(LOCATIONS_EN)
            text = template.format(device=device, article=article, location=location)
        else:
            text = random.choice(CHAT_TEMPLATES_EN)

    return {"text": text, "label": "chat", "lang": lang}


def add_typo(text: str, prob: float = 0.3) -> str:
    """Add random typo to text with given probability."""
    if random.random() > prob or len(text) < 4:
        return text

    words = text.split()
    if not words:
        return text

    # Pick random word to modify
    word_idx = random.randint(0, len(words) - 1)
    word = words[word_idx]

    if len(word) < 3:
        return text

    # Apply random typo type
    typo_type = random.choice(["swap", "duplicate", "omit"])

    if typo_type == "swap" and len(word) > 2:
        # Swap two adjacent characters
        pos = random.randint(0, len(word) - 2)
        word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    elif typo_type == "duplicate":
        # Duplicate a character
        pos = random.randint(0, len(word) - 1)
        word = word[:pos] + word[pos] + word[pos:]
    elif typo_type == "omit" and len(word) > 3:
        # Omit a character
        pos = random.randint(1, len(word) - 2)
        word = word[:pos] + word[pos + 1:]

    words[word_idx] = word
    return " ".join(words)


def generate_dataset(count: int, output_path: Path, typo_rate: float = 0.2, edge_case_rate: float = 0.3):
    """Generate balanced dataset of chat and control examples."""
    examples = []

    # Generate half control, half chat
    control_count = count // 2
    chat_count = count - control_count

    print(f"Generating {control_count} control examples (including ~{int(control_count * edge_case_rate)} edge cases)...")
    for _ in range(control_count):
        lang = random.choice(["fr", "en"])
        example = generate_control_example(lang, edge_case_prob=edge_case_rate)

        # Add typos to some examples
        if random.random() < typo_rate:
            example["text"] = add_typo(example["text"])

        examples.append(example)

    print(f"Generating {chat_count} chat examples (including ~{int(chat_count * edge_case_rate)} edge cases)...")
    for _ in range(chat_count):
        lang = random.choice(["fr", "en"])
        example = generate_chat_example(lang, edge_case_prob=edge_case_rate)

        # Add typos to some examples
        if random.random() < typo_rate:
            example["text"] = add_typo(example["text"])

        examples.append(example)

    # Shuffle
    random.shuffle(examples)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(examples)} examples")
    print(f"  - Control: {sum(1 for e in examples if e['label'] == 'control')}")
    print(f"  - Chat: {sum(1 for e in examples if e['label'] == 'chat')}")
    print(f"  - French: {sum(1 for e in examples if e['lang'] == 'fr')}")
    print(f"  - English: {sum(1 for e in examples if e['lang'] == 'en')}")
    print(f"  - Edge case rate: ~{int(edge_case_rate * 100)}% (implicit commands, status queries, etc.)")
    print(f"  - Saved to: {output_path}")

    # Show sample edge cases
    print(f"\nSample edge cases generated:")
    edge_samples = [e for e in examples[:50] if any(word in e['text'].lower()
                    for word in ['dark', 'noir', 'cold', 'froid', 'are', 'what', 'quelles', 'how many'])][:5]
    for sample in edge_samples:
        print(f"  [{sample['label']:7s}] {sample['text']}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic chat vs control dataset with edge cases")
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Total number of examples to generate (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/routing/synthetic_examples.jsonl"),
        help="Output file path (default: data/routing/synthetic_examples.jsonl)"
    )
    parser.add_argument(
        "--typo-rate",
        type=float,
        default=0.2,
        help="Probability of adding typos to examples (default: 0.2)"
    )
    parser.add_argument(
        "--edge-case-rate",
        type=float,
        default=0.3,
        help="Probability of generating edge case examples (default: 0.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generate_dataset(args.count, args.output, args.typo_rate, args.edge_case_rate)


if __name__ == "__main__":
    main()
