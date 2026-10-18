"""
Seed data for bootstrapping the self-learning router.

This module provides initial examples of CHAT and CONTROL utterances
to initialize the router's centroids before any learning has occurred.
"""

# Seed examples for CHAT classification (conversation, not device control)
CHAT_SEEDS = [
    # Greetings (French)
    "bonjour",
    "bonsoir",
    "salut",
    "bonne nuit",
    "coucou",
    "allô",

    # Greetings (English)
    "hello",
    "hi",
    "good morning",
    "good evening",
    "good night",
    "hey",
    "yo",

    # Thanks (French)
    "merci",
    "merci beaucoup",
    "je te remercie",
    "c'est gentil",

    # Thanks (English)
    "thank you",
    "thanks",
    "thanks a lot",
    "appreciate it",

    # Questions about time (French)
    "quelle heure est-il?",
    "quelle heure il est?",
    "tu as l'heure?",

    # Questions about time (English)
    "what time is it?",
    "do you have the time?",
    "what's the time?",

    # General questions (French)
    "comment ça va?",
    "ça va?",
    "tu vas bien?",
    "où est la cuisine?",
    "c'est où?",
    "qu'est-ce que tu peux faire?",
    "tu peux m'aider?",

    # General questions (English)
    "how are you?",
    "how's it going?",
    "where is the kitchen?",
    "what can you do?",
    "can you help me?",

    # Small talk (French)
    "ça va bien",
    "tout va bien",
    "pas mal",
    "je ne comprends pas",
    "aide-moi",
    "c'est quoi ça?",

    # Small talk (English)
    "I'm fine",
    "all good",
    "not bad",
    "I don't understand",
    "help me",
    "what's this?",

    # Polite closings (French)
    "au revoir",
    "à bientôt",
    "bonne journée",

    # Polite closings (English)
    "goodbye",
    "see you",
    "have a good day",
]

# Seed examples for CONTROL classification (device control commands)
CONTROL_SEEDS = [
    # Turn on/off (French)
    "allume la lumière",
    "éteins la lumière",
    "allume la lampe",
    "éteins la lampe",
    "allume le salon",
    "éteins la cuisine",
    "allume",
    "éteins",

    # Turn on/off (English)
    "turn on the light",
    "turn off the light",
    "switch on the lamp",
    "switch off the lamp",
    "light on",
    "light off",
    "turn on",
    "turn off",

    # Dimming (French)
    "mets la lumière à 50%",
    "règle la luminosité à 75%",
    "baisse la lumière",
    "monte la lumière",
    "diminue l'intensité",
    "augmente l'intensité",

    # Dimming (English)
    "set light to 50%",
    "dim to 75%",
    "lower the light",
    "raise the light",
    "decrease brightness",
    "increase brightness",

    # Rooms (French)
    "allume la lumière du salon",
    "éteins la salle à manger",
    "allume la cuisine",
    "éteins la chambre",
    "allume l'entrée",
    "éteins le bureau",

    # Rooms (English)
    "turn on the living room light",
    "turn off the dining room",
    "turn on kitchen light",
    "turn off bedroom",
    "switch on the hallway",
    "switch off the office",

    # Covers/Blinds (French)
    "ouvre les volets",
    "ferme les volets",
    "ouvre les stores",
    "ferme les stores",
    "monte les volets",
    "descends les stores",

    # Covers/Blinds (English)
    "open the shutters",
    "close the shutters",
    "open the blinds",
    "close the blinds",
    "raise the shades",
    "lower the shades",

    # Specific devices (French)
    "allume la lumière de la salle à manger",
    "éteins la lumière du couloir",
    "mets la salle à manger à 50%",
    "règle le salon à 30%",

    # Specific devices (English)
    "turn on the dining room light",
    "turn off the hallway light",
    "set dining room to 50%",
    "dim the living room to 30%",
]

def get_seed_data():
    """
    Returns the complete seed dataset.

    Returns:
        dict: Dictionary with 'chat' and 'control' keys containing seed examples
    """
    return {
        "chat": CHAT_SEEDS,
        "control": CONTROL_SEEDS
    }


def get_seed_data_by_lang():
    """
    Returns seed data split by language.

    Seed data is designed for maximum centroid separation:
    - CHAT: Pure conversational phrases (greetings, social, emotional)
    - CONTROL: Imperative device commands with action verbs

    Returns:
        dict: Dictionary with 'en' and 'fr' keys, each containing 'chat' and 'control' lists
    """
    # French CHAT seeds - pure conversation, NO action verbs
    fr_chat = [
        # Greetings
        "bonjour", "bonsoir", "salut", "bonne nuit", "coucou", "allô",
        "bonjour comment allez-vous", "salut ça va",
        # Thanks/politeness
        "merci", "merci beaucoup", "je te remercie", "c'est gentil", "de rien",
        "s'il te plaît", "excusez-moi", "pardon",
        # Social questions (NOT about devices)
        "comment ça va", "ça va", "tu vas bien", "comment tu t'appelles",
        "quel temps fait-il", "quelle heure est-il", "quel jour sommes-nous",
        # Emotional responses
        "ça va bien", "tout va bien", "pas mal", "super", "génial", "parfait",
        "je suis content", "je suis fatigué", "j'ai faim", "j'ai soif",
        # Confusion/acknowledgment
        "je ne comprends pas", "d'accord", "ok", "oui", "non", "peut-être",
        "je ne sais pas", "c'est vrai", "vraiment",
        # Casual acknowledgments (common in speech)
        "ouais", "ouaip", "mouais", "mmh", "hum", "ah bon", "bien",
        "compris", "entendu", "très bien", "c'est ça", "voilà",
        # Farewells
        "au revoir", "à bientôt", "bonne journée", "bonne soirée", "à demain",
        # Small talk
        "il fait beau aujourd'hui", "il pleut", "c'est une belle journée",
        "comment s'est passée ta journée", "qu'est-ce que tu fais",
    ]

    # French CONTROL seeds - imperative commands with device/action vocabulary
    fr_control = [
        # Light commands with action verbs
        "allume la lumière", "éteins la lumière", "allume les lumières", "éteins les lumières",
        "allume la lampe", "éteins la lampe", "allume le plafonnier", "éteins le plafonnier",
        "allume tout", "éteins tout", "allume ici", "éteins ici",
        # Dimming commands
        "mets la lumière à 50 pourcent", "règle la luminosité à 75 pourcent",
        "baisse la lumière", "monte la lumière", "augmente la luminosité",
        "diminue l'intensité", "lumière plus forte", "lumière plus faible",
        "mets à fond", "mets au minimum", "lumière tamisée",
        # Room-specific control
        "allume le salon", "éteins la cuisine", "allume la chambre", "éteins le bureau",
        "allume l'entrée", "éteins le couloir", "allume la salle de bain",
        "lumière du salon", "lumière de la cuisine", "lumière de la chambre",
        # Cover/blind commands
        "ouvre les volets", "ferme les volets", "ouvre les stores", "ferme les stores",
        "monte les volets", "descends les stores", "ouvre les rideaux", "ferme les rideaux",
        # Combined commands
        "allume la lumière du salon", "éteins la lumière de la cuisine",
        "mets le salon à 30 pourcent", "règle la chambre à 50 pourcent",
        # Thermostat/climate
        "monte le chauffage", "baisse la climatisation", "règle la température",
    ]

    # English CHAT seeds - pure conversation, NO action verbs
    en_chat = [
        # Greetings
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "good night",
        "hello how are you", "hi there", "hey what's up",
        # Thanks/politeness
        "thank you", "thanks", "thanks a lot", "appreciate it", "you're welcome",
        "please", "excuse me", "sorry", "no problem",
        # Social questions (NOT about devices)
        "how are you", "how are you doing", "what's your name", "how's it going",
        "what's the weather like", "what time is it", "what day is it",
        # Emotional responses
        "I'm fine", "I'm good", "all good", "not bad", "great", "awesome", "perfect",
        "I'm happy", "I'm tired", "I'm hungry", "I'm thirsty",
        # Confusion/acknowledgment
        "I don't understand", "okay", "ok", "yes", "no", "maybe",
        "I don't know", "that's true", "really", "sure",
        # Casual acknowledgments (common in speech)
        "yeah", "yep", "yup", "uh huh", "mhm", "mm hmm", "right",
        "got it", "alright", "all right", "fair enough", "I see",
        # Farewells
        "goodbye", "bye", "see you", "see you later", "have a good day", "take care",
        # Small talk
        "nice weather today", "it's raining", "beautiful day",
        "how was your day", "what are you doing",
    ]

    # English CONTROL seeds - imperative commands with device/action vocabulary
    en_control = [
        # Light commands with action verbs
        "turn on the light", "turn off the light", "turn on the lights", "turn off the lights",
        "switch on the lamp", "switch off the lamp", "turn on the ceiling light",
        "lights on", "lights off", "light on", "light off",
        # Dimming commands
        "set light to 50 percent", "dim to 75 percent", "set brightness to 30",
        "lower the light", "raise the light", "increase brightness", "decrease brightness",
        "brighter", "dimmer", "full brightness", "minimum brightness",
        "make it brighter", "make it darker",
        # Room-specific control
        "turn on the living room", "turn off the kitchen", "turn on bedroom light",
        "switch on the hallway", "switch off the office", "turn on bathroom light",
        "living room light", "kitchen light", "bedroom light",
        # Cover/blind commands
        "open the shutters", "close the shutters", "open the blinds", "close the blinds",
        "raise the shades", "lower the shades", "open the curtains", "close the curtains",
        # Combined commands
        "turn on the living room light", "turn off the kitchen light",
        "set living room to 50 percent", "dim bedroom to 30 percent",
        # Thermostat/climate
        "turn up the heat", "turn down the AC", "set temperature to 72",
    ]

    return {
        "en": {"chat": en_chat, "control": en_control},
        "fr": {"chat": fr_chat, "control": fr_control},
    }

def get_seed_stats():
    """
    Returns statistics about the seed dataset.

    Returns:
        dict: Statistics including counts and balance ratio
    """
    chat_count = len(CHAT_SEEDS)
    control_count = len(CONTROL_SEEDS)
    total = chat_count + control_count
    balance = min(chat_count, control_count) / max(chat_count, control_count)

    return {
        "chat_count": chat_count,
        "control_count": control_count,
        "total": total,
        "balance_ratio": balance,  # 1.0 = perfect balance
        "imbalance": abs(chat_count - control_count)
    }
