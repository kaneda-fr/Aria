"""
Self-learning router for CHAT vs CONTROL classification.

This module implements an embedding-based router that classifies user utterances
as CHAT (conversation), CONTROL (device action), or UNKNOWN (uncertain).

Supports per-language embedding models for improved accuracy.
"""

import logging
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .embeddings import embed_sentence, embed_sentences, cosine_distance, calculate_centroid
from .seed_data import get_seed_data, get_seed_data_by_lang
from .lang_detect import detect_lang, LangCode

logger = logging.getLogger(__name__)

# Default models per language
DEFAULT_MODEL_EN = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_FR = "intfloat/multilingual-e5-small"


@dataclass
class RouterDecision:
    """Result of a routing decision."""
    mode: str  # "chat", "control", or "unknown"
    confidence: float  # 0.0 to 1.0
    distance_to_control: float
    distance_to_chat: float
    reasoning: str
    latency_ms: float
    detected_lang: str = "en"  # "fr" or "en" - detected before embedding


class SelfLearningRouter:
    """
    Embedding-based router that learns from verified outcomes.

    The router maintains per-language centroids for CHAT and CONTROL categories
    and classifies new utterances based on cosine distance to these centroids.
    Uses language-specific embedding models for improved accuracy.
    """

    def __init__(
        self,
        model_name_en: str = DEFAULT_MODEL_EN,
        model_name_fr: str = DEFAULT_MODEL_FR,
        control_threshold: float = 0.70,
        chat_threshold: float = 0.70,
        unknown_threshold: float = 0.50,
        use_seed_data: bool = True,
        lang_detect_method: str = "heuristic"
    ):
        """
        Initialize the router.

        Args:
            model_name_en: Embedding model for English
            model_name_fr: Embedding model for French
            control_threshold: Confidence threshold for CONTROL classification
            chat_threshold: Confidence threshold for CHAT classification
            unknown_threshold: Distance threshold for UNKNOWN classification
            use_seed_data: Whether to initialize with seed examples
            lang_detect_method: Language detection method ("heuristic" or "langid")
        """
        self.model_names = {
            "en": model_name_en,
            "fr": model_name_fr,
        }
        self.control_threshold = control_threshold
        self.chat_threshold = chat_threshold
        self.unknown_threshold = unknown_threshold
        self.lang_detect_method = lang_detect_method

        # Per-language centroids
        self.control_centroids: Dict[str, Optional[np.ndarray]] = {"en": None, "fr": None}
        self.chat_centroids: Dict[str, Optional[np.ndarray]] = {"en": None, "fr": None}

        # Example embeddings (for debugging and monitoring)
        self.control_examples: Dict[str, list] = {"en": [], "fr": []}
        self.chat_examples: Dict[str, list] = {"en": [], "fr": []}

        # Centroid distances per language
        self.centroid_distances: Dict[str, Optional[float]] = {"en": None, "fr": None}

        if use_seed_data:
            self._initialize_from_seed_data()

        logger.info(
            f"Router initialized: control_threshold={control_threshold}, "
            f"chat_threshold={chat_threshold}, unknown_threshold={unknown_threshold}, "
            f"model_en={model_name_en}, model_fr={model_name_fr}"
        )

    def _initialize_from_seed_data(self):
        """Initialize per-language centroids from seed examples."""
        logger.info("Initializing router from seed data (per-language)...")
        start_time = time.time()

        seed_data = get_seed_data_by_lang()

        for lang in ["en", "fr"]:
            model_name = self.model_names[lang]
            chat_seeds = seed_data[lang]["chat"]
            control_seeds = seed_data[lang]["control"]

            logger.info(f"[{lang.upper()}] Generating embeddings for {len(chat_seeds)} CHAT seeds with {model_name}...")
            chat_embeddings = embed_sentences(chat_seeds, model_name=model_name)
            self.chat_centroids[lang] = calculate_centroid(chat_embeddings)
            self.chat_examples[lang] = chat_seeds[:10]

            logger.info(f"[{lang.upper()}] Generating embeddings for {len(control_seeds)} CONTROL seeds...")
            control_embeddings = embed_sentences(control_seeds, model_name=model_name)
            self.control_centroids[lang] = calculate_centroid(control_embeddings)
            self.control_examples[lang] = control_seeds[:10]

            # Validate centroid separation
            if self.control_centroids[lang] is not None and self.chat_centroids[lang] is not None:
                dist = cosine_distance(self.control_centroids[lang], self.chat_centroids[lang])
                self.centroid_distances[lang] = dist
                logger.info(f"[{lang.upper()}] Centroid separation distance: {dist:.3f}")
                if dist < 0.3:
                    logger.warning(
                        f"[{lang.upper()}] Centroids are very close ({dist:.3f}). "
                        "This may lead to poor classification."
                    )

        init_time = time.time() - start_time
        logger.info(f"Router initialized from seed data in {init_time:.2f}s")

    def classify(self, text: str) -> RouterDecision:
        """
        Classify a user utterance as CHAT, CONTROL, or UNKNOWN.

        Detects language BEFORE embedding to select the appropriate model.

        Args:
            text: User utterance to classify

        Returns:
            RouterDecision with mode, confidence, detected_lang, and reasoning
        """
        if not text or not text.strip():
            return RouterDecision(
                mode="unknown",
                confidence=0.0,
                distance_to_control=999.0,
                distance_to_chat=999.0,
                reasoning="Empty input",
                latency_ms=0.0,
                detected_lang="en"
            )

        start_time = time.time()

        # Detect language BEFORE embedding (to select correct model)
        detected_lang = detect_lang(text, method=self.lang_detect_method)

        # Get language-specific model and centroids
        model_name = self.model_names.get(detected_lang, self.model_names["en"])
        control_centroid = self.control_centroids.get(detected_lang)
        chat_centroid = self.chat_centroids.get(detected_lang)

        # Check if centroids are initialized
        if control_centroid is None or chat_centroid is None:
            logger.error(f"Router centroids not initialized for lang={detected_lang}!")
            return RouterDecision(
                mode="unknown",
                confidence=0.0,
                distance_to_control=999.0,
                distance_to_chat=999.0,
                reasoning=f"Router not initialized for {detected_lang}",
                latency_ms=0.0,
                detected_lang=detected_lang
            )

        # Generate embedding for input text using language-specific model
        embedding = embed_sentence(text, model_name=model_name)

        # Calculate distances to centroids
        dist_control = cosine_distance(embedding, control_centroid)
        dist_chat = cosine_distance(embedding, chat_centroid)

        # Determine classification
        mode, confidence, reasoning = self._decide_mode(dist_control, dist_chat)

        latency_ms = (time.time() - start_time) * 1000

        decision = RouterDecision(
            mode=mode,
            confidence=confidence,
            distance_to_control=dist_control,
            distance_to_chat=dist_chat,
            reasoning=reasoning,
            latency_ms=latency_ms,
            detected_lang=detected_lang
        )

        logger.info(
            f"Router decision: {mode.upper()} lang={detected_lang} (confidence={confidence:.2f}, "
            f"dist_control={dist_control:.3f}, dist_chat={dist_chat:.3f}, "
            f"{latency_ms:.1f}ms) | '{text[:50]}...'"
        )

        return decision

    def _decide_mode(self, dist_control: float, dist_chat: float) -> Tuple[str, float, str]:
        """
        Decide the mode based on distances to centroids.

        Args:
            dist_control: Distance to CONTROL centroid
            dist_chat: Distance to CHAT centroid

        Returns:
            Tuple of (mode, confidence, reasoning)
        """
        # Both distances are very high → UNKNOWN
        if dist_control > self.unknown_threshold and dist_chat > self.unknown_threshold:
            return "unknown", 0.0, "Both distances above unknown threshold"

        # Closer to CONTROL
        if dist_control < dist_chat:
            margin = dist_chat - dist_control
            # Confidence based on relative margin
            confidence = min(1.0, margin / (dist_chat + 0.001))

            if confidence >= self.control_threshold:
                return "control", confidence, f"Close to CONTROL (margin={margin:.3f})"
            else:
                return "unknown", confidence, f"Weak CONTROL signal (confidence={confidence:.2f})"

        # Closer to CHAT
        else:
            margin = dist_control - dist_chat
            # Confidence based on relative margin
            confidence = min(1.0, margin / (dist_control + 0.001))

            if confidence >= self.chat_threshold:
                return "chat", confidence, f"Close to CHAT (margin={margin:.3f})"
            else:
                return "unknown", confidence, f"Weak CHAT signal (confidence={confidence:.2f})"

    def update_centroids(
        self,
        lang: str,
        chat_embeddings: Optional[np.ndarray] = None,
        control_embeddings: Optional[np.ndarray] = None
    ):
        """
        Update centroids for a specific language with new confirmed examples.

        Args:
            lang: Language code ("en" or "fr")
            chat_embeddings: Array of CHAT example embeddings
            control_embeddings: Array of CONTROL example embeddings
        """
        if lang not in self.chat_centroids:
            logger.error(f"Unknown language: {lang}")
            return

        if chat_embeddings is not None and len(chat_embeddings) > 0:
            self.chat_centroids[lang] = calculate_centroid(chat_embeddings)
            logger.info(f"[{lang.upper()}] Updated CHAT centroid from {len(chat_embeddings)} examples")

        if control_embeddings is not None and len(control_embeddings) > 0:
            self.control_centroids[lang] = calculate_centroid(control_embeddings)
            logger.info(f"[{lang.upper()}] Updated CONTROL centroid from {len(control_embeddings)} examples")

        # Log new centroid separation
        if self.control_centroids[lang] is not None and self.chat_centroids[lang] is not None:
            dist = cosine_distance(self.control_centroids[lang], self.chat_centroids[lang])
            self.centroid_distances[lang] = dist
            logger.info(f"[{lang.upper()}] New centroid separation: {dist:.3f}")

    def get_stats(self) -> Dict:
        """
        Get router statistics.

        Returns:
            Dictionary with router stats
        """
        return {
            "model_name_en": self.model_names["en"],
            "model_name_fr": self.model_names["fr"],
            "control_threshold": self.control_threshold,
            "chat_threshold": self.chat_threshold,
            "unknown_threshold": self.unknown_threshold,
            "centroid_distance_en": self.centroid_distances.get("en"),
            "centroid_distance_fr": self.centroid_distances.get("fr"),
            "chat_examples_en": self.chat_examples.get("en", []),
            "chat_examples_fr": self.chat_examples.get("fr", []),
            "control_examples_en": self.control_examples.get("en", []),
            "control_examples_fr": self.control_examples.get("fr", []),
            "centroids_initialized": (
                self.control_centroids.get("en") is not None and
                self.chat_centroids.get("en") is not None and
                self.control_centroids.get("fr") is not None and
                self.chat_centroids.get("fr") is not None
            )
        }


# Global router instance (lazy-loaded)
_router: Optional[SelfLearningRouter] = None


def get_router(
    model_name_en: str = DEFAULT_MODEL_EN,
    model_name_fr: str = DEFAULT_MODEL_FR,
    control_threshold: float = 0.70,
    chat_threshold: float = 0.70,
    unknown_threshold: float = 0.50,
    use_seed_data: bool = True,
    lang_detect_method: str = "heuristic"
) -> SelfLearningRouter:
    """
    Get or create the global router instance.

    Args:
        model_name_en: Embedding model for English (default: bge-small-en-v1.5)
        model_name_fr: Embedding model for French (default: multilingual-e5-small)
        control_threshold: Confidence threshold for CONTROL
        chat_threshold: Confidence threshold for CHAT
        unknown_threshold: Distance threshold for UNKNOWN
        use_seed_data: Whether to initialize with seed examples
        lang_detect_method: Language detection method ("heuristic" or "langid")

    Returns:
        The global router instance
    """
    global _router

    if _router is None:
        _router = SelfLearningRouter(
            model_name_en=model_name_en,
            model_name_fr=model_name_fr,
            control_threshold=control_threshold,
            chat_threshold=chat_threshold,
            unknown_threshold=unknown_threshold,
            use_seed_data=use_seed_data,
            lang_detect_method=lang_detect_method
        )

    return _router
