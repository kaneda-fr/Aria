"""
Embedding model management for the self-learning router.

This module handles loading and caching the sentence-transformers model,
generating embeddings for utterances, and managing the embedding cache.
"""

import logging
import time
from typing import List, Optional
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache (supports multiple models)
_models: dict = {}


def load_embedding_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Load the sentence-transformers model.

    Args:
        model_name: Name of the sentence-transformers model to load

    Returns:
        The loaded model

    Raises:
        ImportError: If sentence-transformers is not installed
        Exception: If model loading fails
    """
    global _models

    # Return cached model if already loaded
    if model_name in _models:
        logger.debug(f"Using cached embedding model: {model_name}")
        return _models[model_name]

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()

        model = SentenceTransformer(model_name)
        _models[model_name] = model

        load_time = time.time() - start_time
        logger.info(f"Embedding model loaded in {load_time:.2f}s")

        return model

    except ImportError as e:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        raise ImportError(
            "sentence-transformers package is required. "
            "Install it with: pip install sentence-transformers"
        ) from e

    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def embed_sentence(
    text: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    normalize: bool = True
) -> np.ndarray:
    """
    Generate embedding for a single sentence.

    Args:
        text: The text to embed
        model_name: Name of the model to use
        normalize: Whether to L2-normalize the embedding (recommended for cosine similarity)

    Returns:
        numpy array of shape (embedding_dim,)
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return None

    model = load_embedding_model(model_name)

    start_time = time.time()
    embedding = model.encode(text, normalize_embeddings=normalize)
    embed_time = (time.time() - start_time) * 1000  # Convert to ms

    logger.debug(f"Generated embedding in {embed_time:.1f}ms for: {text[:50]}...")

    return embedding


def embed_sentences(
    texts: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    normalize: bool = True,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for multiple sentences (batched for efficiency).

    Args:
        texts: List of texts to embed
        model_name: Name of the model to use
        normalize: Whether to L2-normalize the embeddings
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (num_texts, embedding_dim)
    """
    if not texts:
        logger.warning("Empty text list provided for embedding")
        return np.array([])

    # Filter out empty texts
    non_empty_texts = [t for t in texts if t and t.strip()]
    if len(non_empty_texts) != len(texts):
        logger.warning(f"Filtered out {len(texts) - len(non_empty_texts)} empty texts")

    if not non_empty_texts:
        return np.array([])

    model = load_embedding_model(model_name)

    start_time = time.time()
    embeddings = model.encode(
        non_empty_texts,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=False
    )
    embed_time = (time.time() - start_time) * 1000  # Convert to ms

    logger.info(
        f"Generated {len(non_empty_texts)} embeddings in {embed_time:.1f}ms "
        f"({embed_time/len(non_empty_texts):.1f}ms per sentence)"
    )

    return embeddings


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0 to 1 if normalized, -1 to 1 otherwise)
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    # If embeddings are already normalized, dot product = cosine similarity
    return float(np.dot(embedding1, embedding2))


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two embeddings.

    Distance = 1 - similarity, so:
    - 0.0 = identical
    - 1.0 = orthogonal
    - 2.0 = opposite

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine distance (0 to 2)
    """
    similarity = cosine_similarity(embedding1, embedding2)
    return 1.0 - similarity


def calculate_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid (mean) of a set of embeddings.

    Args:
        embeddings: Array of shape (num_embeddings, embedding_dim)

    Returns:
        Centroid vector of shape (embedding_dim,), L2-normalized
    """
    if embeddings is None or len(embeddings) == 0:
        return None

    # Calculate mean
    centroid = np.mean(embeddings, axis=0)

    # Normalize to unit length
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid


def get_embedding_dim(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model_name: Name of the model

    Returns:
        Embedding dimension (e.g., 384 for MiniLM-L12-v2)
    """
    model = load_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


# For testing purposes
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Test embedding generation
    test_texts = [
        "bonjour",
        "allume la lumière",
        "hello",
        "turn on the light"
    ]

    logger.info(f"Testing embedding generation with {len(test_texts)} samples")

    # Single embedding
    emb1 = embed_sentence("bonjour")
    logger.info(f"Single embedding shape: {emb1.shape}")

    # Batch embedding
    embs = embed_sentences(test_texts)
    logger.info(f"Batch embeddings shape: {embs.shape}")

    # Test similarity
    emb_bonjour = embed_sentence("bonjour")
    emb_hello = embed_sentence("hello")
    emb_light = embed_sentence("allume la lumière")

    sim_greeting = cosine_similarity(emb_bonjour, emb_hello)
    sim_diff = cosine_similarity(emb_bonjour, emb_light)

    logger.info(f"Similarity (bonjour vs hello): {sim_greeting:.3f}")
    logger.info(f"Similarity (bonjour vs allume): {sim_diff:.3f}")
    logger.info(f"Greetings should be more similar: {sim_greeting > sim_diff}")

    # Test centroid
    centroid = calculate_centroid(embs)
    logger.info(f"Centroid shape: {centroid.shape}")
    logger.info(f"Centroid is normalized: {np.isclose(np.linalg.norm(centroid), 1.0)}")

    logger.info("Embedding tests complete!")
