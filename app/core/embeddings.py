from __future__ import annotations

import hashlib
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 64


def text_to_embedding(text: str) -> list[float]:
    """Deterministic lightweight embedding using hashed character n-grams.

    Placeholder for a real embedding model. Replace with a proper embedding
    API for production use.
    """
    digest = hashlib.sha256(text.lower().encode()).digest()
    raw = np.frombuffer(digest * (EMBEDDING_DIM // len(digest) + 1), dtype=np.uint8)[
        :EMBEDDING_DIM
    ].astype(np.float64)
    norm = np.linalg.norm(raw)
    if norm == 0:
        logger.warning("Zero-norm embedding for text: '%s'", text[:80])
        return [0.0] * EMBEDDING_DIM
    logger.debug("Generated %d-dim embedding for text: '%s'", EMBEDDING_DIM, text[:80])
    return (raw / norm).tolist()


def compute_similarity(embedding_a: list[float], embedding_b: list[float]) -> float:
    a = np.array(embedding_a).reshape(1, -1)
    b = np.array(embedding_b).reshape(1, -1)
    score = float(cosine_similarity(a, b)[0][0])
    logger.debug("Cosine similarity: %.4f", score)
    return score
