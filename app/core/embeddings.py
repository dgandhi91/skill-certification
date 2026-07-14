from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from app.evaluation.providers.base import JudgeProvider

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 64


def _hash_embedding(text: str) -> list[float]:
    """Deterministic lightweight embedding using SHA-256 hash.

    Fallback for providers without a native embedding API (e.g. Anthropic).
    Not semantically meaningful — identical strings match, but paraphrases
    will not. Use a provider with native embeddings for real overlap detection.
    """
    digest = hashlib.sha256(text.lower().encode()).digest()
    raw = np.frombuffer(digest * (EMBEDDING_DIM // len(digest) + 1), dtype=np.uint8)[
        :EMBEDDING_DIM
    ].astype(np.float64)
    norm = np.linalg.norm(raw)
    if norm == 0:
        logger.warning("Zero-norm embedding for text: '%s'", text[:80])
        return [0.0] * EMBEDDING_DIM
    return (raw / norm).tolist()


def text_to_embedding(
    text: str, judge: JudgeProvider | None = None
) -> list[float]:
    """Generate an embedding for the given text.

    When a judge provider is supplied, delegates to its ``embed()`` method
    for real semantic embeddings.  Otherwise falls back to the deterministic
    hash-based approach.
    """
    if judge is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, judge.embed(text))
                embedding = future.result()
        else:
            embedding = asyncio.run(judge.embed(text))

        logger.debug(
            "Generated %d-dim provider embedding for text: '%s'",
            len(embedding),
            text[:80],
        )
        return embedding

    logger.debug(
        "Generated %d-dim hash embedding for text: '%s'", EMBEDDING_DIM, text[:80]
    )
    return _hash_embedding(text)


def compute_similarity(embedding_a: list[float], embedding_b: list[float]) -> float:
    a = np.array(embedding_a).reshape(1, -1)
    b = np.array(embedding_b).reshape(1, -1)
    score = float(cosine_similarity(a, b)[0][0])
    logger.debug("Cosine similarity: %.4f", score)
    return score
