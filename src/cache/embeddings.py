"""Embedding function, pluggable so the cache logic can be tested without OpenAI."""
from __future__ import annotations

import hashlib
from typing import Callable, Protocol

import numpy as np


class Embedder(Protocol):
    def __call__(self, text: str) -> np.ndarray: ...


def openai_embedder(model: str = "text-embedding-3-small") -> Callable[[str], np.ndarray]:
    from openai import OpenAI

    client = OpenAI()

    def embed(text: str) -> np.ndarray:
        resp = client.embeddings.create(model=model, input=text)
        return np.array(resp.data[0].embedding, dtype=float)

    return embed


def deterministic_fake_embedder(dim: int = 64) -> Callable[[str], np.ndarray]:
    """Hashes text into a stable pseudo-random unit vector. Same text -> same vector,
    similar-but-not-identical text -> different vector. For offline tests only."""

    def embed(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.lower().strip().encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim)
        return vec / np.linalg.norm(vec)

    return embed


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
