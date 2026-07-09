"""Semantic cache store: embed prompts, nearest-neighbor lookup, TTL, params scoping.

In-memory numpy index for portability/tests; the docker-compose path swaps this
for Redis + RedisVL for sub-millisecond distributed lookups (same interface)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from src.cache.embeddings import Embedder, cosine_similarity


@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: np.ndarray
    scope_key: str          # system prompt + params hash — prevents cross-contamination
    created_at: float       # injected timestamp
    ttl_seconds: float
    hit_count: int = 0
    model_id: str = ""


@dataclass
class LookupResult:
    hit: bool
    response: str | None
    similarity: float
    entry: CacheEntry | None = None


def scope_key(system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Two identical user prompts with different system prompts / params must NOT
    share a cache entry."""
    raw = f"{system_prompt}|{model}|{temperature}|{max_tokens}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class SemanticCache:
    embedder: Embedder
    threshold: float = 0.95
    default_ttl: float = 86400.0
    entries: list[CacheEntry] = field(default_factory=list)
    near_misses: list[float] = field(default_factory=list)  # similarities just below threshold

    def _valid(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.created_at) < entry.ttl_seconds

    def lookup(self, prompt: str, scope: str, now: float) -> LookupResult:
        query_vec = self.embedder(prompt)
        best: CacheEntry | None = None
        best_sim = 0.0

        for entry in self.entries:
            if entry.scope_key != scope or not self._valid(entry, now):
                continue
            sim = cosine_similarity(query_vec, entry.embedding)
            if sim > best_sim:
                best_sim, best = sim, entry

        if best is not None and best_sim >= self.threshold:
            best.hit_count += 1
            return LookupResult(hit=True, response=best.response, similarity=best_sim, entry=best)

        if best is not None and best_sim >= self.threshold - 0.05:
            self.near_misses.append(best_sim)  # track for threshold tuning
        return LookupResult(hit=False, response=None, similarity=best_sim)

    def store(self, prompt: str, response: str, scope: str, now: float, model_id: str = "", ttl: float | None = None) -> None:
        self.entries.append(CacheEntry(
            prompt=prompt, response=response, embedding=self.embedder(prompt),
            scope_key=scope, created_at=now, ttl_seconds=ttl if ttl is not None else self.default_ttl,
            model_id=model_id,
        ))

    def evict_expired(self, now: float) -> int:
        before = len(self.entries)
        self.entries = [e for e in self.entries if self._valid(e, now)]
        return before - len(self.entries)

    def invalidate_scope(self, scope: str) -> int:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.scope_key != scope]
        return before - len(self.entries)
