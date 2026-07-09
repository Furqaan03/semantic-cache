"""Cache policies: TTL-tier auto-assignment and per-request-type adaptive thresholds."""
from __future__ import annotations

import re

# Time-sensitive prompts should have short TTLs or skip caching entirely.
_TIME_SENSITIVE = re.compile(
    r"\b(today|now|current|latest|this (week|month|year)|right now|breaking|live|as of)\b", re.I
)
_STABLE = re.compile(r"\b(what is|define|explain|how does|history of|formula for)\b", re.I)


def classify_ttl(prompt: str, default_ttl: float = 86400.0) -> float:
    """Long TTL for stable/factual queries, short for time-referencing ones."""
    if _TIME_SENSITIVE.search(prompt):
        return 3600.0          # 1h — references current state
    if _STABLE.search(prompt):
        return 7 * 86400.0     # 7d — stable facts
    return default_ttl


# Different request types tolerate different similarity looseness.
def adaptive_threshold(request_type: str, base: float = 0.95) -> float:
    """Classification can tolerate looser matches (constrained answer space);
    creative generation needs near-exact or shouldn't cache."""
    return {
        "classification": 0.90,
        "extraction": 0.93,
        "qa": 0.95,
        "creative": 0.99,
    }.get(request_type, base)


def should_cache(request_type: str) -> bool:
    return request_type != "creative_no_cache"
