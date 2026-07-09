import numpy as np

from src.cache.embeddings import deterministic_fake_embedder
from src.cache.store import SemanticCache, scope_key


def _cache(threshold=0.95):
    return SemanticCache(embedder=deterministic_fake_embedder(), threshold=threshold)


def test_exact_repeat_is_a_hit():
    cache = _cache()
    scope = scope_key("sys", "gpt-4o-mini", 0.0, 512)
    cache.store("What is Python?", "A language.", scope, now=0.0)
    result = cache.lookup("What is Python?", scope, now=1.0)
    assert result.hit is True
    assert result.response == "A language."


def test_different_prompt_is_a_miss():
    cache = _cache()
    scope = scope_key("sys", "gpt-4o-mini", 0.0, 512)
    cache.store("What is Python?", "A language.", scope, now=0.0)
    result = cache.lookup("Explain quantum entanglement in detail.", scope, now=1.0)
    assert result.hit is False


def test_different_scope_does_not_share_entry():
    cache = _cache()
    scope_a = scope_key("system A", "gpt-4o-mini", 0.0, 512)
    scope_b = scope_key("system B", "gpt-4o-mini", 0.0, 512)
    cache.store("Same prompt", "answer A", scope_a, now=0.0)
    result = cache.lookup("Same prompt", scope_b, now=1.0)
    assert result.hit is False  # same user prompt, different system prompt -> no cross-contamination


def test_expired_entry_is_not_returned():
    cache = _cache()
    scope = scope_key("sys", "gpt-4o-mini", 0.0, 512)
    cache.store("What is Python?", "A language.", scope, now=0.0, ttl=100.0)
    assert cache.lookup("What is Python?", scope, now=50.0).hit is True    # within TTL
    assert cache.lookup("What is Python?", scope, now=150.0).hit is False  # expired


def test_evict_expired_removes_stale():
    cache = _cache()
    scope = scope_key("sys", "gpt-4o-mini", 0.0, 512)
    cache.store("a", "1", scope, now=0.0, ttl=10.0)
    cache.store("b", "2", scope, now=0.0, ttl=1000.0)
    removed = cache.evict_expired(now=100.0)
    assert removed == 1
    assert len(cache.entries) == 1


def test_invalidate_scope():
    cache = _cache()
    scope_a = scope_key("sys A", "m", 0.0, 512)
    scope_b = scope_key("sys B", "m", 0.0, 512)
    cache.store("x", "1", scope_a, now=0.0)
    cache.store("y", "2", scope_b, now=0.0)
    removed = cache.invalidate_scope(scope_a)
    assert removed == 1
    assert all(e.scope_key == scope_b for e in cache.entries)
