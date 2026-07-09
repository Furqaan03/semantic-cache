"""Prometheus metrics for cache performance and cost savings."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

CACHE_HITS = Counter("cache_hits_total", "Cache hits", ["model"])
CACHE_MISSES = Counter("cache_misses_total", "Cache misses", ["model"])
COST_SAVED = Counter("cache_cost_saved_usd_total", "Estimated cost saved by hits")
LATENCY = Histogram("cache_request_latency_seconds", "Request latency", ["outcome"])
CACHE_SIZE = Gauge("cache_entries", "Number of cache entries")
SIMILARITY = Histogram("cache_hit_similarity", "Similarity score of hits/near-misses", ["outcome"])
