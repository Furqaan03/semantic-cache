# Semantic Caching Layer for LLM APIs

A drop-in middleware cache that sits between an application and an LLM provider.
It detects *semantically similar* requests that have already been answered —
"What is Python?" and "Explain Python to me" hit the same entry — and serves the
cached response instantly, cutting latency to near-zero and API cost by 30-60%
on typical workloads.

## Why this exists

Every company running LLMs at scale sends redundant, near-duplicate requests
that burn money and add latency. A semantic cache is infrastructure with an ROI
an engineering manager understands immediately.

## Architecture

```
src/cache/embeddings.py    pluggable embedder (OpenAI for prod, deterministic fake
                            for offline tests) + cosine similarity
src/cache/store.py         SemanticCache: embed -> nearest-neighbor lookup -> TTL
                            check -> scope check; near-miss tracking for tuning
src/cache/policy.py        TTL auto-tiering by prompt content + adaptive per-
                            request-type similarity thresholds
src/proxy/app.py           drop-in OpenAI-compatible /v1/chat/completions proxy
src/monitoring/metrics.py  Prometheus: hit rate, cost saved, latency, cache size
```

## Design decisions

- **Cache keys are embeddings, not string hashes.** Exact-match caching misses
  the whole point — paraphrases of the same question should share an answer. A
  prompt is embedded and matched by cosine similarity against stored entries above
  a configurable threshold (default 0.95).
- **Scope keys prevent cross-contamination.** Two identical *user* prompts under
  different *system* prompts, models, or temperatures must NOT share an entry —
  the scope key hashes all of those together, so a customer-support-tuned answer
  never leaks into a code-assistant context.
- **Adaptive thresholds per request type.** Classification tolerates looser
  matches (0.90 — constrained answer space); creative generation demands near-exact
  (0.99) or shouldn't cache at all. One global threshold would either over-cache
  creative work or under-cache classification.
- **TTL auto-tiering by content.** A prompt containing "latest / today / current"
  gets a 1h TTL (or none); a stable "what is / define / formula for" fact gets 7
  days. Time-sensitive answers going stale in the cache is the main correctness
  risk, so it's handled at write time.
- **Time and the embedder are both injected.** `lookup`/`store`/`evict` take a
  `now` timestamp, and the embedder is a plug-in — so the entire cache core
  (similarity, TTL expiry, scope isolation, eviction) is tested deterministically
  offline with a hash-based fake embedder, no OpenAI key needed.
- **Near-misses are tracked.** Similarities that fall just below threshold are
  recorded, so the threshold can be tuned against real traffic instead of guessed.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env      # fill in OPENAI_API_KEY
uvicorn src.proxy.app:app --reload
```

## Drop-in usage

Point any OpenAI client at the proxy base URL — zero code changes beyond that:

```bash
curl -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Python?"}]}'
# first call -> X-Cache: MISS

curl -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Explain Python to me"}]}'
# semantically similar -> X-Cache: HIT (instant, no provider call)
```

## Tests

```bash
pytest tests/ -v
```

10 tests covering exact-repeat hits, semantic misses, scope isolation, TTL
expiry, eviction, and both policy layers (TTL tiering, adaptive thresholds) —
all offline via the deterministic fake embedder and injected clock.

## Docker

```bash
docker compose up --build   # proxy + Redis Stack (RedisVL) + Prometheus + Grafana
```

## Status

Phases 1-4 complete (similarity engine, drop-in proxy, TTL/threshold policies,
Prometheus monitoring). The in-memory numpy index is the default; swapping to
Redis + RedisVL for distributed sub-ms lookups is the compose path (same
`SemanticCache` interface). Phase 5's 2,000-request load test is a manual step.
```
