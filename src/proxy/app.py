"""Drop-in OpenAI-compatible proxy: change the base URL, get semantic caching free.

A cache hit returns instantly with X-Cache: HIT. A miss forwards to OpenAI,
stores the response, and returns X-Cache: MISS."""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.cache.embeddings import openai_embedder
from src.cache.policy import adaptive_threshold, classify_ttl
from src.cache.store import SemanticCache, scope_key
from src.monitoring import metrics

load_dotenv()

# Rough per-1M-token cost for savings estimate (gpt-4o-mini default).
_COST_PER_1M = 0.60

app = FastAPI(title="Semantic Cache Proxy")
_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            embedder=openai_embedder(),
            threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.95")),
            default_ttl=float(os.getenv("DEFAULT_TTL_SECONDS", "86400")),
        )
    return _cache


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512
    request_type: str = "qa"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> JSONResponse:
    cache = get_cache()
    now = time.time()

    system_prompt = next((m.content for m in req.messages if m.role == "system"), "")
    user_prompt = next((m.content for m in req.messages if m.role == "user"), "")
    scope = scope_key(system_prompt, req.model, req.temperature, req.max_tokens)

    cache.threshold = adaptive_threshold(req.request_type)

    start = time.perf_counter()
    result = cache.lookup(user_prompt, scope, now)
    if result.hit:
        metrics.CACHE_HITS.labels(model=req.model).inc()
        metrics.SIMILARITY.labels(outcome="hit").observe(result.similarity)
        metrics.COST_SAVED.inc(_COST_PER_1M * 0.0008)  # ~ avg request token estimate
        metrics.LATENCY.labels(outcome="hit").observe(time.perf_counter() - start)
        return JSONResponse(
            content=_openai_shape(req.model, result.response),
            headers={"X-Cache": "HIT", "X-Cache-Similarity": f"{result.similarity:.4f}"},
        )

    # Miss -> forward to the real provider.
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=req.model,
        messages=[m.model_dump() for m in req.messages],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    output = resp.choices[0].message.content or ""
    ttl = classify_ttl(user_prompt, cache.default_ttl)
    cache.store(user_prompt, output, scope, now, model_id=req.model, ttl=ttl)

    metrics.CACHE_MISSES.labels(model=req.model).inc()
    metrics.CACHE_SIZE.set(len(cache.entries))
    metrics.LATENCY.labels(outcome="miss").observe(time.perf_counter() - start)
    return JSONResponse(content=_openai_shape(req.model, output), headers={"X-Cache": "MISS"})


def _openai_shape(model: str, content: str) -> dict:
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


@app.get("/v1/cache/stats")
def cache_stats() -> dict:
    cache = get_cache()
    return {"entries": len(cache.entries), "threshold": cache.threshold, "near_misses_tracked": len(cache.near_misses)}


@app.get("/metrics")
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
