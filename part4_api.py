"""
Part 4 – FastAPI Service
========================
Startup:
  uvicorn part4_api:app --host 0.0.0.0 --port 8000

On startup the service loads:
  - The SentenceTransformer embedding model
  - FCM centroids (to assign incoming queries to clusters)
  - The SemanticCache singleton

State is held in-process (single worker).  For multi-worker deployments the
cache would need to be externalised, but the problem statement explicitly
forbids Redis/Memcached, so a shared-memory approach (e.g. multiprocessing
Manager) would be the production path.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from part3_cache import SemanticCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── global state (loaded once at startup) ─────────────────────────────────────

class AppState:
    model:    SentenceTransformer
    centroids: np.ndarray        # shape (k, 50)  – FCM cluster centres in PCA space
    pca:      any                # fitted PCA to project queries into same space
    cache:    SemanticCache
    docs:     list[str]          # cleaned corpus (for result retrieval)
    categories: list[str]


state = AppState()


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup."""
    log.info("Loading embedding model …")
    state.model = SentenceTransformer("all-MiniLM-L6-v2")

    log.info("Loading FCM centroids and fitting PCA …")
    if not Path("fcm_centroids.npy").exists():
        raise RuntimeError("fcm_centroids.npy not found — run part2_cluster.py first")
    state.centroids = np.load("fcm_centroids.npy")   # (k, 50) in PCA space

    # Refit PCA on the reduced embeddings so we can project queries at runtime
    if not Path("embeddings_reduced.npy").exists():
        raise RuntimeError("embeddings_reduced.npy not found — run part2_cluster.py first")
    emb_full     = np.load("embeddings.npy")          # (N, 384) original
    state.pca    = PCA(n_components=50, random_state=42)
    state.pca.fit(emb_full)
    log.info("PCA fitted (50 components)")
    n_clusters = state.centroids.shape[0]

    log.info("Loading corpus …")
    if Path("docs_cleaned.json").exists():
        state.docs       = json.load(open("docs_cleaned.json"))
        state.categories = json.load(open("categories.json"))
    else:
        state.docs, state.categories = [], []

    log.info("Initialising semantic cache …")
    state.cache = SemanticCache(
        similarity_threshold=0.92,
        max_size=1000,
        n_clusters=n_clusters,
    )

    log.info("Service ready ✓")
    yield
    # Cleanup (nothing needed here)


app = FastAPI(
    title="Newsgroups Semantic Search",
    version="1.0.0",
    lifespan=lifespan,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    """Return L2-normalised embedding for a single query string."""
    vec = state.model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec[0]  # shape (384,)


def soft_cluster_assignment(embedding: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Compute fuzzy membership of an arbitrary embedding w.r.t. FCM centroids.

    Uses the standard FCM membership formula:
        u_ic = 1 / sum_j (d_ic / d_ij)^(2/(m-1))
    where d = Euclidean distance to centroid and m=2 (so exponent = 2).

    The embedding is first projected into PCA-50 space (same space used
    during training) to avoid the curse-of-dimensionality collapse.

    Returns (membership_vector, dominant_cluster_id).
    """
    # Project to same 50-dim PCA space used during FCM training
    embedding_50 = state.pca.transform(embedding.reshape(1, -1))[0]
    centroids = state.centroids   # (k, 50)
    m = 2.0                       # same fuzziness exponent used in training

    # Euclidean distances to each centroid
    diffs = centroids - embedding_50[np.newaxis, :]  # (k, 50)
    dists = np.linalg.norm(diffs, axis=1)            # (k,)

    # Guard against zero distance (query == centroid)
    if np.any(dists == 0.0):
        idx = int(np.argmin(dists))
        u = np.zeros(len(dists))
        u[idx] = 1.0
        return u, idx

    # FCM membership formula  (operates in 50-d PCA space)
    exponent = 2.0 / (m - 1)     # = 2.0 when m=2
    inv_dists = 1.0 / dists       # (k,)
    u = (inv_dists ** exponent) / np.sum(inv_dists ** exponent)
    dominant = int(np.argmax(u))
    return u, dominant


def retrieve_result(query: str, dominant_cluster: int) -> str:
    """
    'Compute' a result for a cache miss.

    In a production system this would call a retrieval-augmented generation
    pipeline or a full-text search index.  Here we return the top-3 most
    cosine-similar documents from the corpus whose dominant cluster matches,
    simulating the downstream computation that the cache is protecting.

    This is the expensive operation the cache is designed to avoid.
    """
    if not state.docs:
        return f"[No corpus loaded] Query received: {query}"

    # Load full embedding matrix (would be held in memory in production)
    emb_path = Path("embeddings.npy")
    if not emb_path.exists():
        return f"[embeddings.npy missing] Query: {query}"

    all_embs  = np.load(emb_path)                            # (N, 384)
    dom_path  = Path("dominant_clusters.npy")
    dominant_arr = np.load(dom_path) if dom_path.exists() else None

    query_emb = embed_query(query)

    # Filter to docs in the same dominant cluster for speed
    if dominant_arr is not None:
        mask = dominant_arr == dominant_cluster
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.arange(len(all_embs))
    else:
        idx = np.arange(len(all_embs))

    sims      = all_embs[idx] @ query_emb                    # dot = cosine (normalised)
    top_local = np.argsort(sims)[::-1][:3]
    top_global = idx[top_local]

    snippets = []
    for rank, gi in enumerate(top_global, 1):
        cat  = state.categories[gi] if gi < len(state.categories) else "unknown"
        text = state.docs[gi][:200] if gi < len(state.docs) else ""
        snippets.append(f"[{rank}] ({cat}) {text}")

    return "\n\n".join(snippets)


# ── schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: str
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., gt=0.0, le=1.0)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    """
    Embed the query, check semantic cache, retrieve or reuse result.
    """
    query_text = body.query.strip()

    # 1. Embed
    q_emb = embed_query(query_text)

    # 2. Cluster assignment (soft)
    membership, dominant = soft_cluster_assignment(q_emb)

    # 3. Cache lookup
    hit = state.cache.lookup(q_emb, dominant, membership_vector=membership)

    if hit is not None:
        entry, sim = hit
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(sim, 6),
            result=entry.result,
            dominant_cluster=entry.dominant_cluster,
        )

    # 4. Cache miss → compute result (the expensive step)
    result = retrieve_result(query_text, dominant)

    # 5. Store in cache
    state.cache.store(
        query=query_text,
        query_embedding=q_emb,
        result=result,
        dominant_cluster=dominant,
        membership_vector=membership,
    )

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant,
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Return current cache statistics."""
    s = state.cache.stats
    return CacheStats(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


@app.delete("/cache")
async def flush_cache():
    """Flush the entire cache and reset all stats."""
    state.cache.flush()
    return {"status": "cache flushed", "entries_removed": True}


@app.post("/cache/threshold")
async def set_threshold(body: ThresholdRequest):
    """
    Update the similarity threshold at runtime.
    Useful for exploring the θ trade-off without restarting the service.
    """
    state.cache.set_threshold(body.threshold)
    return {
        "status": "threshold updated",
        "similarity_threshold": body.threshold,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "cache_entries": state.cache.stats["total_entries"]}

@app.post("/debug/similarity")
async def debug_similarity(body: dict):
    q1 = embed_query(body["query1"])
    q2 = embed_query(body["query2"])
    sim = float(np.dot(q1, q2))
    return {
        "similarity": round(sim, 4),
        "threshold": state.cache.similarity_threshold,
        "would_hit": sim >= state.cache.similarity_threshold
    }