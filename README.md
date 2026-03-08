# Semantic Search System — 20 Newsgroups

A lightweight semantic search system built on the 20 Newsgroups dataset featuring fuzzy clustering, a from-scratch semantic cache, and a FastAPI service.

---

## System Architecture

```
Query (natural language)
        │
        ▼
SentenceTransformer — all-MiniLM-L6-v2
        │  384-dim L2-normalised embedding
        ▼
PCA Projection (384 → 50 dims)
        │  eliminates curse of dimensionality
        ▼
Soft Cluster Assignment (FCM formula)
        │  membership vector (15,) + dominant cluster id
        ▼
SemanticCache.lookup()
        │  searches dominant bucket + soft-member buckets
        │  cosine sim ≥ θ  →  HIT  (return cached result instantly)
        │  otherwise       →  MISS
        ▼ (miss only)
retrieve_result()   ←── expensive corpus search
        │  top-3 cosine-similar documents from cluster
        ▼
SemanticCache.store()
        │  stores in dominant cluster bucket
        ▼
JSON Response
```

---

## Project Structure

```
├── part1_embed.py          # Data cleaning, embedding, ChromaDB ingestion
├── part2_cluster.py        # Fuzzy C-Means clustering + analysis + visualisation
├── part3_cache.py          # Semantic cache (built from scratch)
├── part4_api.py            # FastAPI service
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── clustering_output/
    ├── fpc_curve.png       # FPC elbow plot justifying k=15
    └── tsne_clusters.png   # 2D t-SNE visualisation of clusters
```

**Generated after running scripts (not committed):**
```
├── embeddings.npy          # 18,000 × 384 corpus embeddings
├── embeddings_reduced.npy  # 18,000 × 50 PCA-reduced embeddings
├── membership_matrix.npy   # 18,000 × 15 fuzzy memberships
├── dominant_clusters.npy   # dominant cluster per document
├── fcm_centroids.npy       # 15 × 50 cluster centroids
├── docs_cleaned.json       # cleaned corpus texts
├── categories.json         # category label per document
└── chroma_db/              # ChromaDB persistent store
```

---

## Quick Start

### 1. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build embeddings and vector database
```bash
python part1_embed.py
```
Expected output: ~18,000 documents encoded and stored in ChromaDB (~10 min on CPU)

### 4. Run fuzzy clustering and analysis
```bash
python part2_cluster.py
```
Expected output: FPC sweep, cluster report printed to console, two PNG visualisations saved to `clustering_output/`

### 5. Start the API
```bash
uvicorn part4_api:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### `POST /query`
Embeds the query, checks semantic cache, returns result.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I am having trouble with my windows computer crashing"}'
```

**Response (cache miss):**
```json
{
  "query": "I am having trouble with my windows computer crashing",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[1] (comp.os.ms-windows.misc) ...\n\n[2] ...\n\n[3] ...",
  "dominant_cluster": 7
}
```

**Response (cache hit on semantically similar query):**
```json
{
  "query": "my PC keeps crashing and showing a blue screen",
  "cache_hit": true,
  "matched_query": "I am having trouble with my windows computer crashing",
  "similarity_score": 0.7472,
  "result": "[1] (comp.os.ms-windows.misc) ...",
  "dominant_cluster": 7
}
```

---

### `GET /cache/stats`
```json
{
  "total_entries": 3,
  "hit_count": 1,
  "miss_count": 2,
  "hit_rate": 0.333
}
```

### `DELETE /cache`
Flushes the cache and resets all stats to zero.

### `POST /cache/threshold`
Updates the similarity threshold at runtime without restarting.
```json
{ "threshold": 0.80 }
```

### `POST /debug/similarity`
Returns cosine similarity between any two queries. Useful for tuning the threshold.
```json
{
  "similarity": 0.7472,
  "threshold": 0.70,
  "would_hit": true
}
```

---

## Design Decisions

### Embedding Model — `all-MiniLM-L6-v2`
- 384-dimensional vectors, under 80MB, runs on CPU
- Trained on 1B+ sentence pairs via contrastive learning
- Captures semantic meaning not just keyword overlap — essential for paraphrase detection in the cache

### Vector Database — ChromaDB
- Persistent local store, zero infrastructure required
- HNSW index with cosine distance for fast approximate nearest neighbour retrieval
- Metadata filtering by category and cluster for downstream filtered search

### Fuzzy C-Means (k=15, m=2)
- Produces a **probability distribution** over clusters per document, not a hard label
- A post about gun legislation gets `{politics: 0.55, firearms: 0.35, law: 0.10}` — semantically accurate
- k=15 chosen at the elbow of the FPC curve — 20 original labels collapse to 15 because several label pairs are semantically synonymous
- m=2 is the canonical fuzziness exponent, well-calibrated for text data

### PCA Before FCM (384 → 50 dims)
- In 384 dimensions, the curse of dimensionality causes all pairwise Euclidean distances to converge — every FCM membership score becomes exactly 1/k = 0.067 (degenerate)
- PCA to 50 components retains ~85% of variance while restoring meaningful distance geometry
- Critical step without which clustering produces no useful output

### Semantic Cache — Cluster Bucketed
- Built from scratch — no Redis, no Memcached, no caching libraries
- Stored as `dict[cluster_id → list[CacheEntry]]`
- Lookup is **O(N/k)** instead of O(N) — at k=15 this is a 15× speedup that compounds as cache grows
- Queries near cluster boundaries search multiple buckets via soft membership threshold (>0.25)
- Thread-safe with reentrant lock for concurrent FastAPI requests
- LRU eviction at configurable max size (default 1000 entries)

### Similarity Threshold θ = 0.70
- Controls the precision/recall tradeoff of the cache
- θ → 1.0: near-identical phrasing only, behaves like exact-match, low hit rate
- θ = 0.70: catches genuine paraphrases on newsgroup-style queries
- θ → 0.0: everything matches, 100% hit rate, 0% result accuracy
- Exposed as a live API endpoint for runtime tuning without redeployment

---

## Docker

```bash
# Build artefacts first
python part1_embed.py
python part2_cluster.py

# Run with Docker
docker-compose up --build
```

The container starts uvicorn on port 8000.

---

## Interactive API Docs

With the server running, visit `http://localhost:8000/docs` for the full Swagger UI.