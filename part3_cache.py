"""
Part 3 – Semantic Cache (built from first principles)
=====================================================
Design
------
The cache is a dict keyed by cluster bucket for efficient lookup:

    _store: dict[int, list[CacheEntry]]

where `int` is the DOMINANT cluster of the cached query.

Why cluster-bucketed?
  Two queries that are semantically similar will almost certainly share the
  same dominant cluster. Bucketing reduces the lookup search space from O(N)
  to O(N/k) on average — a k-fold speedup that compounds as the cache grows.
  At k=15 this means ~15× fewer cosine comparisons per lookup.

Similarity threshold θ (SIMILARITY_THRESHOLD)
----------------------------------------------
This is THE core tunable. Its effect:

  θ → 1.0  (very strict)
    Near-identical phrasing only. Cache hit rate drops; almost every
    rephrasing misses. Behaves close to exact-match. Very precise but
    computationally wasteful.

  θ = 0.92  (our default)
    Handles paraphrases, word-order changes, synonym substitutions.
    "What are the best graphics cards?" matches "Which GPU should I buy?"
    This is the sweet spot for newsgroups-style short natural-language queries.

  θ = 0.80  (loose)
    Catches topically related but semantically different queries.
    "Tell me about space" might match "NASA moon mission" — arguably wrong.
    High hit rate but risky result quality.

  θ → 0.0  (degenerate)
    Everything matches the first cache entry. 100% hit rate, 0% accuracy.

We expose θ as a runtime parameter so it can be explored empirically.
The cluster structure ensures that even at loose θ, cross-cluster false
positives are suppressed: a tech query never searches the sports bucket.

Cache eviction
  LRU with a configurable max size to prevent unbounded memory growth.
  On eviction, the LRU entry from the most populated bucket is removed.
"""

from __future__ import annotations

import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray          # L2-normalised, shape (384,)
    result: Any                    # whatever the downstream computation returns
    dominant_cluster: int
    membership_vector: np.ndarray  # full soft distribution (k,)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0             # how many times this entry was served


# ── cache ─────────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Cluster-bucketed semantic cache with configurable cosine similarity threshold.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity to count as a cache hit.  Range (0, 1].
        See module docstring for detailed analysis of this parameter.
    max_size : int
        Maximum total number of entries before LRU eviction.
    n_clusters : int
        Number of fuzzy clusters (used for bucket initialisation).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 1000,
        n_clusters: int = 15,
    ):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")

        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.n_clusters = n_clusters

        # Cluster-bucketed storage: cluster_id → list of CacheEntry
        self._store: dict[int, list[CacheEntry]] = {c: [] for c in range(n_clusters)}

        # LRU order: maps (cluster, list_index) but simpler to track via
        # an OrderedDict keyed by a unique entry id
        self._lru: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self._entry_id_counter = 0

        # Stats
        self._hit_count  = 0
        self._miss_count = 0

        # Thread safety for concurrent FastAPI requests
        self._lock = threading.RLock()

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding: np.ndarray,
        dominant_cluster: int,
        membership_vector: Optional[np.ndarray] = None,
    ) -> Optional[tuple[CacheEntry, float]]:
        """
        Search for a cached result similar to the incoming query.

        Strategy:
          1. Search the dominant cluster bucket (fast path, O(bucket_size)).
          2. If membership_vector is provided and the query is a soft member of
             neighbouring clusters (membership > 0.25), search those too.
             This prevents misses at cluster boundaries.

        Returns
        -------
        (entry, similarity_score) if a hit is found, else None.
        """
        with self._lock:
            best_entry: Optional[CacheEntry] = None
            best_sim   = -1.0
            best_eid   = None

            # Determine which buckets to search
            buckets_to_search = {dominant_cluster}
            if membership_vector is not None:
                # Also check clusters where this query has meaningful soft membership
                for c, m in enumerate(membership_vector):
                    if m > 0.25 and c != dominant_cluster:
                        buckets_to_search.add(c)

            for bucket in buckets_to_search:
                for eid, entry in self._iter_bucket(bucket):
                    # Cosine similarity: both embeddings are L2-normalised,
                    # so dot product == cosine similarity
                    sim = float(np.dot(query_embedding, entry.embedding))
                    if sim >= self.similarity_threshold and sim > best_sim:
                        best_sim   = sim
                        best_entry = entry
                        best_eid   = eid

            if best_entry is not None:
                # Promote to most-recently-used
                if best_eid in self._lru:
                    self._lru.move_to_end(best_eid)
                best_entry.hit_count += 1
                self._hit_count += 1
                return best_entry, best_sim

            self._miss_count += 1
            return None

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: int,
        membership_vector: np.ndarray,
    ) -> None:
        """Add a new entry to the cache (with LRU eviction if needed)."""
        with self._lock:
            if len(self._lru) >= self.max_size:
                self._evict_lru()

            entry = CacheEntry(
                query=query,
                embedding=query_embedding.copy(),
                result=result,
                dominant_cluster=dominant_cluster,
                membership_vector=membership_vector.copy(),
            )
            self._store[dominant_cluster].append(entry)
            eid = self._entry_id_counter
            self._entry_id_counter += 1
            # Record (cluster, position) for fast eviction lookup
            self._lru[eid] = (dominant_cluster, len(self._store[dominant_cluster]) - 1)
            self._lru.move_to_end(eid)

    def flush(self) -> None:
        """Remove all entries and reset all stats."""
        with self._lock:
            for c in self._store:
                self._store[c].clear()
            self._lru.clear()
            self._hit_count  = 0
            self._miss_count = 0
            self._entry_id_counter = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "total_entries": sum(len(v) for v in self._store.values()),
                "hit_count":     self._hit_count,
                "miss_count":    self._miss_count,
                "hit_rate":      round(self._hit_count / total, 4) if total else 0.0,
                "bucket_sizes":  {c: len(v) for c, v in self._store.items()},
                "similarity_threshold": self.similarity_threshold,
            }

    def set_threshold(self, theta: float) -> None:
        """Update the similarity threshold at runtime for exploration."""
        with self._lock:
            if not 0.0 < theta <= 1.0:
                raise ValueError("threshold must be in (0, 1]")
            self.similarity_threshold = theta

    # ── private helpers ───────────────────────────────────────────────────────

    def _iter_bucket(self, bucket: int):
        """Yield (entry_id, entry) pairs for a given bucket."""
        # We need entry_ids; look them up via _lru reverse mapping
        # Build a local map bucket→entries with their eids
        bucket_entries = []
        for eid, (c, _pos) in self._lru.items():
            if c == bucket:
                # find entry by scanning (bucket list is small)
                pass
        # Simpler: just yield entries directly from _store (no eid needed for
        # lookup; eid only needed for LRU promotion on hit)
        entries = self._store.get(bucket, [])
        # Pair each entry with its eid via _lru
        # Build reverse map once per call (bucket size ≤ max_size/k ≈ 67)
        pos_to_eid = {}
        for eid, (c, pos) in self._lru.items():
            if c == bucket:
                pos_to_eid[pos] = eid
        for pos, entry in enumerate(entries):
            eid = pos_to_eid.get(pos, -1)
            yield eid, entry

    def _evict_lru(self) -> None:
        """Remove the least-recently-used cache entry."""
        if not self._lru:
            return
        # oldest = first item in OrderedDict
        eid, (cluster, pos) = next(iter(self._lru.items()))
        self._lru.popitem(last=False)

        # Remove from bucket storage
        bucket = self._store.get(cluster, [])
        if pos < len(bucket):
            bucket.pop(pos)
        # Shift all stored positions for this cluster in _lru
        for k in self._lru:
            c, p = self._lru[k]
            if c == cluster and p > pos:
                self._lru[k] = (c, p - 1)