

from __future__ import annotations

import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ── data model 
@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray          
    result: Any                    
    dominant_cluster: int
    membership_vector: np.ndarray  
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0             


# ── cache 

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

        
        self._store: dict[int, list[CacheEntry]] = {c: [] for c in range(n_clusters)}

        
        self._lru: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self._entry_id_counter = 0

        # Stats
        self._hit_count  = 0
        self._miss_count = 0

        # Thread safety for concurrent FastAPI requests
        self._lock = threading.RLock()

    

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
                
                for c, m in enumerate(membership_vector):
                    if m > 0.25 and c != dominant_cluster:
                        buckets_to_search.add(c)

            for bucket in buckets_to_search:
                for eid, entry in self._iter_bucket(bucket):
                    # Cosine similarity: both embeddings are L2-normalised,
                    
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

    

    def _iter_bucket(self, bucket: int):
        """Yield (entry_id, entry) pairs for a given bucket."""
        
        bucket_entries = []
        for eid, (c, _pos) in self._lru.items():
            if c == bucket:
                
                pass
        
        entries = self._store.get(bucket, [])
        
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