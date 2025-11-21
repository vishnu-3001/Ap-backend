"""Simple in-memory cache helpers for LLM responses."""
from __future__ import annotations

import copy
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    created_at: float
    payload: Any


class LLMCache:
    """Naive in-memory cache with TTL and max-size eviction."""

    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 128) -> None:
        self.ttl_seconds = max(0, ttl_seconds)
        self.max_entries = max(1, max_entries)
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None

        if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
            # Expired entry
            self._store.pop(key, None)
            return None

        # LRU bump
        self._store.move_to_end(key)
        return copy.deepcopy(entry.payload)

    def set(self, key: str, payload: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = CacheEntry(time.time(), copy.deepcopy(payload))
            return

        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)

        self._store[key] = CacheEntry(time.time(), copy.deepcopy(payload))

    def clear(self) -> None:
        self._store.clear()


def cache_config_from_env() -> Dict[str, Any]:
    """Read cache configuration from environment variables."""

    ttl_raw = os.getenv("LANGGRAPH_CACHE_TTL", "600")
    size_raw = os.getenv("LANGGRAPH_CACHE_SIZE", "128")

    try:
        ttl = int(ttl_raw)
    except ValueError:
        ttl = 600

    try:
        size = int(size_raw)
    except ValueError:
        size = 128

    return {"ttl_seconds": ttl, "max_entries": size}
