# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""In-memory cache for cell image outputs.

Stores base64-encoded image data with TTL expiration and LRU eviction,
enabling MCP ResourceLink references instead of inline image content.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class CachedImage:
    image_id: str
    data: str
    mime_type: str
    created_at: float = field(default_factory=time.time)


class ImageCache:
    """Thread-safe in-memory image cache with TTL and max-entry eviction."""

    def __init__(self, ttl_seconds: int = 600, max_entries: int = 100):
        self._cache: dict[str, CachedImage] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries

    def store(self, data: str, mime_type: str = "image/png") -> str:
        """Store base64 image data and return a unique image ID."""
        image_id = uuid.uuid4().hex
        entry = CachedImage(image_id=image_id, data=data, mime_type=mime_type)

        with self._lock:
            self._evict_expired()

            # If still at capacity after expiring, evict oldest entries
            while len(self._cache) >= self._max_entries:
                oldest_id = min(self._cache, key=lambda k: self._cache[k].created_at)
                del self._cache[oldest_id]

            self._cache[image_id] = entry

        return image_id

    def get(self, image_id: str) -> CachedImage | None:
        """Retrieve a cached image by ID, or None if expired/missing."""
        with self._lock:
            entry = self._cache.get(image_id)
            if entry is None:
                return None
            if time.time() - entry.created_at > self._ttl_seconds:
                del self._cache[image_id]
                return None
            return entry

    def list_images(self) -> list[CachedImage]:
        """Return all non-expired cached images (for resources/list)."""
        with self._lock:
            self._evict_expired()
            return list(self._cache.values())

    def _evict_expired(self):
        """Remove expired entries. Must be called with lock held."""
        now = time.time()
        expired = [
            image_id
            for image_id, entry in self._cache.items()
            if now - entry.created_at > self._ttl_seconds
        ]
        for image_id in expired:
            del self._cache[image_id]


# Module-level singleton
_image_cache: ImageCache | None = None
_cache_lock = threading.Lock()


def get_image_cache() -> ImageCache:
    """Get the global ImageCache singleton."""
    global _image_cache
    if _image_cache is None:
        with _cache_lock:
            if _image_cache is None:
                _image_cache = ImageCache()
    return _image_cache
