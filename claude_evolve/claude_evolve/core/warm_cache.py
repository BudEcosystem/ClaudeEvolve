"""
Warm-start artifact cache for Claude Evolve.

Persists intermediate computation results between evolution iterations.
When a candidate computes expensive intermediate state (e.g., a matrix,
a trained model checkpoint, search state), it can save this state to the
warm cache. Subsequent iterations load the state and continue from where
the previous iteration left off.

This is CRITICAL for problems where:
- Each iteration recomputes the same base structure
- Search is incremental (SA, tabu, genetic algorithms)
- Evaluation involves expensive preprocessing

Generic interface: stores arbitrary binary/text/numpy data keyed by name.
"""

import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WarmCache:
    """Persists intermediate computation state between evolution iterations.

    Storage layout:
        cache_dir/
        ├── manifest.json       # Index of cached items with metadata
        ├── items/
        │   ├── <key>.npy       # numpy arrays
        │   ├── <key>.json      # JSON-serializable data
        │   └── <key>.txt       # plain text data
    """

    def __init__(self, cache_dir: str, max_items: int = 50) -> None:
        self.cache_dir = cache_dir
        self.max_items = max_items
        self.items_dir = os.path.join(cache_dir, "items")
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        self._access_times_path = os.path.join(cache_dir, "access_times.json")
        self.manifest: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._loaded = False

    def load(self) -> None:
        """Load manifest and access times from disk.

        Creates the cache directory structure if it does not exist.
        If a manifest file exists, its contents are loaded into memory.
        Access times are loaded from a separate file; missing entries
        default to the manifest timestamp.
        """
        os.makedirs(self.items_dir, exist_ok=True)
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)
        if os.path.exists(self._access_times_path):
            with open(self._access_times_path, "r") as f:
                self._access_times = json.load(f)
        # Backfill access times for keys present in manifest but missing
        # from the access-times file (e.g. pre-LRU caches).
        for key, info in self.manifest.items():
            if key not in self._access_times:
                self._access_times[key] = info.get("timestamp", time.time())
        self._loaded = True
        logger.info(
            "Loaded warm cache from %s (%d items)",
            self.cache_dir,
            len(self.manifest),
        )

    def save(self) -> None:
        """Save manifest and access times to disk.

        Creates the cache directory structure if it does not exist and
        writes the current manifest and access times as indented JSON.
        """
        os.makedirs(self.items_dir, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        with open(self._access_times_path, "w") as f:
            json.dump(self._access_times, f, indent=2)

    # ------------------------------------------------------------------
    # LRU eviction helpers
    # ------------------------------------------------------------------

    def _touch(self, key: str) -> None:
        """Update the access time for *key* to the current time."""
        self._access_times[key] = time.time()

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used items until count is within *max_items*."""
        while len(self.manifest) > self.max_items:
            # Find the key with the oldest access time
            oldest_key = min(self._access_times, key=self._access_times.get)
            logger.debug("LRU evicting cache key '%s'", oldest_key)
            self._remove_item(oldest_key)

    def _remove_item(self, key: str) -> None:
        """Remove a single item from manifest, access times, and disk."""
        self.manifest.pop(key, None)
        self._access_times.pop(key, None)
        # Remove all backing files that could exist for this key
        for ext in (".npy", ".json", ".txt"):
            path = os.path.join(self.items_dir, f"{key}{ext}")
            if os.path.exists(path):
                os.remove(path)

    # ------------------------------------------------------------------
    # numpy storage
    # ------------------------------------------------------------------

    def put_numpy(self, key: str, array: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Store a numpy array under *key*.

        Args:
            key: Unique string identifier for this cached item.
            array: The numpy array to persist.
            metadata: Optional dictionary of descriptive metadata (e.g.
                ``description``, ``score``).
        """
        path = os.path.join(self.items_dir, f"{key}.npy")
        np.save(path, array)
        self.manifest[key] = {
            "type": "numpy",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._touch(key)
        self._evict_if_needed()
        self.save()

    def get_numpy(self, key: str) -> Optional[np.ndarray]:
        """Load a numpy array previously stored under *key*.

        Returns ``None`` if the key is not found, is not of numpy type,
        or the backing ``.npy`` file is missing.
        """
        if key not in self.manifest or self.manifest[key]["type"] != "numpy":
            return None
        path = os.path.join(self.items_dir, f"{key}.npy")
        if not os.path.exists(path):
            return None
        self._touch(key)
        return np.load(path)

    # ------------------------------------------------------------------
    # JSON storage
    # ------------------------------------------------------------------

    def put_json(self, key: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Store JSON-serializable *data* under *key*.

        Args:
            key: Unique string identifier for this cached item.
            data: Any JSON-serializable Python object.
            metadata: Optional dictionary of descriptive metadata.
        """
        path = os.path.join(self.items_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        self.manifest[key] = {
            "type": "json",
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._touch(key)
        self._evict_if_needed()
        self.save()

    def get_json(self, key: str) -> Optional[Any]:
        """Load JSON data previously stored under *key*.

        Returns ``None`` if the key is not found, is not of json type,
        or the backing ``.json`` file is missing.
        """
        if key not in self.manifest or self.manifest[key]["type"] != "json":
            return None
        path = os.path.join(self.items_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        self._touch(key)
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Plain-text storage
    # ------------------------------------------------------------------

    def put_text(self, key: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Store plain *text* under *key*.

        Args:
            key: Unique string identifier for this cached item.
            text: The text string to persist.
            metadata: Optional dictionary of descriptive metadata.
        """
        path = os.path.join(self.items_dir, f"{key}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.manifest[key] = {
            "type": "text",
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._touch(key)
        self._evict_if_needed()
        self.save()

    def get_text(self, key: str) -> Optional[str]:
        """Load plain text previously stored under *key*.

        Returns ``None`` if the key is not found, is not of text type,
        or the backing ``.txt`` file is missing.
        """
        if key not in self.manifest or self.manifest[key]["type"] != "text":
            return None
        path = os.path.join(self.items_dir, f"{key}.txt")
        if not os.path.exists(path):
            return None
        self._touch(key)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        """Check if *key* exists in the cache manifest."""
        return key in self.manifest

    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata for a cached item, or ``None`` if absent."""
        if key not in self.manifest:
            return None
        return self.manifest[key].get("metadata", {})

    def keys(self) -> List[str]:
        """List all cached keys."""
        return list(self.manifest.keys())

    def clear(self) -> None:
        """Clear all cached items, removing backing files and resetting the manifest."""
        if os.path.exists(self.items_dir):
            shutil.rmtree(self.items_dir)
        os.makedirs(self.items_dir, exist_ok=True)
        self.manifest = {}
        self._access_times = {}
        self.save()

    # ------------------------------------------------------------------
    # Prompt integration
    # ------------------------------------------------------------------

    def format_for_prompt(self) -> str:
        """Format cache contents for inclusion in evolution prompts.

        Returns a Markdown-formatted string describing available cached
        items so that the evolution agent knows what intermediate state
        can be loaded.  Returns an empty string when the cache is empty.
        """
        if not self.manifest:
            return ""
        lines = [
            "## Warm-Start Cache",
            "",
            "The following intermediate computation results are available from previous iterations:",
            "Load them to avoid recomputing:",
            "",
        ]
        for key, info in self.manifest.items():
            meta = info.get("metadata", {})
            desc = meta.get("description", "")
            score = meta.get("score", "")
            item_type = info.get("type", "unknown")
            entry = f"- **{key}** ({item_type})"
            if desc:
                entry += f": {desc}"
            if score != "":
                entry += f" [score: {score}]"
            lines.append(entry)

        lines.append("")
        lines.append(
            "Use `warm_cache.get_numpy(key)` or `warm_cache.get_json(key)` to load."
        )
        return "\n".join(lines)
