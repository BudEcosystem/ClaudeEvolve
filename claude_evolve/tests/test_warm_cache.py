"""
Tests for the warm-start artifact cache module.

Covers WarmCache initialization, put/get round-trips for numpy/json/text,
has() checks, metadata retrieval, keys listing, clear(), format_for_prompt(),
non-existent key handling, multiple item persistence, and key overwrite.
"""

import json
import os

import numpy as np
import pytest

from claude_evolve.core.warm_cache import WarmCache


# ---------------------------------------------------------------------------
# WarmCache initialization
# ---------------------------------------------------------------------------

class TestWarmCacheInit:
    """Test WarmCache constructor defaults."""

    def test_default_initialization(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        assert cache.cache_dir == str(tmp_path / "cache")
        assert cache.items_dir == os.path.join(str(tmp_path / "cache"), "items")
        assert cache.manifest == {}
        assert cache._loaded is False

    def test_load_creates_directories(self, tmp_path):
        cache_dir = str(tmp_path / "nonexistent" / "deep" / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        assert os.path.isdir(cache_dir)
        assert os.path.isdir(os.path.join(cache_dir, "items"))
        assert cache._loaded is True

    def test_load_empty_directory(self, tmp_path):
        cache_dir = str(tmp_path / "empty_cache")
        os.makedirs(cache_dir)
        cache = WarmCache(cache_dir)
        cache.load()
        assert cache.manifest == {}
        assert cache._loaded is True


# ---------------------------------------------------------------------------
# put_numpy / get_numpy round-trip
# ---------------------------------------------------------------------------

class TestNumpyRoundTrip:
    """Test numpy array storage and retrieval."""

    def test_put_get_1d_array(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        cache.put_numpy("vec", arr)
        loaded = cache.get_numpy("vec")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    def test_put_get_2d_array(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        arr = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int32)
        cache.put_numpy("matrix", arr)
        loaded = cache.get_numpy("matrix")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)
        assert loaded.dtype == np.int32

    def test_put_get_large_array(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        arr = np.random.rand(100, 100)
        cache.put_numpy("big", arr)
        loaded = cache.get_numpy("big")
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, arr)

    def test_put_numpy_with_metadata(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        arr = np.zeros((5, 5))
        cache.put_numpy("zeros", arr, metadata={"description": "Zero matrix", "score": 0.5})
        info = cache.manifest["zeros"]
        assert info["type"] == "numpy"
        assert info["shape"] == [5, 5]
        assert info["dtype"] == "float64"
        assert info["metadata"]["description"] == "Zero matrix"
        assert info["metadata"]["score"] == 0.5

    def test_put_numpy_creates_npy_file(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("test_arr", np.array([1, 2, 3]))
        npy_path = os.path.join(cache.items_dir, "test_arr.npy")
        assert os.path.exists(npy_path)

    def test_put_numpy_persists_manifest(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_numpy("data", np.array([10, 20]))

        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert "data" in cache2.manifest
        loaded = cache2.get_numpy("data")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, np.array([10, 20]))


# ---------------------------------------------------------------------------
# put_json / get_json round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    """Test JSON data storage and retrieval."""

    def test_put_get_dict(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        cache.put_json("config", data)
        loaded = cache.get_json("config")
        assert loaded == data

    def test_put_get_list(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        data = [1, 2, 3, "hello", {"x": True}]
        cache.put_json("items", data)
        loaded = cache.get_json("items")
        assert loaded == data

    def test_put_get_scalar(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("count", 999)
        loaded = cache.get_json("count")
        assert loaded == 999

    def test_put_json_with_metadata(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("state", {"step": 100}, metadata={"description": "SA state"})
        info = cache.manifest["state"]
        assert info["type"] == "json"
        assert info["metadata"]["description"] == "SA state"

    def test_put_json_creates_json_file(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("test_data", {"x": 1})
        json_path = os.path.join(cache.items_dir, "test_data.json")
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)
        assert data == {"x": 1}

    def test_put_json_persists_manifest(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_json("params", {"lr": 0.01})

        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert "params" in cache2.manifest
        loaded = cache2.get_json("params")
        assert loaded == {"lr": 0.01}


# ---------------------------------------------------------------------------
# put_text / get_text round-trip
# ---------------------------------------------------------------------------

class TestTextRoundTrip:
    """Test plain text storage and retrieval."""

    def test_put_get_simple_text(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("log", "Iteration 5: best score 0.82")
        loaded = cache.get_text("log")
        assert loaded == "Iteration 5: best score 0.82"

    def test_put_get_multiline_text(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        text = "line1\nline2\nline3\n"
        cache.put_text("multi", text)
        loaded = cache.get_text("multi")
        assert loaded == text

    def test_put_get_empty_text(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("empty", "")
        loaded = cache.get_text("empty")
        assert loaded == ""

    def test_put_text_with_metadata(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("notes", "Important note", metadata={"description": "Run notes"})
        info = cache.manifest["notes"]
        assert info["type"] == "text"
        assert info["metadata"]["description"] == "Run notes"

    def test_put_text_creates_txt_file(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("test_txt", "hello world")
        txt_path = os.path.join(cache.items_dir, "test_txt.txt")
        assert os.path.exists(txt_path)
        with open(txt_path, "r") as f:
            content = f.read()
        assert content == "hello world"

    def test_put_text_persists_manifest(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_text("msg", "persisted message")

        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert "msg" in cache2.manifest
        loaded = cache2.get_text("msg")
        assert loaded == "persisted message"


# ---------------------------------------------------------------------------
# has() check
# ---------------------------------------------------------------------------

class TestHas:
    """Test has() key presence check."""

    def test_has_returns_false_for_empty_cache(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.has("anything") is False

    def test_has_returns_true_after_put_numpy(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]))
        assert cache.has("arr") is True

    def test_has_returns_true_after_put_json(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("data", {"x": 1})
        assert cache.has("data") is True

    def test_has_returns_true_after_put_text(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("note", "hello")
        assert cache.has("note") is True

    def test_has_returns_false_for_nonexistent_key(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("exists", np.array([1]))
        assert cache.has("does_not_exist") is False


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------

class TestGetMetadata:
    """Test get_metadata retrieval."""

    def test_get_metadata_returns_none_for_missing_key(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.get_metadata("missing") is None

    def test_get_metadata_returns_empty_dict_when_no_metadata(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("plain", {"x": 1})
        meta = cache.get_metadata("plain")
        assert meta == {}

    def test_get_metadata_returns_stored_metadata(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy(
            "arr",
            np.array([1, 2, 3]),
            metadata={"description": "Test array", "score": 0.9},
        )
        meta = cache.get_metadata("arr")
        assert meta is not None
        assert meta["description"] == "Test array"
        assert meta["score"] == 0.9

    def test_get_metadata_for_text_item(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("note", "content", metadata={"author": "test"})
        meta = cache.get_metadata("note")
        assert meta == {"author": "test"}


# ---------------------------------------------------------------------------
# keys() listing
# ---------------------------------------------------------------------------

class TestKeys:
    """Test keys() listing of all cached items."""

    def test_keys_empty_cache(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.keys() == []

    def test_keys_single_item(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("only", 42)
        assert cache.keys() == ["only"]

    def test_keys_multiple_items(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]))
        cache.put_json("data", {"x": 1})
        cache.put_text("note", "hello")
        keys = cache.keys()
        assert sorted(keys) == sorted(["arr", "data", "note"])

    def test_keys_after_overwrite(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("key1", "v1")
        cache.put_json("key1", "v2")
        assert cache.keys() == ["key1"]


# ---------------------------------------------------------------------------
# clear() removes everything
# ---------------------------------------------------------------------------

class TestClear:
    """Test clear() wipes all cached items."""

    def test_clear_empties_manifest(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1, 2, 3]))
        cache.put_json("data", {"x": 1})
        cache.put_text("note", "hello")
        assert len(cache.manifest) == 3
        cache.clear()
        assert cache.manifest == {}
        assert cache.keys() == []

    def test_clear_removes_backing_files(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]))
        cache.put_json("data", {"x": 1})
        npy_path = os.path.join(cache.items_dir, "arr.npy")
        json_path = os.path.join(cache.items_dir, "data.json")
        assert os.path.exists(npy_path)
        assert os.path.exists(json_path)
        cache.clear()
        assert not os.path.exists(npy_path)
        assert not os.path.exists(json_path)

    def test_clear_recreates_items_dir(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("x", 1)
        cache.clear()
        assert os.path.isdir(cache.items_dir)

    def test_clear_persists_empty_manifest(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_json("x", 1)
        cache.clear()

        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert cache2.manifest == {}

    def test_get_after_clear_returns_none(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1, 2, 3]))
        cache.clear()
        assert cache.get_numpy("arr") is None
        assert cache.get_json("arr") is None
        assert cache.get_text("arr") is None


# ---------------------------------------------------------------------------
# format_for_prompt output
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    """Test format_for_prompt Markdown output."""

    def test_empty_cache_returns_empty_string(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.format_for_prompt() == ""

    def test_includes_header(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("data", {"x": 1}, metadata={"description": "Test data"})
        output = cache.format_for_prompt()
        assert "## Warm-Start Cache" in output

    def test_includes_key_and_type(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("matrix", np.eye(3))
        output = cache.format_for_prompt()
        assert "**matrix**" in output
        assert "(numpy)" in output

    def test_includes_description(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("state", {}, metadata={"description": "Search state snapshot"})
        output = cache.format_for_prompt()
        assert "Search state snapshot" in output

    def test_includes_score(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("best", np.array([1]), metadata={"score": 0.95})
        output = cache.format_for_prompt()
        assert "[score: 0.95]" in output

    def test_includes_usage_instructions(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("note", "hello")
        output = cache.format_for_prompt()
        assert "warm_cache.get_numpy(key)" in output
        assert "warm_cache.get_json(key)" in output

    def test_multiple_items_listed(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]), metadata={"description": "Array item"})
        cache.put_json("cfg", {"x": 1}, metadata={"description": "Config item"})
        cache.put_text("log", "text", metadata={"description": "Log item"})
        output = cache.format_for_prompt()
        assert "**arr**" in output
        assert "**cfg**" in output
        assert "**log**" in output

    def test_no_score_omits_bracket(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("data", {}, metadata={"description": "No score"})
        output = cache.format_for_prompt()
        assert "[score:" not in output


# ---------------------------------------------------------------------------
# Non-existent key returns None
# ---------------------------------------------------------------------------

class TestNonExistentKey:
    """Test that retrieving a missing key returns None."""

    def test_get_numpy_missing_key(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.get_numpy("nonexistent") is None

    def test_get_json_missing_key(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.get_json("nonexistent") is None

    def test_get_text_missing_key(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        assert cache.get_text("nonexistent") is None

    def test_get_numpy_wrong_type(self, tmp_path):
        """Requesting numpy for a json key returns None."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("data", {"x": 1})
        assert cache.get_numpy("data") is None

    def test_get_json_wrong_type(self, tmp_path):
        """Requesting json for a numpy key returns None."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]))
        assert cache.get_json("arr") is None

    def test_get_text_wrong_type(self, tmp_path):
        """Requesting text for a numpy key returns None."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]))
        assert cache.get_text("arr") is None

    def test_get_numpy_missing_file(self, tmp_path):
        """Manifest entry exists but backing .npy file was deleted."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("gone", np.array([1, 2, 3]))
        # Manually remove the backing file
        os.remove(os.path.join(cache.items_dir, "gone.npy"))
        assert cache.get_numpy("gone") is None

    def test_get_json_missing_file(self, tmp_path):
        """Manifest entry exists but backing .json file was deleted."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("gone", {"x": 1})
        os.remove(os.path.join(cache.items_dir, "gone.json"))
        assert cache.get_json("gone") is None

    def test_get_text_missing_file(self, tmp_path):
        """Manifest entry exists but backing .txt file was deleted."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("gone", "hello")
        os.remove(os.path.join(cache.items_dir, "gone.txt"))
        assert cache.get_text("gone") is None


# ---------------------------------------------------------------------------
# Multiple items persistence
# ---------------------------------------------------------------------------

class TestMultipleItemsPersistence:
    """Test that multiple items of different types persist correctly."""

    def test_mixed_types_persist_across_load(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_numpy("matrix", np.eye(3), metadata={"description": "Identity"})
        cache.put_json("config", {"temp": 0.5, "steps": 1000})
        cache.put_text("log", "Step 1: initialized\nStep 2: optimized")

        # Reload from disk
        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert len(cache2.manifest) == 3
        assert cache2.has("matrix")
        assert cache2.has("config")
        assert cache2.has("log")

        np.testing.assert_array_equal(cache2.get_numpy("matrix"), np.eye(3))
        assert cache2.get_json("config") == {"temp": 0.5, "steps": 1000}
        assert cache2.get_text("log") == "Step 1: initialized\nStep 2: optimized"

    def test_multiple_numpy_arrays(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        for i in range(5):
            cache.put_numpy(f"arr_{i}", np.full((3, 3), i))
        assert len(cache.keys()) == 5
        for i in range(5):
            loaded = cache.get_numpy(f"arr_{i}")
            assert loaded is not None
            np.testing.assert_array_equal(loaded, np.full((3, 3), i))

    def test_incremental_additions(self, tmp_path):
        cache_dir = str(tmp_path / "cache")

        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_json("item1", "first")

        cache2 = WarmCache(cache_dir)
        cache2.load()
        cache2.put_json("item2", "second")

        cache3 = WarmCache(cache_dir)
        cache3.load()
        assert len(cache3.manifest) == 2
        assert cache3.get_json("item1") == "first"
        assert cache3.get_json("item2") == "second"


# ---------------------------------------------------------------------------
# Overwrite existing key
# ---------------------------------------------------------------------------

class TestOverwriteKey:
    """Test that overwriting an existing key replaces the data."""

    def test_overwrite_numpy(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1, 2, 3]))
        cache.put_numpy("arr", np.array([4, 5, 6]))
        loaded = cache.get_numpy("arr")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, np.array([4, 5, 6]))

    def test_overwrite_json(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("data", {"v": 1})
        cache.put_json("data", {"v": 2})
        loaded = cache.get_json("data")
        assert loaded == {"v": 2}

    def test_overwrite_text(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_text("note", "first")
        cache.put_text("note", "second")
        loaded = cache.get_text("note")
        assert loaded == "second"

    def test_overwrite_updates_metadata(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_numpy("arr", np.array([1]), metadata={"score": 0.5})
        cache.put_numpy("arr", np.array([2]), metadata={"score": 0.9})
        meta = cache.get_metadata("arr")
        assert meta is not None
        assert meta["score"] == 0.9

    def test_overwrite_changes_type(self, tmp_path):
        """Overwriting a key with a different type changes the type in manifest."""
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("item", {"x": 1})
        assert cache.manifest["item"]["type"] == "json"
        cache.put_text("item", "now text")
        assert cache.manifest["item"]["type"] == "text"
        # Old json file may still exist but get_json should return None
        # because manifest type changed
        assert cache.get_json("item") is None
        assert cache.get_text("item") == "now text"

    def test_overwrite_persists(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_json("data", {"v": 1})
        cache.put_json("data", {"v": 2})

        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert cache2.get_json("data") == {"v": 2}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_manifest_is_valid_json_on_disk(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("x", 1)
        cache.put_numpy("y", np.array([2]))
        with open(cache.manifest_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "x" in data
        assert "y" in data

    def test_timestamp_is_set(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("timed", 42)
        info = cache.manifest["timed"]
        assert "timestamp" in info
        assert isinstance(info["timestamp"], float)
        assert info["timestamp"] > 0

    def test_save_to_nested_nonexistent_directory(self, tmp_path):
        cache_dir = str(tmp_path / "a" / "b" / "c" / "cache")
        cache = WarmCache(cache_dir)
        cache.load()
        cache.put_text("deep", "nested data")
        assert os.path.isdir(cache_dir)

        cache2 = WarmCache(cache_dir)
        cache2.load()
        assert cache2.get_text("deep") == "nested data"

    def test_unicode_text_roundtrip(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        text = "Hello \u4e16\u754c \U0001f30d \u00e9\u00e8\u00ea"
        cache.put_text("unicode", text)
        loaded = cache.get_text("unicode")
        assert loaded == text

    def test_numpy_boolean_array(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        arr = np.array([[True, False], [False, True]])
        cache.put_numpy("mask", arr)
        loaded = cache.get_numpy("mask")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)
        assert loaded.dtype == np.bool_

    def test_json_null_value(self, tmp_path):
        cache = WarmCache(str(tmp_path / "cache"))
        cache.load()
        cache.put_json("nullable", None)
        loaded = cache.get_json("nullable")
        assert loaded is None  # JSON null -> Python None
        # But has() should still return True since the key is in manifest
        assert cache.has("nullable")
