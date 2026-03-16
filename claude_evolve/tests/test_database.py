"""
Tests for the MAP-Elites ArtifactDatabase.

Covers initialization, add/get, MAP-Elites placement, island evolution,
sampling, persistence, artifact storage, diversity, feature scaling,
migration, and edge cases.
"""

import json
import os
import tempfile
import unittest

from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.config import DatabaseConfig


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestArtifactDatabaseInit(unittest.TestCase):
    def test_create_empty_database(self):
        db = ArtifactDatabase(DatabaseConfig())
        self.assertEqual(db.size(), 0)
        self.assertIsNone(db.get_best())

    def test_add_initial_artifact(self):
        db = ArtifactDatabase(DatabaseConfig())
        a = Artifact(id="init", content="print(1)", metrics={"combined_score": 0.5})
        db.add(a)
        self.assertEqual(db.size(), 1)
        best = db.get_best()
        self.assertEqual(best.id, "init")

    def test_num_islands_creates_correct_structures(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=7))
        self.assertEqual(len(db.islands), 7)
        self.assertEqual(len(db.island_feature_maps), 7)
        self.assertEqual(len(db.island_best_programs), 7)

    def test_random_seed_determinism(self):
        """Two databases with same seed should sample identically."""
        cfg = DatabaseConfig(num_islands=2, random_seed=999)
        db1 = ArtifactDatabase(cfg)
        db2 = ArtifactDatabase(cfg)
        for i in range(10):
            a = Artifact(
                id=f"p{i}", content=f"code {i}", metrics={"combined_score": i / 10.0}
            )
            db1.add(Artifact(**a.to_dict()))
            db2.add(Artifact(**a.to_dict()))
        # After re-seeding, both should behave identically
        import random

        random.seed(999)
        p1, _ = db1.sample(num_inspirations=2)
        random.seed(999)
        p2, _ = db2.sample(num_inspirations=2)
        self.assertEqual(p1.id, p2.id)


# ---------------------------------------------------------------------------
# MAP-Elites core
# ---------------------------------------------------------------------------
class TestMAPElites(unittest.TestCase):
    def setUp(self):
        self.db = ArtifactDatabase(
            DatabaseConfig(num_islands=2, feature_bins=5, population_size=100)
        )
        self.seed = Artifact(
            id="seed",
            content="x = 1",
            metrics={"combined_score": 0.3},
            complexity=10.0,
            diversity=0.5,
        )
        self.db.add(self.seed)

    def test_add_better_program_updates_best(self):
        better = Artifact(
            id="better",
            content="x = optimized()",
            parent_id="seed",
            generation=1,
            metrics={"combined_score": 0.8},
        )
        self.db.add(better)
        self.assertEqual(self.db.get_best().id, "better")

    def test_feature_grid_placement(self):
        a1 = Artifact(
            id="a1",
            content="short",
            metrics={"combined_score": 0.5},
            complexity=10.0,
            diversity=0.2,
        )
        a2 = Artifact(
            id="a2",
            content="a very long program " * 100,
            metrics={"combined_score": 0.5},
            complexity=500.0,
            diversity=0.9,
        )
        self.db.add(a1)
        self.db.add(a2)
        self.assertGreaterEqual(self.db.size(), 2)

    def test_get_top_programs(self):
        for i in range(10):
            a = Artifact(
                id=f"p{i}",
                content=f"prog {i}",
                metrics={"combined_score": i / 10.0},
            )
            self.db.add(a)
        top = self.db.get_top_programs(3)
        self.assertEqual(len(top), 3)
        self.assertGreaterEqual(
            top[0].metrics["combined_score"], top[1].metrics["combined_score"]
        )

    def test_worse_program_does_not_replace_best(self):
        """Adding a worse program should not change the best."""
        worse = Artifact(
            id="worse",
            content="x = bad()",
            parent_id="seed",
            generation=1,
            metrics={"combined_score": 0.1},
        )
        self.db.add(worse)
        self.assertEqual(self.db.get_best().id, "seed")

    def test_get_returns_none_for_unknown_id(self):
        self.assertIsNone(self.db.get("nonexistent"))

    def test_get_returns_existing_artifact(self):
        a = self.db.get("seed")
        self.assertIsNotNone(a)
        self.assertEqual(a.content, "x = 1")

    def test_duplicate_id_overwrites(self):
        """Adding with the same ID replaces the stored artifact."""
        a = Artifact(id="seed", content="replaced", metrics={"combined_score": 0.9})
        self.db.add(a)
        self.assertEqual(self.db.get("seed").content, "replaced")

    def test_is_better_compares_fitness(self):
        p1 = Artifact(id="p1", content="a", metrics={"combined_score": 0.8})
        p2 = Artifact(id="p2", content="b", metrics={"combined_score": 0.3})
        self.assertTrue(self.db._is_better(p1, p2))
        self.assertFalse(self.db._is_better(p2, p1))

    def test_is_better_no_metrics_uses_timestamp(self):
        p1 = Artifact(id="p1", content="a", metrics={}, timestamp=100.0)
        p2 = Artifact(id="p2", content="b", metrics={}, timestamp=50.0)
        self.assertTrue(self.db._is_better(p1, p2))

    def test_is_better_one_has_metrics(self):
        p1 = Artifact(id="p1", content="a", metrics={"combined_score": 0.1})
        p2 = Artifact(id="p2", content="b", metrics={})
        self.assertTrue(self.db._is_better(p1, p2))
        self.assertFalse(self.db._is_better(p2, p1))


# ---------------------------------------------------------------------------
# Island evolution
# ---------------------------------------------------------------------------
class TestIslandEvolution(unittest.TestCase):
    def setUp(self):
        self.db = ArtifactDatabase(
            DatabaseConfig(
                num_islands=3, migration_interval=5, migration_rate=0.2
            )
        )
        for i in range(9):
            a = Artifact(
                id=f"p{i}",
                content=f"island prog {i}",
                metrics={"combined_score": (i + 1) / 10.0},
            )
            self.db.add(a)

    def test_island_rotation(self):
        island0 = self.db.current_island
        self.db.next_island()
        self.assertNotEqual(self.db.current_island, island0)

    def test_migration_check(self):
        self.assertFalse(self.db.should_migrate())
        for _ in range(5):
            self.db.increment_island_generation()
        self.assertTrue(self.db.should_migrate())

    def test_set_current_island(self):
        self.db.set_current_island(2)
        self.assertEqual(self.db.current_island, 2)

    def test_set_current_island_wraps(self):
        self.db.set_current_island(10)  # wraps mod 3
        self.assertEqual(self.db.current_island, 10 % 3)

    def test_next_island_wraps_around(self):
        self.db.set_current_island(2)  # last island
        next_idx = self.db.next_island()
        self.assertEqual(next_idx, 0)

    def test_island_stats(self):
        stats = self.db.get_island_stats()
        self.assertEqual(len(stats), 3)
        for s in stats:
            self.assertIn("population_size", s)
            self.assertIn("best_score", s)
            self.assertIn("average_score", s)

    def test_migrate_programs(self):
        """Migration should copy top programs to adjacent islands."""
        # Force enough generations
        for _ in range(5):
            self.db.increment_island_generation()
        initial_size = self.db.size()
        self.db.migrate_programs()
        # Migration should have added copies
        self.assertGreaterEqual(self.db.size(), initial_size)

    def test_increment_specific_island(self):
        self.db.increment_island_generation(island_idx=1)
        self.assertEqual(self.db.island_generations[1], 1)
        self.assertEqual(self.db.island_generations[0], 0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
class TestSampling(unittest.TestCase):
    def setUp(self):
        self.db = ArtifactDatabase(
            DatabaseConfig(num_islands=2, population_size=100)
        )
        for i in range(20):
            a = Artifact(
                id=f"s{i}",
                content=f"sample prog {i}" * (i + 1),
                metrics={"combined_score": i / 20.0},
            )
            self.db.add(a)

    def test_sample_returns_parent_and_inspirations(self):
        parent, inspirations = self.db.sample(num_inspirations=3)
        self.assertIsInstance(parent, Artifact)
        self.assertIsInstance(inspirations, list)
        self.assertLessEqual(len(inspirations), 3)

    def test_sample_from_populated_db(self):
        for _ in range(10):
            parent, _ = self.db.sample()
            self.assertIsNotNone(parent)
            self.assertIsNotNone(parent.content)

    def test_sample_from_island(self):
        parent, inspirations = self.db.sample_from_island(0, num_inspirations=2)
        self.assertIsNotNone(parent)
        self.assertIsInstance(inspirations, list)

    def test_sample_from_empty_island_falls_back(self):
        """Sampling from an empty island should fall back to global sample."""
        db = ArtifactDatabase(DatabaseConfig(num_islands=5))
        a = Artifact(id="only", content="x", metrics={"combined_score": 0.5})
        db.add(a)
        # Island 4 is almost certainly empty; should still succeed
        parent, _ = db.sample_from_island(4)
        self.assertIsNotNone(parent)

    def test_sample_single_program_db(self):
        """Sampling from a DB with one program should always return it."""
        db = ArtifactDatabase(DatabaseConfig(num_islands=1))
        a = Artifact(id="solo", content="x", metrics={"combined_score": 0.5})
        db.add(a)
        parent, inspirations = db.sample(num_inspirations=3)
        self.assertEqual(parent.id, "solo")
        # Inspirations may be empty since there's only one program
        self.assertIsInstance(inspirations, list)

    def test_sample_default_inspirations(self):
        """sample() with no num_inspirations should default to 5."""
        parent, inspirations = self.db.sample()
        self.assertIsNotNone(parent)
        self.assertLessEqual(len(inspirations), 5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence(unittest.TestCase):
    def test_save_and_load_json(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=2))
        for i in range(5):
            a = Artifact(
                id=f"persist-{i}",
                content=f"code {i}",
                metrics={"combined_score": i / 5.0},
            )
            db.add(a)
        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            db2 = ArtifactDatabase(DatabaseConfig(num_islands=2))
            db2.load(tmpdir)
            self.assertEqual(db2.size(), db.size())
            self.assertEqual(db2.get_best().id, db.get_best().id)

    def test_load_nonexistent_is_safe(self):
        db = ArtifactDatabase(DatabaseConfig())
        db.load("/nonexistent/path")
        self.assertEqual(db.size(), 0)

    def test_save_load_preserves_metadata(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=2))
        a = Artifact(
            id="meta",
            content="x",
            metrics={"combined_score": 0.5},
            metadata={"custom_key": "custom_val"},
        )
        db.add(a)
        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            db2 = ArtifactDatabase(DatabaseConfig(num_islands=2))
            db2.load(tmpdir)
            loaded = db2.get("meta")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.metadata.get("custom_key"), "custom_val")

    def test_save_load_preserves_island_state(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=3))
        for i in range(6):
            a = Artifact(
                id=f"il{i}",
                content=f"c{i}",
                metrics={"combined_score": i / 6.0},
            )
            db.add(a)
        db.set_current_island(2)
        db.increment_island_generation(island_idx=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            db2 = ArtifactDatabase(DatabaseConfig(num_islands=3))
            db2.load(tmpdir)
            self.assertEqual(db2.current_island, 2)
            self.assertEqual(db2.island_generations[1], 1)

    def test_save_creates_programs_dir(self):
        db = ArtifactDatabase(DatabaseConfig())
        a = Artifact(id="x", content="y", metrics={"combined_score": 0.1})
        db.add(a)
        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "programs")))
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "programs", "x.json"))
            )


# ---------------------------------------------------------------------------
# Artifact storage
# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    def test_store_and_retrieve_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(artifacts_base_path=tmpdir)
            db = ArtifactDatabase(config)
            a = Artifact(
                id="art-1", content="code", metrics={"combined_score": 0.5}
            )
            db.add(a)
            db.store_artifacts(
                "art-1", {"stderr": "warning: x unused", "profile": "fast"}
            )
            retrieved = db.get_artifacts("art-1")
            self.assertEqual(retrieved["stderr"], "warning: x unused")

    def test_missing_artifacts_returns_none(self):
        db = ArtifactDatabase(DatabaseConfig())
        self.assertIsNone(db.get_artifacts("nonexistent"))

    def test_store_artifacts_for_missing_program(self):
        """Storing artifacts for a program that doesn't exist should be safe."""
        db = ArtifactDatabase(DatabaseConfig())
        db.store_artifacts("ghost", {"key": "value"})
        self.assertIsNone(db.get_artifacts("ghost"))

    def test_store_empty_artifacts_is_noop(self):
        db = ArtifactDatabase(DatabaseConfig())
        a = Artifact(id="e", content="x", metrics={"combined_score": 0.1})
        db.add(a)
        db.store_artifacts("e", {})
        # Should not crash; artifacts should be empty/None
        result = db.get_artifacts("e")
        self.assertIsNotNone(result)  # get_artifacts returns {} for existing programs

    def test_large_artifact_goes_to_disk(self):
        """Artifacts bigger than threshold should be written to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                artifacts_base_path=tmpdir, artifact_size_threshold=10
            )
            db = ArtifactDatabase(config)
            a = Artifact(
                id="big", content="code", metrics={"combined_score": 0.5}
            )
            db.add(a)
            db.store_artifacts("big", {"large_data": "x" * 1000})
            retrieved = db.get_artifacts("big")
            self.assertEqual(retrieved["large_data"], "x" * 1000)


# ---------------------------------------------------------------------------
# Feature scaling & diversity
# ---------------------------------------------------------------------------
class TestFeatureScaling(unittest.TestCase):
    def test_update_and_scale_feature(self):
        db = ArtifactDatabase(DatabaseConfig())
        db._update_feature_stats("complexity", 10.0)
        db._update_feature_stats("complexity", 100.0)
        scaled = db._scale_feature_value("complexity", 55.0)
        self.assertAlmostEqual(scaled, 0.5, places=1)

    def test_scale_unknown_feature_clamps(self):
        db = ArtifactDatabase(DatabaseConfig())
        # No stats yet
        self.assertGreaterEqual(db._scale_feature_value("new_dim", 5.0), 0.0)
        self.assertLessEqual(db._scale_feature_value("new_dim", 5.0), 1.0)

    def test_scale_constant_value_returns_half(self):
        db = ArtifactDatabase(DatabaseConfig())
        db._update_feature_stats("x", 42.0)
        # min == max => 0.5
        self.assertAlmostEqual(db._scale_feature_value("x", 42.0), 0.5)

    def test_feature_coords_to_key(self):
        db = ArtifactDatabase(DatabaseConfig())
        key = db._feature_coords_to_key([3, 7])
        self.assertEqual(key, "3-7")

    def test_calculate_complexity_bin_in_range(self):
        db = ArtifactDatabase(DatabaseConfig(feature_bins=10))
        # Feed some values first to build stats
        db._update_feature_stats("complexity", 0.0)
        db._update_feature_stats("complexity", 1000.0)
        b = db._calculate_complexity_bin(500)
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, 10)

    def test_calculate_diversity_bin_in_range(self):
        db = ArtifactDatabase(DatabaseConfig(feature_bins=10))
        db._update_feature_stats("diversity", 0.0)
        db._update_feature_stats("diversity", 100.0)
        b = db._calculate_diversity_bin(50.0)
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, 10)


# ---------------------------------------------------------------------------
# Diversity caching
# ---------------------------------------------------------------------------
class TestDiversityCaching(unittest.TestCase):
    def test_get_cached_diversity_returns_float(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=1))
        for i in range(5):
            a = Artifact(
                id=f"d{i}",
                content=f"diversity test {i}" * (i + 1),
                metrics={"combined_score": 0.5},
            )
            db.add(a)
        p = db.get("d3")
        div = db._get_cached_diversity(p)
        self.assertIsInstance(div, float)

    def test_diversity_cache_is_populated(self):
        db = ArtifactDatabase(DatabaseConfig(num_islands=1))
        for i in range(3):
            a = Artifact(
                id=f"c{i}",
                content=f"cache test {i}" * (i + 1),
                metrics={"combined_score": 0.5},
            )
            db.add(a)
        p = db.get("c2")
        db._get_cached_diversity(p)
        code_hash = hash(p.content)
        self.assertIn(code_hash, db.diversity_cache)


# ---------------------------------------------------------------------------
# Population limit
# ---------------------------------------------------------------------------
class TestPopulationLimit(unittest.TestCase):
    def test_enforce_population_limit(self):
        db = ArtifactDatabase(DatabaseConfig(population_size=5, num_islands=1))
        for i in range(10):
            a = Artifact(
                id=f"pop{i}",
                content=f"p{i}",
                metrics={"combined_score": i / 10.0},
            )
            db.add(a)
        self.assertLessEqual(db.size(), 5)

    def test_best_program_survives_population_limit(self):
        db = ArtifactDatabase(DatabaseConfig(population_size=3, num_islands=1))
        best = Artifact(
            id="best", content="best", metrics={"combined_score": 1.0}
        )
        db.add(best)
        for i in range(10):
            a = Artifact(
                id=f"fill{i}",
                content=f"fill{i}",
                metrics={"combined_score": 0.01 * i},
            )
            db.add(a)
        self.assertIsNotNone(db.get("best"))
        self.assertEqual(db.get_best().id, "best")


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
class TestArchive(unittest.TestCase):
    def test_archive_fills_up_to_limit(self):
        db = ArtifactDatabase(
            DatabaseConfig(archive_size=5, num_islands=1, population_size=200)
        )
        for i in range(20):
            a = Artifact(
                id=f"arch{i}",
                content=f"arch{i}",
                metrics={"combined_score": i / 20.0},
            )
            db.add(a)
        self.assertLessEqual(len(db.archive), 5)

    def test_archive_replaces_worst_with_better(self):
        db = ArtifactDatabase(
            DatabaseConfig(archive_size=3, num_islands=1, population_size=200)
        )
        # Fill archive
        for i in range(3):
            a = Artifact(
                id=f"a{i}",
                content=f"a{i}",
                metrics={"combined_score": 0.1 * (i + 1)},
            )
            db.add(a)
        # Add a much better program
        better = Artifact(
            id="better_arch",
            content="better",
            metrics={"combined_score": 0.9},
        )
        db.add(better)
        self.assertIn("better_arch", db.archive)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    def test_add_artifact_with_no_metrics(self):
        db = ArtifactDatabase(DatabaseConfig())
        a = Artifact(id="nm", content="no metrics")
        db.add(a)
        self.assertEqual(db.size(), 1)

    def test_get_top_programs_empty_db(self):
        db = ArtifactDatabase(DatabaseConfig())
        top = db.get_top_programs(5)
        self.assertEqual(top, [])

    def test_get_best_empty_db(self):
        db = ArtifactDatabase(DatabaseConfig())
        self.assertIsNone(db.get_best())

    def test_get_top_programs_with_metric_filter(self):
        db = ArtifactDatabase(DatabaseConfig())
        for i in range(5):
            a = Artifact(
                id=f"m{i}",
                content=f"m{i}",
                metrics={"combined_score": i / 5.0, "speed": (5 - i) / 5.0},
            )
            db.add(a)
        top_speed = db.get_top_programs(2, metric="speed")
        self.assertEqual(len(top_speed), 2)
        self.assertGreaterEqual(
            top_speed[0].metrics["speed"], top_speed[1].metrics["speed"]
        )

    def test_child_inherits_parent_island(self):
        """A child should be placed in the same island as its parent."""
        db = ArtifactDatabase(DatabaseConfig(num_islands=3))
        parent = Artifact(
            id="parent",
            content="parent code",
            metrics={"combined_score": 0.5},
        )
        db.set_current_island(1)
        db.add(parent)
        parent_island = db.get("parent").metadata.get("island")
        child = Artifact(
            id="child",
            content="child code",
            parent_id="parent",
            metrics={"combined_score": 0.6},
        )
        db.add(child)
        child_island = db.get("child").metadata.get("island")
        self.assertEqual(parent_island, child_island)

    def test_novelty_check_passes_for_self(self):
        """Novelty check should pass when only artifact in island is itself."""
        db = ArtifactDatabase(DatabaseConfig())
        a = Artifact(id="n1", content="x = 1", metrics={"combined_score": 0.5})
        db.add(a)
        # _is_novel should return True since the only island member is itself
        self.assertTrue(db._is_novel("n1", 0))


# ---------------------------------------------------------------------------
# Stagnation-related database methods
# ---------------------------------------------------------------------------
class TestDatabaseStagnation(unittest.TestCase):
    """Tests for stagnation-related database methods."""

    def setUp(self):
        from claude_evolve.config import DatabaseConfig
        self.config = DatabaseConfig(in_memory=True, num_islands=2, population_size=100)
        self.db = ArtifactDatabase(self.config)

    def test_get_score_history_empty(self):
        """Empty database returns empty score history."""
        self.assertEqual(self.db.get_score_history(), [])

    def test_get_score_history_ordered(self):
        """Scores are returned in iteration order."""
        for i, score in enumerate([0.5, 0.7, 0.6, 0.8]):
            a = Artifact(id=f"a{i}", content=f"code{i}", metrics={"combined_score": score})
            self.db.add(a, iteration=i)
        history = self.db.get_score_history()
        self.assertEqual(history, [0.5, 0.7, 0.6, 0.8])

    def test_get_score_history_skips_non_numeric(self):
        """Non-numeric scores are skipped."""
        a1 = Artifact(id="a1", content="c1", metrics={"combined_score": 0.5})
        a2 = Artifact(id="a2", content="c2", metrics={"combined_score": "invalid"})
        a3 = Artifact(id="a3", content="c3", metrics={"other_metric": 0.8})
        self.db.add(a1, iteration=0)
        self.db.add(a2, iteration=1)
        self.db.add(a3, iteration=2)
        history = self.db.get_score_history()
        self.assertEqual(history, [0.5])

    def test_detect_stagnation_no_history(self):
        """detect_stagnation on empty db returns NONE."""
        report = self.db.detect_stagnation()
        from claude_evolve.core.stagnation import StagnationLevel
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_detect_stagnation_improving(self):
        """detect_stagnation with improving scores returns NONE."""
        for i in range(5):
            a = Artifact(id=f"a{i}", content=f"code{i}", metrics={"combined_score": 0.1 * (i + 1)})
            self.db.add(a, iteration=i)
        report = self.db.detect_stagnation()
        from claude_evolve.core.stagnation import StagnationLevel
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_detect_stagnation_plateau(self):
        """detect_stagnation with plateau returns appropriate level."""
        # Best at iteration 0, then 5 iterations of lower scores
        a0 = Artifact(id="a0", content="best", metrics={"combined_score": 0.9})
        self.db.add(a0, iteration=0)
        for i in range(1, 6):
            a = Artifact(id=f"a{i}", content=f"worse{i}", metrics={"combined_score": 0.5})
            self.db.add(a, iteration=i)
        report = self.db.detect_stagnation()
        from claude_evolve.core.stagnation import StagnationLevel
        self.assertEqual(report.level, StagnationLevel.MILD)


# ---------------------------------------------------------------------------
# Novelty (_is_novel) implementation
# ---------------------------------------------------------------------------
class TestIsNovel(unittest.TestCase):
    """Tests for the _is_novel implementation."""

    def setUp(self):
        from claude_evolve.config import DatabaseConfig
        self.config = DatabaseConfig(
            in_memory=True, num_islands=2, population_size=100,
            similarity_threshold=0.95,
        )
        self.db = ArtifactDatabase(self.config)

    def test_novel_when_island_empty(self):
        """Artifact is novel when island is empty."""
        a = Artifact(id="a1", content="code here")
        self.db.artifacts["a1"] = a
        self.assertTrue(self.db._is_novel("a1", 0))

    def test_novel_with_different_content(self):
        """Artifact is novel when content differs significantly."""
        a1 = Artifact(id="a1", content="def foo():\n    return 1\n")
        a2 = Artifact(id="a2", content="def bar():\n    x = 42\n    return x * 2\n")
        self.db.artifacts["a1"] = a1
        self.db.artifacts["a2"] = a2
        self.db.islands[0].add("a1")
        self.assertTrue(self.db._is_novel("a2", 0))

    def test_not_novel_identical_content(self):
        """Identical content is rejected."""
        content = "def foo():\n    return 1\n"
        a1 = Artifact(id="a1", content=content)
        a2 = Artifact(id="a2", content=content)
        self.db.artifacts["a1"] = a1
        self.db.artifacts["a2"] = a2
        self.db.islands[0].add("a1")
        self.assertFalse(self.db._is_novel("a2", 0))

    def test_not_novel_very_similar(self):
        """Very similar content (above threshold) is rejected."""
        base = "\n".join([f"line_{i} = {i}" for i in range(100)])
        # Change just one line
        similar = "\n".join([f"line_{i} = {i}" if i != 50 else "line_50 = 999" for i in range(100)])
        a1 = Artifact(id="a1", content=base)
        a2 = Artifact(id="a2", content=similar)
        self.db.artifacts["a1"] = a1
        self.db.artifacts["a2"] = a2
        self.db.islands[0].add("a1")
        self.assertFalse(self.db._is_novel("a2", 0))

    def test_novel_unknown_artifact_id(self):
        """Unknown artifact ID returns True."""
        self.assertTrue(self.db._is_novel("nonexistent", 0))


# ---------------------------------------------------------------------------
# Line similarity helper
# ---------------------------------------------------------------------------
class TestLineSimilarity(unittest.TestCase):
    """Tests for _line_similarity helper."""

    def test_identical(self):
        self.assertAlmostEqual(ArtifactDatabase._line_similarity("a\nb\nc", "a\nb\nc"), 1.0)

    def test_completely_different(self):
        self.assertAlmostEqual(ArtifactDatabase._line_similarity("a\nb\nc", "x\ny\nz"), 0.0)

    def test_partial_overlap(self):
        sim = ArtifactDatabase._line_similarity("a\nb\nc\nd", "a\nb\nx\ny")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_empty_strings(self):
        self.assertAlmostEqual(ArtifactDatabase._line_similarity("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(ArtifactDatabase._line_similarity("a\nb", ""), 0.0)

    def test_whitespace_normalization(self):
        self.assertAlmostEqual(ArtifactDatabase._line_similarity("  a  \n  b  ", "a\nb"), 1.0)


if __name__ == "__main__":
    unittest.main()
