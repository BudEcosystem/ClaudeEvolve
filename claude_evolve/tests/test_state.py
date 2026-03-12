"""
Tests for the state management layer.

Covers LoopState (evolve.local.md file I/O), StateManager (evolve-state/
directory management), and CheckpointManager (save/restore snapshots).
"""

import json
import os
import tempfile
import time
import unittest

from claude_evolve.config import Config, DatabaseConfig
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.state.checkpoint import CheckpointManager
from claude_evolve.state.loop_state import LoopState
from claude_evolve.state.manager import StateManager


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------
class TestLoopStateCreate(unittest.TestCase):
    """Test LoopState.create() class method and file writing."""

    def test_create_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            ls = LoopState.create(
                path=state_path,
                session_id="test-session-123",
                max_iterations=30,
                target_score=0.95,
                completion_promise="EVOLUTION_TARGET_REACHED",
                prompt="Evolve circle packing algorithm",
            )
            self.assertTrue(os.path.exists(state_path))
            self.assertEqual(ls.iteration, 1)
            self.assertEqual(ls.max_iterations, 30)

    def test_create_sets_default_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            ls = LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=50,
                prompt="test",
            )
            self.assertTrue(ls.active)
            self.assertEqual(ls.iteration, 1)
            self.assertAlmostEqual(ls.best_score, 0.0)
            self.assertEqual(ls.mode, "script")

    def test_create_with_optional_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            ls = LoopState.create(
                path=state_path,
                session_id="s2",
                max_iterations=100,
                target_score=0.99,
                completion_promise="DONE",
                prompt="optimize",
                state_dir=".claude/custom-state",
                evaluator_path="eval.py",
                artifact_path="artifact.py",
                mode="hybrid",
            )
            self.assertEqual(ls.state_dir, ".claude/custom-state")
            self.assertEqual(ls.evaluator_path, "eval.py")
            self.assertEqual(ls.artifact_path, "artifact.py")
            self.assertEqual(ls.mode, "hybrid")

    def test_create_writes_valid_file(self):
        """The file created should be re-readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="roundtrip",
                max_iterations=10,
                prompt="roundtrip test",
            )
            ls = LoopState.read(state_path)
            self.assertEqual(ls.session_id, "roundtrip")
            self.assertEqual(ls.prompt, "roundtrip test")


class TestLoopStateRead(unittest.TestCase):
    """Test LoopState.read() class method."""

    def test_read_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                target_score=0.9,
                completion_promise="DONE",
                prompt="test prompt",
            )
            ls = LoopState.read(state_path)
            self.assertEqual(ls.iteration, 1)
            self.assertEqual(ls.max_iterations, 10)
            self.assertAlmostEqual(ls.target_score, 0.9)
            self.assertEqual(ls.prompt, "test prompt")

    def test_read_preserves_all_frontmatter_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="full-test",
                max_iterations=25,
                target_score=0.88,
                completion_promise="ALL_DONE",
                prompt="full test",
                evaluator_path="evaluator.py",
                artifact_path="program.py",
                mode="critic",
            )
            ls = LoopState.read(state_path)
            self.assertEqual(ls.session_id, "full-test")
            self.assertEqual(ls.completion_promise, "ALL_DONE")
            self.assertEqual(ls.evaluator_path, "evaluator.py")
            self.assertEqual(ls.artifact_path, "program.py")
            self.assertEqual(ls.mode, "critic")

    def test_read_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            LoopState.read("/nonexistent/path/evolve.local.md")


class TestLoopStateMutations(unittest.TestCase):
    """Test LoopState mutation methods."""

    def test_increment_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="test",
            )
            ls = LoopState.read(state_path)
            ls.increment_iteration()
            ls.write(state_path)
            ls2 = LoopState.read(state_path)
            self.assertEqual(ls2.iteration, 2)

    def test_increment_iteration_multiple_times(self):
        ls = LoopState(
            iteration=1,
            max_iterations=50,
            prompt="x",
            session_id="s",
        )
        for _ in range(10):
            ls.increment_iteration()
        self.assertEqual(ls.iteration, 11)

    def test_update_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="original",
            )
            ls = LoopState.read(state_path)
            ls.update_prompt("new iteration prompt")
            ls.write(state_path)
            ls2 = LoopState.read(state_path)
            self.assertEqual(ls2.prompt, "new iteration prompt")

    def test_update_best_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="test",
            )
            ls = LoopState.read(state_path)
            ls.update_best_score(0.75)
            ls.write(state_path)
            ls2 = LoopState.read(state_path)
            self.assertAlmostEqual(ls2.best_score, 0.75)

    def test_update_best_score_does_not_decrease(self):
        """update_best_score should only increase, never decrease."""
        ls = LoopState(
            iteration=5,
            max_iterations=50,
            prompt="x",
            session_id="s",
            best_score=0.8,
        )
        ls.update_best_score(0.5)
        self.assertAlmostEqual(ls.best_score, 0.8)
        ls.update_best_score(0.95)
        self.assertAlmostEqual(ls.best_score, 0.95)

    def test_deactivate(self):
        ls = LoopState(
            iteration=1,
            max_iterations=10,
            prompt="x",
            session_id="s",
            active=True,
        )
        ls.active = False
        self.assertFalse(ls.active)


class TestLoopStateCompletion(unittest.TestCase):
    """Test completion/termination checks."""

    def test_is_max_iterations_reached_false(self):
        ls = LoopState(iteration=5, max_iterations=10, prompt="x", session_id="s")
        self.assertFalse(ls.is_max_iterations_reached())

    def test_is_max_iterations_reached_true(self):
        ls = LoopState(iteration=10, max_iterations=10, prompt="x", session_id="s")
        self.assertTrue(ls.is_max_iterations_reached())

    def test_is_max_iterations_reached_exceeded(self):
        ls = LoopState(iteration=15, max_iterations=10, prompt="x", session_id="s")
        self.assertTrue(ls.is_max_iterations_reached())

    def test_is_target_reached_false(self):
        ls = LoopState(
            iteration=3, max_iterations=50, prompt="x", session_id="s",
            target_score=0.9,
        )
        self.assertFalse(ls.is_target_reached(0.5))

    def test_is_target_reached_true(self):
        ls = LoopState(
            iteration=3, max_iterations=50, prompt="x", session_id="s",
            target_score=0.9,
        )
        self.assertTrue(ls.is_target_reached(0.95))

    def test_is_target_reached_exact(self):
        ls = LoopState(
            iteration=3, max_iterations=50, prompt="x", session_id="s",
            target_score=0.9,
        )
        self.assertTrue(ls.is_target_reached(0.9))

    def test_is_target_reached_no_target(self):
        """When no target_score is set, is_target_reached always returns False."""
        ls = LoopState(
            iteration=3, max_iterations=50, prompt="x", session_id="s",
        )
        self.assertFalse(ls.is_target_reached(1.0))


class TestLoopStateFileFormat(unittest.TestCase):
    """Test that the file format is correct markdown with YAML frontmatter."""

    def test_file_starts_with_frontmatter_delimiter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="hello world",
            )
            with open(state_path, "r") as f:
                content = f.read()
            self.assertTrue(content.startswith("---\n"))
            # Should have closing frontmatter delimiter
            parts = content.split("---\n")
            self.assertGreaterEqual(len(parts), 3)

    def test_prompt_is_in_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt="My special prompt content",
            )
            with open(state_path, "r") as f:
                content = f.read()
            self.assertIn("My special prompt content", content)

    def test_multiline_prompt_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "evolve.local.md")
            prompt = "Line 1\nLine 2\nLine 3"
            LoopState.create(
                path=state_path,
                session_id="s1",
                max_iterations=10,
                prompt=prompt,
            )
            ls = LoopState.read(state_path)
            self.assertEqual(ls.prompt, prompt)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------
class TestStateManagerInitialize(unittest.TestCase):
    """Test StateManager.initialize() method."""

    def test_initialize_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="print('hello')",
                artifact_type="python",
            )
            self.assertTrue(os.path.exists(os.path.join(state_dir, "database.json")))
            self.assertTrue(os.path.exists(os.path.join(state_dir, "config.json")))

    def test_initialize_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "deep", "nested", "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            self.assertTrue(os.path.isdir(state_dir))

    def test_initialize_seeds_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            self.assertEqual(sm.database.size(), 1)

    def test_initialize_stores_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            cfg = Config(max_iterations=42)
            sm.initialize(
                config=cfg,
                initial_content="x = 1",
                artifact_type="python",
            )
            self.assertEqual(sm.config.max_iterations, 42)


class TestStateManagerLoadSave(unittest.TestCase):
    """Test StateManager load/save cycle."""

    def test_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="print('hello')",
                artifact_type="python",
            )
            sm2 = StateManager(state_dir)
            sm2.load()
            self.assertIsNotNone(sm2.database)
            self.assertEqual(sm2.database.size(), 1)

    def test_save_updates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            # Add another artifact, then save
            a = Artifact(
                id="new-artifact",
                content="x = 2",
                artifact_type="python",
                metrics={"combined_score": 0.5},
            )
            sm.database.add(a)
            sm.save()
            # Reload and verify
            sm2 = StateManager(state_dir)
            sm2.load()
            self.assertEqual(sm2.database.size(), 2)

    def test_load_nonexistent_raises(self):
        sm = StateManager("/nonexistent/path/evolve-state")
        with self.assertRaises(FileNotFoundError):
            sm.load()

    def test_get_database_accessor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x=1",
                artifact_type="python",
            )
            db = sm.get_database()
            self.assertIsInstance(db, ArtifactDatabase)
            self.assertEqual(db.size(), 1)

    def test_get_config_accessor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            cfg = Config(max_iterations=77)
            sm.initialize(
                config=cfg,
                initial_content="x=1",
                artifact_type="python",
            )
            loaded_cfg = sm.get_config()
            self.assertIsInstance(loaded_cfg, Config)
            self.assertEqual(loaded_cfg.max_iterations, 77)


class TestStateManagerIterationContext(unittest.TestCase):
    """Test iteration context read/write."""

    def test_write_and_read_iteration_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x=1",
                artifact_type="python",
            )
            sm.write_iteration_context("# Iteration 1 context here")
            content = sm.read_iteration_context()
            self.assertIn("Iteration 1", content)

    def test_read_iteration_context_when_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x=1",
                artifact_type="python",
            )
            content = sm.read_iteration_context()
            self.assertEqual(content, "")

    def test_overwrite_iteration_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x=1",
                artifact_type="python",
            )
            sm.write_iteration_context("first context")
            sm.write_iteration_context("second context")
            content = sm.read_iteration_context()
            self.assertEqual(content, "second context")


class TestStateManagerBestArtifact(unittest.TestCase):
    """Test best artifact writing."""

    def test_write_best_artifact_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x=1",
                artifact_type="python",
            )
            sm.write_best_artifact("optimized_x = 2", "python")
            best_path = os.path.join(state_dir, "best_artifact.py")
            self.assertTrue(os.path.exists(best_path))
            with open(best_path, "r") as f:
                self.assertEqual(f.read(), "optimized_x = 2")

    def test_write_best_artifact_javascript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="const x = 1;",
                artifact_type="javascript",
            )
            sm.write_best_artifact("const x = 2;", "javascript")
            best_path = os.path.join(state_dir, "best_artifact.js")
            self.assertTrue(os.path.exists(best_path))

    def test_write_best_artifact_generic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="text content",
                artifact_type="text",
            )
            sm.write_best_artifact("optimized text", "text")
            best_path = os.path.join(state_dir, "best_artifact.txt")
            self.assertTrue(os.path.exists(best_path))


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------
class TestCheckpointManagerSave(unittest.TestCase):
    """Test CheckpointManager.save()."""

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(checkpoint_interval=5),
                initial_content="x = 1",
                artifact_type="python",
            )
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=5)
            checkpoint_dir = os.path.join(state_dir, "checkpoints", "iter_005")
            self.assertTrue(os.path.isdir(checkpoint_dir))

    def test_save_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=5)
            cp.save(sm.database, iteration=10)
            cp.save(sm.database, iteration=15)
            checkpoints = cp.list_checkpoints()
            self.assertEqual(len(checkpoints), 3)


class TestCheckpointManagerList(unittest.TestCase):
    """Test CheckpointManager.list_checkpoints()."""

    def test_list_checkpoints_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            os.makedirs(state_dir, exist_ok=True)
            cp = CheckpointManager(state_dir)
            checkpoints = cp.list_checkpoints()
            self.assertEqual(len(checkpoints), 0)

    def test_list_checkpoints_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=5)
            checkpoints = cp.list_checkpoints()
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]["iteration"], 5)
            self.assertIn("timestamp", checkpoints[0])
            self.assertIn("num_artifacts", checkpoints[0])

    def test_list_checkpoints_sorted_by_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=15)
            cp.save(sm.database, iteration=5)
            cp.save(sm.database, iteration=10)
            checkpoints = cp.list_checkpoints()
            iterations = [c["iteration"] for c in checkpoints]
            self.assertEqual(iterations, [5, 10, 15])


class TestCheckpointManagerRestore(unittest.TestCase):
    """Test CheckpointManager.restore()."""

    def test_restore_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=10)
            db2 = cp.restore(iteration=10, config=sm.config)
            self.assertEqual(db2.size(), sm.database.size())

    def test_restore_preserves_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            # Add a better artifact
            a = Artifact(
                id="best-one",
                content="optimized = True",
                artifact_type="python",
                metrics={"combined_score": 0.9},
            )
            sm.database.add(a)
            cp = CheckpointManager(state_dir)
            cp.save(sm.database, iteration=5)
            db2 = cp.restore(iteration=5, config=sm.config)
            self.assertIsNotNone(db2.get_best())
            self.assertEqual(db2.get_best().id, "best-one")

    def test_restore_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            os.makedirs(state_dir, exist_ok=True)
            cp = CheckpointManager(state_dir)
            with self.assertRaises(FileNotFoundError):
                cp.restore(iteration=999, config=Config())

    def test_restore_different_iterations(self):
        """Restoring from different iterations yields correct state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "evolve-state")
            sm = StateManager(state_dir)
            sm.initialize(
                config=Config(),
                initial_content="x = 1",
                artifact_type="python",
            )
            cp = CheckpointManager(state_dir)

            # Save at iteration 5 with 1 artifact
            cp.save(sm.database, iteration=5)

            # Add more artifacts and save at iteration 10
            for i in range(3):
                a = Artifact(
                    id=f"extra-{i}",
                    content=f"extra {i}",
                    artifact_type="python",
                    metrics={"combined_score": 0.3 + i * 0.1},
                )
                sm.database.add(a)
            cp.save(sm.database, iteration=10)

            # Restore iteration 5 - should have 1 artifact
            db5 = cp.restore(iteration=5, config=sm.config)
            self.assertEqual(db5.size(), 1)

            # Restore iteration 10 - should have 4 artifacts
            db10 = cp.restore(iteration=10, config=sm.config)
            self.assertEqual(db10.size(), 4)


if __name__ == "__main__":
    unittest.main()
