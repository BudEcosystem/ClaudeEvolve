"""Tests for IterationOrchestrator -- the final integration wiring all modules."""

import json
import os
import tempfile

from claude_evolve.config import Config
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.orchestrator import IterationOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _art(content: str = "def solve(): pass", score: float = 0.5) -> Artifact:
    """Create a test artifact with combined_score set."""
    a = Artifact(
        id=Artifact.generate_id(),
        content=content,
        artifact_type="python",
        metrics={"combined_score": score},
    )
    return a


def _make_orch(state_dir: str, config: Config | None = None) -> IterationOrchestrator:
    """Create an orchestrator with default config and a seed artifact."""
    config = config or Config()
    orch = IterationOrchestrator(state_dir=state_dir, config=config)
    seed = _art("def solve(): pass", score=0.5)
    orch.db.add(seed)
    return orch


# ---------------------------------------------------------------------------
# prepare_next_iteration
# ---------------------------------------------------------------------------

def test_prepare_next_iteration_returns_complete_context():
    """Orchestrator should produce a complete context dict for the builder."""
    with tempfile.TemporaryDirectory() as d:
        orch = _make_orch(d)
        ctx = orch.prepare_next_iteration(iteration=1)

        assert 'parent' in ctx
        assert 'strategy_name' in ctx
        assert 'exploration_intensity' in ctx
        assert 'stagnation_level' in ctx
        assert 'scratchpad_text' in ctx
        assert 'failures_text' in ctx
        assert 'reflection_text' in ctx
        assert 'meta_guidance' in ctx
        # Exploration intensity should be a float
        assert isinstance(ctx['exploration_intensity'], float)


def test_prepare_next_iteration_empty_database():
    """Orchestrator should handle an empty database gracefully."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)
        # No artifacts added -- db is empty
        ctx = orch.prepare_next_iteration(iteration=1)

        assert ctx['parent'] is None
        assert ctx['comparison'] is None
        assert 'strategy_name' in ctx
        assert isinstance(ctx['exploration_intensity'], float)


# ---------------------------------------------------------------------------
# process_submission
# ---------------------------------------------------------------------------

def test_process_submission_updates_signal():
    """Orchestrator should update G_t signal after submission."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)

        # Setup manifest as if prepare_next_iteration ran
        orch._save_manifest({
            "parent_score": 0.5,
            "selected_strategy_id": "default-incremental",
            "parent_island_id": 0,
            "iteration": 1,
            "run_id": "test",
        })

        result = orch.process_submission("def solve(): return 1", {"combined_score": 0.8})

        assert orch.signal.g_t > 0.0  # Improvement recorded
        assert result['improved'] is True
        assert result['score'] == 0.8
        assert result['parent_score'] == 0.5


def test_process_submission_records_ucb_outcome():
    """UCB arm visit count should increase after submission."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)

        # The strategy_id must exist as a UCB arm
        strategy_id = list(orch.ucb.arms.keys())[0]
        initial_visits = orch.ucb.arms[strategy_id].visit_count

        orch._save_manifest({
            "parent_score": 0.5,
            "selected_strategy_id": strategy_id,
            "parent_island_id": 0,
            "iteration": 1,
            "run_id": "test",
        })

        orch.process_submission("def solve(): return 1", {"combined_score": 0.7})

        assert orch.ucb.arms[strategy_id].visit_count == initial_visits + 1


def test_process_submission_captures_failure():
    """Failures should be recorded to recent_failures.json."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)

        orch._save_manifest({
            "parent_score": 0.8,
            "selected_strategy_id": "default-incremental",
            "parent_island_id": 0,
            "iteration": 2,
            "run_id": "test",
        })

        # Submit with a significant regression (0.3 < 0.8 * 0.95 = 0.76)
        orch.process_submission("def solve(): return 0", {"combined_score": 0.3})

        failures_path = os.path.join(d, 'recent_failures.json')
        assert os.path.exists(failures_path)
        with open(failures_path, encoding='utf-8') as f:
            failures = json.load(f)
        assert len(failures) >= 1
        assert failures[-1]['approach'] == 'default-incremental'
        assert failures[-1]['score'] == 0.3


def test_process_submission_captures_failure_on_error():
    """An error metric should also trigger failure capture."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)

        orch._save_manifest({
            "parent_score": 0.5,
            "selected_strategy_id": "default-creative",
            "parent_island_id": 0,
            "iteration": 3,
            "run_id": "test",
        })

        orch.process_submission("bad code", {"combined_score": 0.6, "error": "SyntaxError"})

        failures_path = os.path.join(d, 'recent_failures.json')
        assert os.path.exists(failures_path)
        with open(failures_path, encoding='utf-8') as f:
            failures = json.load(f)
        assert any(f.get('error') == 'SyntaxError' for f in failures)


def test_process_submission_increments_offspring_count():
    """Parent artifact's offspring_count should increase after submission."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)

        # Add a parent artifact to the db
        parent = _art("def solve(): return 42", score=0.5)
        orch.db.add(parent)
        assert parent.offspring_count == 0

        orch._save_manifest({
            "parent_score": 0.5,
            "selected_strategy_id": "default-incremental",
            "parent_artifact_id": parent.id,
            "parent_island_id": 0,
            "iteration": 1,
            "run_id": "test",
        })

        orch.process_submission("def solve(): return 99", {"combined_score": 0.7})

        assert orch.db.artifacts[parent.id].offspring_count == 1


# ---------------------------------------------------------------------------
# Manifest roundtrip
# ---------------------------------------------------------------------------

def test_iteration_manifest_roundtrip():
    """save_manifest / load_manifest should survive a round-trip."""
    with tempfile.TemporaryDirectory() as d:
        config = Config()
        orch = IterationOrchestrator(state_dir=d, config=config)

        manifest_data = {
            "iteration": 7,
            "run_id": "roundtrip-test",
            "selected_strategy_id": "default-hybrid",
            "parent_artifact_id": "abc-123",
            "parent_score": 0.42,
            "parent_island_id": 2,
        }
        orch._save_manifest(manifest_data)
        loaded = orch._load_manifest()

        assert loaded == manifest_data


# ---------------------------------------------------------------------------
# Meta-guidance trigger
# ---------------------------------------------------------------------------

def test_meta_guidance_triggers_on_total_stagnation():
    """When G_t is 0 everywhere, meta-guidance text should appear."""
    with tempfile.TemporaryDirectory() as d:
        orch = _make_orch(d)

        # Force all island g_t values below meta_threshold
        orch.signal.g_t = 0.0
        orch.signal.per_island_g_t = {"0": 0.01, "1": 0.02}
        orch.signal.meta_threshold = 0.12

        ctx = orch.prepare_next_iteration(iteration=1)

        assert 'BREAKTHROUGH REQUIRED' in ctx['meta_guidance']


def test_meta_guidance_absent_when_improving():
    """When improvement is active, no meta-guidance should appear."""
    with tempfile.TemporaryDirectory() as d:
        orch = _make_orch(d)

        # Simulate strong improvement on island 0
        orch.signal.g_t = 0.5
        orch.signal.per_island_g_t = {"0": 0.5}
        orch.signal.meta_threshold = 0.12

        ctx = orch.prepare_next_iteration(iteration=1)

        assert ctx['meta_guidance'] == ""
