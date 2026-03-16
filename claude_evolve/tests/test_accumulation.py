"""
Tests for the multi-iteration accumulation, problem decomposition,
and evaluation caching features.

Covers:
1. New strategies exist in DEFAULT_STRATEGIES
2. Accumulation strategy has correct fields
3. Decomposition strategy has correct fields
4. Decomposition template is registered
5. Eval cache hint fragment exists
"""

import json
import os
import tempfile
import shutil

import pytest
from click.testing import CliRunner

from claude_evolve.core.strategy import DEFAULT_STRATEGIES, Strategy
from claude_evolve.prompt.templates import (
    DECOMPOSITION_USER_TEMPLATE,
    TemplateManager,
    _INLINE_DEFAULTS,
    _INLINE_FRAGMENTS,
)
from claude_evolve.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_strategy_by_id(strategy_id: str) -> Strategy:
    """Find a strategy in DEFAULT_STRATEGIES by id."""
    for s in DEFAULT_STRATEGIES:
        if s.id == strategy_id:
            return s
    return None


# ---------------------------------------------------------------------------
# 1. New strategies exist in DEFAULT_STRATEGIES
# ---------------------------------------------------------------------------

class TestNewStrategiesExist:
    """Verify the new strategies are present in DEFAULT_STRATEGIES."""

    def test_accumulate_strategy_present(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert "default-accumulate" in ids

    def test_decompose_strategy_present(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert "default-decompose" in ids

    def test_total_default_strategies_count(self):
        # Original 5 + accumulate + decompose = 7
        assert len(DEFAULT_STRATEGIES) == 7

    def test_all_ids_still_unique(self):
        ids = [s.id for s in DEFAULT_STRATEGIES]
        assert len(ids) == len(set(ids))

    def test_all_names_still_unique(self):
        names = [s.name for s in DEFAULT_STRATEGIES]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# 2. Accumulation strategy has correct fields
# ---------------------------------------------------------------------------

class TestAccumulationStrategy:
    """Verify the accumulation strategy fields."""

    def test_accumulation_strategy_id(self):
        s = _get_strategy_by_id("default-accumulate")
        assert s is not None
        assert s.id == "default-accumulate"

    def test_accumulation_strategy_name(self):
        s = _get_strategy_by_id("default-accumulate")
        assert s.name == "Multi-Iteration Accumulation"

    def test_accumulation_strategy_description_contains_warm_cache(self):
        s = _get_strategy_by_id("default-accumulate")
        assert "warm cache" in s.description.lower()

    def test_accumulation_strategy_description_contains_warm_start(self):
        s = _get_strategy_by_id("default-accumulate")
        assert "warm_cache" in s.description

    def test_accumulation_strategy_generation_approach(self):
        s = _get_strategy_by_id("default-accumulate")
        assert s.generation_approach == "diff"

    def test_accumulation_strategy_template_key(self):
        s = _get_strategy_by_id("default-accumulate")
        assert s.template_key == "diff_user"

    def test_accumulation_strategy_exploration_weight(self):
        s = _get_strategy_by_id("default-accumulate")
        assert s.exploration_weight == pytest.approx(0.3)

    def test_accumulation_strategy_research_focus(self):
        s = _get_strategy_by_id("default-accumulate")
        assert "incremental optimization" in s.research_focus
        assert "warm-start" in s.research_focus

    def test_accumulation_strategy_serializable(self):
        s = _get_strategy_by_id("default-accumulate")
        d = s.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        restored = Strategy.from_dict(json.loads(serialized))
        assert restored.id == s.id
        assert restored.name == s.name


# ---------------------------------------------------------------------------
# 3. Decomposition strategy has correct fields
# ---------------------------------------------------------------------------

class TestDecompositionStrategy:
    """Verify the decomposition strategy fields."""

    def test_decomposition_strategy_id(self):
        s = _get_strategy_by_id("default-decompose")
        assert s is not None
        assert s.id == "default-decompose"

    def test_decomposition_strategy_name(self):
        s = _get_strategy_by_id("default-decompose")
        assert s.name == "Problem Decomposition"

    def test_decomposition_strategy_description_contains_sub_problems(self):
        s = _get_strategy_by_id("default-decompose")
        assert "sub-problem" in s.description.lower()

    def test_decomposition_strategy_generation_approach(self):
        s = _get_strategy_by_id("default-decompose")
        assert s.generation_approach == "full_rewrite"

    def test_decomposition_strategy_template_key(self):
        s = _get_strategy_by_id("default-decompose")
        assert s.template_key == "decomposition_user"

    def test_decomposition_strategy_exploration_weight(self):
        s = _get_strategy_by_id("default-decompose")
        assert s.exploration_weight == pytest.approx(0.6)

    def test_decomposition_strategy_research_focus(self):
        s = _get_strategy_by_id("default-decompose")
        assert "problem structure" in s.research_focus
        assert "divide-and-conquer" in s.research_focus

    def test_decomposition_strategy_serializable(self):
        s = _get_strategy_by_id("default-decompose")
        d = s.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        restored = Strategy.from_dict(json.loads(serialized))
        assert restored.id == s.id
        assert restored.name == s.name


# ---------------------------------------------------------------------------
# 4. Decomposition template is registered
# ---------------------------------------------------------------------------

class TestDecompositionTemplate:
    """Verify DECOMPOSITION_USER_TEMPLATE is defined and registered."""

    def test_decomposition_template_exists_as_module_constant(self):
        assert DECOMPOSITION_USER_TEMPLATE is not None
        assert isinstance(DECOMPOSITION_USER_TEMPLATE, str)
        assert len(DECOMPOSITION_USER_TEMPLATE) > 100

    def test_decomposition_template_in_inline_defaults(self):
        assert "decomposition_user" in _INLINE_DEFAULTS

    def test_decomposition_template_matches_module_constant(self):
        assert _INLINE_DEFAULTS["decomposition_user"] == DECOMPOSITION_USER_TEMPLATE

    def test_decomposition_template_contains_required_placeholders(self):
        required = [
            "{fitness_score}",
            "{feature_coords}",
            "{improvement_areas}",
            "{artifacts}",
            "{evolution_history}",
            "{language}",
            "{current_program}",
            "{feature_dimensions}",
        ]
        for placeholder in required:
            assert placeholder in DECOMPOSITION_USER_TEMPLATE, (
                f"Missing placeholder {placeholder} in DECOMPOSITION_USER_TEMPLATE"
            )

    def test_decomposition_template_contains_decomposition_steps(self):
        assert "independent" in DECOMPOSITION_USER_TEMPLATE.lower()
        assert "component" in DECOMPOSITION_USER_TEMPLATE.lower()

    def test_template_manager_has_decomposition_template(self):
        tm = TemplateManager()
        assert tm.has_template("decomposition_user")

    def test_template_manager_returns_decomposition_template(self):
        tm = TemplateManager()
        template = tm.get_template("decomposition_user")
        assert template == DECOMPOSITION_USER_TEMPLATE


# ---------------------------------------------------------------------------
# 5. Eval cache hint fragment exists
# ---------------------------------------------------------------------------

class TestEvalCacheHintFragment:
    """Verify the eval_cache_hint fragment is registered."""

    def test_eval_cache_hint_in_inline_fragments(self):
        assert "eval_cache_hint" in _INLINE_FRAGMENTS

    def test_eval_cache_hint_contains_optimization_keyword(self):
        hint = _INLINE_FRAGMENTS["eval_cache_hint"]
        assert "OPTIMIZATION" in hint

    def test_eval_cache_hint_mentions_warm_cache(self):
        hint = _INLINE_FRAGMENTS["eval_cache_hint"]
        assert "warm cache" in hint.lower()

    def test_eval_cache_hint_mentions_deterministic(self):
        hint = _INLINE_FRAGMENTS["eval_cache_hint"]
        assert "deterministic" in hint.lower()

    def test_template_manager_has_eval_cache_hint_fragment(self):
        tm = TemplateManager()
        assert tm.has_fragment("eval_cache_hint")

    def test_template_manager_returns_eval_cache_hint(self):
        tm = TemplateManager()
        result = tm.get_fragment("eval_cache_hint")
        assert "OPTIMIZATION" in result


# ---------------------------------------------------------------------------
# 6. CLI cache-eval command
# ---------------------------------------------------------------------------

class TestCacheEvalCli:
    """Test the cache-eval CLI command."""

    def setup_method(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_cache_eval_creates_cache_file(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "cache-eval", "--state-dir", state_dir,
            "--n", "10", "--result", '{"score": 0.95}',
        ])
        assert result.exit_code == 0, result.output
        cache_path = os.path.join(state_dir, "eval_cache.json")
        assert os.path.exists(cache_path)

    def test_cache_eval_stores_correct_data(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "cache-eval", "--state-dir", state_dir,
            "--n", "42", "--result", '{"verified": true, "clique_free": true}',
        ])
        assert result.exit_code == 0, result.output
        cache_path = os.path.join(state_dir, "eval_cache.json")
        with open(cache_path, "r") as f:
            cache = json.load(f)
        assert "42" in cache
        assert cache["42"]["verified"] is True

    def test_cache_eval_outputs_json_status(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "cache-eval", "--state-dir", state_dir,
            "--n", "5", "--result", '{"score": 1.0}',
        ])
        assert result.exit_code == 0, result.output
        output = json.loads(result.output.strip())
        assert output["status"] == "cached"
        assert output["n"] == 5
        assert output["cache_size"] == 1

    def test_cache_eval_accumulates_multiple_entries(self):
        state_dir = os.path.join(self.tmpdir, "state")
        for n in [10, 20, 30]:
            result = self.runner.invoke(main, [
                "cache-eval", "--state-dir", state_dir,
                "--n", str(n), "--result", json.dumps({"n": n, "ok": True}),
            ])
            assert result.exit_code == 0, result.output

        cache_path = os.path.join(state_dir, "eval_cache.json")
        with open(cache_path, "r") as f:
            cache = json.load(f)
        assert len(cache) == 3
        assert "10" in cache
        assert "20" in cache
        assert "30" in cache

    def test_cache_eval_invalid_json_fails(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "cache-eval", "--state-dir", state_dir,
            "--n", "5", "--result", "not-json",
        ])
        assert result.exit_code != 0

    def test_cache_eval_overwrites_existing_n(self):
        state_dir = os.path.join(self.tmpdir, "state")
        # First write
        self.runner.invoke(main, [
            "cache-eval", "--state-dir", state_dir,
            "--n", "42", "--result", '{"version": 1}',
        ])
        # Overwrite
        self.runner.invoke(main, [
            "cache-eval", "--state-dir", state_dir,
            "--n", "42", "--result", '{"version": 2}',
        ])
        cache_path = os.path.join(state_dir, "eval_cache.json")
        with open(cache_path, "r") as f:
            cache = json.load(f)
        assert cache["42"]["version"] == 2
        assert len(cache) == 1

    def test_cache_eval_help(self):
        result = self.runner.invoke(main, ["cache-eval", "--help"])
        assert result.exit_code == 0
        assert "--n" in result.output
        assert "--result" in result.output
