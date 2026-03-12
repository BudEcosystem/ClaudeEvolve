import os
import tempfile
import unittest
import yaml
from claude_evolve.config import (
    Config,
    DatabaseConfig,
    EvaluatorConfig,
    PromptConfig,
    EvolutionConfig,
    EvolutionTraceConfig,
    load_config,
    _resolve_env_vars,
)


class TestDatabaseConfig(unittest.TestCase):
    def test_defaults(self):
        c = DatabaseConfig()
        self.assertEqual(c.population_size, 1000)
        self.assertEqual(c.num_islands, 5)
        self.assertAlmostEqual(c.elite_selection_ratio, 0.1)
        self.assertAlmostEqual(c.exploration_ratio, 0.2)
        self.assertAlmostEqual(c.exploitation_ratio, 0.7)
        self.assertEqual(c.feature_dimensions, ["complexity", "diversity"])
        self.assertEqual(c.feature_bins, 10)
        self.assertEqual(c.migration_interval, 50)

    def test_custom_values(self):
        c = DatabaseConfig(
            population_size=500,
            num_islands=3,
            feature_dimensions=["score", "complexity", "diversity"],
            feature_bins={"score": 20, "complexity": 10, "diversity": 10},
        )
        self.assertEqual(c.population_size, 500)
        self.assertEqual(c.num_islands, 3)
        self.assertEqual(len(c.feature_dimensions), 3)
        self.assertIsInstance(c.feature_bins, dict)

    def test_archive_and_selection(self):
        c = DatabaseConfig()
        self.assertEqual(c.archive_size, 100)
        self.assertEqual(c.diversity_metric, "edit_distance")
        self.assertEqual(c.diversity_reference_size, 20)

    def test_migration_params(self):
        c = DatabaseConfig()
        self.assertEqual(c.migration_interval, 50)
        self.assertAlmostEqual(c.migration_rate, 0.1)

    def test_artifact_storage(self):
        c = DatabaseConfig()
        self.assertIsNone(c.artifacts_base_path)
        self.assertEqual(c.artifact_size_threshold, 32 * 1024)
        self.assertTrue(c.cleanup_old_artifacts)
        self.assertEqual(c.artifact_retention_days, 30)


class TestEvaluatorConfig(unittest.TestCase):
    def test_defaults(self):
        c = EvaluatorConfig()
        self.assertEqual(c.timeout, 300)
        self.assertTrue(c.cascade_evaluation)
        self.assertEqual(c.mode, "script")
        self.assertEqual(c.max_retries, 3)
        self.assertEqual(c.parallel_evaluations, 1)

    def test_script_mode(self):
        c = EvaluatorConfig(mode="script")
        self.assertEqual(c.mode, "script")

    def test_critic_mode(self):
        c = EvaluatorConfig(mode="critic")
        self.assertEqual(c.mode, "critic")

    def test_hybrid_mode(self):
        c = EvaluatorConfig(mode="hybrid")
        self.assertEqual(c.mode, "hybrid")

    def test_cascade_thresholds(self):
        c = EvaluatorConfig()
        self.assertEqual(c.cascade_thresholds, [0.5, 0.75, 0.9])

    def test_resource_limits(self):
        c = EvaluatorConfig(memory_limit_mb=2048, cpu_limit=4.0)
        self.assertEqual(c.memory_limit_mb, 2048)
        self.assertAlmostEqual(c.cpu_limit, 4.0)


class TestPromptConfig(unittest.TestCase):
    def test_defaults(self):
        c = PromptConfig()
        self.assertIsNone(c.template_dir)
        self.assertEqual(c.num_top_programs, 3)
        self.assertEqual(c.num_diverse_programs, 2)
        self.assertTrue(c.use_template_stochasticity)
        self.assertTrue(c.include_artifacts)

    def test_custom_template_dir(self):
        c = PromptConfig(template_dir="/tmp/templates")
        self.assertEqual(c.template_dir, "/tmp/templates")

    def test_diff_summary_defaults(self):
        c = PromptConfig()
        self.assertEqual(c.diff_summary_max_line_len, 100)
        self.assertEqual(c.diff_summary_max_lines, 30)

    def test_artifact_rendering(self):
        c = PromptConfig()
        self.assertEqual(c.max_artifact_bytes, 20 * 1024)
        self.assertTrue(c.artifact_security_filter)


class TestEvolutionConfig(unittest.TestCase):
    def test_diff_based_default(self):
        c = EvolutionConfig()
        self.assertTrue(c.diff_based)
        self.assertEqual(c.max_content_length, 50000)

    def test_custom_values(self):
        c = EvolutionConfig(diff_based=False, max_content_length=100000)
        self.assertFalse(c.diff_based)
        self.assertEqual(c.max_content_length, 100000)

    def test_diff_pattern(self):
        c = EvolutionConfig()
        self.assertIn("SEARCH", c.diff_pattern)
        self.assertIn("REPLACE", c.diff_pattern)


class TestEvolutionTraceConfig(unittest.TestCase):
    def test_defaults(self):
        c = EvolutionTraceConfig()
        self.assertFalse(c.enabled)
        self.assertEqual(c.format, "jsonl")
        self.assertFalse(c.include_code)
        self.assertTrue(c.include_prompts)
        self.assertIsNone(c.output_path)
        self.assertEqual(c.buffer_size, 10)
        self.assertFalse(c.compress)

    def test_custom(self):
        c = EvolutionTraceConfig(enabled=True, format="json", include_code=True)
        self.assertTrue(c.enabled)
        self.assertEqual(c.format, "json")
        self.assertTrue(c.include_code)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.max_iterations, 50)
        self.assertIsInstance(c.database, DatabaseConfig)
        self.assertIsInstance(c.evaluator, EvaluatorConfig)
        self.assertIsInstance(c.prompt, PromptConfig)
        self.assertIsInstance(c.evolution, EvolutionConfig)
        self.assertIsInstance(c.evolution_trace, EvolutionTraceConfig)
        self.assertIsNone(c.target_score)
        self.assertEqual(c.artifact_type, "python")
        self.assertEqual(c.output_dir, "./evolve_output")

    def test_general_settings(self):
        c = Config()
        self.assertEqual(c.checkpoint_interval, 100)
        self.assertEqual(c.log_level, "INFO")
        self.assertEqual(c.random_seed, 42)

    def test_early_stopping(self):
        c = Config()
        self.assertIsNone(c.early_stopping_patience)
        self.assertAlmostEqual(c.convergence_threshold, 0.001)
        self.assertEqual(c.early_stopping_metric, "combined_score")

    def test_from_dict(self):
        d = {
            "max_iterations": 100,
            "target_score": 0.95,
            "database": {"num_islands": 3},
            "evaluator": {"timeout": 600, "mode": "critic"},
        }
        c = Config.from_dict(d)
        self.assertEqual(c.max_iterations, 100)
        self.assertAlmostEqual(c.target_score, 0.95)
        self.assertEqual(c.database.num_islands, 3)
        self.assertEqual(c.evaluator.mode, "critic")

    def test_from_dict_nested_evolution(self):
        d = {
            "evolution": {
                "diff_based": False,
                "max_content_length": 80000,
            },
        }
        c = Config.from_dict(d)
        self.assertFalse(c.evolution.diff_based)
        self.assertEqual(c.evolution.max_content_length, 80000)

    def test_from_dict_preserves_defaults(self):
        """Fields not in dict should keep their default values."""
        d = {"max_iterations": 200}
        c = Config.from_dict(d)
        self.assertEqual(c.max_iterations, 200)
        self.assertEqual(c.database.population_size, 1000)
        self.assertEqual(c.evaluator.timeout, 300)
        self.assertEqual(c.artifact_type, "python")

    def test_from_yaml_file(self):
        config_data = {
            "max_iterations": 30,
            "artifact_type": "markdown",
            "database": {"population_size": 500},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            c = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertEqual(c.max_iterations, 30)
        self.assertEqual(c.artifact_type, "markdown")
        self.assertEqual(c.database.population_size, 500)

    def test_from_yaml_resolves_template_dir(self):
        """Template dir should be resolved relative to the yaml file location."""
        config_data = {
            "prompt": {"template_dir": "my_templates"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir="/tmp"
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            c = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertTrue(c.prompt.template_dir.startswith("/tmp"))
        self.assertTrue(c.prompt.template_dir.endswith("my_templates"))

    def test_to_dict_roundtrip(self):
        c = Config(max_iterations=77, target_score=0.8)
        d = c.to_dict()
        c2 = Config.from_dict(d)
        self.assertEqual(c2.max_iterations, 77)
        self.assertAlmostEqual(c2.target_score, 0.8)

    def test_to_yaml_roundtrip(self):
        c = Config(max_iterations=55, artifact_type="rust", target_score=0.99)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            c.to_yaml(f.name)
            c2 = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertEqual(c2.max_iterations, 55)
        self.assertEqual(c2.artifact_type, "rust")
        self.assertAlmostEqual(c2.target_score, 0.99)

    def test_env_var_resolution(self):
        os.environ["TEST_EVOLVE_VAR"] = "resolved_value"
        d = {"output_dir": "${TEST_EVOLVE_VAR}/output"}
        c = Config.from_dict(d)
        self.assertIn("resolved_value", c.output_dir)
        del os.environ["TEST_EVOLVE_VAR"]

    def test_env_var_full_match(self):
        os.environ["TEST_FULL_VAR"] = "/some/path"
        d = {"output_dir": "${TEST_FULL_VAR}"}
        c = Config.from_dict(d)
        self.assertEqual(c.output_dir, "/some/path")
        del os.environ["TEST_FULL_VAR"]

    def test_env_var_missing_raises(self):
        d = {"output_dir": "${NONEXISTENT_VAR_12345}"}
        with self.assertRaises(ValueError):
            Config.from_dict(d)

    def test_random_seed_propagation(self):
        """Config random_seed should propagate to database if database has no seed."""
        d = {"random_seed": 123, "database": {"random_seed": None}}
        c = Config.from_dict(d)
        self.assertEqual(c.database.random_seed, 123)

    def test_invalid_diff_pattern_raises(self):
        d = {"evolution": {"diff_pattern": "[invalid"}}
        with self.assertRaises(ValueError):
            Config.from_dict(d)


class TestLoadConfig(unittest.TestCase):
    def test_load_defaults(self):
        """load_config with no path returns defaults."""
        c = load_config()
        self.assertIsInstance(c, Config)
        self.assertEqual(c.max_iterations, 50)

    def test_load_from_yaml(self):
        config_data = {"max_iterations": 42, "target_score": 0.75}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            c = load_config(f.name)
        os.unlink(f.name)
        self.assertEqual(c.max_iterations, 42)
        self.assertAlmostEqual(c.target_score, 0.75)

    def test_load_nonexistent_returns_default(self):
        c = load_config("/tmp/nonexistent_config_file_12345.yaml")
        self.assertIsInstance(c, Config)
        self.assertEqual(c.max_iterations, 50)


class TestResolveEnvVars(unittest.TestCase):
    def test_none_passthrough(self):
        self.assertIsNone(_resolve_env_vars(None))

    def test_no_var(self):
        self.assertEqual(_resolve_env_vars("plain string"), "plain string")

    def test_full_match(self):
        os.environ["_TEST_RES_1"] = "hello"
        self.assertEqual(_resolve_env_vars("${_TEST_RES_1}"), "hello")
        del os.environ["_TEST_RES_1"]

    def test_embedded_match(self):
        os.environ["_TEST_RES_2"] = "world"
        self.assertEqual(
            _resolve_env_vars("hello/${_TEST_RES_2}/foo"),
            "hello/world/foo",
        )
        del os.environ["_TEST_RES_2"]

    def test_missing_var_raises(self):
        with self.assertRaises(ValueError):
            _resolve_env_vars("${_MISSING_VAR_99999}")

    def test_non_string_passthrough(self):
        self.assertEqual(_resolve_env_vars(42), 42)
        self.assertEqual(_resolve_env_vars(3.14), 3.14)
        self.assertTrue(_resolve_env_vars(True))


if __name__ == "__main__":
    unittest.main()
