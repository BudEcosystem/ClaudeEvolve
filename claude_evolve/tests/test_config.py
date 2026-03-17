import os
import tempfile
import unittest
import yaml
from claude_evolve.config import (
    Config,
    CrossRunMemoryConfig,
    DatabaseConfig,
    DiagnosticsConfig,
    EvaluatorConfig,
    ImprovementSignalConfig,
    PromptConfig,
    EvolutionConfig,
    EvolutionTraceConfig,
    ReflectionConfig,
    ResearchConfig,
    ScratchpadConfig,
    SelectionConfig,
    StagnationConfig,
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


class TestStagnationConfig(unittest.TestCase):
    def test_defaults(self):
        c = StagnationConfig()
        self.assertEqual(c.mild_threshold, 3)
        self.assertEqual(c.moderate_threshold, 6)
        self.assertEqual(c.severe_threshold, 11)
        self.assertEqual(c.critical_threshold, 20)
        self.assertAlmostEqual(c.score_tolerance, 0.001)
        self.assertAlmostEqual(c.exploration_boost_mild, 0.1)
        self.assertAlmostEqual(c.exploration_boost_moderate, 0.2)
        self.assertAlmostEqual(c.exploration_boost_severe, 0.3)
        self.assertAlmostEqual(c.exploration_boost_critical, 0.5)
        self.assertTrue(c.enabled)

    def test_custom_values(self):
        c = StagnationConfig(
            mild_threshold=5,
            severe_threshold=15,
            score_tolerance=0.01,
            enabled=False,
        )
        self.assertEqual(c.mild_threshold, 5)
        self.assertEqual(c.severe_threshold, 15)
        self.assertAlmostEqual(c.score_tolerance, 0.01)
        self.assertFalse(c.enabled)

    def test_threshold_ordering(self):
        """Default thresholds should increase: mild < moderate < severe < critical."""
        c = StagnationConfig()
        self.assertLess(c.mild_threshold, c.moderate_threshold)
        self.assertLess(c.moderate_threshold, c.severe_threshold)
        self.assertLess(c.severe_threshold, c.critical_threshold)


class TestCrossRunMemoryConfig(unittest.TestCase):
    def test_defaults(self):
        c = CrossRunMemoryConfig()
        self.assertTrue(c.enabled)
        self.assertEqual(c.memory_dir, "cross_run_memory")
        self.assertEqual(c.max_learnings, 100)
        self.assertEqual(c.max_failed_approaches, 50)
        self.assertAlmostEqual(c.improvement_threshold, 0.01)

    def test_custom_values(self):
        c = CrossRunMemoryConfig(
            enabled=False,
            memory_dir="custom_memory",
            max_learnings=200,
            max_failed_approaches=75,
            improvement_threshold=0.05,
        )
        self.assertFalse(c.enabled)
        self.assertEqual(c.memory_dir, "custom_memory")
        self.assertEqual(c.max_learnings, 200)
        self.assertEqual(c.max_failed_approaches, 75)
        self.assertAlmostEqual(c.improvement_threshold, 0.05)


class TestResearchConfig(unittest.TestCase):
    def test_defaults(self):
        c = ResearchConfig()
        self.assertFalse(c.enabled)
        self.assertEqual(c.trigger, "on_stagnation")
        self.assertEqual(c.periodic_interval, 10)
        self.assertEqual(c.max_web_searches, 5)
        self.assertTrue(c.persist_findings)
        self.assertEqual(c.research_log_file, "research_log.json")

    def test_custom_values(self):
        c = ResearchConfig(
            enabled=True,
            trigger="always",
            periodic_interval=5,
            max_web_searches=10,
            persist_findings=False,
            research_log_file="custom_research.json",
        )
        self.assertTrue(c.enabled)
        self.assertEqual(c.trigger, "always")
        self.assertEqual(c.periodic_interval, 5)
        self.assertEqual(c.max_web_searches, 10)
        self.assertFalse(c.persist_findings)
        self.assertEqual(c.research_log_file, "custom_research.json")


class TestDiagnosticsConfig(unittest.TestCase):
    def test_defaults(self):
        c = DiagnosticsConfig()
        self.assertFalse(c.enabled)
        self.assertEqual(c.trigger, "on_stagnation")
        self.assertEqual(c.min_stagnation_level, "mild")
        self.assertTrue(c.persist_reports)
        self.assertEqual(c.diagnostic_report_file, "diagnostic_report.json")

    def test_custom_values(self):
        c = DiagnosticsConfig(
            enabled=True,
            trigger="always",
            min_stagnation_level="moderate",
            persist_reports=False,
            diagnostic_report_file="custom_diagnostic.json",
        )
        self.assertTrue(c.enabled)
        self.assertEqual(c.trigger, "always")
        self.assertEqual(c.min_stagnation_level, "moderate")
        self.assertFalse(c.persist_reports)
        self.assertEqual(c.diagnostic_report_file, "custom_diagnostic.json")


class TestImprovementSignalConfig(unittest.TestCase):
    def test_defaults(self):
        c = ImprovementSignalConfig()
        self.assertTrue(c.enabled)
        self.assertAlmostEqual(c.rho, 0.95)
        self.assertAlmostEqual(c.i_min, 0.1)
        self.assertAlmostEqual(c.i_max, 0.7)
        self.assertAlmostEqual(c.meta_threshold, 0.12)

    def test_custom_values(self):
        c = ImprovementSignalConfig(
            enabled=False, rho=0.9, i_min=0.05, i_max=0.8, meta_threshold=0.2
        )
        self.assertFalse(c.enabled)
        self.assertAlmostEqual(c.rho, 0.9)
        self.assertAlmostEqual(c.i_min, 0.05)
        self.assertAlmostEqual(c.i_max, 0.8)
        self.assertAlmostEqual(c.meta_threshold, 0.2)


class TestSelectionConfig(unittest.TestCase):
    def test_defaults(self):
        c = SelectionConfig()
        self.assertEqual(c.strategy_selection, "ucb1")
        self.assertAlmostEqual(c.ucb_c, 1.414)
        self.assertAlmostEqual(c.ucb_decay, 0.95)
        self.assertEqual(c.parent_selection, "power_law")
        self.assertAlmostEqual(c.novelty_gate_min_novelty, 0.05)

    def test_custom_values(self):
        c = SelectionConfig(
            strategy_selection="random",
            ucb_c=2.0,
            ucb_decay=0.9,
            parent_selection="tournament",
            novelty_gate_min_novelty=0.1,
        )
        self.assertEqual(c.strategy_selection, "random")
        self.assertAlmostEqual(c.ucb_c, 2.0)
        self.assertAlmostEqual(c.ucb_decay, 0.9)
        self.assertEqual(c.parent_selection, "tournament")
        self.assertAlmostEqual(c.novelty_gate_min_novelty, 0.1)


class TestScratchpadConfig(unittest.TestCase):
    def test_defaults(self):
        c = ScratchpadConfig()
        self.assertTrue(c.enabled)
        self.assertEqual(c.synthesis_interval, 10)

    def test_custom_values(self):
        c = ScratchpadConfig(enabled=False, synthesis_interval=20)
        self.assertFalse(c.enabled)
        self.assertEqual(c.synthesis_interval, 20)


class TestReflectionConfig(unittest.TestCase):
    def test_defaults(self):
        c = ReflectionConfig()
        self.assertTrue(c.enabled)
        self.assertEqual(c.max_short_reflections, 20)
        self.assertEqual(c.synthesis_interval, 5)

    def test_custom_values(self):
        c = ReflectionConfig(
            enabled=False, max_short_reflections=50, synthesis_interval=10
        )
        self.assertFalse(c.enabled)
        self.assertEqual(c.max_short_reflections, 50)
        self.assertEqual(c.synthesis_interval, 10)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.max_iterations, 50)
        self.assertIsInstance(c.database, DatabaseConfig)
        self.assertIsInstance(c.evaluator, EvaluatorConfig)
        self.assertIsInstance(c.prompt, PromptConfig)
        self.assertIsInstance(c.evolution, EvolutionConfig)
        self.assertIsInstance(c.evolution_trace, EvolutionTraceConfig)
        self.assertIsInstance(c.stagnation, StagnationConfig)
        self.assertIsInstance(c.cross_run_memory, CrossRunMemoryConfig)
        self.assertIsInstance(c.research, ResearchConfig)
        self.assertIsInstance(c.diagnostics, DiagnosticsConfig)
        self.assertIsInstance(c.improvement_signal, ImprovementSignalConfig)
        self.assertIsInstance(c.selection, SelectionConfig)
        self.assertIsInstance(c.scratchpad, ScratchpadConfig)
        self.assertIsInstance(c.reflection, ReflectionConfig)
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

    def test_from_dict_stagnation_overrides(self):
        d = {
            "stagnation": {
                "mild_threshold": 5,
                "severe_threshold": 15,
                "score_tolerance": 0.01,
                "enabled": False,
            },
        }
        c = Config.from_dict(d)
        self.assertEqual(c.stagnation.mild_threshold, 5)
        self.assertEqual(c.stagnation.severe_threshold, 15)
        self.assertAlmostEqual(c.stagnation.score_tolerance, 0.01)
        self.assertFalse(c.stagnation.enabled)
        # Unspecified fields keep defaults
        self.assertEqual(c.stagnation.moderate_threshold, 6)
        self.assertEqual(c.stagnation.critical_threshold, 20)

    def test_from_dict_cross_run_memory_overrides(self):
        d = {
            "cross_run_memory": {
                "enabled": False,
                "memory_dir": "my_memory",
                "max_learnings": 200,
            },
        }
        c = Config.from_dict(d)
        self.assertFalse(c.cross_run_memory.enabled)
        self.assertEqual(c.cross_run_memory.memory_dir, "my_memory")
        self.assertEqual(c.cross_run_memory.max_learnings, 200)
        # Unspecified fields keep defaults
        self.assertEqual(c.cross_run_memory.max_failed_approaches, 50)
        self.assertAlmostEqual(c.cross_run_memory.improvement_threshold, 0.01)

    def test_to_dict_includes_stagnation_and_cross_run_memory(self):
        c = Config()
        d = c.to_dict()
        self.assertIn("stagnation", d)
        self.assertIn("cross_run_memory", d)
        self.assertEqual(d["stagnation"]["mild_threshold"], 3)
        self.assertEqual(d["stagnation"]["enabled"], True)
        self.assertEqual(d["cross_run_memory"]["enabled"], True)
        self.assertEqual(d["cross_run_memory"]["memory_dir"], "cross_run_memory")

    def test_to_dict_includes_research_and_diagnostics(self):
        c = Config()
        d = c.to_dict()
        self.assertIn("research", d)
        self.assertIn("diagnostics", d)
        self.assertEqual(d["research"]["enabled"], False)
        self.assertEqual(d["research"]["trigger"], "on_stagnation")
        self.assertEqual(d["diagnostics"]["enabled"], False)
        self.assertEqual(d["diagnostics"]["trigger"], "on_stagnation")

    def test_from_dict_research_overrides(self):
        d = {
            "research": {
                "enabled": True,
                "trigger": "always",
                "periodic_interval": 5,
                "max_web_searches": 10,
            },
        }
        c = Config.from_dict(d)
        self.assertTrue(c.research.enabled)
        self.assertEqual(c.research.trigger, "always")
        self.assertEqual(c.research.periodic_interval, 5)
        self.assertEqual(c.research.max_web_searches, 10)
        # Unspecified fields keep defaults
        self.assertTrue(c.research.persist_findings)
        self.assertEqual(c.research.research_log_file, "research_log.json")

    def test_from_dict_diagnostics_overrides(self):
        d = {
            "diagnostics": {
                "enabled": True,
                "trigger": "always",
                "min_stagnation_level": "moderate",
            },
        }
        c = Config.from_dict(d)
        self.assertTrue(c.diagnostics.enabled)
        self.assertEqual(c.diagnostics.trigger, "always")
        self.assertEqual(c.diagnostics.min_stagnation_level, "moderate")
        # Unspecified fields keep defaults
        self.assertTrue(c.diagnostics.persist_reports)
        self.assertEqual(c.diagnostics.diagnostic_report_file, "diagnostic_report.json")

    def test_config_has_new_sections_with_correct_defaults(self):
        """Config should include improvement_signal, selection, scratchpad, reflection."""
        config = Config()
        self.assertTrue(hasattr(config, "improvement_signal"))
        self.assertTrue(hasattr(config, "selection"))
        self.assertTrue(hasattr(config, "scratchpad"))
        self.assertTrue(hasattr(config, "reflection"))
        self.assertAlmostEqual(config.improvement_signal.rho, 0.95)
        self.assertEqual(config.selection.strategy_selection, "ucb1")
        self.assertEqual(config.scratchpad.synthesis_interval, 10)
        self.assertEqual(config.reflection.max_short_reflections, 20)

    def test_from_dict_improvement_signal_overrides(self):
        d = {"improvement_signal": {"rho": 0.9, "i_min": 0.05, "enabled": False}}
        c = Config.from_dict(d)
        self.assertAlmostEqual(c.improvement_signal.rho, 0.9)
        self.assertAlmostEqual(c.improvement_signal.i_min, 0.05)
        self.assertFalse(c.improvement_signal.enabled)
        # Unspecified fields keep defaults
        self.assertAlmostEqual(c.improvement_signal.i_max, 0.7)
        self.assertAlmostEqual(c.improvement_signal.meta_threshold, 0.12)

    def test_from_dict_selection_overrides(self):
        d = {"selection": {"ucb_c": 2.0, "parent_selection": "tournament"}}
        c = Config.from_dict(d)
        self.assertAlmostEqual(c.selection.ucb_c, 2.0)
        self.assertEqual(c.selection.parent_selection, "tournament")
        # Unspecified fields keep defaults
        self.assertEqual(c.selection.strategy_selection, "ucb1")
        self.assertAlmostEqual(c.selection.ucb_decay, 0.95)
        self.assertAlmostEqual(c.selection.novelty_gate_min_novelty, 0.05)

    def test_from_dict_scratchpad_overrides(self):
        d = {"scratchpad": {"enabled": False, "synthesis_interval": 20}}
        c = Config.from_dict(d)
        self.assertFalse(c.scratchpad.enabled)
        self.assertEqual(c.scratchpad.synthesis_interval, 20)

    def test_from_dict_reflection_overrides(self):
        d = {"reflection": {"max_short_reflections": 50, "synthesis_interval": 10}}
        c = Config.from_dict(d)
        self.assertEqual(c.reflection.max_short_reflections, 50)
        self.assertEqual(c.reflection.synthesis_interval, 10)
        # Unspecified fields keep defaults
        self.assertTrue(c.reflection.enabled)

    def test_from_dict_with_multiple_new_sections(self):
        """Config.from_dict should deserialize multiple new sections at once."""
        d = {
            "improvement_signal": {"rho": 0.9},
            "selection": {"ucb_c": 2.0},
            "scratchpad": {"synthesis_interval": 15},
            "reflection": {"max_short_reflections": 30},
        }
        c = Config.from_dict(d)
        self.assertAlmostEqual(c.improvement_signal.rho, 0.9)
        self.assertAlmostEqual(c.selection.ucb_c, 2.0)
        self.assertEqual(c.scratchpad.synthesis_interval, 15)
        self.assertEqual(c.reflection.max_short_reflections, 30)

    def test_to_dict_includes_new_sections(self):
        c = Config()
        d = c.to_dict()
        self.assertIn("improvement_signal", d)
        self.assertIn("selection", d)
        self.assertIn("scratchpad", d)
        self.assertIn("reflection", d)
        self.assertAlmostEqual(d["improvement_signal"]["rho"], 0.95)
        self.assertTrue(d["improvement_signal"]["enabled"])
        self.assertEqual(d["selection"]["strategy_selection"], "ucb1")
        self.assertAlmostEqual(d["selection"]["ucb_c"], 1.414)
        self.assertTrue(d["scratchpad"]["enabled"])
        self.assertEqual(d["scratchpad"]["synthesis_interval"], 10)
        self.assertTrue(d["reflection"]["enabled"])
        self.assertEqual(d["reflection"]["max_short_reflections"], 20)
        self.assertEqual(d["reflection"]["synthesis_interval"], 5)

    def test_to_dict_from_dict_roundtrip_new_sections(self):
        """to_dict -> from_dict should preserve new section values."""
        c = Config(
            improvement_signal=ImprovementSignalConfig(rho=0.8, i_min=0.2),
            selection=SelectionConfig(ucb_c=3.0, parent_selection="tournament"),
            scratchpad=ScratchpadConfig(enabled=False, synthesis_interval=25),
            reflection=ReflectionConfig(max_short_reflections=40, synthesis_interval=8),
        )
        d = c.to_dict()
        c2 = Config.from_dict(d)
        self.assertAlmostEqual(c2.improvement_signal.rho, 0.8)
        self.assertAlmostEqual(c2.improvement_signal.i_min, 0.2)
        self.assertAlmostEqual(c2.selection.ucb_c, 3.0)
        self.assertEqual(c2.selection.parent_selection, "tournament")
        self.assertFalse(c2.scratchpad.enabled)
        self.assertEqual(c2.scratchpad.synthesis_interval, 25)
        self.assertEqual(c2.reflection.max_short_reflections, 40)
        self.assertEqual(c2.reflection.synthesis_interval, 8)

    def test_yaml_roundtrip_with_new_sections(self):
        """YAML save/load should preserve new section values."""
        c = Config(
            improvement_signal=ImprovementSignalConfig(rho=0.85),
            selection=SelectionConfig(ucb_c=2.5),
            scratchpad=ScratchpadConfig(synthesis_interval=15),
            reflection=ReflectionConfig(max_short_reflections=30),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            c.to_yaml(f.name)
            c2 = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertAlmostEqual(c2.improvement_signal.rho, 0.85)
        self.assertAlmostEqual(c2.selection.ucb_c, 2.5)
        self.assertEqual(c2.scratchpad.synthesis_interval, 15)
        self.assertEqual(c2.reflection.max_short_reflections, 30)

    def test_yaml_roundtrip_with_new_configs(self):
        c = Config(
            max_iterations=25,
            stagnation=StagnationConfig(mild_threshold=4, enabled=False),
            cross_run_memory=CrossRunMemoryConfig(
                memory_dir="custom_dir", max_learnings=50
            ),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            c.to_yaml(f.name)
            c2 = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertEqual(c2.max_iterations, 25)
        self.assertEqual(c2.stagnation.mild_threshold, 4)
        self.assertFalse(c2.stagnation.enabled)
        self.assertEqual(c2.cross_run_memory.memory_dir, "custom_dir")
        self.assertEqual(c2.cross_run_memory.max_learnings, 50)

    def test_yaml_roundtrip_with_research_and_diagnostics(self):
        c = Config(
            max_iterations=30,
            research=ResearchConfig(
                enabled=True, trigger="periodic", periodic_interval=5
            ),
            diagnostics=DiagnosticsConfig(
                enabled=True, min_stagnation_level="moderate"
            ),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            c.to_yaml(f.name)
            c2 = Config.from_yaml(f.name)
        os.unlink(f.name)
        self.assertEqual(c2.max_iterations, 30)
        self.assertTrue(c2.research.enabled)
        self.assertEqual(c2.research.trigger, "periodic")
        self.assertEqual(c2.research.periodic_interval, 5)
        self.assertTrue(c2.diagnostics.enabled)
        self.assertEqual(c2.diagnostics.min_stagnation_level, "moderate")

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
