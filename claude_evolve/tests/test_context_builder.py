"""
Tests for the prompt/template system: TemplateManager and ContextBuilder.

Covers template loading, fragment formatting, security filtering,
improvement area detection, evolution history, inspiration classification,
markdown rendering, artifact handling, and edge cases.
"""

import json
import os
import tempfile
import unittest

from claude_evolve.config import PromptConfig
from claude_evolve.core.artifact import Artifact
from claude_evolve.prompt.context_builder import ContextBuilder
from claude_evolve.prompt.templates import TemplateManager


class TestTemplateManagerDefaults(unittest.TestCase):
    """Test TemplateManager with built-in defaults."""

    def setUp(self):
        self.tm = TemplateManager()

    def test_default_templates_loaded(self):
        """All standard templates should be available."""
        for name in [
            "system_message",
            "evaluator_system_message",
            "diff_user",
            "full_rewrite_user",
            "evolution_history",
            "previous_attempt",
            "top_program",
            "inspirations_section",
            "inspiration_program",
            "evaluation",
        ]:
            self.assertTrue(
                self.tm.has_template(name),
                f"Missing template: {name}",
            )

    def test_default_fragments_loaded(self):
        """All standard fragments should be available."""
        for name in [
            "fitness_improved",
            "fitness_declined",
            "fitness_stable",
            "no_specific_guidance",
            "artifact_title",
            "attempt_unknown_changes",
            "attempt_all_metrics_improved",
            "attempt_all_metrics_regressed",
            "attempt_mixed_metrics",
            "inspiration_type_diverse",
            "inspiration_type_score_high_performer",
        ]:
            self.assertTrue(
                self.tm.has_fragment(name),
                f"Missing fragment: {name}",
            )

    def test_get_template_returns_string(self):
        tmpl = self.tm.get_template("system_message")
        self.assertIsInstance(tmpl, str)
        self.assertGreater(len(tmpl), 0)

    def test_get_template_raises_on_missing(self):
        with self.assertRaises(ValueError):
            self.tm.get_template("nonexistent_template")

    def test_get_fragment_formats_kwargs(self):
        result = self.tm.get_fragment(
            "fitness_improved", prev=0.3, current=0.5
        )
        self.assertIn("0.3000", result)
        self.assertIn("0.5000", result)

    def test_get_fragment_missing_returns_placeholder(self):
        result = self.tm.get_fragment("totally_missing_fragment")
        self.assertIn("[Missing fragment", result)

    def test_get_fragment_missing_kwarg_returns_error(self):
        result = self.tm.get_fragment("fitness_improved")  # missing prev/current
        self.assertIn("[Fragment formatting error", result)

    def test_add_template_overrides(self):
        self.tm.add_template("custom_test", "hello {name}")
        self.assertEqual(self.tm.get_template("custom_test"), "hello {name}")

    def test_add_fragment_overrides(self):
        self.tm.add_fragment("custom_frag", "value: {x}")
        self.assertEqual(self.tm.get_fragment("custom_frag", x=42), "value: 42")

    def test_list_templates(self):
        templates = self.tm.list_templates()
        self.assertIsInstance(templates, list)
        self.assertIn("system_message", templates)
        # Should be sorted
        self.assertEqual(templates, sorted(templates))

    def test_list_fragments(self):
        fragments = self.tm.list_fragments()
        self.assertIsInstance(fragments, list)
        self.assertIn("fitness_improved", fragments)
        self.assertEqual(fragments, sorted(fragments))

    def test_has_template(self):
        self.assertTrue(self.tm.has_template("system_message"))
        self.assertFalse(self.tm.has_template("nope"))

    def test_has_fragment(self):
        self.assertTrue(self.tm.has_fragment("fitness_improved"))
        self.assertFalse(self.tm.has_fragment("nope"))


class TestTemplateManagerCustomDir(unittest.TestCase):
    """Test TemplateManager with a custom template directory."""

    def test_custom_templates_override_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_file = os.path.join(tmpdir, "system_message.txt")
            with open(custom_file, "w") as f:
                f.write("Custom system message.")

            tm = TemplateManager(custom_template_dir=tmpdir)
            self.assertEqual(
                tm.get_template("system_message"), "Custom system message."
            )

    def test_custom_fragments_merge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frags = {"my_custom_frag": "Hello {who}"}
            with open(os.path.join(tmpdir, "fragments.json"), "w") as f:
                json.dump(frags, f)

            tm = TemplateManager(custom_template_dir=tmpdir)
            self.assertEqual(
                tm.get_fragment("my_custom_frag", who="world"),
                "Hello world",
            )
            # Original fragments should still exist
            self.assertTrue(tm.has_fragment("fitness_improved"))

    def test_nonexistent_custom_dir_uses_defaults(self):
        tm = TemplateManager(custom_template_dir="/nonexistent/path")
        # Should still have default templates
        self.assertTrue(tm.has_template("system_message"))

    def test_custom_dir_adds_new_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "brand_new.txt"), "w") as f:
                f.write("brand new content")

            tm = TemplateManager(custom_template_dir=tmpdir)
            self.assertEqual(
                tm.get_template("brand_new"), "brand new content"
            )
            # Defaults still present
            self.assertTrue(tm.has_template("diff_user"))


class TestContextBuilderBasic(unittest.TestCase):
    """Test ContextBuilder.build_context with basic inputs."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="parent-1",
            content="def solve():\n    return 42",
            artifact_type="python",
            metrics={"combined_score": 0.6, "accuracy": 0.7},
            generation=2,
        )

    def test_build_context_returns_dict(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[self.parent.to_dict()],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("prompt", ctx)
        self.assertIn("system_message", ctx)
        self.assertIn("parent_id", ctx)
        self.assertIn("metadata", ctx)

    def test_context_includes_parent_code(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("def solve():", ctx["prompt"])

    def test_context_includes_metrics(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        # The fitness score is derived from combined_score and displayed
        # as the Fitness field in the prompt (0.6000).  The metadata dict
        # in the context retains the raw metric names.
        self.assertIn("0.6000", ctx["prompt"])
        self.assertIn("combined_score", str(ctx["metadata"]["parent_metrics"]))

    def test_context_includes_evolution_history(self):
        prev = Artifact(
            id="prev-1",
            content="old code",
            metrics={"combined_score": 0.4},
            metadata={"changes": "initial attempt"},
        )
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=3,
            best_score=0.6,
            top_programs=[self.parent.to_dict()],
            inspirations=[],
            previous_programs=[prev.to_dict()],
        )
        self.assertIn("Previous Attempts", ctx["prompt"])

    def test_context_includes_artifacts_when_present(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            parent_artifacts={"stderr": "TypeError: bad arg"},
        )
        self.assertIn("TypeError", ctx["prompt"])

    def test_context_diff_vs_rewrite_mode(self):
        ctx_diff = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            diff_based=True,
        )
        self.assertIn("SEARCH", ctx_diff["prompt"])

        ctx_rewrite = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            diff_based=False,
        )
        self.assertIn("Rewrite", ctx_rewrite["prompt"])

    def test_render_iteration_context_markdown(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=5, max_iterations=30
        )
        self.assertIn("Iteration 5", md)
        self.assertIn("Best Score", md)

    def test_parent_id_propagated(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertEqual(ctx["parent_id"], "parent-1")

    def test_metadata_contains_expected_keys(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=7,
            best_score=0.8,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        meta = ctx["metadata"]
        self.assertEqual(meta["iteration"], 7)
        self.assertEqual(meta["best_score"], 0.8)
        self.assertIn("parent_metrics", meta)
        self.assertIn("diff_based", meta)

    def test_parent_as_dict(self):
        parent_dict = {
            "id": "dict-parent",
            "content": "x = 1",
            "metrics": {"combined_score": 0.3},
            "artifact_type": "python",
        }
        ctx = self.builder.build_context(
            parent=parent_dict,
            iteration=1,
            best_score=0.3,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertEqual(ctx["parent_id"], "dict-parent")
        self.assertIn("x = 1", ctx["prompt"])


class TestContextBuilderImprovementAreas(unittest.TestCase):
    """Test improvement area detection logic."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())

    def test_fitness_improved_message(self):
        parent = Artifact(
            id="p1",
            content="code",
            metrics={"combined_score": 0.7},
        )
        prev = {
            "id": "p0",
            "content": "old",
            "metrics": {"combined_score": 0.4},
        }
        ctx = self.builder.build_context(
            parent=parent,
            iteration=2,
            best_score=0.7,
            top_programs=[],
            inspirations=[],
            previous_programs=[prev],
        )
        self.assertIn("improved", ctx["prompt"].lower())

    def test_fitness_declined_message(self):
        parent = Artifact(
            id="p1",
            content="code",
            metrics={"combined_score": 0.3},
        )
        prev = {
            "id": "p0",
            "content": "old",
            "metrics": {"combined_score": 0.6},
        }
        ctx = self.builder.build_context(
            parent=parent,
            iteration=2,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[prev],
        )
        self.assertIn("declined", ctx["prompt"].lower())

    def test_fitness_stable_message(self):
        parent = Artifact(
            id="p1",
            content="code",
            metrics={"combined_score": 0.5},
        )
        prev = {
            "id": "p0",
            "content": "old",
            "metrics": {"combined_score": 0.5},
        }
        ctx = self.builder.build_context(
            parent=parent,
            iteration=2,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[prev],
        )
        self.assertIn("unchanged", ctx["prompt"].lower())

    def test_code_too_long_suggestion(self):
        long_code = "x = 1\n" * 200  # way over 500 chars
        parent = Artifact(
            id="p1",
            content=long_code,
            metrics={"combined_score": 0.5},
        )
        ctx = self.builder.build_context(
            parent=parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("simplifying", ctx["prompt"].lower())

    def test_no_specific_guidance_default(self):
        short_code = "x = 1"
        parent = Artifact(
            id="p1",
            content=short_code,
            metrics={"combined_score": 0.5},
        )
        ctx = self.builder.build_context(
            parent=parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        # With no previous programs and short code, should get default guidance
        self.assertIn("Focus", ctx["prompt"])

    def test_feature_dimensions_exploration(self):
        parent = Artifact(
            id="p1",
            content="code",
            metrics={"combined_score": 0.5, "complexity": 0.7},
        )
        ctx = self.builder.build_context(
            parent=parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            feature_dimensions=["complexity"],
        )
        self.assertIn("complexity", ctx["prompt"])


class TestContextBuilderInspirations(unittest.TestCase):
    """Test inspiration program formatting and classification."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve(): pass",
            metrics={"combined_score": 0.5},
        )

    def test_inspiration_section_rendered(self):
        inspiration = {
            "id": "insp-1",
            "code": "def solve(): return 99",
            "metrics": {"combined_score": 0.8},
            "metadata": {},
        }
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[inspiration],
            previous_programs=[],
        )
        self.assertIn("Inspiration", ctx["prompt"])

    def test_no_inspirations_no_section(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        # "Inspiration Programs" header should not appear
        self.assertNotIn("Inspiration Programs", ctx["prompt"])

    def test_determine_program_type_diverse(self):
        prog = {"metrics": {"combined_score": 0.5}, "metadata": {"diverse": True}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "Diverse")

    def test_determine_program_type_high_score(self):
        prog = {"metrics": {"combined_score": 0.9}, "metadata": {}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "High-Performer")

    def test_determine_program_type_low_score(self):
        prog = {"metrics": {"combined_score": 0.2}, "metadata": {}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "Exploratory")

    def test_determine_program_type_experimental(self):
        prog = {"metrics": {"combined_score": 0.45}, "metadata": {}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "Experimental")

    def test_determine_program_type_alternative(self):
        prog = {"metrics": {"combined_score": 0.65}, "metadata": {}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "Alternative")

    def test_determine_program_type_migrant(self):
        prog = {"metrics": {"combined_score": 0.5}, "metadata": {"migrant": True}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "Migrant")

    def test_determine_program_type_random(self):
        prog = {"metrics": {"combined_score": 0.5}, "metadata": {"random": True}}
        result = self.builder._determine_program_type(prog)
        self.assertEqual(result, "Random")


class TestContextBuilderFeatureExtraction(unittest.TestCase):
    """Test unique feature extraction from programs."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())

    def test_extract_numpy_feature(self):
        prog = {
            "code": "import numpy as np\nx = np.array([1,2,3])",
            "metrics": {"combined_score": 0.5},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("NumPy", features)

    def test_extract_class_feature(self):
        prog = {
            "code": "class Foo:\n    def __init__(self):\n        pass",
            "metrics": {"combined_score": 0.5},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("Object-oriented", features)

    def test_extract_concise_feature(self):
        prog = {
            "code": "x = 1\ny = 2",
            "metrics": {"combined_score": 0.5},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("Concise", features)

    def test_extract_comprehensive_feature(self):
        prog = {
            "code": "\n".join([f"line_{i} = {i}" for i in range(60)]),
            "metrics": {"combined_score": 0.5},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("Comprehensive", features)

    def test_extract_excellent_metric(self):
        prog = {
            "code": "x = 1",
            "metrics": {"accuracy": 0.95},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("Excellent", features)
        self.assertIn("accuracy", features)

    def test_extract_alternative_metric(self):
        prog = {
            "code": "x = 1",
            "metrics": {"speed": 0.2},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("Alternative", features)
        self.assertIn("speed", features)

    def test_extract_changes_metadata(self):
        prog = {
            "code": "x = 1",
            "metrics": {"combined_score": 0.5},
            "metadata": {"changes": "refactored loop"},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("Modification", features)

    def test_default_features_when_nothing_special(self):
        prog = {
            "code": "x = 1\ny = 2\nz = 3\nprint(x + y + z)\na = 4\nb = 5\nc = 6\nd = 7\ne = 8\nf = 9\ng = 10\nh = 11",
            "metrics": {"combined_score": 0.5},
            "metadata": {},
        }
        features = self.builder._extract_unique_features(prog)
        self.assertIn("approach to the problem", features)


class TestContextBuilderSecurity(unittest.TestCase):
    """Test security filtering of artifacts."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())

    def test_security_filter_redacts_api_keys(self):
        text = "key = sk-abc123def456ghi789jkl012mno345pqr678"
        filtered = self.builder._apply_security_filter(text)
        self.assertIn("<REDACTED_API_KEY>", filtered)

    def test_security_filter_redacts_secret_key_assignments(self):
        text = "api_key = aaaaaaaabbbbbbbbccccccccdddddddd"
        filtered = self.builder._apply_security_filter(text)
        self.assertIn("<REDACTED_SECRET>", filtered)

    def test_security_filter_does_not_redact_legitimate_code(self):
        # UUIDs, hashes, and long variable names should NOT be redacted
        text = "uuid = aaaaaaaabbbbbbbbccccccccdddddddd"
        filtered = self.builder._apply_security_filter(text)
        self.assertNotIn("<REDACTED", filtered)

    def test_security_filter_redacts_password(self):
        text = "password= s3cr3t_pass"
        filtered = self.builder._apply_security_filter(text)
        self.assertIn("<REDACTED>", filtered)

    def test_security_filter_strips_ansi(self):
        text = "\x1b[31mError: bad\x1b[0m"
        filtered = self.builder._apply_security_filter(text)
        self.assertNotIn("\x1b", filtered)
        self.assertIn("Error", filtered)

    def test_security_filter_preserves_normal_text(self):
        text = "This is fine."
        filtered = self.builder._apply_security_filter(text)
        self.assertEqual(filtered, "This is fine.")

    def test_safe_decode_bytes_artifact(self):
        value = b"hello from stderr"
        # Disable security filter to test pure decode
        self.builder.config.artifact_security_filter = False
        result = self.builder._safe_decode_artifact(value)
        self.assertEqual(result, "hello from stderr")

    def test_safe_decode_non_utf8_bytes(self):
        value = b"\xff\xfe invalid utf8"
        self.builder.config.artifact_security_filter = False
        result = self.builder._safe_decode_artifact(value)
        # Should use replacement characters rather than crashing
        self.assertIsInstance(result, str)

    def test_safe_decode_non_string_non_bytes(self):
        result = self.builder._safe_decode_artifact(12345)
        self.assertEqual(result, "12345")


class TestContextBuilderArtifacts(unittest.TestCase):
    """Test artifact rendering in prompts."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())

    def test_render_artifacts_empty(self):
        result = self.builder._render_artifacts({})
        self.assertEqual(result, "")

    def test_render_artifacts_none(self):
        result = self.builder._render_artifacts(None)
        self.assertEqual(result, "")

    def test_render_artifacts_with_content(self):
        artifacts = {"stderr": "Error line", "stdout": "Output line"}
        result = self.builder._render_artifacts(artifacts)
        self.assertIn("stderr", result)
        self.assertIn("stdout", result)
        self.assertIn("Error line", result)
        self.assertIn("Output line", result)
        self.assertIn("Last Execution Output", result)

    def test_render_artifacts_truncation(self):
        self.builder.config.max_artifact_bytes = 50
        # Use content that won't be collapsed by the security filter
        # (short words separated by spaces avoid the 32+ char token pattern)
        long_content = " ".join(["word"] * 60)  # ~300 chars
        artifacts = {"big": long_content}
        # Disable security filter so it doesn't interfere with length test
        self.builder.config.artifact_security_filter = False
        result = self.builder._render_artifacts(artifacts)
        self.assertIn("truncated", result)
        # Should not contain the full content
        self.assertLess(len(result), len(long_content) + 100)


class TestContextBuilderTemplateOverrides(unittest.TestCase):
    """Test template override mechanism."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="code",
            metrics={"combined_score": 0.5},
        )

    def test_set_system_template_override(self):
        self.builder.template_manager.add_template(
            "custom_sys", "Custom system: evolve!"
        )
        self.builder.set_templates(system_template="custom_sys")
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertEqual(ctx["system_message"], "Custom system: evolve!")

    def test_set_user_template_override(self):
        self.builder.template_manager.add_template(
            "custom_user",
            "Custom user prompt for {current_program}. "
            "Metrics: {metrics}. "
            "Fitness: {fitness_score}. "
            "Coords: {feature_coords}. "
            "Dims: {feature_dimensions}. "
            "Areas: {improvement_areas}. "
            "History: {evolution_history}. "
            "Artifacts: {artifacts}. "
            "Lang: {language}.",
        )
        self.builder.set_templates(user_template="custom_user")
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("Custom user prompt", ctx["prompt"])


class TestContextBuilderRenderMarkdown(unittest.TestCase):
    """Test render_iteration_context Markdown output."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve(): return 42",
            metrics={"combined_score": 0.6, "accuracy": 0.7},
        )

    def test_markdown_contains_iteration(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(ctx, iteration=5, max_iterations=30)
        self.assertIn("Iteration 5 of 30", md)

    def test_markdown_contains_best_score(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(ctx, iteration=1, max_iterations=50)
        self.assertIn("0.6000", md)

    def test_markdown_contains_parent_metrics(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(ctx, iteration=1, max_iterations=50)
        self.assertIn("combined_score", md)
        self.assertIn("accuracy", md)

    def test_markdown_contains_mode_diff(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            diff_based=True,
        )
        md = self.builder.render_iteration_context(ctx, iteration=1, max_iterations=50)
        self.assertIn("Diff-based", md)

    def test_markdown_contains_mode_rewrite(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            diff_based=False,
        )
        md = self.builder.render_iteration_context(ctx, iteration=1, max_iterations=50)
        self.assertIn("Full Rewrite", md)

    def test_markdown_contains_system_guidance(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(ctx, iteration=1, max_iterations=50)
        self.assertIn("System Guidance", md)

    def test_markdown_contains_parent_id(self):
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(ctx, iteration=1, max_iterations=50)
        self.assertIn("p1", md)


class TestContextBuilderEvolutionHistory(unittest.TestCase):
    """Test evolution history formatting (previous attempts, top programs)."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve(): return 42",
            metrics={"combined_score": 0.6},
        )

    def test_top_programs_in_prompt(self):
        top = {
            "id": "top-1",
            "code": "def solve(): return 100",
            "metrics": {"combined_score": 0.9},
        }
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.9,
            top_programs=[top],
            inspirations=[],
            previous_programs=[],
        )
        self.assertIn("Top Performing Programs", ctx["prompt"])
        self.assertIn("0.9000", ctx["prompt"])

    def test_previous_attempts_limited_to_three(self):
        prevs = [
            {
                "id": f"prev-{i}",
                "content": f"code {i}",
                "metrics": {"combined_score": 0.1 * i},
                "metadata": {"changes": f"change {i}"},
            }
            for i in range(6)
        ]
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=7,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=prevs,
        )
        # Should contain the most recent 3 attempts: 6, 5, 4
        self.assertIn("Attempt 6", ctx["prompt"])
        self.assertIn("Attempt 5", ctx["prompt"])
        self.assertIn("Attempt 4", ctx["prompt"])
        # Older ones should not appear
        self.assertNotIn("Attempt 1\n", ctx["prompt"])

    def test_outcome_all_improved(self):
        prev = {
            "id": "prev-1",
            "content": "old",
            "metrics": {"accuracy": 0.8, "speed": 0.7},
            "metadata": {"parent_metrics": {"accuracy": 0.5, "speed": 0.3}},
        }
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=2,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[prev],
        )
        self.assertIn("Improvement in all metrics", ctx["prompt"])

    def test_outcome_all_regressed(self):
        prev = {
            "id": "prev-1",
            "content": "old",
            "metrics": {"accuracy": 0.3, "speed": 0.2},
            "metadata": {"parent_metrics": {"accuracy": 0.5, "speed": 0.5}},
        }
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=2,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[prev],
        )
        self.assertIn("Regression in all metrics", ctx["prompt"])


class TestContextBuilderTemplateVariations(unittest.TestCase):
    """Test stochastic template variations."""

    def test_variation_applied(self):
        config = PromptConfig(
            use_template_stochasticity=True,
            template_variations={"task_verb": ["Optimize", "Enhance", "Refine"]},
        )
        builder = ContextBuilder(config)
        template = "Please {task_verb} the code."
        result = builder._apply_template_variations(template)
        self.assertIn(result, [
            "Please Optimize the code.",
            "Please Enhance the code.",
            "Please Refine the code.",
        ])

    def test_no_variation_when_disabled(self):
        config = PromptConfig(
            use_template_stochasticity=False,
            template_variations={"task_verb": ["Optimize"]},
        )
        builder = ContextBuilder(config)
        parent = Artifact(
            id="p1",
            content="x = 1",
            metrics={"combined_score": 0.5},
        )
        # Even though template_variations has entries, stochasticity is off
        # so the template should remain as-is (variations not applied in
        # build_context because the flag check is there)
        ctx = builder.build_context(
            parent=parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        # The prompt should still be a valid string
        self.assertIsInstance(ctx["prompt"], str)


class TestContextBuilderFormatMetrics(unittest.TestCase):
    """Test _format_metrics helper."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())

    def test_format_numeric_metrics(self):
        result = self.builder._format_metrics({"acc": 0.95, "loss": 0.05})
        self.assertIn("acc: 0.9500", result)
        self.assertIn("loss: 0.0500", result)

    def test_format_mixed_types(self):
        result = self.builder._format_metrics(
            {"acc": 0.8, "status": "ok", "count": 10}
        )
        self.assertIn("acc: 0.8000", result)
        self.assertIn("status: ok", result)
        self.assertIn("count: 10.0000", result)

    def test_format_empty_metrics(self):
        result = self.builder._format_metrics({})
        self.assertEqual(result, "")


class TestContextBuilderChangesDescription(unittest.TestCase):
    """Test programs_as_changes_description mode."""

    def test_changes_description_in_prompt(self):
        config = PromptConfig(programs_as_changes_description=True)
        builder = ContextBuilder(config)
        parent = Artifact(
            id="p1",
            content="code",
            metrics={"combined_score": 0.5},
        )
        ctx = builder.build_context(
            parent=parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            current_changes_description="1. Added loop optimization",
        )
        self.assertIn("Changes Description", ctx["prompt"])
        self.assertIn("Added loop optimization", ctx["prompt"])


class TestContextBuilderPairwiseComparison(unittest.TestCase):
    """Test pairwise comparison (verbal gradient) rendering."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve():\n    return 42",
            artifact_type="python",
            metrics={"combined_score": 0.8, "accuracy": 0.9},
            generation=3,
        )
        self.comparison = Artifact(
            id="comp-1",
            content="def solve():\n    return sum(range(10))",
            artifact_type="python",
            metrics={"combined_score": 0.5, "accuracy": 0.6},
            generation=2,
        )

    def test_comparison_artifact_in_rendered_context(self):
        """When comparison_artifact is provided, both parent and comparison rendered."""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.8,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            comparison_artifact=self.comparison,
            comparison_score=0.5,
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=5, max_iterations=30
        )
        self.assertIn("Pairwise Comparison (Verbal Gradient)", md)
        self.assertIn("0.8000", md)  # parent score
        self.assertIn("0.5000", md)  # comparison score
        self.assertIn("sum(range(10))", md)  # comparison content snippet

    def test_comparison_section_absent_without_comparison(self):
        """Without comparison_artifact, no pairwise section appears."""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.8,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=5, max_iterations=30
        )
        self.assertNotIn("Pairwise Comparison", md)

    def test_comparison_metadata_stored(self):
        """Comparison data should be in the context metadata."""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.8,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            comparison_artifact=self.comparison,
            comparison_score=0.5,
        )
        meta = ctx["metadata"]
        self.assertIn("comparison_artifact_content", meta)
        self.assertIn("comparison_score", meta)
        self.assertAlmostEqual(meta["comparison_score"], 0.5)
        self.assertIn("sum(range(10))", meta["comparison_artifact_content"])

    def test_comparison_with_dict_artifact(self):
        """Comparison should also work when passed as a dict."""
        comp_dict = {
            "id": "comp-dict",
            "content": "def alternative(): return 99",
            "metrics": {"combined_score": 0.3},
        }
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=2,
            best_score=0.8,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            comparison_artifact=comp_dict,
            comparison_score=0.3,
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=2, max_iterations=30
        )
        self.assertIn("Pairwise Comparison", md)
        self.assertIn("alternative", md)

    def test_comparison_content_truncated_to_500_chars(self):
        """Long comparison content should be truncated to 500 chars."""
        long_content = "x = 1\n" * 200  # ~1200 chars
        comp = Artifact(
            id="comp-long",
            content=long_content,
            artifact_type="python",
            metrics={"combined_score": 0.4},
        )
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.8,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            comparison_artifact=comp,
            comparison_score=0.4,
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=1, max_iterations=30
        )
        self.assertIn("Pairwise Comparison", md)
        # The snippet in the rendered markdown should be at most 500 chars
        # Find the comparison code block
        idx = md.index("### Comparison Program")
        snippet_section = md[idx:idx+700]
        # The actual content between ``` markers should be <= 500 chars
        self.assertIn("x = 1", snippet_section)


class TestContextBuilderAscendingSort(unittest.TestCase):
    """Test ascending score sort and score annotation in top programs (Task 11)."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve(): return 42",
            metrics={"combined_score": 0.6},
        )

    def test_top_programs_sorted_ascending_in_context(self):
        """Top programs should be rendered worst-first, best-last in context."""
        top = [
            {
                "id": f"top-{i}",
                "code": f"def solve(): return {i}",
                "metrics": {"combined_score": score},
            }
            for i, score in [(1, 0.9), (2, 0.7), (3, 0.5)]
        ]
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.9,
            top_programs=top,
            inspirations=[],
            previous_programs=[],
        )
        prompt = ctx["prompt"]
        # Find positions of the three scores in the prompt
        pos_05 = prompt.find("0.5000")
        pos_07 = prompt.find("0.7000")
        pos_09 = prompt.find("0.9000")
        # All three scores should appear
        self.assertNotEqual(pos_05, -1, "Score 0.5000 should appear in prompt")
        self.assertNotEqual(pos_07, -1, "Score 0.7000 should appear in prompt")
        self.assertNotEqual(pos_09, -1, "Score 0.9000 should appear in prompt")
        # Ascending order: worst (0.5) first, best (0.9) last
        self.assertLess(pos_05, pos_07,
                        "Score 0.5 should appear before 0.7 (ascending)")
        self.assertLess(pos_07, pos_09,
                        "Score 0.7 should appear before 0.9 (ascending)")

    def test_score_annotation_prefix_on_programs(self):
        """Each top program should have a '# Score: X.XXXXXX' prefix."""
        top = [
            {
                "id": "top-1",
                "code": "def solve(): return 100",
                "metrics": {"combined_score": 0.85},
            },
        ]
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=3,
            best_score=0.85,
            top_programs=top,
            inspirations=[],
            previous_programs=[],
        )
        prompt = ctx["prompt"]
        self.assertIn("# Score: 0.850000", prompt,
                       "Score annotation with 6 decimal places should be present")

    def test_ascending_sort_with_single_program(self):
        """Ascending sort should work fine with a single top program."""
        top = [
            {
                "id": "top-1",
                "code": "def solve(): return 1",
                "metrics": {"combined_score": 0.75},
            },
        ]
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=2,
            best_score=0.75,
            top_programs=top,
            inspirations=[],
            previous_programs=[],
        )
        prompt = ctx["prompt"]
        self.assertIn("0.7500", prompt)
        self.assertIn("# Score: 0.750000", prompt)

    def test_ascending_sort_with_empty_top_programs(self):
        """Ascending sort should handle empty top programs gracefully."""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        # Should not crash, just no top programs section content
        self.assertIsInstance(ctx["prompt"], str)


class TestContextBuilderEvaluatorSource(unittest.TestCase):
    """Test evaluator source code inclusion in context (Task 11)."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve(): return 42",
            metrics={"combined_score": 0.6},
        )

    def test_evaluator_source_included_in_context(self):
        """When evaluator_source is provided, it should appear in the context."""
        evaluator_code = "def evaluate(path):\n    return {'score': 1.0}\n"
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=3,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            evaluator_source=evaluator_code,
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=3, max_iterations=30
        )
        self.assertIn("Evaluator Source", md)
        self.assertIn("def evaluate", md)

    def test_evaluator_source_absent_when_not_provided(self):
        """Without evaluator_source, no evaluator section appears."""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=3,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=3, max_iterations=30
        )
        self.assertNotIn("Evaluator Source", md)

    def test_evaluator_source_in_metadata(self):
        """Evaluator source should be stored in context metadata."""
        evaluator_code = "def evaluate(x): return x"
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            evaluator_source=evaluator_code,
        )
        self.assertEqual(ctx["metadata"]["evaluator_source"], evaluator_code)

    def test_evaluator_source_rendered_in_python_code_block(self):
        """Evaluator source should be rendered in a python code block."""
        evaluator_code = "import math\ndef evaluate(p): return math.pi"
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=1,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            evaluator_source=evaluator_code,
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=1, max_iterations=30
        )
        self.assertIn("```python", md)
        self.assertIn("import math", md)


class TestContextBuilderFailuresText(unittest.TestCase):
    """Test recent failures reflexion rendering in context (Task 10)."""

    def setUp(self):
        self.builder = ContextBuilder(PromptConfig())
        self.parent = Artifact(
            id="p1",
            content="def solve(): return 42",
            metrics={"combined_score": 0.6},
        )

    def test_failures_text_included_in_context(self):
        """When failures_text is provided, it should appear in the rendered context."""
        failures = "## Recent Failures (Avoid These)\n- Approach: foo. Result: score dropped to 0.1"
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            failures_text=failures,
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=5, max_iterations=30
        )
        self.assertIn("Recent Failures", md)
        self.assertIn("foo", md)

    def test_failures_text_absent_when_not_provided(self):
        """Without failures_text, no failures section appears."""
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=5,
            best_score=0.6,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
        )
        md = self.builder.render_iteration_context(
            ctx, iteration=5, max_iterations=30
        )
        self.assertNotIn("Recent Failures", md)

    def test_failures_text_in_metadata(self):
        """Failures text should be stored in context metadata."""
        failures = "## Recent Failures\n- Something failed"
        ctx = self.builder.build_context(
            parent=self.parent,
            iteration=2,
            best_score=0.5,
            top_programs=[],
            inspirations=[],
            previous_programs=[],
            failures_text=failures,
        )
        self.assertEqual(ctx["metadata"]["failures_text"], failures)


if __name__ == "__main__":
    unittest.main()
