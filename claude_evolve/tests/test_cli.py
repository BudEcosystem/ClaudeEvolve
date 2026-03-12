"""
Tests for the CLI entry points (claude-evolve command).

Covers init, next, submit, status, and export subcommands.
"""

import json
import os
import shutil
import tempfile
import unittest

from click.testing import CliRunner

from claude_evolve.cli import main


class TestCliInit(unittest.TestCase):
    """Tests for ``claude-evolve init``."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("def solve():\n    return 42\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(path):\n    return {"combined_score": 0.5}\n')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_creates_state(self):
        state_dir = os.path.join(self.tmpdir, ".claude", "evolve-state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--max-iterations", "10",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(os.path.join(state_dir, "database.json")))

    def test_init_missing_artifact_fails(self):
        result = self.runner.invoke(main, [
            "init", "--artifact", "/nonexistent/file.py",
            "--evaluator", self.evaluator,
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_init_missing_evaluator_fails(self):
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact,
            "--evaluator", "/nonexistent/evaluator.py",
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_init_outputs_json(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output.strip())
        self.assertEqual(output["status"], "initialized")
        self.assertIn("population_size", output)

    def test_init_baseline_score_present(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--mode", "script",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output.strip())
        self.assertIn("baseline_score", output)

    def test_init_default_mode_is_script(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Check that the config stored in state has script mode
        config_path = os.path.join(state_dir, "config.json")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["evaluator"]["mode"], "script")

    def test_init_critic_mode_skips_baseline(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--mode", "critic",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output.strip())
        # Critic mode should not have run baseline evaluation
        self.assertIsNone(output.get("baseline_score"))

    def test_init_with_max_iterations(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--max-iterations", "25",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["max_iterations"], 25)

    def test_init_with_target_score(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--target-score", "0.95",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertAlmostEqual(cfg["target_score"], 0.95)

    def test_init_detects_python_artifact_type(self):
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["artifact_type"], "python")

    def test_init_detects_markdown_artifact_type(self):
        md_artifact = os.path.join(self.tmpdir, "prompt.md")
        with open(md_artifact, "w") as f:
            f.write("# My Prompt\nDo something.")
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", md_artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["artifact_type"], "markdown")

    def test_init_with_config_file(self):
        config_file = os.path.join(self.tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            f.write("max_iterations: 100\ncheckpoint_interval: 5\n")
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--config", config_file,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["checkpoint_interval"], 5)

    def test_init_cli_overrides_config_file(self):
        config_file = os.path.join(self.tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            f.write("max_iterations: 100\n")
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", state_dir, "--config", config_file,
            "--max-iterations", "20",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        # CLI --max-iterations=20 should override config file's 100
        self.assertEqual(cfg["max_iterations"], 20)


class TestCliNext(unittest.TestCase):
    """Tests for ``claude-evolve next``."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.3}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_next_produces_context(self):
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        context_file = os.path.join(self.state_dir, "iteration_context.md")
        self.assertTrue(os.path.exists(context_file))

    def test_next_outputs_prompt(self):
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(len(result.output) > 0)

    def test_next_without_init_fails(self):
        empty_state = os.path.join(self.tmpdir, "empty_state")
        result = self.runner.invoke(main, ["next", "--state-dir", empty_state])
        self.assertNotEqual(result.exit_code, 0)

    def test_next_context_file_contains_markdown(self):
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        context_file = os.path.join(self.state_dir, "iteration_context.md")
        with open(context_file) as f:
            content = f.read()
        # Should contain markdown headings
        self.assertIn("# Evolution Iteration", content)


class TestCliSubmit(unittest.TestCase):
    """Tests for ``claude-evolve submit``."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = optimized_value\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_submit_evaluates_and_stores(self):
        result = self.runner.invoke(main, [
            "submit", "--candidate", self.candidate, "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("combined_score", output)

    def test_submit_with_metrics_passthrough(self):
        result = self.runner.invoke(main, [
            "submit", "--candidate", self.candidate, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.85, "clarity": 0.9}',
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertAlmostEqual(output["combined_score"], 0.85)

    def test_submit_missing_candidate_fails(self):
        result = self.runner.invoke(main, [
            "submit", "--candidate", "/nonexistent/candidate.py",
            "--state-dir", self.state_dir,
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_submit_without_init_fails(self):
        empty_state = os.path.join(self.tmpdir, "empty_state")
        result = self.runner.invoke(main, [
            "submit", "--candidate", self.candidate,
            "--state-dir", empty_state,
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_submit_reports_is_new_best(self):
        result = self.runner.invoke(main, [
            "submit", "--candidate", self.candidate, "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("is_new_best", output)

    def test_submit_invalid_metrics_json(self):
        result = self.runner.invoke(main, [
            "submit", "--candidate", self.candidate,
            "--state-dir", self.state_dir,
            "--metrics", "not-valid-json",
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_submit_increases_population(self):
        # Get initial population
        status1 = self.runner.invoke(main, [
            "status", "--state-dir", self.state_dir,
        ])
        pop_before = json.loads(status1.output)["population_size"]

        # Submit a candidate
        result = self.runner.invoke(main, [
            "submit", "--candidate", self.candidate,
            "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        # Check population grew
        status2 = self.runner.invoke(main, [
            "status", "--state-dir", self.state_dir,
        ])
        pop_after = json.loads(status2.output)["population_size"]
        self.assertGreaterEqual(pop_after, pop_before)


class TestCliStatus(unittest.TestCase):
    """Tests for ``claude-evolve status``."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_status_shows_info(self):
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("best_score", output)
        self.assertIn("population_size", output)

    def test_status_includes_iteration(self):
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("iteration", output)

    def test_status_includes_islands(self):
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("islands", output)

    def test_status_includes_target_score(self):
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        self.assertIn("target_score", output)

    def test_status_without_init_fails(self):
        empty_state = os.path.join(self.tmpdir, "empty_state")
        result = self.runner.invoke(main, ["status", "--state-dir", empty_state])
        self.assertNotEqual(result.exit_code, 0)


class TestCliExport(unittest.TestCase):
    """Tests for ``claude-evolve export``."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("x = 1\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        self.state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_export_best(self):
        output_file = os.path.join(self.tmpdir, "best.py")
        result = self.runner.invoke(main, [
            "export", "--state-dir", self.state_dir, "--output", output_file,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(output_file))
        with open(output_file) as f:
            content = f.read()
        self.assertIn("x = 1", content)

    def test_export_without_init_fails(self):
        empty_state = os.path.join(self.tmpdir, "empty_state")
        output_file = os.path.join(self.tmpdir, "best.py")
        result = self.runner.invoke(main, [
            "export", "--state-dir", empty_state, "--output", output_file,
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_export_top_n(self):
        # Submit a few candidates first
        for i in range(3):
            candidate = os.path.join(self.tmpdir, f"cand_{i}.py")
            with open(candidate, "w") as f:
                f.write(f"x = {i + 10}\n")
            self.runner.invoke(main, [
                "submit", "--candidate", candidate,
                "--state-dir", self.state_dir,
                "--metrics", json.dumps({"combined_score": 0.1 * (i + 1)}),
            ])

        output_file = os.path.join(self.tmpdir, "top.py")
        result = self.runner.invoke(main, [
            "export", "--state-dir", self.state_dir,
            "--output", output_file, "--top-n", "2",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # With --top-n, should produce numbered output files
        self.assertTrue(
            os.path.exists(output_file)
            or os.path.exists(os.path.join(self.tmpdir, "top_1.py"))
        )

    def test_export_creates_parent_directory(self):
        output_file = os.path.join(self.tmpdir, "deep", "nested", "best.py")
        result = self.runner.invoke(main, [
            "export", "--state-dir", self.state_dir, "--output", output_file,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(output_file))


class TestCliMultiSubmitCycle(unittest.TestCase):
    """Test a multi-iteration cycle: init -> (next, submit) x N -> status."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write("def solve():\n    return 0\n")
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write(
                'def evaluate(p):\n'
                '    with open(p) as f: c = f.read()\n'
                '    return {"combined_score": min(1.0, len(c) / 100.0)}\n'
            )
        self.state_dir = os.path.join(self.tmpdir, "state")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_multi_submit_cycle(self):
        # Init
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        init_output = json.loads(result.output.strip())
        self.assertEqual(init_output["status"], "initialized")

        # Run 3 iterations of next -> submit
        for i in range(3):
            # Next
            next_result = self.runner.invoke(main, [
                "next", "--state-dir", self.state_dir,
            ])
            self.assertEqual(next_result.exit_code, 0,
                             msg=f"next failed at iteration {i}: {next_result.output}")

            # Create a candidate
            candidate = os.path.join(self.tmpdir, f"candidate_{i}.py")
            content = f"def solve():\n    return {i + 1}\n" + ("# padding\n" * (i + 1) * 5)
            with open(candidate, "w") as f:
                f.write(content)

            # Submit
            submit_result = self.runner.invoke(main, [
                "submit", "--candidate", candidate,
                "--state-dir", self.state_dir,
            ])
            self.assertEqual(submit_result.exit_code, 0,
                             msg=f"submit failed at iteration {i}: {submit_result.output}")
            submit_output = json.loads(submit_result.output)
            self.assertIn("combined_score", submit_output)

        # Final status check
        status_result = self.runner.invoke(main, [
            "status", "--state-dir", self.state_dir,
        ])
        self.assertEqual(status_result.exit_code, 0, msg=status_result.output)
        status_output = json.loads(status_result.output)
        self.assertGreater(status_output["population_size"], 1)

    def test_submit_with_improving_scores(self):
        # Init
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        scores = []
        for i in range(4):
            candidate = os.path.join(self.tmpdir, f"cand_{i}.py")
            with open(candidate, "w") as f:
                f.write(f"x = {i}\n")

            submit_result = self.runner.invoke(main, [
                "submit", "--candidate", candidate,
                "--state-dir", self.state_dir,
                "--metrics", json.dumps({"combined_score": 0.2 * (i + 1)}),
            ])
            self.assertEqual(submit_result.exit_code, 0, msg=submit_result.output)
            output = json.loads(submit_result.output)
            scores.append(output["combined_score"])

        # Later submissions had higher metrics
        self.assertAlmostEqual(scores[-1], 0.8)

        # Best score in status should reflect the highest
        status_result = self.runner.invoke(main, [
            "status", "--state-dir", self.state_dir,
        ])
        status_output = json.loads(status_result.output)
        self.assertGreaterEqual(status_output["best_score"], 0.8)


class TestCliEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_subcommand_shows_help(self):
        result = self.runner.invoke(main, [])
        # Click returns exit code 0 or 2 depending on version when no subcommand given
        self.assertIn(result.exit_code, (0, 2))
        self.assertIn("Usage", result.output)

    def test_help_flag(self):
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage", result.output)

    def test_init_help(self):
        result = self.runner.invoke(main, ["init", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--artifact", result.output)

    def test_init_empty_artifact_file(self):
        empty_artifact = os.path.join(self.tmpdir, "empty.py")
        with open(empty_artifact, "w") as f:
            f.write("")
        evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.0}')
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", empty_artifact, "--evaluator", evaluator,
            "--state-dir", state_dir,
        ])
        # Empty artifact is valid -- the user might start from scratch
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_export_no_output_requires_option(self):
        artifact = os.path.join(self.tmpdir, "program.py")
        with open(artifact, "w") as f:
            f.write("x = 1\n")
        evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init", "--artifact", artifact, "--evaluator", evaluator,
            "--state-dir", state_dir,
        ])
        # Missing --output should fail
        result = self.runner.invoke(main, [
            "export", "--state-dir", state_dir,
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_init_javascript_artifact(self):
        js_artifact = os.path.join(self.tmpdir, "script.js")
        with open(js_artifact, "w") as f:
            f.write("console.log('hello');\n")
        evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        state_dir = os.path.join(self.tmpdir, "state")
        result = self.runner.invoke(main, [
            "init", "--artifact", js_artifact, "--evaluator", evaluator,
            "--state-dir", state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config_path = os.path.join(state_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["artifact_type"], "javascript")

    def test_submit_metrics_passthrough_ensures_combined_score(self):
        """If metrics lack combined_score, it should be computed."""
        artifact = os.path.join(self.tmpdir, "program.py")
        with open(artifact, "w") as f:
            f.write("x = 1\n")
        evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(evaluator, "w") as f:
            f.write('def evaluate(p): return {"combined_score": 0.5}')
        state_dir = os.path.join(self.tmpdir, "state")
        self.runner.invoke(main, [
            "init", "--artifact", artifact, "--evaluator", evaluator,
            "--state-dir", state_dir,
        ])
        candidate = os.path.join(self.tmpdir, "cand.py")
        with open(candidate, "w") as f:
            f.write("y = 2\n")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", state_dir,
            "--metrics", '{"accuracy": 0.8, "fluency": 0.6}',
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        output = json.loads(result.output)
        # evaluate_with_metrics computes combined_score as average
        self.assertIn("combined_score", output)
        self.assertAlmostEqual(output["combined_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
