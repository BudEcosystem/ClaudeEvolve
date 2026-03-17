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


class TestCliCrossRunMemory(unittest.TestCase):
    """Tests for cross-run memory population in init/next/submit cycle."""

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

    def test_init_creates_metadata_with_run_id(self):
        """init should create metadata.json with a run_id."""
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        metadata_path = os.path.join(self.state_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_path),
                        "metadata.json should be created by init")
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.assertIn("run_id", metadata)
        self.assertEqual(len(metadata["run_id"]), 8,
                         "run_id should be an 8-char hex string")

    def test_init_run_id_is_unique_across_runs(self):
        """Each init should produce a different run_id."""
        run_ids = []
        for i in range(3):
            sd = os.path.join(self.tmpdir, f"state_{i}")
            self.runner.invoke(main, [
                "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
                "--state-dir", sd, "--max-iterations", "5",
            ])
            metadata_path = os.path.join(sd, "metadata.json")
            with open(metadata_path) as f:
                run_ids.append(json.load(f)["run_id"])
        self.assertEqual(len(set(run_ids)), 3,
                         "Each init should produce a unique run_id")

    def test_next_writes_iteration_manifest(self):
        """next should write current_iteration.json with iteration metadata."""
        # Init
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        # Next
        result = self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        manifest_path = os.path.join(self.state_dir, "current_iteration.json")
        self.assertTrue(os.path.exists(manifest_path),
                        "current_iteration.json should be written by next")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertIn("iteration", manifest)
        self.assertIn("run_id", manifest)
        self.assertIn("selected_strategy_id", manifest)
        self.assertIn("parent_artifact_id", manifest)
        self.assertIn("parent_score", manifest)
        self.assertIn("parent_island_id", manifest)

    def test_submit_populates_memory_on_failure(self):
        """When submit score <= parent, failed approach should be recorded."""
        # Init + Next to create manifest
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        # Submit with a low score (worse than the parent)
        candidate = os.path.join(self.tmpdir, "bad_candidate.py")
        with open(candidate, "w") as f:
            f.write("# tiny")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.01}',
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Check that cross-run memory was populated
        memory_dir = os.path.join(self.state_dir, "cross_run_memory")
        failed_path = os.path.join(memory_dir, "failed_approaches.json")
        self.assertTrue(os.path.exists(failed_path),
                        "failed_approaches.json should exist after a failed submit")
        with open(failed_path) as f:
            failed = json.load(f)
        self.assertGreater(len(failed), 0,
                           "At least one failed approach should be recorded")

    def test_submit_populates_memory_on_success(self):
        """When submit score > parent, successful strategy should be recorded."""
        # Init + Next to create manifest
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        # Submit with a high score (better than parent)
        candidate = os.path.join(self.tmpdir, "good_candidate.py")
        with open(candidate, "w") as f:
            f.write("def solve():\n    return 42\n" + "# padding\n" * 20)
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.99}',
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Check that cross-run memory has a successful strategy
        memory_dir = os.path.join(self.state_dir, "cross_run_memory")
        strategies_path = os.path.join(memory_dir, "strategies.json")
        self.assertTrue(os.path.exists(strategies_path),
                        "strategies.json should exist after a successful submit")
        with open(strategies_path) as f:
            strategies = json.load(f)
        self.assertGreater(len(strategies), 0,
                           "At least one successful strategy should be recorded")

    def test_submit_memory_includes_run_id(self):
        """Memory entries should include the run_id from metadata."""
        # Init + Next + Submit
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(candidate, "w") as f:
            f.write("x = 1")
        self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.01}',
        ])
        # Read the run_id from metadata
        with open(os.path.join(self.state_dir, "metadata.json")) as f:
            run_id = json.load(f)["run_id"]
        # Read the memory
        memory_dir = os.path.join(self.state_dir, "cross_run_memory")
        failed_path = os.path.join(memory_dir, "failed_approaches.json")
        with open(failed_path) as f:
            failed = json.load(f)
        self.assertTrue(
            any(fa.get("run_id") == run_id for fa in failed),
            f"Failed approach should contain run_id={run_id}",
        )


class TestCliStrategyOutcome(unittest.TestCase):
    """Tests for strategy outcome recording in submit."""

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

    def test_submit_records_strategy_outcome(self):
        """Strategy outcome should be recorded with score delta after submit."""
        # Init + Next (creates strategies.json and current_iteration.json)
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        # Read the manifest to know which strategy was selected
        with open(os.path.join(self.state_dir, "current_iteration.json")) as f:
            manifest = json.load(f)
        strategy_id = manifest["selected_strategy_id"]
        parent_score = manifest["parent_score"]

        # Read strategies before submit
        strategies_path = os.path.join(self.state_dir, "strategies.json")
        with open(strategies_path) as f:
            strategies_before = json.load(f)
        strategy_before = next(
            (s for s in strategies_before if s["id"] == strategy_id), None
        )
        times_used_before = strategy_before["times_used"] if strategy_before else 0

        # Submit with a known score
        submit_score = 0.75
        candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(candidate, "w") as f:
            f.write("def solve():\n    return 42\n")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", json.dumps({"combined_score": submit_score}),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        # Read strategies after submit
        with open(strategies_path) as f:
            strategies_after = json.load(f)
        strategy_after = next(
            (s for s in strategies_after if s["id"] == strategy_id), None
        )
        self.assertIsNotNone(strategy_after,
                             f"Strategy {strategy_id} should still exist")
        self.assertEqual(strategy_after["times_used"], times_used_before + 1,
                         "times_used should be incremented by 1")
        # The score delta (score - parent_score) should be in score_history
        expected_delta = submit_score - parent_score
        self.assertIn(expected_delta, strategy_after["score_history"],
                      f"score_history should contain delta {expected_delta}")

    def test_submit_records_negative_delta_for_worse_score(self):
        """A submit with score < parent should record a negative delta."""
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        with open(os.path.join(self.state_dir, "current_iteration.json")) as f:
            manifest = json.load(f)
        strategy_id = manifest["selected_strategy_id"]
        parent_score = manifest["parent_score"]

        # Submit with a score lower than parent
        submit_score = 0.001
        candidate = os.path.join(self.tmpdir, "bad_candidate.py")
        with open(candidate, "w") as f:
            f.write("# bad")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", json.dumps({"combined_score": submit_score}),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        strategies_path = os.path.join(self.state_dir, "strategies.json")
        with open(strategies_path) as f:
            strategies = json.load(f)
        strategy = next((s for s in strategies if s["id"] == strategy_id), None)
        self.assertIsNotNone(strategy)
        expected_delta = submit_score - parent_score
        self.assertIn(expected_delta, strategy["score_history"])
        # Delta should be negative (or at most zero)
        self.assertLessEqual(expected_delta, 0.0)

    def test_submit_without_manifest_still_succeeds(self):
        """Submit should not crash if current_iteration.json is missing."""
        self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "10",
        ])
        # Skip next (no manifest written) -- submit directly
        candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(candidate, "w") as f:
            f.write("x = 1")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.5}',
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)


class TestCliResearchTrigger(unittest.TestCase):
    """Tests for research trigger gating in the next command."""

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

    def _init_with_research(self, trigger="on_stagnation"):
        """Helper: init with research enabled and a given trigger."""
        config_file = os.path.join(self.tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(
                f"max_iterations: 100\n"
                f"research:\n"
                f"  enabled: true\n"
                f"  trigger: {trigger}\n"
                f"  research_log_file: research_log.json\n"
            )
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--config", config_file,
        ])
        assert result.exit_code == 0, result.output

    def _populate_research_log(self):
        """Populate a research log with a finding so format_for_prompt() returns text."""
        import time
        log_path = os.path.join(self.state_dir, "research_log.json")
        log_data = {
            "findings": [
                {
                    "id": "test-finding-1",
                    "iteration": 1,
                    "timestamp": time.time(),
                    "approach_name": "Test Approach XYZ",
                    "description": "A test research approach for verification.",
                    "novelty": "high",
                    "implementation_hint": "Use technique XYZ.",
                    "source_url": "",
                    "was_tried": False,
                    "outcome_score": None,
                    "metadata": {},
                }
            ],
            "theoretical_bounds": {},
            "key_papers": [],
            "approaches_to_avoid": [],
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f)

    def test_next_excludes_research_when_no_stagnation(self):
        """Research text should NOT appear when trigger is on_stagnation and stagnation is none."""
        self._init_with_research(trigger="on_stagnation")
        self._populate_research_log()

        result = self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # The research finding text should NOT appear in the output
        # because stagnation is "none" and trigger is "on_stagnation"
        self.assertNotIn("Test Approach XYZ", result.output,
                         "Research findings should be excluded when no stagnation")

    def test_next_includes_research_when_trigger_always(self):
        """Research text should appear when trigger is 'always'."""
        self._init_with_research(trigger="always")
        self._populate_research_log()

        result = self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # The research finding text should appear in the output
        self.assertIn("Test Approach XYZ", result.output,
                      "Research findings should be included when trigger is 'always'")

    def test_next_excludes_research_when_trigger_never(self):
        """Research text should NOT appear when trigger is 'never'."""
        self._init_with_research(trigger="never")
        self._populate_research_log()

        result = self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertNotIn("Test Approach XYZ", result.output,
                         "Research findings should be excluded when trigger is 'never'")

    def test_next_includes_research_on_periodic_at_interval(self):
        """Research text should appear on periodic trigger at the right interval."""
        config_file = os.path.join(self.tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(
                "max_iterations: 100\n"
                "research:\n"
                "  enabled: true\n"
                "  trigger: periodic\n"
                "  periodic_interval: 1\n"
                "  research_log_file: research_log.json\n"
            )
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--config", config_file,
        ])
        assert result.exit_code == 0, result.output
        self._populate_research_log()

        result = self.runner.invoke(main, [
            "next", "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # periodic_interval=1 means every iteration should trigger
        self.assertIn("Test Approach XYZ", result.output,
                      "Research findings should be included for periodic trigger")


if __name__ == "__main__":
    unittest.main()
