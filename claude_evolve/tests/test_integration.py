# tests/test_integration.py
"""End-to-end test: init -> next -> submit -> next -> submit -> export"""
import json
import os
import shutil
import tempfile
import unittest
from click.testing import CliRunner
from claude_evolve.cli import main


class TestFullEvolutionCycle(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()

        # Initial program
        self.artifact = os.path.join(self.tmpdir, "program.py")
        with open(self.artifact, "w") as f:
            f.write('''
def pack_circles(n=10):
    import random
    circles = []
    for i in range(n):
        x = random.random()
        y = random.random()
        r = 0.01
        circles.append((x, y, r))
    return circles
''')

        # Evaluator
        self.evaluator = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.evaluator, "w") as f:
            f.write('''
def evaluate(artifact_path):
    with open(artifact_path) as f:
        content = f.read()
    has_optimization = "optimize" in content.lower() or "numpy" in content.lower()
    base_score = 0.3
    if has_optimization:
        base_score = 0.7
    if "grid" in content.lower():
        base_score = 0.9
    return {
        "combined_score": base_score,
        "code_quality": min(len(content) / 500.0, 1.0),
    }
''')

        self.state_dir = os.path.join(self.tmpdir, "state")

    def test_full_cycle(self):
        # 1. Init
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--max-iterations", "5",
        ])
        self.assertEqual(result.exit_code, 0, msg=f"Init failed: {result.output}")
        init_output = json.loads(result.output.strip())
        self.assertEqual(init_output["status"], "initialized")

        # 2. First next
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0, msg=f"Next failed: {result.output}")
        context_file = os.path.join(self.state_dir, "iteration_context.md")
        self.assertTrue(os.path.exists(context_file))

        # 3. Simulate Claude producing a better candidate
        candidate1 = os.path.join(self.tmpdir, "candidate1.py")
        with open(candidate1, "w") as f:
            f.write('''
import numpy as np
def pack_circles(n=10):
    # Optimized version using grid placement
    circles = []
    grid_size = int(np.ceil(np.sqrt(n)))
    r = 0.4 / grid_size
    for i in range(n):
        x = (i % grid_size + 0.5) / grid_size
        y = (i // grid_size + 0.5) / grid_size
        circles.append((x, y, r))
    return circles
''')

        # 4. Submit candidate 1
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate1, "--state-dir", self.state_dir
        ])
        self.assertEqual(result.exit_code, 0, msg=f"Submit failed: {result.output}")
        metrics1 = json.loads(result.output)
        self.assertGreater(metrics1["combined_score"], 0.3)

        # 5. Second next
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0)

        # 6. Submit candidate 2 with critic metrics
        candidate2 = os.path.join(self.tmpdir, "candidate2.py")
        with open(candidate2, "w") as f:
            f.write("# grid-based optimal solution\ndef pack_circles(): pass")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate2, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.95, "elegance": 0.8}',
        ])
        self.assertEqual(result.exit_code, 0)
        metrics2 = json.loads(result.output)
        self.assertAlmostEqual(metrics2["combined_score"], 0.95)

        # 7. Status check
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0)
        status = json.loads(result.output)
        self.assertGreaterEqual(status["best_score"], 0.9)
        self.assertGreaterEqual(status["population_size"], 3)

        # 8. Export best
        best_output = os.path.join(self.tmpdir, "best.py")
        result = self.runner.invoke(main, [
            "export", "--state-dir", self.state_dir, "--output", best_output
        ])
        self.assertEqual(result.exit_code, 0)
        with open(best_output) as f:
            best_code = f.read()
        self.assertTrue(len(best_code) > 0)

    def test_multiple_submit_improving_scores(self):
        """Test that submitting multiple increasingly better candidates works"""
        # Init
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir,
        ])
        self.assertEqual(result.exit_code, 0)

        scores = []
        for i in range(5):
            # Next
            result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
            self.assertEqual(result.exit_code, 0)

            # Submit with increasing scores (each candidate is sufficiently distinct
            # to pass the novelty gate)
            algorithms = ["bubble_sort", "merge_sort", "quick_sort", "heap_sort", "radix_sort"]
            candidate = os.path.join(self.tmpdir, f"cand_{i}.py")
            with open(candidate, "w") as f:
                f.write(f"# Approach: {algorithms[i]} implementation v{i}\n"
                        f"import {['os','sys','math','json','re'][i]}\n"
                        f"def {algorithms[i]}(data):\n"
                        f"    return sorted(data, key=lambda x: x ** {i+1})\n")
            result = self.runner.invoke(main, [
                "submit", "--candidate", candidate, "--state-dir", self.state_dir,
                "--metrics", json.dumps({"combined_score": 0.5 + i * 0.1}),
            ])
            self.assertEqual(result.exit_code, 0)
            output = json.loads(result.output)
            scores.append(output["combined_score"])

        # Best score should be the last one
        result = self.runner.invoke(main, ["status", "--state-dir", self.state_dir])
        status = json.loads(result.output)
        self.assertAlmostEqual(status["best_score"], 0.9)

    def test_critic_mode_full_cycle(self):
        """Test full cycle using critic mode (no script evaluator for candidates)"""
        result = self.runner.invoke(main, [
            "init", "--artifact", self.artifact, "--evaluator", self.evaluator,
            "--state-dir", self.state_dir, "--mode", "critic",
        ])
        self.assertEqual(result.exit_code, 0)

        # Next
        result = self.runner.invoke(main, ["next", "--state-dir", self.state_dir])
        self.assertEqual(result.exit_code, 0)

        # Submit with critic metrics
        candidate = os.path.join(self.tmpdir, "critic_cand.py")
        with open(candidate, "w") as f:
            f.write("# critic-evaluated candidate")
        result = self.runner.invoke(main, [
            "submit", "--candidate", candidate, "--state-dir", self.state_dir,
            "--metrics", '{"combined_score": 0.88, "coherence": 0.92}',
        ])
        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)
        self.assertAlmostEqual(output["combined_score"], 0.88)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
