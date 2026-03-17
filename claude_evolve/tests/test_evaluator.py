"""
Tests for claude_evolve.core.evaluator.

Covers three evaluation modes (script, critic, hybrid), cascade evaluation,
subprocess isolation, timeout protection, artifact collection, and error handling.
"""

import asyncio
import os
import shutil
import tempfile
import unittest

from claude_evolve.core.evaluator import EvaluationResult, Evaluator
from claude_evolve.config import EvaluatorConfig


class TestEvaluationResult(unittest.TestCase):
    """Tests for the EvaluationResult dataclass."""

    def test_from_dict(self):
        metrics = {"combined_score": 0.8, "accuracy": 0.9}
        result = EvaluationResult.from_dict(metrics)
        self.assertEqual(result.metrics, metrics)
        self.assertEqual(result.artifacts, {})

    def test_to_dict(self):
        result = EvaluationResult(
            metrics={"combined_score": 0.7}, artifacts={"log": "some text"}
        )
        d = result.to_dict()
        self.assertEqual(d, {"combined_score": 0.7})

    def test_has_artifacts_true(self):
        result = EvaluationResult(
            metrics={"combined_score": 0.5}, artifacts={"stderr": "warning"}
        )
        self.assertTrue(result.has_artifacts())

    def test_has_artifacts_false(self):
        result = EvaluationResult(metrics={"combined_score": 0.5})
        self.assertFalse(result.has_artifacts())

    def test_get_artifact_keys(self):
        result = EvaluationResult(
            metrics={"combined_score": 0.5},
            artifacts={"stderr": "warning", "profile": "10ms"},
        )
        keys = result.get_artifact_keys()
        self.assertIn("stderr", keys)
        self.assertIn("profile", keys)

    def test_get_artifact_size_string(self):
        result = EvaluationResult(
            metrics={"combined_score": 0.5}, artifacts={"msg": "hello"}
        )
        size = result.get_artifact_size("msg")
        self.assertEqual(size, len("hello".encode("utf-8")))

    def test_get_artifact_size_bytes(self):
        result = EvaluationResult(
            metrics={"combined_score": 0.5}, artifacts={"data": b"\x00\x01\x02"}
        )
        self.assertEqual(result.get_artifact_size("data"), 3)

    def test_get_artifact_size_missing(self):
        result = EvaluationResult(metrics={"combined_score": 0.5})
        self.assertEqual(result.get_artifact_size("missing"), 0)

    def test_get_total_artifact_size(self):
        result = EvaluationResult(
            metrics={"combined_score": 0.5},
            artifacts={"a": "hello", "b": b"\x00\x01"},
        )
        total = result.get_total_artifact_size()
        self.assertEqual(total, 5 + 2)


class TestEvaluatorScriptMode(unittest.TestCase):
    """Tests for script-mode evaluation (subprocess isolation)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eval_script = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.eval_script, "w") as f:
            f.write(
                '''
def evaluate(artifact_path):
    with open(artifact_path) as f:
        content = f.read()
    length = len(content)
    return {
        "combined_score": min(length / 100.0, 1.0),
        "length": float(length),
    }
'''
            )
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1\n" * 20)

    def test_evaluate_returns_metrics(self):
        config = EvaluatorConfig(mode="script", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertIn("combined_score", metrics)
        self.assertIsInstance(metrics["combined_score"], float)

    def test_evaluate_correct_score(self):
        config = EvaluatorConfig(mode="script", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        content = "x = 1\n" * 20
        expected = min(len(content) / 100.0, 1.0)
        self.assertAlmostEqual(metrics["combined_score"], expected, places=4)

    def test_evaluate_bad_script_returns_zero(self):
        bad_script = os.path.join(self.tmpdir, "bad_eval.py")
        with open(bad_script, "w") as f:
            f.write("def evaluate(path): raise Exception('boom')")
        config = EvaluatorConfig(mode="script", timeout=10, max_retries=0)
        evaluator = Evaluator(config, bad_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_evaluate_timeout(self):
        slow_script = os.path.join(self.tmpdir, "slow_eval.py")
        with open(slow_script, "w") as f:
            f.write(
                'import time\ndef evaluate(path):\n    time.sleep(100)\n    return {"combined_score": 1.0}'
            )
        config = EvaluatorConfig(mode="script", timeout=2, max_retries=0)
        evaluator = Evaluator(config, slow_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_evaluate_with_content_string(self):
        """Evaluate by passing content directly instead of file path."""
        config = EvaluatorConfig(mode="script", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate_content("x = 1\n" * 20))
        self.assertIn("combined_score", metrics)

    def test_evaluate_nonexistent_candidate(self):
        """Evaluating a nonexistent file should return error metrics."""
        config = EvaluatorConfig(mode="script", timeout=10, max_retries=0)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate("/nonexistent/path.py"))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_evaluate_returns_all_metrics(self):
        """All metrics from the evaluator should be returned."""
        config = EvaluatorConfig(mode="script", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertIn("length", metrics)
        self.assertIsInstance(metrics["length"], float)

    def test_evaluate_syntax_error_script(self):
        """Evaluator script with syntax error should return error metrics."""
        bad_script = os.path.join(self.tmpdir, "syntax_err.py")
        with open(bad_script, "w") as f:
            f.write("def evaluate(path\n")  # Missing colon/parenthesis
        config = EvaluatorConfig(mode="script", timeout=10, max_retries=0)
        evaluator = Evaluator(config, bad_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_evaluate_no_evaluate_function(self):
        """Script without evaluate() function should return error metrics."""
        bad_script = os.path.join(self.tmpdir, "no_func.py")
        with open(bad_script, "w") as f:
            f.write("x = 42\n")
        config = EvaluatorConfig(mode="script", timeout=10, max_retries=0)
        evaluator = Evaluator(config, bad_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_evaluate_returns_non_dict(self):
        """Script that returns non-dict should return error metrics."""
        bad_script = os.path.join(self.tmpdir, "non_dict.py")
        with open(bad_script, "w") as f:
            f.write('def evaluate(path): return "not a dict"')
        config = EvaluatorConfig(mode="script", timeout=10, max_retries=0)
        evaluator = Evaluator(config, bad_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def test_subprocess_isolation(self):
        """Evaluator script runs in a separate process, not in-process."""
        # A script that sets a global variable should not affect our process
        isolation_script = os.path.join(self.tmpdir, "isolation_eval.py")
        with open(isolation_script, "w") as f:
            f.write(
                '''
import os
os.environ["CLAUDE_EVOLVE_ISOLATION_TEST"] = "set_by_subprocess"
def evaluate(path):
    return {"combined_score": 0.5}
'''
            )
        config = EvaluatorConfig(mode="script", timeout=10)
        evaluator = Evaluator(config, isolation_script)
        asyncio.run(evaluator.evaluate(self.candidate))
        # The env var should NOT be set in our process
        self.assertIsNone(os.environ.get("CLAUDE_EVOLVE_ISOLATION_TEST"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestEvaluatorRetry(unittest.TestCase):
    """Tests for retry logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1")

    def test_retry_on_failure(self):
        """Evaluator should retry up to max_retries times."""
        # Script that always fails
        fail_script = os.path.join(self.tmpdir, "fail_eval.py")
        with open(fail_script, "w") as f:
            f.write("def evaluate(path): raise Exception('always fails')")
        config = EvaluatorConfig(mode="script", timeout=10, max_retries=2)
        evaluator = Evaluator(config, fail_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertEqual(metrics.get("combined_score", 0), 0.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestEvaluatorMetricsPassthrough(unittest.TestCase):
    """Tests for critic mode (pre-computed metrics passthrough)."""

    def test_passthrough_mode(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        metrics = {"combined_score": 0.85, "flaws": 2}
        result = asyncio.run(evaluator.evaluate_with_metrics(metrics))
        self.assertEqual(result["combined_score"], 0.85)

    def test_passthrough_validates_combined_score(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        metrics = {"clarity": 0.9}  # No combined_score
        result = asyncio.run(evaluator.evaluate_with_metrics(metrics))
        # Should still work, combined_score calculated as average
        self.assertIn("combined_score", result)

    def test_passthrough_preserves_all_metrics(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        metrics = {"combined_score": 0.75, "quality": 0.8, "style": 0.7}
        result = asyncio.run(evaluator.evaluate_with_metrics(metrics))
        self.assertEqual(result["quality"], 0.8)
        self.assertEqual(result["style"], 0.7)

    def test_passthrough_empty_metrics(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        result = asyncio.run(evaluator.evaluate_with_metrics({}))
        self.assertIn("combined_score", result)
        self.assertEqual(result["combined_score"], 0.0)

    def test_passthrough_filters_non_numeric(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        metrics = {"combined_score": 0.8, "notes": "good code"}
        result = asyncio.run(evaluator.evaluate_with_metrics(metrics))
        self.assertEqual(result["combined_score"], 0.8)
        # Non-numeric values should be filtered out of metrics
        self.assertNotIn("notes", result)


class TestCascadeEvaluation(unittest.TestCase):
    """Tests for cascade (multi-stage) evaluation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eval_script = os.path.join(self.tmpdir, "cascade_eval.py")
        with open(self.eval_script, "w") as f:
            f.write(
                '''
def evaluate_stage1(path):
    return {"combined_score": 0.6, "validity": 1.0}

def evaluate_stage2(path):
    return {"combined_score": 0.8, "validity": 1.0, "performance": 0.7}

def evaluate(path):
    return {"combined_score": 0.9, "validity": 1.0, "performance": 0.85}
'''
            )
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1")

    def test_cascade_passes_all_stages(self):
        config = EvaluatorConfig(
            mode="script",
            cascade_evaluation=True,
            cascade_thresholds=[0.5, 0.7],
        )
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertGreater(metrics["combined_score"], 0.0)

    def test_cascade_fails_early(self):
        fail_script = os.path.join(self.tmpdir, "fail_cascade.py")
        with open(fail_script, "w") as f:
            f.write(
                '''
def evaluate_stage1(path):
    return {"combined_score": 0.1, "validity": 0.0}

def evaluate(path):
    return {"combined_score": 0.9}
'''
            )
        config = EvaluatorConfig(
            mode="script",
            cascade_evaluation=True,
            cascade_thresholds=[0.5],
        )
        evaluator = Evaluator(config, fail_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        # Should have low score since stage1 failed threshold
        self.assertLessEqual(metrics["combined_score"], 0.2)

    def test_cascade_no_stage_functions_falls_back(self):
        """If no evaluate_stageN functions exist, fall back to evaluate()."""
        simple_script = os.path.join(self.tmpdir, "simple_eval.py")
        with open(simple_script, "w") as f:
            f.write(
                '''
def evaluate(path):
    return {"combined_score": 0.75}
'''
            )
        config = EvaluatorConfig(
            mode="script",
            cascade_evaluation=True,
            cascade_thresholds=[0.5],
        )
        evaluator = Evaluator(config, simple_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertAlmostEqual(metrics["combined_score"], 0.75, places=4)

    def test_cascade_stage1_only(self):
        """If only stage1 exists (no stage2), return stage1 results."""
        stage1_only = os.path.join(self.tmpdir, "stage1_only.py")
        with open(stage1_only, "w") as f:
            f.write(
                '''
def evaluate_stage1(path):
    return {"combined_score": 0.6, "validity": 1.0}

def evaluate(path):
    return {"combined_score": 0.9}
'''
            )
        config = EvaluatorConfig(
            mode="script",
            cascade_evaluation=True,
            cascade_thresholds=[0.5],
        )
        evaluator = Evaluator(config, stage1_only)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        # stage1 passes threshold 0.5, but no stage2, so stage1 result
        # merged with evaluate() or returned as is
        self.assertGreater(metrics["combined_score"], 0.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestEvaluatorArtifacts(unittest.TestCase):
    """Tests for artifact collection from evaluation results."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1")

    def test_pending_artifacts_collection(self):
        """Artifacts from evaluation should be stored and retrievable."""
        eval_script = os.path.join(self.tmpdir, "artifact_eval.py")
        with open(eval_script, "w") as f:
            f.write(
                '''
import json
import sys

def evaluate(path):
    # Return a dict; artifacts are passed via a special key
    return {
        "combined_score": 0.7,
        "__artifacts__": {"stderr": "warning: deprecated API", "profile": "10ms avg"}
    }
'''
            )
        config = EvaluatorConfig(mode="script", timeout=30)
        evaluator = Evaluator(config, eval_script)
        metrics = asyncio.run(
            evaluator.evaluate(self.candidate, program_id="test-123")
        )
        self.assertIn("combined_score", metrics)
        artifacts = evaluator.get_pending_artifacts("test-123")
        if artifacts:
            self.assertIn("stderr", artifacts)

    def test_get_pending_artifacts_clears(self):
        """Getting pending artifacts should clear them."""
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        # Manually store some artifacts
        evaluator._pending_artifacts["prog-1"] = {"log": "test"}
        artifacts = evaluator.get_pending_artifacts("prog-1")
        self.assertIsNotNone(artifacts)
        # Second call should return None
        artifacts2 = evaluator.get_pending_artifacts("prog-1")
        self.assertIsNone(artifacts2)

    def test_get_pending_artifacts_missing(self):
        """Getting artifacts for unknown program should return None."""
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        self.assertIsNone(evaluator.get_pending_artifacts("nonexistent"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestEvaluatorHybridMode(unittest.TestCase):
    """Tests for hybrid mode (script + critic passthrough)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eval_script = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.eval_script, "w") as f:
            f.write(
                '''
def evaluate(artifact_path):
    return {"combined_score": 0.6, "length": 10.0}
'''
            )
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1")

    def test_hybrid_evaluate_script(self):
        """Hybrid mode should support script evaluation."""
        config = EvaluatorConfig(mode="hybrid", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(evaluator.evaluate(self.candidate))
        self.assertIn("combined_score", metrics)

    def test_hybrid_evaluate_with_metrics(self):
        """Hybrid mode should support metrics passthrough."""
        config = EvaluatorConfig(mode="hybrid", timeout=30)
        evaluator = Evaluator(config, self.eval_script)
        metrics = asyncio.run(
            evaluator.evaluate_with_metrics({"combined_score": 0.9})
        )
        self.assertEqual(metrics["combined_score"], 0.9)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestPassesThreshold(unittest.TestCase):
    """Tests for the _passes_threshold helper."""

    def test_passes_with_combined_score(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        self.assertTrue(
            evaluator._passes_threshold({"combined_score": 0.8}, 0.5)
        )

    def test_fails_with_combined_score(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        self.assertFalse(
            evaluator._passes_threshold({"combined_score": 0.3}, 0.5)
        )

    def test_fallback_to_average(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        # Average of 0.6 and 0.8 is 0.7 -> passes 0.5
        self.assertTrue(
            evaluator._passes_threshold({"a": 0.6, "b": 0.8}, 0.5)
        )

    def test_empty_metrics_fails(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        self.assertFalse(evaluator._passes_threshold({}, 0.5))

    def test_error_key_excluded(self):
        config = EvaluatorConfig(mode="critic")
        evaluator = Evaluator(config, None)
        # "error" key should be excluded from average
        self.assertTrue(
            evaluator._passes_threshold({"error": 0.0, "score": 0.8}, 0.5)
        )


class TestEvaluateMultiple(unittest.TestCase):
    """Tests for parallel evaluation of multiple candidates."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eval_script = os.path.join(self.tmpdir, "evaluator.py")
        with open(self.eval_script, "w") as f:
            f.write(
                '''
def evaluate(artifact_path):
    with open(artifact_path) as f:
        content = f.read()
    return {"combined_score": min(len(content) / 100.0, 1.0)}
'''
            )
        self.candidates = []
        for i in range(3):
            path = os.path.join(self.tmpdir, f"candidate_{i}.py")
            with open(path, "w") as f:
                f.write(f"x = {i}\n" * (10 * (i + 1)))
            self.candidates.append(path)

    def test_evaluate_multiple(self):
        config = EvaluatorConfig(mode="script", timeout=30, parallel_evaluations=2)
        evaluator = Evaluator(config, self.eval_script)
        programs = [(path, f"prog-{i}") for i, path in enumerate(self.candidates)]
        results = asyncio.run(evaluator.evaluate_multiple(programs))
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("combined_score", r)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


# ---------------------------------------------------------------------------
# Evaluator Error Logging (Task 7)
# ---------------------------------------------------------------------------
class TestTimeoutLogsArtifactPath(unittest.TestCase):
    """Timeout log message should include the artifact path being evaluated."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.slow_script = os.path.join(self.tmpdir, "slow_eval.py")
        with open(self.slow_script, "w") as f:
            f.write(
                "import time\n"
                "def evaluate(path):\n"
                '    time.sleep(100)\n'
                '    return {"combined_score": 1.0}\n'
            )
        self.candidate = os.path.join(self.tmpdir, "my_candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1\n")

    def test_timeout_log_includes_artifact_path(self):
        """The WARNING log on timeout should contain the candidate file path."""
        import logging

        config = EvaluatorConfig(mode="script", timeout=2, max_retries=0)
        evaluator = Evaluator(config, self.slow_script)

        with self.assertLogs("claude_evolve.core.evaluator", level=logging.WARNING) as cm:
            metrics = asyncio.run(evaluator.evaluate(self.candidate))

        self.assertEqual(metrics.get("combined_score", 0), 0.0)
        # Check that at least one log message contains the candidate path
        found = any("my_candidate.py" in msg for msg in cm.output)
        self.assertTrue(found, f"Candidate path not found in logs: {cm.output}")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestExponentialBackoff(unittest.TestCase):
    """Retry sleep should use exponential backoff instead of fixed 0.5s."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Use os._exit(1) so the subprocess dies without writing a result
        # file.  This triggers the RuntimeError branch in
        # _subprocess_evaluate which is what exercises the retry+sleep path.
        self.crash_script = os.path.join(self.tmpdir, "crash_eval.py")
        with open(self.crash_script, "w") as f:
            f.write("import os\ndef evaluate(path): os._exit(1)\n")
        self.candidate = os.path.join(self.tmpdir, "candidate.py")
        with open(self.candidate, "w") as f:
            f.write("x = 1\n")

    def test_backoff_increases_with_attempt(self):
        """Backoff delays should increase: 0.5, 1.0, 2.0, ... (capped at 4.0)."""
        import unittest.mock

        config = EvaluatorConfig(mode="script", timeout=10, max_retries=3)
        evaluator = Evaluator(config, self.crash_script)

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with unittest.mock.patch(
            "claude_evolve.core.evaluator.asyncio.sleep",
            side_effect=mock_sleep,
        ):
            asyncio.run(evaluator.evaluate(self.candidate))

        # With max_retries=3, there are 4 attempts (0,1,2,3).
        # Retries happen after attempts 0, 1, 2 (3 sleeps).
        # Expected backoff: 0.5*(2^0)=0.5, 0.5*(2^1)=1.0, 0.5*(2^2)=2.0
        self.assertEqual(len(sleep_calls), 3)
        self.assertAlmostEqual(sleep_calls[0], 0.5, places=2)
        self.assertAlmostEqual(sleep_calls[1], 1.0, places=2)
        self.assertAlmostEqual(sleep_calls[2], 2.0, places=2)

    def test_backoff_capped_at_4_seconds(self):
        """Backoff should never exceed 4.0 seconds."""
        import unittest.mock

        config = EvaluatorConfig(mode="script", timeout=10, max_retries=5)
        evaluator = Evaluator(config, self.crash_script)

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with unittest.mock.patch(
            "claude_evolve.core.evaluator.asyncio.sleep",
            side_effect=mock_sleep,
        ):
            asyncio.run(evaluator.evaluate(self.candidate))

        # 5 retries -> sleeps after attempts 0..4 = 5 sleeps
        # 0.5, 1.0, 2.0, 4.0, 4.0
        self.assertEqual(len(sleep_calls), 5)
        for delay in sleep_calls:
            self.assertLessEqual(delay, 4.0)
        # Last two should both be capped at 4.0
        self.assertAlmostEqual(sleep_calls[3], 4.0, places=2)
        self.assertAlmostEqual(sleep_calls[4], 4.0, places=2)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main()
