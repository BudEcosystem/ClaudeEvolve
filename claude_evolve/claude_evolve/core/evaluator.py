"""
Evaluator for claude_evolve.

Runs user-provided evaluation scripts in isolated subprocesses.
Supports three modes:
  - script:  Run evaluator.py in a subprocess with timeout protection.
  - critic:  Accept pre-computed metrics from Claude's critic agent.
  - hybrid:  Supports both script evaluation and metrics passthrough.

Subprocess isolation is critical for security -- evaluation code never
executes in the main process.

Extracted from OpenEvolve's evaluator.py and adapted for Claude Evolve:
- Removed LLM-based evaluation (_llm_evaluate) since Claude IS the LLM.
- Added evaluate_with_metrics() for critic mode passthrough.
- All script evaluation uses subprocess (not in-process importlib).
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from claude_evolve.config import EvaluatorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts.

    Maintains backward compatibility with the existing dict[str, float] contract
    while adding a side-channel for arbitrary artifacts (text or binary data).

    For custom MAP-Elites features, metric values must be raw continuous scores
    (e.g., actual counts, percentages, continuous measurements), NOT pre-computed
    bin indices.  The database handles all binning internally.
    """

    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility."""
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility -- return just metrics."""
        return self.metrics

    def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts."""
        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:
        """Get list of artifact keys."""
        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes."""
        if key not in self.artifacts:
            return 0
        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes."""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())


# ---------------------------------------------------------------------------
# Runner script template (executed in subprocess)
# ---------------------------------------------------------------------------

_RUNNER_SCRIPT_TEMPLATE = textwrap.dedent(
    r'''
import importlib.util
import json
import os
import sys
import traceback

def main():
    eval_file = sys.argv[1]
    artifact_path = sys.argv[2]
    result_path = sys.argv[3]
    function_name = sys.argv[4] if len(sys.argv) > 4 else "evaluate"

    try:
        # Add the evaluator's directory to sys.path for local imports
        eval_dir = os.path.dirname(os.path.abspath(eval_file))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        spec = importlib.util.spec_from_file_location("_eval_module", eval_file)
        if spec is None or spec.loader is None:
            result = {"error": 0.0, "__error_msg__": f"Cannot load spec from {eval_file}"}
            with open(result_path, "w") as f:
                json.dump(result, f)
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            result = {"error": 0.0, "__error_msg__": f"No '{function_name}' function in {eval_file}"}
            with open(result_path, "w") as f:
                json.dump(result, f)
            return

        func = getattr(module, function_name)
        raw_result = func(artifact_path)

        # Normalize result to a JSON-serializable dict
        if isinstance(raw_result, dict):
            # Extract artifacts if present (special __artifacts__ key)
            artifacts = raw_result.pop("__artifacts__", None)
            result = {}
            for k, v in raw_result.items():
                if isinstance(v, (int, float)):
                    result[k] = float(v)
                elif isinstance(v, bool):
                    result[k] = 1.0 if v else 0.0
            if artifacts and isinstance(artifacts, dict):
                result["__artifacts__"] = {}
                for ak, av in artifacts.items():
                    result["__artifacts__"][ak] = str(av)
        else:
            result = {"error": 0.0, "__error_msg__": f"evaluate() returned {type(raw_result).__name__}, expected dict"}

        with open(result_path, "w") as f:
            json.dump(result, f)

    except Exception as e:
        result = {"error": 0.0, "__error_msg__": f"{type(e).__name__}: {e}"}
        try:
            with open(result_path, "w") as f:
                json.dump(result, f)
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """
    Evaluates candidate artifacts and assigns scores.

    The evaluator runs user-provided evaluation scripts in isolated
    subprocesses for security and reliability.  Three modes are supported:

    - script:  Run an external evaluator script.
    - critic:  Accept pre-computed metrics (from Claude's critic agent).
    - hybrid:  Supports both script and metrics passthrough.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: Optional[str],
        suffix: str = ".py",
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.program_suffix = suffix

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        # Validate configuration
        if config.mode == "script" and evaluation_file is None:
            raise ValueError("evaluation_file is required for script mode")
        if config.mode == "script" and not os.path.exists(evaluation_file):
            raise FileNotFoundError(
                f"Evaluation file not found: {evaluation_file}"
            )

        # Write the runner script once to a temp location for reuse
        self._runner_script_path: Optional[str] = None
        if config.mode in ("script", "hybrid") and evaluation_file is not None:
            self._runner_script_path = self._create_runner_script()

        logger.info(
            f"Initialized evaluator mode={config.mode} "
            f"eval_file={evaluation_file}"
        )

    def _create_runner_script(self) -> str:
        """Write the subprocess runner script to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".py", prefix="claude_evolve_runner_")
        with os.fdopen(fd, "w") as f:
            f.write(_RUNNER_SCRIPT_TEMPLATE)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        artifact_path: str,
        program_id: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate an artifact file and return metrics.

        The evaluation script is run in an isolated subprocess with timeout
        protection.  For cascade evaluation, stages are run sequentially;
        a candidate must pass each stage's threshold to continue.

        Args:
            artifact_path: Path to the candidate artifact file.
            program_id:    Optional ID for logging and artifact tracking.

        Returns:
            Dictionary mapping metric names to float scores.
        """
        if self.config.mode == "critic":
            logger.warning(
                "evaluate() called in critic mode; use evaluate_with_metrics()"
            )
            return {"error": 0.0}

        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        last_exception: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if self.config.cascade_evaluation:
                    result = await self._cascade_evaluate(artifact_path)
                else:
                    result = await self._subprocess_evaluate(
                        artifact_path, "evaluate"
                    )

                eval_result = self._process_evaluation_result(result)

                # Store artifacts if present
                if program_id and eval_result.has_artifacts():
                    self._pending_artifacts[program_id] = dict(eval_result.artifacts)

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated{program_id_str} in {elapsed:.2f}s: "
                    f"{eval_result.metrics}"
                )
                return eval_result.metrics

            except asyncio.TimeoutError:
                logger.warning(
                    f"Evaluation timed out after {self.config.timeout}s"
                )
                if program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": "true",
                        "timeout_duration": str(self.config.timeout),
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }
                return {"error": 0.0, "combined_score": 0.0}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} "
                    f"failed{program_id_str}: {e}"
                )
                if program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": str(attempt + 1),
                    }
                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.5)

        logger.error(
            f"All evaluation attempts failed{program_id_str}. "
            f"Last error: {last_exception}"
        )
        return {"error": 0.0, "combined_score": 0.0}

    async def evaluate_content(
        self,
        content: str,
        program_id: str = "",
        suffix: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate raw content by writing it to a temp file first.

        Convenience wrapper around evaluate() for when you have the artifact
        content as a string rather than a file path.

        Args:
            content:    The artifact content to evaluate.
            program_id: Optional ID for logging and artifact tracking.
            suffix:     File suffix (defaults to self.program_suffix).

        Returns:
            Dictionary mapping metric names to float scores.
        """
        ext = suffix or self.program_suffix
        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="claude_evolve_artifact_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            return await self.evaluate(tmp_path, program_id=program_id)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def evaluate_with_metrics(
        self,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Accept pre-computed metrics from Claude's critic agent.

        For critic and hybrid modes.  Validates and normalizes the metrics,
        ensuring a ``combined_score`` key is present.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            Validated dictionary mapping metric names to float scores.
        """
        # Filter to numeric values only
        validated: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                validated[key] = float(value)

        # Ensure combined_score exists
        if "combined_score" not in validated:
            numeric_values = [v for v in validated.values()]
            if numeric_values:
                validated["combined_score"] = sum(numeric_values) / len(
                    numeric_values
                )
            else:
                validated["combined_score"] = 0.0

        return validated

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple candidate artifacts concurrently.

        Args:
            programs: List of (artifact_path_or_content, program_id) tuples.
                      If the path does not exist as a file, it is treated
                      as raw content.

        Returns:
            List of metric dictionaries (one per program).
        """
        sem = asyncio.Semaphore(self.config.parallel_evaluations)

        async def _bounded_evaluate(path: str, pid: str) -> Dict[str, float]:
            async with sem:
                if os.path.isfile(path):
                    return await self.evaluate(path, program_id=pid)
                else:
                    return await self.evaluate_content(path, program_id=pid)

        tasks = [
            asyncio.ensure_future(_bounded_evaluate(path, pid))
            for path, pid in programs
        ]
        return list(await asyncio.gather(*tasks))

    def get_pending_artifacts(
        self, program_id: str
    ) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program.

        Returns:
            Artifacts dictionary or None if no artifacts are pending.
        """
        return self._pending_artifacts.pop(program_id, None)

    # ------------------------------------------------------------------
    # Subprocess evaluation
    # ------------------------------------------------------------------

    async def _subprocess_evaluate(
        self,
        artifact_path: str,
        function_name: str = "evaluate",
    ) -> Dict[str, Any]:
        """
        Run the evaluation script in an isolated subprocess.

        The runner script imports the evaluator module, calls the specified
        function with the artifact path, and writes the result as JSON to a
        temp file which we read back.

        Args:
            artifact_path: Path to the candidate artifact file.
            function_name: Name of the function to call in the evaluator.

        Returns:
            Dictionary of evaluation results.

        Raises:
            asyncio.TimeoutError: If evaluation exceeds the configured timeout.
            RuntimeError:         If the subprocess exits with a non-zero code
                                  and no result file was written.
        """
        # Create a temp file for the result
        fd, result_path = tempfile.mkstemp(
            suffix=".json", prefix="claude_evolve_result_"
        )
        os.close(fd)

        try:
            cmd = [
                sys.executable,
                self._runner_script_path,
                self.evaluation_file,
                artifact_path,
                result_path,
                function_name,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.communicate()
                except Exception:
                    pass
                raise

            # Read the result file
            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                with open(result_path, "r") as f:
                    result = json.load(f)

                # Check for error messages from the runner
                error_msg = result.pop("__error_msg__", None)
                if error_msg and "error" in result:
                    logger.warning(f"Evaluation error: {error_msg}")

                return result
            else:
                # No result file produced
                stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
                raise RuntimeError(
                    f"Subprocess produced no result (exit code {proc.returncode}). "
                    f"stderr: {stderr_text[:500]}"
                )

        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    # ------------------------------------------------------------------
    # Cascade evaluation
    # ------------------------------------------------------------------

    async def _cascade_evaluate(
        self,
        artifact_path: str,
    ) -> Dict[str, Any]:
        """
        Run cascade evaluation with increasingly thorough stages.

        If the evaluator script defines ``evaluate_stage1(path)``, that is run
        first.  If the score passes the first threshold, ``evaluate_stage2``
        is run (if it exists), and so on.  If no stageN functions exist, falls
        back to a single ``evaluate(path)`` call.

        Args:
            artifact_path: Path to the candidate artifact file.

        Returns:
            Dictionary of metrics (possibly merged from multiple stages).
        """
        # Check which stage functions the evaluator script has
        available_functions = await self._probe_evaluator_functions()

        if "evaluate_stage1" not in available_functions:
            # No cascade functions, fall back to direct evaluate
            return await self._subprocess_evaluate(artifact_path, "evaluate")

        # Stage 1
        try:
            stage1_result = await self._subprocess_evaluate(
                artifact_path, "evaluate_stage1"
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Stage 1 timed out after {self.config.timeout}s"
            )
            return {"combined_score": 0.0, "error": 0.0, "stage1_passed": 0.0}
        except Exception as e:
            logger.error(f"Error in stage 1: {e}")
            return {"combined_score": 0.0, "error": 0.0, "stage1_passed": 0.0}

        stage1_metrics = self._extract_metrics(stage1_result)

        # Check threshold for stage 1
        thresholds = self.config.cascade_thresholds
        if thresholds and not self._passes_threshold(
            stage1_metrics, thresholds[0]
        ):
            return stage1_metrics

        # Stage 2
        if "evaluate_stage2" not in available_functions:
            # Only stage1 exists; return its results
            return stage1_metrics

        try:
            stage2_result = await self._subprocess_evaluate(
                artifact_path, "evaluate_stage2"
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Stage 2 timed out after {self.config.timeout}s"
            )
            stage1_metrics["stage2_passed"] = 0.0
            return stage1_metrics
        except Exception as e:
            logger.error(f"Error in stage 2: {e}")
            stage1_metrics["stage2_passed"] = 0.0
            return stage1_metrics

        stage2_metrics = self._extract_metrics(stage2_result)

        # Merge stage 1 and 2
        merged = dict(stage1_metrics)
        for key, val in stage2_metrics.items():
            if isinstance(val, (int, float)) and key != "error":
                merged[key] = float(val)

        # Check threshold for stage 2 -> 3
        if (
            len(thresholds) >= 2
            and not self._passes_threshold(merged, thresholds[1])
        ):
            return merged

        # Stage 3 (via evaluate())
        if "evaluate" not in available_functions:
            return merged

        try:
            stage3_result = await self._subprocess_evaluate(
                artifact_path, "evaluate"
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Stage 3 (evaluate) timed out after {self.config.timeout}s"
            )
            merged["stage3_passed"] = 0.0
            return merged
        except Exception as e:
            logger.error(f"Error in stage 3 (evaluate): {e}")
            merged["stage3_passed"] = 0.0
            return merged

        stage3_metrics = self._extract_metrics(stage3_result)
        for key, val in stage3_metrics.items():
            if isinstance(val, (int, float)) and key != "error":
                merged[key] = float(val)

        return merged

    async def _probe_evaluator_functions(self) -> set:
        """
        Discover which functions the evaluator script defines.

        Runs a small subprocess that imports the evaluator module and prints
        the names of callable attributes.

        Returns:
            Set of function names found in the evaluator module.
        """
        probe_code = textwrap.dedent(
            f'''
import importlib.util, os, sys, json
eval_dir = os.path.dirname(os.path.abspath("{self.evaluation_file}"))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)
spec = importlib.util.spec_from_file_location("_probe", "{self.evaluation_file}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
funcs = [name for name in dir(mod) if callable(getattr(mod, name)) and not name.startswith("_")]
print(json.dumps(funcs))
'''
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                probe_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=10
            )
            if proc.returncode == 0 and stdout:
                names = json.loads(stdout.decode("utf-8").strip())
                return set(names)
        except Exception as e:
            logger.warning(f"Failed to probe evaluator functions: {e}")

        # Fallback: assume evaluate exists
        return {"evaluate"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Convert a raw result dict into an EvaluationResult.

        Handles the special ``__artifacts__`` key for artifact extraction.
        """
        if isinstance(result, EvaluationResult):
            return result

        if isinstance(result, dict):
            artifacts = {}
            # Extract __artifacts__ if present
            raw_artifacts = result.pop("__artifacts__", None)
            if isinstance(raw_artifacts, dict):
                artifacts = raw_artifacts

            # Filter to numeric metrics
            metrics: Dict[str, float] = {}
            for key, val in result.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    metrics[key] = float(val)

            return EvaluationResult(metrics=metrics, artifacts=artifacts)

        logger.warning(f"Unexpected evaluation result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0, "combined_score": 0.0})

    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from a raw result dict."""
        metrics: Dict[str, float] = {}
        for key, val in result.items():
            if key == "__artifacts__":
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                metrics[key] = float(val)
        return metrics

    def _passes_threshold(
        self, metrics: Dict[str, float], threshold: float
    ) -> bool:
        """
        Check if metrics pass a score threshold.

        Uses ``combined_score`` if available; otherwise falls back to
        averaging all numeric metrics except ``error``.
        """
        if not metrics:
            return False

        if "combined_score" in metrics:
            score = metrics["combined_score"]
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        # Fallback: average all numeric metrics except 'error'
        valid = []
        for name, value in metrics.items():
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid.append(float(value))
                except (TypeError, ValueError):
                    continue

        if not valid:
            return False

        return (sum(valid) / len(valid)) >= threshold

    def __del__(self):
        """Clean up the runner script temp file."""
        if (
            hasattr(self, "_runner_script_path")
            and self._runner_script_path
            and os.path.exists(self._runner_script_path)
        ):
            try:
                os.unlink(self._runner_script_path)
            except Exception:
                pass
