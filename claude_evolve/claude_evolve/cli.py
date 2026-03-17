"""
CLI entry points for claude-evolve.

Provides the ``claude-evolve`` command group with subcommands:
    init, next, submit, status, export.

These are the interface between the stop hook / Claude Code plugin layer
and the Python package internals.  All JSON output goes to stdout via
``click.echo()``; diagnostic text goes to stderr via ``click.echo(..., err=True)``.
"""

import asyncio
import json
import logging
import os
import re
import sys
import uuid
from typing import Optional

import click

from claude_evolve.config import Config, EvaluatorConfig, load_config
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.core.evaluator import Evaluator
from claude_evolve.prompt.context_builder import ContextBuilder
from claude_evolve.state.checkpoint import CheckpointManager
from claude_evolve.state.manager import StateManager
from claude_evolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension -> artifact_type reverse map (inverse of StateManager._EXTENSION_MAP)
# ---------------------------------------------------------------------------
_EXTENSION_TO_TYPE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".rb": "ruby",
    ".sh": "shell",
    ".bash": "bash",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
    ".txt": "text",
}


def _detect_artifact_type(path: str) -> str:
    """Detect artifact type from file extension.

    Falls back to ``"text"`` for unrecognised extensions.
    """
    _, ext = os.path.splitext(path)
    return _EXTENSION_TO_TYPE.get(ext.lower(), "text")


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------

@click.group()
def main():
    """claude-evolve: Evolutionary artifact optimization for Claude Code."""
    pass


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@main.command()
@click.option("--artifact", required=True, type=click.Path(),
              help="Path to the artifact file to evolve.")
@click.option("--evaluator", required=True, type=click.Path(),
              help="Path to the evaluator script (evaluator.py or eval_prompt.md).")
@click.option("--mode", type=click.Choice(["script", "critic", "hybrid"]),
              default="script", help="Evaluation mode (default: script).")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Optional YAML config file.")
@click.option("--max-iterations", type=int, default=None,
              help="Maximum number of iterations (default: 50).")
@click.option("--target-score", type=float, default=None,
              help="Optional early stop threshold score.")
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--keep-state", is_flag=True, default=False,
              help="Keep existing state directory (don't clear). Useful for resuming.")
def init(artifact, evaluator, mode, config_path, max_iterations, target_score, state_dir, keep_state):
    """Initialize a new evolution run."""
    # Validate files exist
    if not os.path.exists(artifact):
        click.echo(f"Error: Artifact file not found: {artifact}", err=True)
        raise SystemExit(1)
    if not os.path.exists(evaluator):
        click.echo(f"Error: Evaluator file not found: {evaluator}", err=True)
        raise SystemExit(1)

    # Read initial artifact content
    with open(artifact, "r") as f:
        initial_content = f.read()

    # Detect artifact type from extension
    artifact_type = _detect_artifact_type(artifact)

    # Load config from file or defaults, then apply CLI overrides
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Apply CLI overrides
    if max_iterations is not None:
        config.max_iterations = max_iterations
    if target_score is not None:
        config.target_score = target_score
    config.evaluator.mode = mode
    config.artifact_type = artifact_type

    # Create StateManager and initialize
    sm = StateManager(state_dir)
    sm.initialize(
        config=config,
        initial_content=initial_content,
        artifact_type=artifact_type,
        evaluator_path=os.path.abspath(evaluator),
        fresh=not keep_state,
    )

    # Persist evaluator path separately so submit can find it
    _save_evaluator_path(state_dir, os.path.abspath(evaluator))

    # Generate and persist a unique run_id for cross-run memory tracking
    run_id = uuid.uuid4().hex[:8]
    metadata_path = os.path.join(state_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    metadata["run_id"] = run_id
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Run baseline evaluation for script mode
    baseline_score = None
    if mode == "script":
        try:
            eval_config = config.evaluator
            ev = Evaluator(
                config=eval_config,
                evaluation_file=os.path.abspath(evaluator),
                suffix=os.path.splitext(artifact)[1] or ".py",
            )
            metrics = _run_async(ev.evaluate_content(initial_content))
            baseline_score = metrics.get("combined_score", 0.0)

            # Update the seed artifact metrics in the database.
            # The seed is already in db.artifacts; update its metrics in-place
            # and re-add it so that best-tracking and feature grid are updated.
            db = sm.get_database()
            seed = db.get_best()
            if seed is not None:
                seed.metrics = metrics
                db.add(seed, iteration=0)
        except Exception as e:
            click.echo(f"Warning: Baseline evaluation failed: {e}", err=True)
            baseline_score = 0.0

    # Auto-seed from cross-run memory if available
    if config.cross_run_memory.enabled:
        memory_dir = os.path.join(state_dir, config.cross_run_memory.memory_dir)
        if os.path.exists(memory_dir):
            from claude_evolve.core.memory import CrossRunMemory
            memory = CrossRunMemory(
                memory_dir=memory_dir,
                max_learnings=config.cross_run_memory.max_learnings,
                max_failed_approaches=config.cross_run_memory.max_failed_approaches,
            )
            memory.load()
            # Log that cross-run memory was found
            click.echo(f"Found cross-run memory with {len(memory.learnings)} learnings, "
                       f"{len(memory.failed_approaches)} failed approaches", err=True)

    # Save state including any baseline metrics
    sm.save()

    # Output JSON status
    db = sm.get_database()
    output = {
        "status": "initialized",
        "population_size": db.size(),
    }
    if baseline_score is not None:
        output["baseline_score"] = baseline_score

    click.echo(json.dumps(output))


# ---------------------------------------------------------------------------
# next
# ---------------------------------------------------------------------------

@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
def next(state_dir):
    """Prepare the next iteration context (called by stop hook)."""
    # Load state
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = sm.get_config()
    db = sm.get_database()

    # Check if max iterations exceeded
    next_iteration = db.last_iteration + 1
    if config.max_iterations > 0 and next_iteration > config.max_iterations:
        click.echo(f"Max iterations ({config.max_iterations}) reached. No more iterations.", err=True)
        raise SystemExit(1)

    # --- Orchestrator: unified next-phase logic ---
    from claude_evolve.core.orchestrator import IterationOrchestrator
    orch = IterationOrchestrator(state_dir=state_dir, config=config, db=db)
    orch_ctx = orch.prepare_next_iteration(iteration=next_iteration)

    parent = orch_ctx['parent']

    # Fall back to db.sample() when orchestrator returns no parent (empty db edge case)
    if parent is None:
        try:
            parent, inspirations = db.sample()
        except Exception as e:
            click.echo(f"Error sampling from database: {e}", err=True)
            raise SystemExit(1)
    else:
        # Get inspirations via db.sample() -- orchestrator handles parent selection
        try:
            _unused_parent, inspirations = db.sample()
        except Exception:
            inspirations = []

    # Get top programs and previous programs (recent artifacts sorted by iteration)
    top_programs = db.get_top_programs(n=config.prompt.num_top_programs)
    top_programs_dicts = [a.to_dict() for a in top_programs]

    # Build "previous programs" -- recently added artifacts sorted by timestamp
    all_artifacts = sorted(
        db.artifacts.values(),
        key=lambda a: a.timestamp,
        reverse=True,
    )
    # Take the most recent artifacts (excluding the parent) as "previous"
    previous_programs = []
    for a in all_artifacts:
        if a.id != parent.id:
            previous_programs.append(a.to_dict())
        if len(previous_programs) >= 5:
            break

    inspiration_dicts = [a.to_dict() for a in inspirations]

    # Get parent artifacts (evaluation artifacts)
    parent_artifacts = db.get_artifacts(parent.id)

    # Build context
    best = db.get_best()
    best_score = 0.0
    if best is not None:
        best_score = best.metrics.get("combined_score", 0.0)

    # Stagnation detection (v2) -- use orchestrator's stagnation_level
    stagnation_report = None
    if config.stagnation.enabled:
        stagnation_report = db.detect_stagnation(config.stagnation)

    # Cross-run memory (v2)
    cross_run_memory_text = None
    if config.cross_run_memory.enabled:
        memory_dir = os.path.join(state_dir, config.cross_run_memory.memory_dir)
        if os.path.exists(memory_dir):
            from claude_evolve.core.memory import CrossRunMemory
            memory = CrossRunMemory(
                memory_dir=memory_dir,
                max_learnings=config.cross_run_memory.max_learnings,
                max_failed_approaches=config.cross_run_memory.max_failed_approaches,
            )
            memory.load()
            cross_run_memory_text = memory.format_for_prompt()

    # Research findings (v2 phase 2) -- only include when trigger fires
    research_text = None
    if config.research.enabled:
        research_log_path = os.path.join(state_dir, config.research.research_log_file)
        from claude_evolve.core.research import ResearchLog
        research_log = ResearchLog(research_log_path)
        research_log.load()

        stagnation_level_str = stagnation_report.level.value if stagnation_report else "none"
        if research_log.should_research(db.last_iteration + 1, stagnation_level_str, config.research):
            research_text = research_log.format_for_prompt()

    # Strategy text from orchestrator
    strategy_text = orch_ctx['strategy_name']

    # Warm-start cache
    warm_cache_text = None
    warm_cache_dir = os.path.join(state_dir, "warm_cache")
    if os.path.exists(warm_cache_dir):
        from claude_evolve.core.warm_cache import WarmCache
        warm_cache = WarmCache(warm_cache_dir)
        warm_cache.load()
        warm_cache_text = warm_cache.format_for_prompt()

    # Stepping stones -- diverse intermediate solutions for inspiration
    stepping_stones_text = None
    stones_path = os.path.join(state_dir, "stepping_stones.json")
    if os.path.exists(stones_path):
        from claude_evolve.core.novelty import SteppingStonesArchive
        try:
            with open(stones_path, "r") as f:
                stones_data = json.load(f)
            archive = SteppingStonesArchive.from_list(stones_data)
            parent_content = parent.content if hasattr(parent, 'content') else parent.get('content', '')
            stepping_stones_text = archive.format_for_prompt(
                parent_content, config.artifact_type
            )
        except Exception:
            pass

    # Load evaluator source so the LLM can see how candidates are scored
    evaluator_source = None
    evaluator_path = _find_evaluator_path(state_dir, config)
    if evaluator_path and os.path.exists(evaluator_path):
        try:
            with open(evaluator_path, "r", encoding="utf-8") as f:
                evaluator_lines = f.readlines()[:200]
            evaluator_source = "".join(evaluator_lines)
        except Exception:
            pass

    # Extract parent rationale for thought-code coevolution
    parent_rationale = parent.rationale if parent and hasattr(parent, 'rationale') else None

    ctx_builder = ContextBuilder(config.prompt)
    ctx = ctx_builder.build_context(
        parent=parent,
        iteration=db.last_iteration + 1,
        best_score=best_score,
        top_programs=top_programs_dicts,
        inspirations=inspiration_dicts,
        previous_programs=previous_programs,
        language=config.artifact_type,
        diff_based=config.evolution.diff_based,
        parent_artifacts=parent_artifacts,
        feature_dimensions=config.database.feature_dimensions,
        stagnation_report=stagnation_report,
        cross_run_memory_text=cross_run_memory_text if cross_run_memory_text else None,
        research_text=research_text if research_text else None,
        strategy_text=strategy_text,
        warm_cache_text=warm_cache_text if warm_cache_text else None,
        stepping_stones_text=stepping_stones_text if stepping_stones_text else None,
        comparison_artifact=orch_ctx['comparison'],
        failures_text=orch_ctx['failures_text'] if orch_ctx['failures_text'] else None,
        evaluator_source=evaluator_source if evaluator_source else None,
        parent_rationale=parent_rationale,
        scratchpad_text=orch_ctx['scratchpad_text'] if orch_ctx['scratchpad_text'] else None,
        reflection_text=orch_ctx['reflection_text'] if orch_ctx['reflection_text'] else None,
        meta_guidance=orch_ctx['meta_guidance'] if orch_ctx['meta_guidance'] else None,
    )

    # Render iteration context
    rendered = ctx_builder.render_iteration_context(
        ctx=ctx,
        iteration=db.last_iteration + 1,
        max_iterations=config.max_iterations,
    )

    # Orchestrator already writes current_iteration.json via _save_manifest

    # Write to state dir
    sm.write_iteration_context(rendered)
    sm.save()

    # Print prompt text to stdout for the stop hook
    click.echo(rendered)


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
def diagnose(state_dir):
    """Run stagnation detection and output a diagnostic report."""
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = sm.get_config()
    db = sm.get_database()

    report = db.detect_stagnation(config.stagnation)

    output = {
        "level": report.level.value,
        "iterations_stagnant": report.iterations_stagnant,
        "best_score": report.best_score,
        "exploration_ratio_boost": report.exploration_ratio_boost,
        "suggested_strategy": report.suggested_strategy,
        "diagnosis": report.diagnosis,
        "recommendations": report.recommendations,
        "failed_approaches": report.failed_approaches,
    }

    click.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# research-log
# ---------------------------------------------------------------------------

@main.command("research-log")
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--findings", required=True, type=str,
              help="JSON string of research findings to append.")
def research_log(state_dir, findings):
    """Append research findings to the research log (called by researcher agent)."""
    import time
    from claude_evolve.core.research import ResearchFinding, ResearchLog

    # Load state to get config
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = sm.get_config()
    db = sm.get_database()

    # Parse findings JSON
    try:
        raw_findings = json.loads(findings)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid findings JSON: {e}", err=True)
        raise SystemExit(1)

    if not isinstance(raw_findings, list):
        raw_findings = [raw_findings]

    # Load research log
    log_path = os.path.join(state_dir, config.research.research_log_file)
    rlog = ResearchLog(log_path)
    rlog.load()

    # Add each finding
    added_count = 0
    current_iteration = db.last_iteration
    for raw in raw_findings:
        if not isinstance(raw, dict):
            click.echo(f"Warning: Skipping non-dict finding: {raw}", err=True)
            continue

        # Ensure required fields have defaults
        finding_id = raw.get("id", f"research-{current_iteration}-{added_count}")
        finding = ResearchFinding(
            id=finding_id,
            iteration=raw.get("iteration", current_iteration),
            timestamp=raw.get("timestamp", time.time()),
            approach_name=raw.get("approach_name", raw.get("name", "Unknown")),
            description=raw.get("description", ""),
            novelty=raw.get("novelty", "medium"),
            implementation_hint=raw.get("implementation_hint", ""),
            source_url=raw.get("source_url", ""),
            was_tried=raw.get("was_tried", False),
            outcome_score=raw.get("outcome_score"),
            metadata=raw.get("metadata", {}),
        )
        rlog.add_finding(finding)
        added_count += 1

    # Handle theoretical bounds, key papers, and approaches to avoid
    # if present in the top-level JSON
    if isinstance(raw_findings, list) and len(raw_findings) > 0:
        first = raw_findings[0]
        if isinstance(first, dict):
            if "theoretical_bounds" in first and isinstance(first["theoretical_bounds"], dict):
                rlog.theoretical_bounds.update(first["theoretical_bounds"])
            if "key_papers" in first and isinstance(first["key_papers"], list):
                rlog.key_papers.extend(first["key_papers"])
            if "approaches_to_avoid" in first and isinstance(first["approaches_to_avoid"], list):
                rlog.approaches_to_avoid.extend(first["approaches_to_avoid"])

    # Save
    if config.research.persist_findings:
        rlog.save()

    click.echo(json.dumps({
        "status": "appended",
        "findings_added": added_count,
        "total_findings": len(rlog.findings),
    }))


# ---------------------------------------------------------------------------
# cache-eval
# ---------------------------------------------------------------------------

@main.command("cache-eval")
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--n", type=int, required=True, help="The n value that was verified")
@click.option("--result", required=True, type=str, help="JSON result for this n")
def cache_eval(state_dir, n, result):
    """Cache evaluation result for a specific n value (avoids re-evaluation)."""
    # Parse the result JSON
    try:
        result_data = json.loads(result)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid result JSON: {e}", err=True)
        raise SystemExit(1)

    # Ensure state directory exists
    os.makedirs(state_dir, exist_ok=True)

    # Load existing eval cache or create new one
    cache_path = os.path.join(state_dir, "eval_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Store the result keyed by n
    cache[str(n)] = result_data

    # Write back
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    click.echo(json.dumps({
        "status": "cached",
        "n": n,
        "cache_size": len(cache),
    }))


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state")
@click.option("--artifact", required=True, type=click.Path(exists=True),
              help="Path to artifact file to seed into population")
@click.option("--metrics", default=None, type=str,
              help="Optional pre-computed metrics as JSON")
@click.option("--description", default="Manually seeded artifact",
              help="Description of the seeded artifact")
def seed(state_dir, artifact, metrics, description):
    """Seed a known-good artifact into the evolution population.

    Use this to inject previously discovered solutions (from cross-run memory,
    external sources, or manual construction) into the current evolution run.
    """
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = sm.get_config()
    db = sm.get_database()

    with open(artifact, "r") as f:
        content = f.read()

    parsed_metrics = {}
    if metrics:
        try:
            parsed_metrics = json.loads(metrics)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid metrics JSON: {e}", err=True)
            raise SystemExit(1)

    new_artifact = Artifact(
        id=Artifact.generate_id(),
        content=content,
        artifact_type=config.artifact_type,
        generation=0,
        metrics=parsed_metrics,
        metadata={
            "source": "manual_seed",
            "description": description,
        },
    )

    db.add(new_artifact, iteration=db.last_iteration)
    sm.save()

    click.echo(json.dumps({
        "status": "seeded",
        "artifact_id": new_artifact.id,
        "metrics": parsed_metrics,
    }))


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--key", required=True, help="Cache key to inspect.")
def cache(state_dir, key):
    """Inspect warm cache contents."""
    from claude_evolve.core.warm_cache import WarmCache

    warm_cache_dir = os.path.join(state_dir, "warm_cache")
    wc = WarmCache(warm_cache_dir)
    wc.load()

    if not wc.has(key):
        click.echo(json.dumps({
            "status": "not_found",
            "key": key,
            "available_keys": wc.keys(),
        }))
        return

    info = wc.manifest[key]
    output = {
        "status": "found",
        "key": key,
        "type": info.get("type", "unknown"),
        "timestamp": info.get("timestamp"),
        "metadata": info.get("metadata", {}),
    }

    # Include type-specific details
    if info["type"] == "numpy":
        output["shape"] = info.get("shape")
        output["dtype"] = info.get("dtype")
    elif info["type"] == "json":
        data = wc.get_json(key)
        if data is not None:
            # Include a preview for small data
            preview = json.dumps(data)
            if len(preview) > 500:
                preview = preview[:500] + "..."
            output["preview"] = preview
    elif info["type"] == "text":
        text = wc.get_text(key)
        if text is not None:
            preview = text[:500] + "..." if len(text) > 500 else text
            output["preview"] = preview

    click.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# cache-put
# ---------------------------------------------------------------------------

@main.command("cache-put")
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--key", required=True, help="Cache key to store under.")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True),
              help="Path to the file to cache.")
@click.option("--type", "data_type", type=click.Choice(["numpy", "json", "text"]),
              default="numpy", help="Data type of the file (default: numpy).")
@click.option("--description", default="", help="Human-readable description of cached item.")
@click.option("--score", type=float, default=None, help="Associated score for this cached item.")
def cache_put(state_dir, key, file_path, data_type, description, score):
    """Save a file to the warm cache (called by candidate scripts).

    Stores the specified file in the warm-start cache under the given key.
    Subsequent evolution iterations can load this cached state to avoid
    recomputing expensive intermediate results.
    """
    import numpy as np

    from claude_evolve.core.warm_cache import WarmCache

    warm_cache_dir = os.path.join(state_dir, "warm_cache")
    wc = WarmCache(warm_cache_dir)
    wc.load()

    metadata = {}
    if description:
        metadata["description"] = description
    if score is not None:
        metadata["score"] = score

    if data_type == "numpy":
        try:
            array = np.load(file_path)
        except Exception as e:
            click.echo(f"Error: Failed to load numpy file: {e}", err=True)
            raise SystemExit(1)
        wc.put_numpy(key, array, metadata=metadata)
        click.echo(json.dumps({
            "status": "cached",
            "key": key,
            "type": "numpy",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }))
    elif data_type == "json":
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            click.echo(f"Error: Failed to load JSON file: {e}", err=True)
            raise SystemExit(1)
        wc.put_json(key, data, metadata=metadata)
        click.echo(json.dumps({
            "status": "cached",
            "key": key,
            "type": "json",
        }))
    elif data_type == "text":
        try:
            with open(file_path, "r") as f:
                text = f.read()
        except Exception as e:
            click.echo(f"Error: Failed to read text file: {e}", err=True)
            raise SystemExit(1)
        wc.put_text(key, text, metadata=metadata)
        click.echo(json.dumps({
            "status": "cached",
            "key": key,
            "type": "text",
            "length": len(text),
        }))


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

@main.command()
@click.option("--candidate", required=True, type=click.Path(),
              help="Path to the candidate artifact produced by Claude.")
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--metrics", default=None, type=str,
              help="Optional pre-computed metrics as JSON (for critic mode).")
def submit(candidate, state_dir, metrics):
    """Submit a candidate artifact for evaluation."""
    # Validate candidate exists
    if not os.path.exists(candidate):
        click.echo(f"Error: Candidate file not found: {candidate}", err=True)
        raise SystemExit(1)

    # Load state
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = sm.get_config()
    db = sm.get_database()

    # Read candidate content
    with open(candidate, "r") as f:
        candidate_content = f.read()

    # Extract rationale for thought-code coevolution
    # Supports optional comment prefixes (# // /* etc.) before markers
    rationale_match = re.search(
        r'RATIONALE-START\s*\n(.*?)\n[^\n]*RATIONALE-END',
        candidate_content, re.DOTALL,
    )
    rationale = rationale_match.group(1).strip() if rationale_match else None

    # Pre-evaluation novelty gate — reject near-duplicates before
    # spending evaluator resources.  Skip when the archive is too
    # small (< 3 members) to make a meaningful comparison.
    min_novelty = config.selection.novelty_gate_min_novelty
    top_artifacts = db.get_top_programs(n=10)
    if len(top_artifacts) >= 3:
        from claude_evolve.core.novelty import compute_novelty

        for archive_art in top_artifacts:
            novelty = compute_novelty(
                candidate_content,
                archive_art.content,
                artifact_type=config.artifact_type,
            )
            if novelty < min_novelty:
                click.echo(json.dumps({
                    "rejected": True,
                    "reason": (
                        f"Candidate too similar to existing solution "
                        f"{archive_art.id} (novelty: {novelty:.3f}, "
                        f"threshold: {min_novelty}). "
                        f"Try a more novel approach."
                    ),
                }))
                return

    # Evaluate
    raw_metrics = None  # Preserved for failure reflexion (may contain non-numeric fields)
    if metrics is not None:
        # Critic mode: parse pre-computed metrics
        try:
            raw_metrics = json.loads(metrics)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid metrics JSON: {e}", err=True)
            raise SystemExit(1)

        eval_config = config.evaluator
        ev = Evaluator(
            config=EvaluatorConfig(mode="critic"),
            evaluation_file=None,
        )
        validated_metrics = _run_async(ev.evaluate_with_metrics(raw_metrics))
    else:
        # Script mode: run evaluator subprocess
        # Find the evaluator path from the stored config
        evaluator_path = _find_evaluator_path(state_dir, config)
        if evaluator_path is None:
            click.echo(
                "Error: Cannot find evaluator script. "
                "Re-run init or pass --metrics for critic mode.",
                err=True,
            )
            raise SystemExit(1)

        ev = Evaluator(
            config=config.evaluator,
            evaluation_file=evaluator_path,
            suffix=os.path.splitext(candidate)[1] or ".py",
        )
        validated_metrics = _run_async(ev.evaluate(candidate))

    # Determine parent (use best or most recent)
    best = db.get_best()
    parent_id = best.id if best else None
    parent_generation = best.generation if best else 0

    # Create new artifact
    new_artifact = Artifact(
        id=Artifact.generate_id(),
        content=candidate_content,
        artifact_type=config.artifact_type,
        parent_id=parent_id,
        generation=parent_generation + 1,
        metrics=validated_metrics,
        metadata={
            "source": "claude_submit",
            "candidate_path": os.path.abspath(candidate),
        },
    )
    # Attach extracted rationale for thought-code coevolution
    new_artifact.rationale = rationale

    # Add to database
    current_iteration = db.last_iteration + 1
    db.add(new_artifact, iteration=current_iteration)

    # Handle island migration if due
    if db.should_migrate():
        db.migrate_programs()
        db.last_migration_generation = max(db.island_generations)

    # Save checkpoint if interval reached
    if config.checkpoint_interval > 0 and current_iteration % config.checkpoint_interval == 0:
        cp_manager = CheckpointManager(state_dir)
        cp_manager.save(db, current_iteration)

    # Check if this is a new best
    new_best = db.get_best()
    is_new_best = new_best is not None and new_best.id == new_artifact.id

    # Update best artifact file if new best
    if is_new_best:
        sm.write_best_artifact(candidate_content, config.artifact_type)

    # Add to stepping stones archive (preserves diverse intermediates)
    try:
        from claude_evolve.core.novelty import SteppingStonesArchive
        stones_path = os.path.join(state_dir, "stepping_stones.json")
        if os.path.exists(stones_path):
            with open(stones_path, "r") as f:
                archive = SteppingStonesArchive.from_list(json.load(f))
        else:
            archive = SteppingStonesArchive(max_size=50, novelty_threshold=0.3)
        archive.try_add(
            candidate_content, validated_metrics,
            config.artifact_type,
            metadata={"iteration": current_iteration, "is_new_best": is_new_best},
        )
        with open(stones_path, "w") as f:
            json.dump(archive.to_list(), f)
    except Exception:
        pass  # Non-critical — don't fail submit on stepping stone error

    # --- Orchestrator: unified submit-phase logic ---
    # Handles cross-run memory population, UCB strategy recording,
    # failure capture, improvement signal update, and offspring tracking.
    try:
        from claude_evolve.core.orchestrator import IterationOrchestrator
        orch = IterationOrchestrator(state_dir=state_dir, config=config, db=db)
        # Merge raw error into metrics so orchestrator can see it
        submit_metrics = dict(validated_metrics)
        if raw_metrics is not None and raw_metrics.get("error"):
            submit_metrics.setdefault("error", raw_metrics["error"])
        orch.process_submission(candidate_content, submit_metrics)
    except Exception as e:
        click.echo(f"Warning: Orchestrator submit processing failed: {e}", err=True)

    # Save state
    sm.save()

    # Build output
    output = dict(validated_metrics)
    output["is_new_best"] = is_new_best

    click.echo(json.dumps(output))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
def status(state_dir):
    """Show evolution progress."""
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = sm.get_config()
    db = sm.get_database()

    best = db.get_best()
    best_score = 0.0
    if best is not None:
        best_score = best.metrics.get("combined_score", 0.0)

    island_stats = db.get_island_stats()

    output = {
        "iteration": db.last_iteration,
        "best_score": best_score,
        "population_size": db.size(),
        "target_score": config.target_score,
        "max_iterations": config.max_iterations,
        "islands": island_stats,
    }

    click.echo(json.dumps(output))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@main.command()
@click.option("--state-dir", type=click.Path(), default=".claude/evolve-state",
              help="Path to the evolution state directory.")
@click.option("--output", required=True, type=click.Path(),
              help="Output file path for the exported artifact.")
@click.option("--top-n", type=int, default=None,
              help="Export the top N artifacts (numbered files).")
def export(state_dir, output, top_n):
    """Export the best artifact(s)."""
    sm = StateManager(state_dir)
    try:
        sm.load()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    db = sm.get_database()

    if top_n is not None and top_n > 1:
        # Export top N artifacts as numbered files
        top_artifacts = db.get_top_programs(n=top_n)
        if not top_artifacts:
            click.echo("Error: No artifacts to export.", err=True)
            raise SystemExit(1)

        base, ext = os.path.splitext(output)
        for i, artifact in enumerate(top_artifacts):
            numbered_path = f"{base}_{i + 1}{ext}"
            output_dir = os.path.dirname(numbered_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(numbered_path, "w") as f:
                f.write(artifact.content)

        click.echo(json.dumps({
            "status": "exported",
            "count": len(top_artifacts),
            "files": [f"{base}_{i + 1}{ext}" for i in range(len(top_artifacts))],
        }))
    else:
        # Export single best artifact
        best = db.get_best()
        if best is None:
            click.echo("Error: No artifacts to export.", err=True)
            raise SystemExit(1)

        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output, "w") as f:
            f.write(best.content)

        click.echo(json.dumps({
            "status": "exported",
            "file": output,
            "score": best.metrics.get("combined_score", 0.0),
        }))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_evaluator_path(state_dir: str, evaluator_path: str) -> None:
    """Persist the evaluator path to a file in the state directory."""
    path = os.path.join(state_dir, "evaluator_path.txt")
    with open(path, "w") as f:
        f.write(evaluator_path)


def _find_evaluator_path(state_dir: str, config: Config) -> Optional[str]:
    """Locate the evaluator script path from the state directory.

    The evaluator path is stored by ``init`` as ``evaluator_path.txt``.
    """
    # Primary: dedicated evaluator_path.txt file
    path_file = os.path.join(state_dir, "evaluator_path.txt")
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            evaluator_path = f.read().strip()
        if evaluator_path and os.path.exists(evaluator_path):
            return evaluator_path

    return None
