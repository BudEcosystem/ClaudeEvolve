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
import sys
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
def init(artifact, evaluator, mode, config_path, max_iterations, target_score, state_dir):
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
    )

    # Persist evaluator path separately so submit can find it
    _save_evaluator_path(state_dir, os.path.abspath(evaluator))

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

            # Update the seed artifact metrics in the database
            db = sm.get_database()
            seed = db.get_best()
            if seed is not None:
                seed.metrics = metrics
                db._update_best(seed)
        except Exception as e:
            click.echo(f"Warning: Baseline evaluation failed: {e}", err=True)
            baseline_score = 0.0

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

    # Sample parent + inspirations
    try:
        parent, inspirations = db.sample()
    except Exception as e:
        click.echo(f"Error sampling from database: {e}", err=True)
        raise SystemExit(1)

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
    )

    # Render iteration context
    rendered = ctx_builder.render_iteration_context(
        ctx=ctx,
        iteration=db.last_iteration + 1,
        max_iterations=config.max_iterations,
    )

    # Write to state dir
    sm.write_iteration_context(rendered)
    sm.save()

    # Print prompt text to stdout for the stop hook
    click.echo(rendered)


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

    # Evaluate
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
