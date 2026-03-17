"""
Context builder for claude_evolve prompt system.

Constructs per-iteration context for Claude Code during evolution.
Unlike OpenEvolve's PromptSampler (which builds raw LLM API messages),
ContextBuilder produces structured dicts and Markdown for a tool-using
agent that reads iteration_context.md at the start of each iteration.

Adapted from OpenEvolve's prompt/sampler.py.
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional, Union

from claude_evolve.config import PromptConfig
from claude_evolve.prompt.templates import TemplateManager
from claude_evolve.utils.metrics_utils import (
    format_feature_coordinates,
    get_fitness_score,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds iteration context for Claude Code evolution runs.

    The primary interface is ``build_context(...)`` which returns a dict
    with keys ``prompt``, ``system_message``, ``parent_id``, and
    ``metadata``.  The ``render_iteration_context(...)`` method turns
    that dict into a Markdown document written to disk for Claude to
    read at the start of each iteration.
    """

    def __init__(self, config: PromptConfig) -> None:
        self.config = config
        self.template_manager = TemplateManager(
            custom_template_dir=config.template_dir
        )

        # Optional per-instance template overrides
        self.system_template_override: Optional[str] = None
        self.user_template_override: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_templates(
        self,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        """Override template selection for system and/or user messages.

        Args:
            system_template: Template name to use for the system message.
            user_template: Template name to use for the user prompt.
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(
            "Set custom templates: system=%s, user=%s",
            system_template,
            user_template,
        )

    def build_context(
        self,
        parent: Any,
        iteration: int,
        best_score: float,
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        previous_programs: List[Dict[str, Any]],
        language: str = "python",
        diff_based: bool = True,
        template_key: Optional[str] = None,
        parent_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        feature_dimensions: Optional[List[str]] = None,
        current_changes_description: Optional[str] = None,
        stagnation_report: Optional[Any] = None,
        cross_run_memory_text: Optional[str] = None,
        research_text: Optional[str] = None,
        strategy_text: Optional[str] = None,
        warm_cache_text: Optional[str] = None,
        stepping_stones_text: Optional[str] = None,
        comparison_artifact: Optional[Any] = None,
        comparison_score: float = 0.0,
        failures_text: Optional[str] = None,
        evaluator_source: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the full context dict for one evolution iteration.

        Args:
            parent: The parent Artifact (or Artifact-like object with
                ``.id``, ``.content``, ``.metrics``, ``.artifact_type``,
                and ``.to_dict()`` methods). May also be a dict.
            iteration: Current iteration number.
            best_score: Best fitness score seen so far.
            top_programs: Dicts of top-performing programs.
            inspirations: Dicts of diverse inspiration programs.
            previous_programs: Dicts of recent previous attempts.
            language: Programming language of the artifact.
            diff_based: Whether to use diff-based (SEARCH/REPLACE) or
                full-rewrite mode.
            template_key: Explicit override for the user template name.
            parent_artifacts: Evaluation artifacts (stderr, stdout, etc.)
                from the parent program's last evaluation.
            feature_dimensions: MAP-Elites feature dimension names.
            current_changes_description: Changes description text for
                the parent program.
            stagnation_report: Optional StagnationReport from the
                StagnationEngine. When present, stagnation guidance is
                injected into the prompt.
            cross_run_memory_text: Optional pre-formatted text from
                CrossRunMemory.format_for_prompt(). When present,
                cross-run learnings are injected into the prompt.
            research_text: Optional pre-formatted text from
                ResearchLog.format_for_prompt(). When present,
                research findings are injected into the prompt.
            warm_cache_text: Optional pre-formatted text from
                WarmCache.format_for_prompt(). When present,
                warm-start cache info is injected into the prompt.
            comparison_artifact: Optional artifact (or dict) to use
                for pairwise comparison (verbal gradient). When
                provided along with comparison_score, a "Pairwise
                Comparison" section is rendered showing both parent
                and comparison code snippets with scores.
            comparison_score: Fitness score of the comparison artifact.
            failures_text: Optional pre-formatted text of recent
                failures. When present, failure reflexion is injected
                into the prompt so the LLM avoids repeating mistakes.
            evaluator_source: Optional source code of the evaluator
                script (first 200 lines). When present, it is shown
                to the LLM so it can understand how candidates are scored.
            **kwargs: Extra keys forwarded into the user template.

        Returns:
            Dict with keys:
                ``prompt``: The assembled user prompt string.
                ``system_message``: The system message string.
                ``parent_id``: ID of the selected parent artifact.
                ``metadata``: Auxiliary info (iteration, best_score, etc.)
        """
        # Normalise parent to a dict-like access pattern
        parent_dict = self._normalize_parent(parent)
        parent_id = parent_dict.get("id", "unknown")
        current_program = parent_dict.get("content", parent_dict.get("code", ""))
        program_metrics = parent_dict.get("metrics", {})
        artifact_type = parent_dict.get("artifact_type", language)

        feature_dimensions = feature_dimensions or []

        # --- Select templates ---
        if template_key:
            user_template_key = template_key
        elif self.user_template_override:
            user_template_key = self.user_template_override
        else:
            user_template_key = "diff_user" if diff_based else "full_rewrite_user"

        user_template = self.template_manager.get_template(user_template_key)

        # System message
        if self.system_template_override:
            system_message = self.template_manager.get_template(
                self.system_template_override
            )
        else:
            system_message = self.config.system_message
            if self.template_manager.has_template(system_message):
                system_message = self.template_manager.get_template(system_message)

        # Optional changes-description wrapper
        if self.config.programs_as_changes_description:
            if self.config.system_message_changes_description:
                changes_desc = self.config.system_message_changes_description.strip()
            else:
                changes_desc = self.template_manager.get_template(
                    "system_message_changes_description"
                )
            system_message = self.template_manager.get_template(
                "system_message_with_changes_description"
            ).format(
                system_message=system_message,
                system_message_changes_description=changes_desc,
            )

        # --- Build prompt components ---
        metrics_str = self._format_metrics(program_metrics)
        improvement_areas = self._identify_improvement_areas(
            current_program, "", program_metrics, previous_programs, feature_dimensions
        )
        evolution_history = self._format_evolution_history(
            previous_programs,
            top_programs,
            inspirations,
            artifact_type,
            feature_dimensions,
        )

        artifacts_section = ""
        if self.config.include_artifacts and parent_artifacts:
            artifacts_section = self._render_artifacts(parent_artifacts)

        # Stochastic template variations
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Feature/fitness numbers
        fitness_score = get_fitness_score(program_metrics, feature_dimensions)
        feature_coords = format_feature_coordinates(
            program_metrics, feature_dimensions
        )

        # --- Assemble user message ---
        user_message = user_template.format(
            metrics=metrics_str,
            fitness_score=f"{fitness_score:.4f}",
            feature_coords=feature_coords,
            feature_dimensions=(
                ", ".join(feature_dimensions) if feature_dimensions else "None"
            ),
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=artifact_type,
            artifacts=artifacts_section,
            **kwargs,
        )

        # Optional changes-description wrapper for user message
        if self.config.programs_as_changes_description:
            desc = (current_changes_description or "").rstrip()
            user_message = self.template_manager.get_template(
                "user_message_with_changes_description"
            ).format(user_message=user_message, changes_description=desc)

        metadata = {
            "iteration": iteration,
            "best_score": best_score,
            "parent_metrics": program_metrics,
            "diff_based": diff_based,
            "language": artifact_type,
            "feature_dimensions": feature_dimensions,
        }

        # Stagnation report (v2)
        if stagnation_report is not None:
            metadata["stagnation_report"] = stagnation_report

        # Cross-run memory text (v2)
        if cross_run_memory_text:
            metadata["cross_run_memory_text"] = cross_run_memory_text

        # Research findings text (v2 phase 2)
        if research_text:
            metadata["research_text"] = research_text

        # Strategy directive text (v2 phase 3)
        if strategy_text:
            metadata["strategy_text"] = strategy_text

        # Warm-start cache text
        if warm_cache_text:
            metadata["warm_cache_text"] = warm_cache_text

        # Stepping stones text
        if stepping_stones_text:
            metadata["stepping_stones_text"] = stepping_stones_text

        # Pairwise comparison (verbal gradient)
        if comparison_artifact is not None:
            comp_dict = self._normalize_parent(comparison_artifact)
            comp_content = comp_dict.get("content", comp_dict.get("code", ""))
            metadata["comparison_artifact_content"] = comp_content
            metadata["comparison_score"] = comparison_score

        # Recent failures for reflexion
        if failures_text:
            metadata["failures_text"] = failures_text

        # Evaluator source code for scoring context
        if evaluator_source:
            metadata["evaluator_source"] = evaluator_source

        return {
            "prompt": user_message,
            "system_message": system_message,
            "parent_id": parent_id,
            "metadata": metadata,
        }

    def render_iteration_context(
        self,
        ctx: Dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Render a context dict into a Markdown document.

        This is what Claude reads as ``iteration_context.md`` at the
        start of each evolution iteration.

        Args:
            ctx: Context dict produced by ``build_context``.
            iteration: Current iteration number.
            max_iterations: Total iterations configured.

        Returns:
            Markdown string.
        """
        metadata = ctx.get("metadata", {})
        best_score = metadata.get("best_score", 0.0)
        parent_metrics = metadata.get("parent_metrics", {})
        diff_based = metadata.get("diff_based", True)
        parent_id = ctx.get("parent_id", "unknown")

        lines: list[str] = []
        lines.append(f"# Evolution Iteration {iteration} of {max_iterations}")
        lines.append("")
        lines.append(f"**Best Score So Far:** {best_score:.4f}")
        lines.append(f"**Parent ID:** `{parent_id}`")
        lines.append(
            f"**Mode:** {'Diff-based (SEARCH/REPLACE)' if diff_based else 'Full Rewrite'}"
        )
        lines.append("")

        # Parent metrics summary
        if parent_metrics:
            lines.append("## Parent Metrics")
            lines.append("")
            for name, value in parent_metrics.items():
                if isinstance(value, (int, float)):
                    try:
                        lines.append(f"- {name}: {value:.4f}")
                    except (ValueError, TypeError):
                        lines.append(f"- {name}: {value}")
                else:
                    lines.append(f"- {name}: {value}")
            lines.append("")

        # Stagnation report section (v2)
        stagnation_report = metadata.get("stagnation_report")
        if stagnation_report is not None:
            stagnation_section = self._render_stagnation_section(stagnation_report)
            if stagnation_section:
                lines.append(stagnation_section)
                lines.append("")

        # Cross-run memory section (v2)
        cross_run_text = metadata.get("cross_run_memory_text")
        if cross_run_text:
            memory_template = self.template_manager.get_template("cross_run_memory")
            lines.append(memory_template.format(memory_content=cross_run_text))
            lines.append("")

        # Research findings section (v2 phase 2)
        research_text = metadata.get("research_text")
        if research_text:
            research_template = self.template_manager.get_template("research_findings")
            lines.append(research_template.format(research_content=research_text))
            lines.append("")

        # Strategy directive section (v2 phase 3)
        strategy_text = metadata.get("strategy_text")
        if strategy_text:
            lines.append(strategy_text)
            lines.append("")

        # Warm-start cache section
        warm_cache_text = metadata.get("warm_cache_text")
        if warm_cache_text:
            warm_cache_template = self.template_manager.get_template("warm_cache")
            lines.append(warm_cache_template.format(warm_cache_content=warm_cache_text))
            lines.append("")

        # Stepping stones section
        stepping_stones_text = metadata.get("stepping_stones_text")
        if stepping_stones_text:
            lines.append(stepping_stones_text)
            lines.append("")

        # Pairwise comparison (verbal gradient)
        comparison_content = metadata.get("comparison_artifact_content")
        if comparison_content is not None:
            comparison_score = metadata.get("comparison_score", 0.0)
            parent_score_val = parent_metrics.get("combined_score", 0.0)
            lines.append("## Pairwise Comparison (Verbal Gradient)")
            lines.append("")
            lines.append(
                f"The parent (score {parent_score_val:.4f}) and this comparison "
                f"(score {comparison_score:.4f}) differ in their approach. "
                f"Consider what makes the higher-scoring one better."
            )
            lines.append("")
            lines.append("### Comparison Program")
            snippet = comparison_content[:500]
            lines.append(f"```\n{snippet}\n```")
            lines.append("")

        # Embed the full prompt
        lines.append("## Evolution Prompt")
        lines.append("")
        lines.append(ctx.get("prompt", ""))
        lines.append("")

        # System guidance
        lines.append("---")
        lines.append("")
        lines.append("## System Guidance")
        lines.append("")
        lines.append(ctx.get("system_message", ""))
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_stagnation_section(self, report: Any) -> str:
        """Render a StagnationReport into a Markdown section for the prompt.

        Args:
            report: A StagnationReport object with level, iterations_stagnant,
                    best_score, diagnosis, recommendations, suggested_strategy,
                    and failed_approaches attributes.

        Returns:
            Rendered Markdown string, or empty string if level is NONE.
        """
        # Import here to avoid circular dependency
        from claude_evolve.core.stagnation import StagnationLevel

        level = report.level
        if level == StagnationLevel.NONE:
            return ""

        # Get strategy guidance from fragment
        strategy_key = f"strategy_{report.suggested_strategy}"
        if self.template_manager.has_fragment(strategy_key):
            strategy_guidance = self.template_manager.get_fragment(strategy_key)
        else:
            strategy_guidance = report.suggested_strategy

        # Format recommendations as bullet list
        recommendations = "\n".join(f"- {r}" for r in report.recommendations)

        # Format failed approaches section
        failed_section = ""
        if report.failed_approaches:
            header = self.template_manager.get_fragment("stagnation_failed_approaches_header")
            items = "\n".join(
                self.template_manager.get_fragment("stagnation_failed_approach_item", approach=fa)
                for fa in report.failed_approaches
            )
            failed_section = f"{header}{items}\n"

        # Use the stagnation guidance template
        template = self.template_manager.get_template("stagnation_guidance")
        return template.format(
            stagnation_level=level.value.upper(),
            iterations_stagnant=report.iterations_stagnant,
            best_score=report.best_score,
            diagnosis=report.diagnosis,
            recommendations=recommendations,
            strategy_guidance=strategy_guidance,
            failed_approaches_section=failed_section,
        )

    @staticmethod
    def _normalize_parent(parent: Any) -> Dict[str, Any]:
        """Convert a parent (Artifact or dict) to a plain dict."""
        if isinstance(parent, dict):
            return parent
        if hasattr(parent, "to_dict"):
            return parent.to_dict()
        # Fallback: try attribute access
        result: Dict[str, Any] = {}
        for attr in ("id", "content", "code", "metrics", "artifact_type", "metadata"):
            if hasattr(parent, attr):
                result[attr] = getattr(parent, attr)
        return result

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format a metrics dict for prompt inclusion.

        Each metric is rendered as ``- name: value``. Numeric values
        are formatted to four decimal places.
        """
        formatted_parts: list[str] = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, Any],
        previous_programs: List[Dict[str, Any]],
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Analyse metrics and code to suggest improvement areas.

        Returns a newline-separated list of bullet points.
        """
        improvement_areas: list[str] = []
        feature_dimensions = feature_dimensions or []

        current_fitness = get_fitness_score(metrics, feature_dimensions)

        # Compare fitness to last previous program
        if previous_programs:
            prev_metrics = previous_programs[-1].get("metrics", {})
            prev_fitness = get_fitness_score(prev_metrics, feature_dimensions)

            if current_fitness > prev_fitness:
                msg = self.template_manager.get_fragment(
                    "fitness_improved", prev=prev_fitness, current=current_fitness
                )
                improvement_areas.append(msg)
            elif current_fitness < prev_fitness:
                msg = self.template_manager.get_fragment(
                    "fitness_declined", prev=prev_fitness, current=current_fitness
                )
                improvement_areas.append(msg)
            elif abs(current_fitness - prev_fitness) < 1e-6:
                msg = self.template_manager.get_fragment(
                    "fitness_stable", current=current_fitness
                )
                improvement_areas.append(msg)

        # Feature exploration note
        if feature_dimensions:
            feature_coords = format_feature_coordinates(metrics, feature_dimensions)
            if feature_coords == "":
                msg = self.template_manager.get_fragment("no_feature_coordinates")
            else:
                msg = self.template_manager.get_fragment(
                    "exploring_region", features=feature_coords
                )
            improvement_areas.append(msg)

        # Code length check
        threshold = (
            self.config.suggest_simplification_after_chars
        )
        if threshold and len(current_program) > threshold:
            msg = self.template_manager.get_fragment(
                "code_too_long", threshold=threshold
            )
            improvement_areas.append(msg)

        # Default guidance when nothing else applies
        if not improvement_areas:
            improvement_areas.append(
                self.template_manager.get_fragment("no_specific_guidance")
            )

        return "\n".join(f"- {area}" for area in improvement_areas)

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Assemble the evolution history block for the prompt."""
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template(
            "previous_attempt"
        )
        top_program_template = self.template_manager.get_template("top_program")

        # --- Previous attempts (most-recent-first, up to 3) ---
        previous_attempts_str = ""
        selected_previous = previous_programs[
            -min(3, len(previous_programs)) :
        ]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = (
                program.get("changes_description")
                or program.get("metadata", {}).get(
                    "changes",
                    self.template_manager.get_fragment("attempt_unknown_changes"),
                )
            )

            # Format performance metrics
            performance_parts: list[str] = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome by comparing to parent metrics
            parent_metrics = program.get("metadata", {}).get("parent_metrics", {})
            outcome = self.template_manager.get_fragment("attempt_mixed_metrics")

            program_metrics = program.get("metrics", {})
            numeric_improved: list[bool] = []
            numeric_regressed: list[bool] = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)
                if isinstance(prog_value, (int, float)) and isinstance(
                    parent_value, (int, float)
                ):
                    numeric_improved.append(prog_value > parent_value)
                    numeric_regressed.append(prog_value < parent_value)

            if numeric_improved and all(numeric_improved):
                outcome = self.template_manager.get_fragment(
                    "attempt_all_metrics_improved"
                )
            elif numeric_regressed and all(numeric_regressed):
                outcome = self.template_manager.get_fragment(
                    "attempt_all_metrics_regressed"
                )

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # --- Top programs ---
        top_programs_str = ""
        selected_top = top_programs[
            : min(self.config.num_top_programs, len(top_programs))
        ]

        for i, program in enumerate(selected_top):
            use_changes = self.config.programs_as_changes_description
            program_code = (
                program.get("changes_description", "")
                if use_changes
                else program.get("code", program.get("content", ""))
            )
            if not program_code:
                program_code = (
                    "<missing changes_description>" if use_changes else ""
                )

            score = get_fitness_score(
                program.get("metrics", {}), feature_dimensions or []
            )

            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    prefix = self.template_manager.get_fragment(
                        "top_program_metrics_prefix"
                    )
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"{prefix} {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"{prefix} {name} ({value})")
                    else:
                        key_features.append(f"{prefix} {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=(
                        "text"
                        if self.config.programs_as_changes_description
                        else language
                    ),
                    program_snippet=program_code,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # --- Diverse programs (sampled from remaining top_programs) ---
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            remaining = top_programs[self.config.num_top_programs :]
            num_diverse = min(self.config.num_diverse_programs, len(remaining))
            if num_diverse > 0:
                diverse_programs = random.sample(remaining, num_diverse)
                diverse_programs_str += (
                    "\n\n## "
                    + self.template_manager.get_fragment("diverse_programs_title")
                    + "\n\n"
                )

                for i, program in enumerate(diverse_programs):
                    use_changes = self.config.programs_as_changes_description
                    program_code = (
                        program.get("changes_description", "")
                        if use_changes
                        else program.get("code", program.get("content", ""))
                    )
                    if not program_code:
                        program_code = (
                            "<missing changes_description>"
                            if use_changes
                            else ""
                        )

                    score = get_fitness_score(
                        program.get("metrics", {}), feature_dimensions or []
                    )

                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            self.template_manager.get_fragment(
                                "diverse_program_metrics_prefix"
                            )
                            + f" {name}"
                            for name in list(
                                program.get("metrics", {}).keys()
                            )[:2]
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=(
                                "text"
                                if self.config.programs_as_changes_description
                                else language
                            ),
                            program_snippet=program_code,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        combined_programs_str = top_programs_str + diverse_programs_str

        # --- Inspirations section ---
        inspirations_section_str = self._format_inspirations_section(
            inspirations, language, feature_dimensions
        )

        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self,
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Format the inspirations section of the prompt."""
        if not inspirations:
            return ""

        section_template = self.template_manager.get_template("inspirations_section")
        program_template = self.template_manager.get_template("inspiration_program")

        programs_str = ""

        for i, program in enumerate(inspirations):
            use_changes = self.config.programs_as_changes_description
            program_code = (
                program.get("changes_description", "")
                if use_changes
                else program.get("code", program.get("content", ""))
            )
            if not program_code:
                program_code = (
                    "<missing changes_description>" if use_changes else ""
                )

            score = get_fitness_score(
                program.get("metrics", {}), feature_dimensions or []
            )
            program_type = self._determine_program_type(
                program, feature_dimensions or []
            )
            unique_features = self._extract_unique_features(program)

            programs_str += (
                program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=(
                        "text"
                        if self.config.programs_as_changes_description
                        else language
                    ),
                    program_snippet=program_code,
                    unique_features=unique_features,
                )
                + "\n\n"
            )

        return section_template.format(
            inspiration_programs=programs_str.strip()
        )

    def _determine_program_type(
        self,
        program: Dict[str, Any],
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Classify an inspiration program by type label.

        Checks metadata markers first, then falls back to score-based
        classification.
        """
        metadata = program.get("metadata", {})
        score = get_fitness_score(
            program.get("metrics", {}), feature_dimensions or []
        )

        if metadata.get("diverse", False):
            return self.template_manager.get_fragment("inspiration_type_diverse")
        if metadata.get("migrant", False):
            return self.template_manager.get_fragment("inspiration_type_migrant")
        if metadata.get("random", False):
            return self.template_manager.get_fragment("inspiration_type_random")

        if score >= 0.8:
            return self.template_manager.get_fragment(
                "inspiration_type_score_high_performer"
            )
        elif score >= 0.6:
            return self.template_manager.get_fragment(
                "inspiration_type_score_alternative"
            )
        elif score >= 0.4:
            return self.template_manager.get_fragment(
                "inspiration_type_score_experimental"
            )
        else:
            return self.template_manager.get_fragment(
                "inspiration_type_score_exploratory"
            )

    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """Extract notable features of an inspiration program for display."""
        features: list[str] = []

        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if (
                isinstance(changes, str)
                and self.config.include_changes_under_chars
                and len(changes) < self.config.include_changes_under_chars
            ):
                features.append(
                    self.template_manager.get_fragment(
                        "inspiration_changes_prefix", changes=changes
                    )
                )

        # Standout metric values
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(
                        self.template_manager.get_fragment(
                            "inspiration_metrics_excellent",
                            metric_name=metric_name,
                            value=value,
                        )
                    )
                elif value <= 0.3:
                    features.append(
                        self.template_manager.get_fragment(
                            "inspiration_metrics_alternative",
                            metric_name=metric_name,
                        )
                    )

        # Simple code-based heuristics
        code = program.get("code", program.get("content", ""))
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append(
                    self.template_manager.get_fragment("inspiration_code_with_class")
                )
            if "numpy" in code_lower or "np." in code_lower:
                features.append(
                    self.template_manager.get_fragment("inspiration_code_with_numpy")
                )
            if "for" in code_lower and "while" in code_lower:
                features.append(
                    self.template_manager.get_fragment(
                        "inspiration_code_with_mixed_iteration"
                    )
                )
            if (
                self.config.concise_implementation_max_lines
                and len(code.split("\n")) <= self.config.concise_implementation_max_lines
            ):
                features.append(
                    self.template_manager.get_fragment(
                        "inspiration_code_with_concise_line"
                    )
                )
            elif (
                self.config.comprehensive_implementation_min_lines
                and len(code.split("\n"))
                >= self.config.comprehensive_implementation_min_lines
            ):
                features.append(
                    self.template_manager.get_fragment(
                        "inspiration_code_with_comprehensive_line"
                    )
                )

        # Default when no features extracted
        if not features:
            program_type = self._determine_program_type(program)
            features.append(
                self.template_manager.get_fragment(
                    "inspiration_no_features_postfix",
                    program_type=program_type,
                )
            )

        feature_limit = self.config.num_top_programs
        return ", ".join(features[:feature_limit])

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to template placeholders."""
        result = template
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen)
        return result

    def _render_artifacts(
        self, artifacts: Dict[str, Union[str, bytes]]
    ) -> str:
        """Render evaluation artifacts for prompt inclusion.

        Each artifact is placed in a fenced code block under its key
        name. Long artifacts are truncated to ``max_artifact_bytes``.
        """
        if not artifacts:
            return ""

        sections: list[str] = []

        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            if len(content) > self.config.max_artifact_bytes:
                content = (
                    content[: self.config.max_artifact_bytes] + "\n... (truncated)"
                )
            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            title = self.template_manager.get_fragment("artifact_title")
            return "## " + title + "\n\n" + "\n\n".join(sections)
        return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """Safely decode an artifact value to a string.

        Applies the security filter when ``artifact_security_filter``
        is enabled in the config.
        """
        if isinstance(value, str):
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:
        """Remove ANSI escapes and redact common secret patterns."""
        # Strip ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Redact common secret patterns (targeted to avoid matching legitimate code
        # like UUIDs, hashes, variable names, or hex strings in artifacts)
        secret_patterns = [
            (r"sk-[A-Za-z0-9]{20,}", "<REDACTED_API_KEY>"),
            (r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?", "<REDACTED_SECRET>"),
            (r"password\s*[=:]\s*[^\s]+", "password=<REDACTED>"),
            (r"(?:bearer|token)\s+[A-Za-z0-9_\-.]{20,}", "token <REDACTED>"),
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
