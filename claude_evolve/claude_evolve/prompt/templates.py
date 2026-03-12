"""
Template manager for claude_evolve prompt system.

Handles loading, cascading overrides, and formatting of prompt templates
and fragments used to construct per-iteration context for Claude Code.

Extracted and adapted from OpenEvolve's prompt/templates.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline default templates (used as a final fallback when the default
# template directory is missing or a specific file is absent)
# ---------------------------------------------------------------------------

BASE_SYSTEM_TEMPLATE = (
    "You are an expert software developer tasked with iteratively improving a codebase.\n"
    "Your goal is to maximize the FITNESS SCORE while exploring diverse solutions across feature dimensions.\n"
    "The system maintains a collection of diverse programs - both high fitness AND diversity are valuable."
)

BASE_EVALUATOR_SYSTEM_TEMPLATE = (
    "You are an expert code reviewer.\n"
    "Your job is to analyze the provided code and evaluate it systematically."
)

DIFF_USER_TEMPLATE = (
    "# Current Program Information\n"
    "- Fitness: {fitness_score}\n"
    "- Feature coordinates: {feature_coords}\n"
    "- Focus areas: {improvement_areas}\n"
    "\n"
    "{artifacts}\n"
    "\n"
    "# Program Evolution History\n"
    "{evolution_history}\n"
    "\n"
    "# Current Program\n"
    "```{language}\n"
    "{current_program}\n"
    "```\n"
    "\n"
    "# Task\n"
    "Suggest improvements to the program that will improve its FITNESS SCORE.\n"
    "The system maintains diversity across these dimensions: {feature_dimensions}\n"
    "Different solutions with similar fitness but different features are valuable.\n"
    "\n"
    "You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:\n"
    "\n"
    "<<<<<<< SEARCH\n"
    "# Original code to find and replace (must match exactly)\n"
    "=======\n"
    "# New replacement code\n"
    ">>>>>>> REPLACE\n"
    "\n"
    "Example of valid diff format:\n"
    "<<<<<<< SEARCH\n"
    "for i in range(m):\n"
    "    for j in range(p):\n"
    "        for k in range(n):\n"
    "            C[i, j] += A[i, k] * B[k, j]\n"
    "=======\n"
    "# Reorder loops for better memory access pattern\n"
    "for i in range(m):\n"
    "    for k in range(n):\n"
    "        for j in range(p):\n"
    "            C[i, j] += A[i, k] * B[k, j]\n"
    ">>>>>>> REPLACE\n"
    "\n"
    "You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.\n"
    "Be thoughtful about your changes and explain your reasoning thoroughly.\n"
    "\n"
    "IMPORTANT: Do not rewrite the entire program - focus on targeted improvements."
)

FULL_REWRITE_USER_TEMPLATE = (
    "# Current Program Information\n"
    "- Fitness: {fitness_score}\n"
    "- Feature coordinates: {feature_coords}\n"
    "- Focus areas: {improvement_areas}\n"
    "\n"
    "{artifacts}\n"
    "\n"
    "# Program Evolution History\n"
    "{evolution_history}\n"
    "\n"
    "# Current Program\n"
    "```{language}\n"
    "{current_program}\n"
    "```\n"
    "\n"
    "# Task\n"
    "Rewrite the program to improve its FITNESS SCORE.\n"
    "The system maintains diversity across these dimensions: {feature_dimensions}\n"
    "Different solutions with similar fitness but different features are valuable.\n"
    "Provide the complete new program code.\n"
    "\n"
    "IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs\n"
    "as the original program, but with improved internal implementation.\n"
    "\n"
    "```{language}\n"
    "# Your rewritten program here\n"
    "```"
)

EVOLUTION_HISTORY_TEMPLATE = (
    "## Previous Attempts\n"
    "\n"
    "{previous_attempts}\n"
    "\n"
    "## Top Performing Programs\n"
    "\n"
    "{top_programs}\n"
    "\n"
    "{inspirations_section}"
)

PREVIOUS_ATTEMPT_TEMPLATE = (
    "### Attempt {attempt_number}\n"
    "- Changes: {changes}\n"
    "- Metrics: {performance}\n"
    "- Outcome: {outcome}"
)

TOP_PROGRAM_TEMPLATE = (
    "### Program {program_number} (Score: {score})\n"
    "```{language}\n"
    "{program_snippet}\n"
    "```\n"
    "Key features: {key_features}"
)

INSPIRATIONS_SECTION_TEMPLATE = (
    "## Inspiration Programs\n"
    "\n"
    "These programs represent diverse approaches and creative solutions "
    "that may inspire new ideas:\n"
    "\n"
    "{inspiration_programs}"
)

INSPIRATION_PROGRAM_TEMPLATE = (
    "### Inspiration {program_number} (Score: {score}, Type: {program_type})\n"
    "```{language}\n"
    "{program_snippet}\n"
    "```\n"
    "Unique approach: {unique_features}"
)

EVALUATION_TEMPLATE = (
    "Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:\n"
    "1. Readability: How easy is the code to read and understand?\n"
    "2. Maintainability: How easy would the code be to maintain and modify?\n"
    "3. Efficiency: How efficient is the code in terms of time and space complexity?\n"
    "\n"
    "For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.\n"
    "\n"
    "Code to evaluate:\n"
    "```python\n"
    "{current_program}\n"
    "```\n"
    "\n"
    "Return your evaluation as a JSON object with the following format:\n"
    "{{\n"
    '    "readability": [score],\n'
    '    "maintainability": [score],\n'
    '    "efficiency": [score],\n'
    '    "reasoning": "[brief explanation of scores]"\n'
    "}}"
)

# Inline fallback lookup used when files are unavailable
_INLINE_DEFAULTS: Dict[str, str] = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
}

# Inline fallback fragments
_INLINE_FRAGMENTS: Dict[str, str] = {
    "fitness_improved": "Fitness improved: {prev:.4f} -> {current:.4f}",
    "fitness_declined": (
        "Fitness declined: {prev:.4f} -> {current:.4f}. "
        "Consider revising recent changes."
    ),
    "fitness_stable": "Fitness unchanged at {current:.4f}",
    "exploring_region": "Exploring {features} region of solution space",
    "metrics_label": "Metrics: {metrics}",
    "outcome_all_improved": "All metrics improved",
    "outcome_all_regressed": "All metrics regressed",
    "outcome_mixed": "Mixed results",
    "outcome_fitness_improved": "Fitness improved (exploring new features)",
    "key_features_prefix": "Strong in",
    "code_too_long": (
        "Consider simplifying - code length exceeds {threshold} characters"
    ),
    "no_specific_guidance": (
        "Focus on improving fitness while maintaining diversity"
    ),
    "metrics_improved": (
        "Metrics showing improvement: {metrics}. "
        "Consider continuing with similar approaches."
    ),
    "metrics_regressed": (
        "Metrics showing changes: {metrics}. "
        "Consider different approaches in these areas."
    ),
    "code_simplification": (
        "Consider simplifying the code to improve readability and maintainability"
    ),
    "default_improvement": (
        "Focus on improving the fitness score while exploring diverse solutions"
    ),
    "no_feature_coordinates": "No feature coordinates",
    "artifact_title": "Last Execution Output",
    "diverse_programs_title": "Diverse Programs",
    "attempt_unknown_changes": "Unknown changes",
    "attempt_all_metrics_improved": "Improvement in all metrics",
    "attempt_all_metrics_regressed": "Regression in all metrics",
    "attempt_mixed_metrics": "Mixed results",
    "top_program_metrics_prefix": "Performs well on",
    "diverse_program_metrics_prefix": "Alternative approach to",
    "inspiration_type_diverse": "Diverse",
    "inspiration_type_migrant": "Migrant",
    "inspiration_type_random": "Random",
    "inspiration_type_score_high_performer": "High-Performer",
    "inspiration_type_score_alternative": "Alternative",
    "inspiration_type_score_experimental": "Experimental",
    "inspiration_type_score_exploratory": "Exploratory",
    "inspiration_changes_prefix": "Modification: {changes}",
    "inspiration_metrics_excellent": "Excellent {metric_name} ({value:.3f})",
    "inspiration_metrics_alternative": "Alternative {metric_name} approach",
    "inspiration_code_with_class": "Object-oriented approach",
    "inspiration_code_with_numpy": "NumPy-based implementation",
    "inspiration_code_with_mixed_iteration": "Mixed iteration strategies",
    "inspiration_code_with_concise_line": "Concise implementation",
    "inspiration_code_with_comprehensive_line": "Comprehensive implementation",
    "inspiration_no_features_postfix": "{program_type} approach to the problem",
}


class TemplateManager:
    """Manages prompt templates with cascading override support.

    Loading priority (later overrides earlier):
      1. Inline defaults (always available)
      2. Default template files shipped with the package
      3. Custom template directory (user-provided)

    Templates are plain-text files (``*.txt``) looked up by stem name.
    Fragments are short reusable strings stored in ``fragments.json``.
    """

    def __init__(self, custom_template_dir: Optional[str] = None) -> None:
        self.default_dir = Path(__file__).parent / "default_templates"
        self.custom_dir = Path(custom_template_dir) if custom_template_dir else None

        # Seed with inline defaults so templates are always available
        self.templates: Dict[str, str] = dict(_INLINE_DEFAULTS)
        self.fragments: Dict[str, str] = dict(_INLINE_FRAGMENTS)

        # Layer 2: default files shipped with package
        self._load_from_directory(self.default_dir)

        # Layer 3: custom user overrides
        if self.custom_dir:
            if self.custom_dir.exists():
                self._load_from_directory(self.custom_dir)
            else:
                logger.warning(
                    "Custom template directory does not exist: %s, using defaults.",
                    self.custom_dir,
                )

    def _load_from_directory(self, directory: Path) -> None:
        """Load all templates and fragments from a directory.

        ``.txt`` files are loaded as templates (stem -> content).
        ``fragments.json`` is merged into the fragment dictionary.
        """
        if not directory.exists():
            return

        # Load .txt templates
        for txt_file in directory.glob("*.txt"):
            template_name = txt_file.stem
            with open(txt_file, "r", encoding="utf-8") as f:
                self.templates[template_name] = f.read()

        # Load fragments.json if present
        fragments_file = directory / "fragments.json"
        if fragments_file.exists():
            with open(fragments_file, "r", encoding="utf-8") as f:
                loaded_fragments = json.load(f)
                self.fragments.update(loaded_fragments)

    def get_template(self, name: str) -> str:
        """Return a template by name.

        Raises:
            ValueError: If the template name is not found.
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def get_fragment(self, name: str, **kwargs) -> str:
        """Return a formatted fragment.

        If the fragment is missing, returns a placeholder string.
        If formatting fails (missing keys), returns an error string.
        """
        if name not in self.fragments:
            return f"[Missing fragment: {name}]"
        try:
            return self.fragments[name].format(**kwargs)
        except KeyError as e:
            return f"[Fragment formatting error: {e}]"

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template at runtime."""
        self.templates[template_name] = template

    def add_fragment(self, fragment_name: str, fragment: str) -> None:
        """Add or update a fragment at runtime."""
        self.fragments[fragment_name] = fragment

    def list_templates(self) -> list:
        """Return sorted list of available template names."""
        return sorted(self.templates.keys())

    def list_fragments(self) -> list:
        """Return sorted list of available fragment names."""
        return sorted(self.fragments.keys())

    def has_template(self, name: str) -> bool:
        """Check whether a template exists."""
        return name in self.templates

    def has_fragment(self, name: str) -> bool:
        """Check whether a fragment exists."""
        return name in self.fragments
