"""
Configuration handling for claude_evolve.

Extracted from OpenEvolve's config.py, adapted for Claude Code integration.
LLM configuration is removed since Claude IS the LLM.
Adds evaluator mode support (script/critic/hybrid) and evolution config.
"""

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dacite
import yaml


# Pattern that matches ${VAR} references anywhere in a string.
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value):
    """
    Resolve ${VAR} environment variable references in a string value.

    Supports both full-string matches (e.g., "${HOME}") and embedded
    references (e.g., "${HOME}/output"). Multiple references in one
    string are supported.

    Args:
        value: The value that may contain ${VAR} syntax. Non-string
               values are returned unchanged.

    Returns:
        The resolved value with environment variables expanded, or the
        original value if no match or not a string.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    def _replace_match(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable {var_name} is not set")
        return env_value

    return _ENV_VAR_PATTERN.sub(_replace_match, value)


def _resolve_env_vars_recursive(data):
    """
    Walk a nested dict/list structure and resolve all ${VAR} references
    in string values.
    """
    if isinstance(data, dict):
        return {k: _resolve_env_vars_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars_recursive(item) for item in data]
    if isinstance(data, str):
        return _resolve_env_vars(data)
    return data


@dataclass
class PromptConfig:
    """Configuration for prompt generation.

    Controls how prompts are assembled, including template selection,
    example sampling, and artifact rendering.  LLM-specific fields
    (system_message content, evaluator_system_message) are kept as
    template identifiers but do not configure an LLM backend.
    """

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Large-codebase mode: compact changes descriptions
    programs_as_changes_description: bool = False
    system_message_changes_description: Optional[str] = None
    initial_changes_description: str = ""

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting (not yet implemented)
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = 500
    include_changes_under_chars: Optional[int] = 100
    concise_implementation_max_lines: Optional[int] = 10
    comprehensive_implementation_min_lines: Optional[int] = 50

    # Diff summary formatting for "Previous Attempts" section
    diff_summary_max_line_len: int = 100
    diff_summary_max_lines: int = 30


@dataclass
class DatabaseConfig:
    """Configuration for the MAP-Elites program database.

    Controls population sizing, island topology, feature-map dimensions,
    selection ratios, migration, and artifact storage.
    """

    # General settings
    db_path: Optional[str] = None
    in_memory: bool = True

    # Prompt and response logging
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"

    # Feature map dimensions for MAP-Elites
    # Built-in: "complexity", "diversity", "score" (always available)
    # Custom: Any metric from your evaluator (must be continuous values)
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
    )
    feature_bins: Union[int, Dict[str, int]] = 10
    diversity_reference_size: int = 20

    # Migration parameters for island-based evolution
    migration_interval: int = 50
    migration_rate: float = 0.1

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None
    artifact_size_threshold: int = 32 * 1024  # 32KB
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30
    max_snapshot_artifacts: Optional[int] = 100

    # Similarity
    similarity_threshold: float = 0.99


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation.

    Supports three modes:
      - "script": Run an external evaluator script (default).
      - "critic": Use Claude to evaluate artifacts.
      - "hybrid": Run the script, then use Claude for qualitative feedback.
    """

    # Evaluation mode
    mode: str = "script"  # Options: "script", "critic", "hybrid"

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits (not yet enforced)
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Cascade evaluation
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    distributed: bool = False

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program


@dataclass
class EvolutionConfig:
    """Configuration for the evolution strategy.

    Controls diff-based mutation, content length limits, and the
    search/replace pattern used for diff application.
    """

    diff_based: bool = True
    max_content_length: int = 50000
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"


@dataclass
class EvolutionTraceConfig:
    """Configuration for evolution trace logging."""

    enabled: bool = False
    format: str = "jsonl"  # Options: "jsonl", "json", "hdf5"
    include_code: bool = False
    include_prompts: bool = True
    output_path: Optional[str] = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class StagnationConfig:
    """Configuration for stagnation detection engine."""

    # Iteration thresholds for stagnation levels
    mild_threshold: int = 3
    moderate_threshold: int = 6
    severe_threshold: int = 11
    critical_threshold: int = 20

    # Score comparison tolerance
    score_tolerance: float = 0.001

    # Exploration ratio boosts per level
    exploration_boost_mild: float = 0.1
    exploration_boost_moderate: float = 0.2
    exploration_boost_severe: float = 0.3
    exploration_boost_critical: float = 0.5

    # Enable/disable stagnation detection
    enabled: bool = True


@dataclass
class CrossRunMemoryConfig:
    """Configuration for cross-run memory persistence."""

    # Enable/disable cross-run memory
    enabled: bool = True

    # Directory for memory storage (relative to state_dir)
    memory_dir: str = "cross_run_memory"

    # Maximum number of learnings to persist
    max_learnings: int = 100

    # Maximum number of failed approaches to remember
    max_failed_approaches: int = 50

    # Score improvement threshold to count as a "learning"
    improvement_threshold: float = 0.01


@dataclass
class ResearchConfig:
    """Configuration for the research agent integration.

    Controls when the researcher agent is triggered and how its findings
    are persisted and injected into evolution prompts.
    """

    enabled: bool = False
    trigger: str = "on_stagnation"  # "always", "on_stagnation", "periodic", "never"
    periodic_interval: int = 10
    max_web_searches: int = 5
    persist_findings: bool = True
    research_log_file: str = "research_log.json"


@dataclass
class DiagnosticsConfig:
    """Configuration for the diagnostician agent integration.

    Controls when the diagnostician agent is triggered and how its
    reports are persisted.
    """

    enabled: bool = False
    trigger: str = "on_stagnation"  # "always", "on_stagnation", "never"
    min_stagnation_level: str = "mild"  # Minimum level to trigger
    persist_reports: bool = True
    diagnostic_report_file: str = "diagnostic_report.json"


@dataclass
class Config:
    """Master configuration for claude_evolve.

    Aggregates all sub-configurations and provides serialization helpers
    (``from_yaml``, ``from_dict``, ``to_dict``, ``to_yaml``).

    Unlike OpenEvolve's Config, this omits ``LLMConfig`` entirely because
    Claude Code *is* the LLM -- there is no external model to configure.
    """

    # General settings
    max_iterations: int = 50
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42

    # Claude-evolve specific
    target_score: Optional[float] = None
    artifact_type: str = "python"
    output_dir: str = "./evolve_output"

    # Component configurations
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    cross_run_memory: CrossRunMemoryConfig = field(default_factory=CrossRunMemoryConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # Early stopping settings
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    # Parallel controller settings
    max_tasks_per_child: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file.

        Environment variable references (``${VAR}``) in string values are
        resolved.  A relative ``prompt.template_dir`` is resolved against
        the directory containing the YAML file.
        """
        config_path = Path(path).resolve()
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = cls.from_dict(config_dict)

        # Resolve template_dir relative to config file location
        if config.prompt.template_dir:
            template_path = Path(config.prompt.template_dir)
            if not template_path.is_absolute():
                config.prompt.template_dir = str(
                    (config_path.parent / template_path).resolve()
                )

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Build a ``Config`` from a plain dictionary.

        Nested dicts are recursively mapped to the corresponding
        dataclass.  String values containing ``${VAR}`` are resolved
        via environment variables.  The ``evolution.diff_pattern`` is
        validated as a legal regex.
        """
        # Resolve environment variables in all string values
        config_dict = _resolve_env_vars_recursive(config_dict)

        # Validate diff_pattern if provided
        evolution_dict = config_dict.get("evolution", {})
        if isinstance(evolution_dict, dict) and "diff_pattern" in evolution_dict:
            try:
                re.compile(evolution_dict["diff_pattern"])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in diff_pattern: {e}")

        config: Config = dacite.from_dict(
            data_class=cls,
            data=config_dict,
            config=dacite.Config(
                cast=[List, Union],
            ),
        )

        # Propagate random_seed to database if database seed is None
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire config tree to a plain dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or return defaults.

    If ``config_path`` is ``None`` or points to a non-existent file,
    a default ``Config`` is returned.
    """
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config()
