"""
Artifact dataclass for Claude Evolve.

Generalization of OpenEvolve's Program dataclass.
An artifact can be any text: code, prompts, configs, prose.
"""
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    """Represents an evolvable artifact in the database."""

    # Identification
    id: str
    content: str
    artifact_type: str = "python"

    # Evolution lineage
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # MAP-Elites features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    changes_description: str = ""

    # Evaluation artifacts (stderr, critic feedback, etc.)
    eval_artifacts: Optional[Dict[str, str]] = None

    # Prompt logging (optional)
    prompts: Optional[Dict[str, Any]] = None

    # Embedding for novelty (optional)
    embedding: Optional[List[float]] = None

    # Thought-code coevolution: rationale explaining the approach
    rationale: Optional[str] = None

    # Offspring count for power-law selection weighting
    offspring_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        # Backward compat: OpenEvolve uses 'code'/'language'
        if "code" in data and "content" not in data:
            data = {**data, "content": data.pop("code")}
        if "language" in data and "artifact_type" not in data:
            data = {**data, "artifact_type": data.pop("language")}

        # Handle missing changes_description
        if "changes_description" not in data:
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                data = {
                    **data,
                    "changes_description": metadata.get("changes_description")
                    or metadata.get("changes", ""),
                }
            else:
                data = {**data, "changes_description": ""}

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())
