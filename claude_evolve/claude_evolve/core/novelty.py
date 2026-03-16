"""
Universal novelty and diversity system for Claude Evolve.

Works across ALL artifact types: code (Python, JS, Rust), text (prompts, docs),
structured (YAML, JSON, SQL), and hybrid. Does NOT assume code structure.

Three layers of novelty measurement:
1. Structural similarity — token-level n-gram overlap (works for any text)
2. Behavioral similarity — metric fingerprint comparison (works if evaluator returns metrics)
3. Semantic fingerprint — lightweight topic/concept extraction (works for prose)

The system auto-selects the appropriate comparison strategy based on artifact type
and available information.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token extraction — artifact-type-aware but universal
# ---------------------------------------------------------------------------

def tokenize(content: str, artifact_type: str = "text") -> List[str]:
    """Extract tokens from content, appropriate for the artifact type.

    For code: splits on whitespace and punctuation, preserving identifiers.
    For prose: splits on word boundaries, lowercased.
    For structured (YAML/JSON): extracts key-value tokens.
    For any type: falls back to word-level tokenization.

    Returns a list of tokens (lowercase, stripped).
    """
    if not content:
        return []

    if artifact_type in ("python", "javascript", "typescript", "rust", "go",
                         "java", "c", "cpp", "ruby", "shell", "bash"):
        # Code: split on non-alphanumeric, keep identifiers
        tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s\w]', content)
        return [t.lower() for t in tokens if len(t) > 1]

    elif artifact_type in ("yaml", "yml", "json", "toml"):
        # Structured: extract keys and significant values
        tokens = re.findall(r'[a-zA-Z_]\w*|"[^"]*"|\'[^\']*\'|[0-9]+(?:\.[0-9]+)?', content)
        return [t.lower().strip('"\'') for t in tokens if len(t) > 0]

    elif artifact_type == "sql":
        # SQL: keywords, table names, column names
        tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+(?:\.[0-9]+)?|[<>=!]+|[,;()]', content)
        return [t.lower() for t in tokens if len(t) > 0]

    else:
        # Text/markdown/prose: word-level, lowercased
        tokens = re.findall(r'\b\w+\b', content.lower())
        return tokens


# ---------------------------------------------------------------------------
# N-gram extraction — universal structural fingerprint
# ---------------------------------------------------------------------------

def ngrams(tokens: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
    """Extract n-grams from a token list.

    If there are fewer tokens than n, falls back to smaller n-grams
    (bigrams, then unigrams) to still capture structure.
    """
    if not tokens:
        return set()
    # Fall back to smaller n if tokens are too short
    effective_n = min(n, len(tokens))
    if effective_n < 1:
        return set()
    return {tuple(tokens[i:i+effective_n]) for i in range(len(tokens) - effective_n + 1)}


# ---------------------------------------------------------------------------
# Structural similarity — works for ANY text artifact
# ---------------------------------------------------------------------------

def structural_similarity(content_a: str, content_b: str,
                          artifact_type: str = "text") -> float:
    """Compute structural similarity between two artifacts using token n-gram overlap.

    This is the universal replacement for line-based Jaccard. Instead of comparing
    lines (code-biased), it compares token trigrams which capture local structure
    regardless of artifact type.

    For code: captures function signatures, variable patterns, algorithmic structure.
    For prose: captures phrase patterns, sentence fragments, idea sequences.
    For configs: captures key-value patterns, nesting structure.

    Returns:
        Float between 0.0 (completely different) and 1.0 (identical).
    """
    if content_a == content_b:
        return 1.0
    if not content_a or not content_b:
        return 0.0

    tokens_a = tokenize(content_a, artifact_type)
    tokens_b = tokenize(content_b, artifact_type)

    if not tokens_a and not tokens_b:
        return 0.0  # Both empty after tokenization = no structure to compare
    if not tokens_a or not tokens_b:
        return 0.0

    # Use multiple n-gram sizes for robustness:
    # Unigrams catch vocabulary overlap, trigrams catch structural patterns
    total_sim = 0.0
    n_levels = 0

    for n_size in [1, 2, 3]:
        ng_a = ngrams(tokens_a, n_size)
        ng_b = ngrams(tokens_b, n_size)
        if ng_a and ng_b:
            intersection = len(ng_a & ng_b)
            union = len(ng_a | ng_b)
            if union > 0:
                total_sim += intersection / union
                n_levels += 1

    return total_sim / max(n_levels, 1)


# ---------------------------------------------------------------------------
# Behavioral similarity — metric fingerprint comparison
# ---------------------------------------------------------------------------

def behavioral_similarity(metrics_a: Dict[str, Any],
                          metrics_b: Dict[str, Any]) -> float:
    """Compute similarity between two artifacts based on their evaluation metrics.

    This captures whether two artifacts BEHAVE the same way, regardless of
    whether their content looks similar. Two very different programs that
    produce identical metrics are behaviorally identical.

    For code: same performance profile = same behavior.
    For prompts: same clarity/effectiveness scores = same behavior.
    For configs: same system metrics = same behavior.

    Returns:
        Float between 0.0 (completely different behavior) and 1.0 (identical).
    """
    if not metrics_a or not metrics_b:
        return 0.0

    # Extract numeric metrics from both
    shared_keys = set()
    for k in metrics_a:
        if k in metrics_b:
            va = metrics_a[k]
            vb = metrics_b[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                shared_keys.add(k)

    if not shared_keys:
        return 0.0

    # Compute normalized distance across shared metrics
    total_diff = 0.0
    for k in shared_keys:
        va = float(metrics_a[k])
        vb = float(metrics_b[k])
        # Normalize by max to get relative difference
        max_val = max(abs(va), abs(vb), 1e-10)
        total_diff += abs(va - vb) / max_val

    avg_diff = total_diff / len(shared_keys)

    # Convert distance to similarity (0 distance = 1.0 similarity)
    return max(0.0, 1.0 - avg_diff)


# ---------------------------------------------------------------------------
# Semantic fingerprint — lightweight concept extraction
# ---------------------------------------------------------------------------

def semantic_fingerprint(content: str, artifact_type: str = "text") -> Set[str]:
    """Extract a lightweight semantic fingerprint from content.

    Instead of full embeddings (which require ML models), this extracts
    meaningful "concepts" that represent what the artifact does/says:

    For code: function names, class names, imported modules, algorithm keywords
    For prose: key noun phrases, topic words (filtered by frequency)
    For configs: top-level keys, significant values
    For SQL: table names, operation types

    Returns a set of concept strings.
    """
    if not content:
        return set()

    concepts = set()

    if artifact_type in ("python",):
        # Extract Python-specific semantic features
        concepts.update(re.findall(r'def\s+(\w+)', content))
        concepts.update(re.findall(r'class\s+(\w+)', content))
        concepts.update(re.findall(r'import\s+(\w+)', content))
        concepts.update(re.findall(r'from\s+(\w+)', content))
        # Algorithm keywords
        for kw in ['sort', 'search', 'optimize', 'minimize', 'maximize',
                    'greedy', 'dynamic', 'recursive', 'iterate', 'linear',
                    'quadratic', 'exponential', 'numpy', 'scipy', 'torch']:
            if kw in content.lower():
                concepts.add(f"algo:{kw}")

    elif artifact_type in ("javascript", "typescript"):
        concepts.update(re.findall(r'function\s+(\w+)', content))
        concepts.update(re.findall(r'class\s+(\w+)', content))
        concepts.update(re.findall(r'(?:import|require)\s*\(?\s*[\'"](\w+)', content))

    elif artifact_type in ("yaml", "yml", "json", "toml"):
        # Top-level keys
        concepts.update(re.findall(r'^(\w+)\s*:', content, re.MULTILINE))
        concepts.update(re.findall(r'"(\w+)"\s*:', content))

    elif artifact_type == "sql":
        # Table names and operations
        concepts.update(re.findall(r'(?:FROM|JOIN|INTO|UPDATE)\s+(\w+)', content, re.IGNORECASE))
        for op in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']:
            if op in content.upper():
                concepts.add(f"op:{op.lower()}")

    else:
        # Prose/markdown: extract significant words (appear 2+ times, length > 4)
        words = re.findall(r'\b\w{5,}\b', content.lower())
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        concepts.update(w for w, c in word_counts.items() if c >= 2)
        # Section headers
        concepts.update(re.findall(r'^#+\s+(.+)$', content, re.MULTILINE))

    return concepts


def semantic_similarity(content_a: str, content_b: str,
                        artifact_type: str = "text") -> float:
    """Compute semantic similarity using concept fingerprints.

    Returns:
        Float between 0.0 (completely different concepts) and 1.0 (identical).
    """
    fp_a = semantic_fingerprint(content_a, artifact_type)
    fp_b = semantic_fingerprint(content_b, artifact_type)

    if not fp_a and not fp_b:
        return 1.0
    if not fp_a or not fp_b:
        return 0.0

    intersection = len(fp_a & fp_b)
    union = len(fp_a | fp_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Combined novelty score — universal, multi-layer
# ---------------------------------------------------------------------------

def compute_novelty(content_a: str, content_b: str,
                    artifact_type: str = "text",
                    metrics_a: Optional[Dict[str, Any]] = None,
                    metrics_b: Optional[Dict[str, Any]] = None,
                    weights: Optional[Dict[str, float]] = None) -> float:
    """Compute universal novelty score between two artifacts.

    Combines structural, behavioral, and semantic similarity with configurable
    weights. Higher novelty = more different = more novel.

    Args:
        content_a, content_b: The artifact content strings.
        artifact_type: Type of artifact for type-aware tokenization.
        metrics_a, metrics_b: Optional evaluation metrics for behavioral comparison.
        weights: Optional weight dict with keys 'structural', 'behavioral', 'semantic'.
                 Defaults to equal weighting, with behavioral given extra weight
                 when metrics are available.

    Returns:
        Float between 0.0 (identical) and 1.0 (completely novel).
    """
    if weights is None:
        if metrics_a and metrics_b:
            weights = {"structural": 0.4, "behavioral": 0.35, "semantic": 0.25}
        else:
            weights = {"structural": 0.6, "behavioral": 0.0, "semantic": 0.4}

    similarity = 0.0
    total_weight = 0.0

    # Structural layer (always available)
    if weights.get("structural", 0) > 0:
        w = weights["structural"]
        similarity += w * structural_similarity(content_a, content_b, artifact_type)
        total_weight += w

    # Behavioral layer (available when metrics exist)
    if weights.get("behavioral", 0) > 0 and metrics_a and metrics_b:
        w = weights["behavioral"]
        similarity += w * behavioral_similarity(metrics_a, metrics_b)
        total_weight += w

    # Semantic layer (always available)
    if weights.get("semantic", 0) > 0:
        w = weights["semantic"]
        similarity += w * semantic_similarity(content_a, content_b, artifact_type)
        total_weight += w

    if total_weight > 0:
        similarity /= total_weight

    # Novelty = 1 - similarity
    return 1.0 - similarity


# ---------------------------------------------------------------------------
# Stepping stones archive — preserves interesting intermediates
# ---------------------------------------------------------------------------

class SteppingStonesArchive:
    """Maintains a diverse archive of intermediate solutions (stepping stones).

    Unlike the main population which keeps the BEST per cell, this archive
    keeps INTERESTING solutions — those that opened new regions of the
    solution space, even if they weren't the highest-scoring.

    Works for any artifact type.
    """

    def __init__(self, max_size: int = 50, novelty_threshold: float = 0.3):
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold
        self.stones: List[Dict[str, Any]] = []

    def try_add(self, content: str, metrics: Dict[str, Any],
                artifact_type: str = "text",
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Try to add a solution as a stepping stone.

        A solution is added if it's sufficiently novel compared to all
        existing stones (novelty > threshold). If the archive is full,
        the oldest stone is evicted.

        Returns True if added, False if rejected (too similar).
        """
        for stone in self.stones:
            novelty = compute_novelty(
                content, stone["content"],
                artifact_type=artifact_type,
                metrics_a=metrics,
                metrics_b=stone.get("metrics", {}),
            )
            if novelty < self.novelty_threshold:
                return False

        entry = {
            "content": content,
            "metrics": metrics,
            "artifact_type": artifact_type,
            "metadata": metadata or {},
        }

        if len(self.stones) >= self.max_size:
            self.stones.pop(0)

        self.stones.append(entry)
        return True

    def get_inspirations(self, current_content: str,
                         artifact_type: str = "text",
                         n: int = 3) -> List[Dict[str, Any]]:
        """Get the most diverse stepping stones relative to current content.

        Selects stones that are DIFFERENT from the current solution,
        providing maximum inspiration for novel directions.
        """
        if not self.stones:
            return []

        scored = []
        for stone in self.stones:
            novelty = compute_novelty(
                current_content, stone["content"],
                artifact_type=artifact_type,
            )
            scored.append((novelty, stone))

        scored.sort(key=lambda x: -x[0])
        return [stone for _, stone in scored[:n]]

    def format_for_prompt(self, current_content: str,
                          artifact_type: str = "text",
                          max_items: int = 3) -> str:
        """Format stepping stones for inclusion in evolution prompts."""
        inspirations = self.get_inspirations(current_content, artifact_type, max_items)
        if not inspirations:
            return ""

        lines = ["## Stepping Stones (Diverse Intermediate Solutions)", ""]
        lines.append("These are interesting intermediate solutions from earlier in the search.")
        lines.append("They may contain ideas worth combining with the current approach:")
        lines.append("")

        for i, stone in enumerate(inspirations, 1):
            score = stone["metrics"].get("combined_score", "N/A")
            lines.append(f"### Stepping Stone {i} (score: {score})")
            # Show a preview, not the full content
            preview = stone["content"][:500]
            if len(stone["content"]) > 500:
                preview += "\n... (truncated)"
            lines.append(f"```\n{preview}\n```")
            lines.append("")

        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize for persistence."""
        return list(self.stones)

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]],
                  max_size: int = 50,
                  novelty_threshold: float = 0.3) -> "SteppingStonesArchive":
        """Deserialize from persistence."""
        archive = cls(max_size=max_size, novelty_threshold=novelty_threshold)
        archive.stones = data[:max_size]
        return archive
