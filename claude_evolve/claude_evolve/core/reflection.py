"""Pairwise reflection / verbal gradients engine (inspired by ReEvo, NeurIPS 2024).

Generates short-term reflections by comparing artifact pairs using semantic
fingerprint set differences. Accumulates into long-term reflections via
concept frequency counting. Fully deterministic -- no LLM calls.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from claude_evolve.core.artifact import Artifact
from claude_evolve.core.novelty import semantic_fingerprint


@dataclass
class ShortReflection:
    better_id: str
    worse_id: str
    better_score: float
    worse_score: float
    insight: str
    iteration: int


class ReflectionEngine:
    """Generates and accumulates pairwise reflections."""

    def __init__(self, state_dir: str, max_short: int = 20, synthesis_interval: int = 5):
        self.state_dir = state_dir
        self.max_short = max_short
        self.synthesis_interval = synthesis_interval
        self.short_reflections: list[ShortReflection] = []
        self.long_reflection: str = ""

    def generate_short_reflection(self, better: Artifact, worse: Artifact) -> ShortReflection:
        """Compare two artifacts using semantic fingerprint differences."""
        better_fp = semantic_fingerprint(better.content, better.artifact_type)
        worse_fp = semantic_fingerprint(worse.content, worse.artifact_type)
        added = sorted(better_fp - worse_fp)[:3]
        removed = sorted(worse_fp - better_fp)[:3]

        parts = []
        if added:
            parts.append(f"Better adds: {', '.join(added)}")
        if removed:
            parts.append(f"Drops: {', '.join(removed)}")
        if not parts:
            # Fallback: line count comparison
            b_lines = better.content.count('\n')
            w_lines = worse.content.count('\n')
            parts.append(f"Better has {b_lines} lines vs {w_lines}")

        insight = ". ".join(parts)
        better_score = better.metrics.get('combined_score', 0.0) if hasattr(better, 'metrics') and better.metrics else 0.0
        worse_score = worse.metrics.get('combined_score', 0.0) if hasattr(worse, 'metrics') and worse.metrics else 0.0

        ref = ShortReflection(
            better_id=better.id, worse_id=worse.id,
            better_score=better_score, worse_score=worse_score,
            insight=insight, iteration=0,
        )
        self.short_reflections.append(ref)
        self.short_reflections = self.short_reflections[-self.max_short:]
        return ref

    def accumulate_long_reflection(self, current_iteration: int) -> Optional[str]:
        """Every synthesis_interval iterations, compress short reflections."""
        if current_iteration <= 0 or current_iteration % self.synthesis_interval != 0:
            return None
        if not self.short_reflections:
            return None

        # Count concept occurrences across all insights
        concept_counter: Counter = Counter()
        for ref in self.short_reflections:
            for word in ref.insight.split():
                word = word.strip(".,:")
                if len(word) > 3 and word.isalpha():
                    concept_counter[word] += 1

        top_concepts = [c for c, _ in concept_counter.most_common(5)]
        if top_concepts:
            self.long_reflection = f"Key patterns: {', '.join(top_concepts)}"
        return self.long_reflection

    def format_for_prompt(self) -> str:
        """Format for injection into iteration context."""
        parts = []
        if self.short_reflections:
            latest = self.short_reflections[-1]
            parts.append(f"## Verbal Gradient\n{latest.insight}")
        if self.long_reflection:
            parts.append(f"## Accumulated Wisdom\n{self.long_reflection}")
        return "\n\n".join(parts)

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            'short_reflections': [
                {'better_id': r.better_id, 'worse_id': r.worse_id,
                 'better_score': r.better_score, 'worse_score': r.worse_score,
                 'insight': r.insight, 'iteration': r.iteration}
                for r in self.short_reflections
            ],
            'long_reflection': self.long_reflection,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load_from(cls, path: str) -> 'ReflectionEngine':
        eng = cls(os.path.dirname(path))
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            eng.short_reflections = [
                ShortReflection(**r) for r in data.get('short_reflections', [])
            ]
            eng.long_reflection = data.get('long_reflection', '')
        return eng
