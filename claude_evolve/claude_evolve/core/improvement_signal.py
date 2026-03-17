"""Continuous improvement signal G_t (inspired by AdaEvolve).

Replaces discrete stagnation levels with a continuous exponential moving average
of improvement magnitude. Drives exploration intensity, strategy selection
modulation, and meta-guidance triggering from a single unified signal.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from claude_evolve.core.stagnation import StagnationLevel


@dataclass
class ImprovementSignal:
    g_t: float = 0.0
    rho: float = 0.95
    i_min: float = 0.1
    i_max: float = 0.7
    meta_threshold: float = 0.12
    per_island_g_t: dict = field(default_factory=dict)

    def update(self, child_score: float, parent_score: float, island_id: int) -> None:
        """Update G_t after an evaluation.

        delta = max((child - parent) / max(parent, 1e-10), 0)
        g_t = rho * g_t + (1 - rho) * delta
        """
        delta = max((child_score - parent_score) / max(parent_score, 1e-10), 0.0)
        delta = min(delta, 10.0)  # Cap to prevent G_t explosion when parent_score ~0
        self.g_t = self.rho * self.g_t + (1 - self.rho) * delta

        # Per-island signal
        island_key = str(island_id)
        prev = self.per_island_g_t.get(island_key, 0.0)
        self.per_island_g_t[island_key] = self.rho * prev + (1 - self.rho) * delta

    @property
    def exploration_intensity(self) -> float:
        """Exploration intensity derived from G_t.

        I_t = I_min + (I_max - I_min) / (1 + G_t / 0.05)
        """
        return self.i_min + (self.i_max - self.i_min) / (1.0 + self.g_t / 0.05)

    def should_trigger_meta_guidance(self) -> bool:
        """True when ALL islands have g_t <= meta_threshold."""
        if not self.per_island_g_t:
            return self.g_t <= self.meta_threshold
        return all(v <= self.meta_threshold for v in self.per_island_g_t.values())

    def derive_stagnation_level(self) -> StagnationLevel:
        """Backward-compatible stagnation level from G_t."""
        if self.g_t > 0.1:
            return StagnationLevel.NONE
        elif self.g_t > 0.05:
            return StagnationLevel.MILD
        elif self.g_t > 0.02:
            return StagnationLevel.MODERATE
        elif self.g_t > 0.005:
            return StagnationLevel.SEVERE
        else:
            return StagnationLevel.CRITICAL

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'g_t': self.g_t,
                'rho': self.rho,
                'i_min': self.i_min,
                'i_max': self.i_max,
                'meta_threshold': self.meta_threshold,
                'per_island_g_t': self.per_island_g_t,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ImprovementSignal':
        if not os.path.exists(path):
            return cls()
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        sig = cls(
            g_t=data.get('g_t', 0.0),
            rho=data.get('rho', 0.95),
            i_min=data.get('i_min', 0.1),
            i_max=data.get('i_max', 0.7),
            meta_threshold=data.get('meta_threshold', 0.12),
        )
        sig.per_island_g_t = data.get('per_island_g_t', {})
        return sig
