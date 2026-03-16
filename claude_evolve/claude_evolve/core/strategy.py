"""
Strategy management for Claude Evolve.

Maintains a population of text-based strategies that guide how candidates
are generated. Strategies are themselves evolved: the most successful ones
survive and get mutated/crossed to produce new ones.

Inspired by: CodeEvolve (meta-prompting), DSPy MIPROv2 (prompt optimization),
EvoPrompt (evolutionary prompt optimization).
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """A text-based strategy for candidate generation."""
    id: str
    name: str
    description: str  # Natural language description of the approach
    generation_approach: str = "diff"  # "diff", "full_rewrite", "solver_hybrid", "from_scratch"
    research_focus: str = ""  # What aspects to research
    template_key: str = "diff_user"  # Which template to prefer
    exploration_weight: float = 0.5  # 0=exploit, 1=explore
    score_history: List[float] = field(default_factory=list)
    times_used: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)

    @property
    def best_score(self) -> float:
        return max(self.score_history) if self.score_history else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        from dataclasses import fields as dc_fields
        valid = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


# Default strategies that are always available
DEFAULT_STRATEGIES = [
    Strategy(
        id="default-incremental",
        name="Incremental Improvement",
        description="Make small, targeted improvements to the current best. Focus on the weakest metric and make minimal changes to address it.",
        generation_approach="diff",
        template_key="diff_user",
        exploration_weight=0.2,
    ),
    Strategy(
        id="default-creative",
        name="Creative Leap",
        description="Ignore the current approach entirely. Think about what a domain expert would do differently. Try a completely novel algorithm or technique.",
        generation_approach="full_rewrite",
        template_key="full_rewrite_user",
        exploration_weight=0.9,
    ),
    Strategy(
        id="default-hybrid",
        name="Hybrid Synthesis",
        description="Study the top 3 programs and inspiration programs. Identify the best technique from each. Synthesize a new solution combining their strengths.",
        generation_approach="diff",
        template_key="diff_user",
        exploration_weight=0.5,
    ),
    Strategy(
        id="default-research-first",
        name="Research-Driven",
        description="Spend 80% of effort on understanding the problem deeply. Search for papers, theoretical bounds, and known approaches. Then implement the most promising finding.",
        generation_approach="full_rewrite",
        template_key="full_rewrite_user",
        exploration_weight=0.7,
        research_focus="theoretical bounds, state-of-the-art algorithms, related work",
    ),
    Strategy(
        id="default-solver",
        name="Solver Hybrid",
        description="Formulate the problem as a constraint satisfaction problem. Use SAT/SMT solvers, LP, or scipy.optimize. The LLM formulates; the solver searches.",
        generation_approach="solver_hybrid",
        template_key="diff_user",
        exploration_weight=0.6,
        research_focus="constraint formulation, SAT encoding, mathematical programming",
    ),
    Strategy(
        id="default-accumulate",
        name="Multi-Iteration Accumulation",
        description=(
            "Instead of starting fresh, load the best result from the warm cache "
            "and continue optimizing from where the last iteration left off. "
            "Save your improved result back to the warm cache before submitting. "
            "This allows sustained computation across hundreds of iterations. "
            "Use: warm_cache.get_numpy('best_solution') to load, then run more "
            "SA/tabu/search iterations, then warm_cache.put_numpy('best_solution', result)."
        ),
        generation_approach="diff",
        template_key="diff_user",
        exploration_weight=0.3,
        research_focus="incremental optimization, warm-start continuation",
    ),
    Strategy(
        id="default-decompose",
        name="Problem Decomposition",
        description=(
            "Break the problem into independent sub-problems. "
            "For R(5,5): separate 'find best K_42 base' from 'optimize vertex extension' from 'refine full graph'. "
            "For business problems: separate 'identify constraints' from 'optimize objective'. "
            "Solve each sub-problem independently, then combine."
        ),
        generation_approach="full_rewrite",
        template_key="decomposition_user",
        exploration_weight=0.6,
        research_focus="problem structure, independent components, divide-and-conquer",
    ),
]


class StrategyManager:
    """Manages a population of evolution strategies.

    Strategies are loaded from/saved to a JSON file. The manager provides
    selection (weighted by score), and tracking of strategy outcomes.
    """

    def __init__(self, strategies_path: str):
        self.strategies_path = strategies_path
        self.strategies: List[Strategy] = []
        self._loaded = False

    def load(self) -> None:
        """Load strategies from disk, or initialize with defaults."""
        if os.path.exists(self.strategies_path):
            with open(self.strategies_path, "r") as f:
                data = json.load(f)
            self.strategies = [Strategy.from_dict(d) for d in data]
        else:
            self.strategies = [Strategy.from_dict(s.to_dict()) for s in DEFAULT_STRATEGIES]
        self._loaded = True

    def save(self) -> None:
        """Save strategies to disk."""
        os.makedirs(os.path.dirname(self.strategies_path) or ".", exist_ok=True)
        with open(self.strategies_path, "w") as f:
            json.dump([s.to_dict() for s in self.strategies], f, indent=2)

    def select_strategy(self, stagnation_level: str = "none") -> Strategy:
        """Select a strategy based on stagnation level and past performance.

        At low stagnation: favor high-scoring strategies (exploit).
        At high stagnation: favor high-exploration strategies (explore).
        """
        if not self.strategies:
            self.strategies = [Strategy.from_dict(s.to_dict()) for s in DEFAULT_STRATEGIES]

        # Weight strategies based on stagnation level
        stagnation_weights = {
            "none": 0.2,       # Favor exploitation
            "mild": 0.4,
            "moderate": 0.6,
            "severe": 0.8,
            "critical": 0.95,  # Almost pure exploration
        }
        explore_weight = stagnation_weights.get(stagnation_level, 0.5)

        import random

        # Score each strategy
        scored = []
        for s in self.strategies:
            # Blend performance score with exploration weight
            perf_score = s.avg_score if s.score_history else 0.5  # Unknown = neutral
            explore_score = s.exploration_weight

            # Bonus for untried strategies
            novelty_bonus = 0.3 if s.times_used == 0 else 0.0

            combined = (1 - explore_weight) * perf_score + explore_weight * explore_score + novelty_bonus
            scored.append((s, combined))

        # Softmax-like selection (temperature based on stagnation)
        total = sum(score for _, score in scored)
        if total <= 0:
            return random.choice(self.strategies)

        r = random.random() * total
        cumulative = 0.0
        for strategy, score in scored:
            cumulative += score
            if cumulative >= r:
                return strategy

        return scored[-1][0]

    def record_outcome(self, strategy_id: str, score: float) -> None:
        """Record the outcome of using a strategy."""
        for s in self.strategies:
            if s.id == strategy_id:
                s.score_history.append(score)
                s.times_used += 1
                break

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy to the population."""
        self.strategies.append(strategy)

    def get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by ID."""
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        return None

    def format_for_prompt(self, selected_strategy: Strategy) -> str:
        """Format the selected strategy as guidance for the evolution prompt."""
        lines = [
            "## Strategy Directive",
            "",
            f"**Strategy:** {selected_strategy.name}",
            f"**Approach:** {selected_strategy.generation_approach}",
            f"**Exploration level:** {selected_strategy.exploration_weight:.1f}",
            "",
            f"{selected_strategy.description}",
        ]

        if selected_strategy.research_focus:
            lines.extend([
                "",
                f"**Research focus:** {selected_strategy.research_focus}",
            ])

        if selected_strategy.score_history:
            lines.extend([
                "",
                f"**Track record:** Used {selected_strategy.times_used} times, "
                f"avg score: {selected_strategy.avg_score:.4f}, "
                f"best: {selected_strategy.best_score:.4f}",
            ])

        return "\n".join(lines)
