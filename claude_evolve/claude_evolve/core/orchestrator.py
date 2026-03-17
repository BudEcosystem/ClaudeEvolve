"""IterationOrchestrator: wires all feature modules for next/submit lifecycle.

Coordinates: ImprovementSignal, UCBStrategySelector, MetaScratchpad,
ReflectionEngine, CrossRunMemory, StrategyManager, ArtifactDatabase.
"""

import json
import os
from typing import Optional

from claude_evolve.config import Config
from claude_evolve.core.artifact import Artifact
from claude_evolve.core.database import ArtifactDatabase
from claude_evolve.core.improvement_signal import ImprovementSignal
from claude_evolve.core.memory import CrossRunMemory
from claude_evolve.core.reflection import ReflectionEngine
from claude_evolve.core.scratchpad import MetaScratchpad
from claude_evolve.core.strategy import DEFAULT_STRATEGIES, StrategyManager
from claude_evolve.core.ucb_selector import UCBStrategySelector


class IterationOrchestrator:
    """Coordinates all feature modules for the evolution lifecycle."""

    def __init__(self, state_dir: str, config: Config, db: Optional[ArtifactDatabase] = None):
        self.state_dir = state_dir
        self.config = config
        os.makedirs(state_dir, exist_ok=True)
        self.db = db or ArtifactDatabase(config.database)

        # Load improvement signal
        self.signal = ImprovementSignal.load(
            os.path.join(state_dir, 'improvement_signal.json'))

        # Load meta-scratchpad
        self.scratchpad = MetaScratchpad(
            state_dir, config.scratchpad.synthesis_interval)

        # Load reflection engine
        self.reflection = ReflectionEngine.load_from(
            os.path.join(state_dir, 'reflections.json'))

        # UCB selector -- initialize with strategy IDs from StrategyManager
        ucb_path = os.path.join(state_dir, 'strategy_bandit.json')
        if os.path.exists(ucb_path):
            self.ucb = UCBStrategySelector.load(ucb_path)
        else:
            strategy_path = os.path.join(state_dir, 'strategies.json')
            strategy_mgr = StrategyManager(strategy_path)
            strategy_mgr.load()
            self.ucb = UCBStrategySelector(
                [s.id for s in strategy_mgr.strategies],
                c=config.selection.ucb_c,
                decay=config.selection.ucb_decay)

        # Cross-run memory
        memory_dir = os.path.join(state_dir, config.cross_run_memory.memory_dir)
        self.memory = CrossRunMemory(memory_dir)
        self.memory.load()

    def prepare_next_iteration(self, iteration: int) -> dict:
        """Run all next-phase logic. Returns context dict for the builder."""
        ei = self.signal.exploration_intensity
        stag_level = self.signal.derive_stagnation_level()

        # Strategy selection
        if self.config.selection.strategy_selection == "ucb1":
            strategy_id = self.ucb.select(ei)
        else:
            strategy_id = "default-incremental"

        # Parent selection
        island_id = iteration % max(len(self.db.islands), 1)
        if self.config.selection.parent_selection == "power_law":
            parent = self.db.select_parent_power_law(
                island_id, ei, self.config.improvement_signal.i_max)
        else:
            if self.db.artifacts:
                parent, _inspirations = self.db.sample()
            else:
                parent = None

        # Comparison artifact for pairwise reflection
        comparison = None
        if parent:
            top = self.db.get_top_programs(n=5)
            parent_score = parent.metrics.get('combined_score', 0) if parent.metrics else 0
            candidates = [a for a in top if a.id != parent.id]
            if candidates:
                candidates.sort(key=lambda a: abs(
                    (a.metrics.get('combined_score', 0) if a.metrics else 0) - parent_score
                ), reverse=True)
                comparison = candidates[0]

        # Generate short reflection
        if parent and comparison:
            parent_cs = (parent.metrics or {}).get('combined_score', 0)
            comp_cs = (comparison.metrics or {}).get('combined_score', 0)
            if parent_cs >= comp_cs:
                self.reflection.generate_short_reflection(parent, comparison)
            else:
                self.reflection.generate_short_reflection(comparison, parent)
        self.reflection.accumulate_long_reflection(iteration)

        # Meta-scratchpad
        scratchpad_text = self.scratchpad.load() or ""
        if self.scratchpad.should_synthesize(iteration):
            top_arts = self.db.get_top_programs(n=5)
            scores = [
                a.metrics.get('combined_score', 0) for a in top_arts
            ] if top_arts else []
            failed = []
            failures_path = os.path.join(self.state_dir, 'recent_failures.json')
            if os.path.exists(failures_path):
                with open(failures_path, encoding='utf-8') as f:
                    failed = [e.get('approach', '') for e in json.load(f)]
            scratchpad_text = self.scratchpad.synthesize(top_arts, scores, failed)
            self.scratchpad.save(scratchpad_text)

        # Recent failures
        failures_text = ""
        failures_path = os.path.join(self.state_dir, 'recent_failures.json')
        if os.path.exists(failures_path):
            with open(failures_path, encoding='utf-8') as f:
                failures = json.load(f)
            if failures:
                lines = ["## Recent Failures (Avoid These)"]
                for fe in failures:
                    err = fe.get('error') or f"score dropped to {fe.get('score', '?')}"
                    lines.append(f"- Approach: {fe.get('approach', '?')}. Result: {err}")
                failures_text = "\n".join(lines)

        # Meta-guidance check
        meta_guidance = ""
        if self.signal.should_trigger_meta_guidance():
            meta_guidance = (
                "## BREAKTHROUGH REQUIRED\n"
                "All search directions have stagnated. You MUST try a fundamentally "
                "different algorithmic approach. Analyze the evaluator source and current "
                "best to identify completely untried strategies. Do NOT make incremental changes."
            )

        # Save iteration manifest
        self._save_manifest({
            "iteration": iteration,
            "run_id": self._get_run_id(),
            "selected_strategy_id": strategy_id,
            "parent_artifact_id": parent.id if parent else None,
            "parent_score": (parent.metrics or {}).get('combined_score', 0.0) if parent else 0.0,
            "parent_island_id": island_id,
        })

        # Save module states
        self.signal.save(os.path.join(self.state_dir, 'improvement_signal.json'))
        self.ucb.save(os.path.join(self.state_dir, 'strategy_bandit.json'))
        self.reflection.save(os.path.join(self.state_dir, 'reflections.json'))

        return {
            'parent': parent,
            'comparison': comparison,
            'strategy_name': strategy_id,
            'exploration_intensity': ei,
            'stagnation_level': stag_level,
            'scratchpad_text': scratchpad_text,
            'failures_text': failures_text,
            'reflection_text': self.reflection.format_for_prompt(),
            'meta_guidance': meta_guidance,
        }

    def process_submission(self, candidate_content: str, metrics: dict) -> dict:
        """Run all submit-phase logic. Returns result dict."""
        manifest = self._load_manifest()
        score = metrics.get('combined_score', 0.0)
        parent_score = manifest.get('parent_score', 0.0)
        strategy_id = manifest.get('selected_strategy_id', 'unknown')
        island_id = manifest.get('parent_island_id', 0)
        run_id = manifest.get('run_id', 'unknown')
        iteration = manifest.get('iteration', 0)

        # Update G_t signal
        self.signal.update(score, parent_score, island_id)

        # Record UCB outcome
        self.ucb.record(strategy_id, score - parent_score)

        # Update cross-run memory
        if score <= parent_score:
            self.memory.add_failed_approach(
                description=f"Strategy '{strategy_id}' on iteration {iteration}",
                score=score, iteration=iteration, run_id=run_id)
        else:
            self.memory.add_strategy(
                name=strategy_id,
                description=f"Improved from {parent_score:.4f} to {score:.4f}",
                score=score, run_id=run_id)
        self.memory.save()

        # Capture failure for reflexion
        if score < parent_score * 0.95 or metrics.get('error'):
            self._capture_failure(strategy_id, metrics.get('error'), score, parent_score, iteration)

        # Increment parent offspring count
        parent_id = manifest.get('parent_artifact_id')
        if parent_id and parent_id in self.db.artifacts:
            self.db.artifacts[parent_id].offspring_count += 1

        # Save states
        self.signal.save(os.path.join(self.state_dir, 'improvement_signal.json'))
        self.ucb.save(os.path.join(self.state_dir, 'strategy_bandit.json'))

        return {'score': score, 'parent_score': parent_score, 'improved': score > parent_score}

    def _save_manifest(self, data: dict) -> None:
        path = os.path.join(self.state_dir, 'current_iteration.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _load_manifest(self) -> dict:
        path = os.path.join(self.state_dir, 'current_iteration.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _get_run_id(self) -> str:
        meta_path = os.path.join(self.state_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                return json.load(f).get('run_id', 'unknown')
        return 'unknown'

    def _capture_failure(self, strategy, error, score, parent_score, iteration):
        path = os.path.join(self.state_dir, 'recent_failures.json')
        failures = []
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                failures = json.load(f)
        failures.append({
            "approach": strategy, "error": error,
            "score": score, "parent_score": parent_score,
            "iteration": iteration,
        })
        failures = failures[-5:]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(failures, f)
