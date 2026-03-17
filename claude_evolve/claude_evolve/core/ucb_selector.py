"""UCB1 bandit for adaptive strategy selection.

Inspired by ShinkaEvolve (UCB1 with exponential reward) and AdaEvolve
(exploration modulation). Uses capped linear reward to prevent single
lucky outcomes from dominating.
"""

import json
import math
import os
from dataclasses import dataclass, field


@dataclass
class StrategyArm:
    strategy_id: str
    total_reward: float = 0.0
    visit_count: int = 0
    decayed_reward: float = 0.0


class UCBStrategySelector:
    """UCB1-based strategy selection with exploration modulation."""

    def __init__(self, strategy_ids: list, c: float = 1.414, decay: float = 0.95):
        self.arms: dict[str, StrategyArm] = {
            sid: StrategyArm(strategy_id=sid) for sid in strategy_ids
        }
        self.c = c
        self.decay = decay
        self.total_selections = 0

    def select(self, exploration_intensity: float = 0.5) -> str:
        """Select strategy using UCB1 with exploration modulation."""
        self.total_selections += 1

        # Always select unvisited arms first
        unvisited = [a for a in self.arms.values() if a.visit_count == 0]
        if unvisited:
            return unvisited[0].strategy_id

        c_adj = self.c * (0.5 + exploration_intensity)
        n = self.total_selections

        best_score = -float('inf')
        best_id = list(self.arms.keys())[0]

        for arm in self.arms.values():
            if arm.visit_count == 0:
                return arm.strategy_id
            avg_reward = arm.decayed_reward / arm.visit_count
            ucb_bonus = c_adj * math.sqrt(math.log(n) / arm.visit_count)
            ucb_score = avg_reward + ucb_bonus
            if ucb_score > best_score:
                best_score = ucb_score
                best_id = arm.strategy_id

        return best_id

    def record(self, strategy_id: str, score_delta: float) -> None:
        """Record outcome with capped linear reward and global decay."""
        reward = min(max(score_delta, 0.0), 1.0)

        # Decay all arms
        for arm in self.arms.values():
            arm.decayed_reward *= self.decay

        # Update selected arm
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward = min(arm.total_reward + reward, arm.visit_count + 1)
            arm.visit_count += 1
            arm.decayed_reward += reward

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'c': self.c, 'decay': self.decay,
            'total_selections': self.total_selections,
            'arms': {
                sid: {'total_reward': a.total_reward, 'visit_count': a.visit_count,
                      'decayed_reward': a.decayed_reward}
                for sid, a in self.arms.items()
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'UCBStrategySelector':
        if not os.path.exists(path):
            return cls([])
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        sel = cls(list(data.get('arms', {}).keys()),
                  c=data.get('c', 1.414), decay=data.get('decay', 0.95))
        sel.total_selections = data.get('total_selections', 0)
        for sid, arm_data in data.get('arms', {}).items():
            if sid in sel.arms:
                sel.arms[sid].total_reward = arm_data.get('total_reward', 0.0)
                sel.arms[sid].visit_count = arm_data.get('visit_count', 0)
                sel.arms[sid].decayed_reward = arm_data.get('decayed_reward', 0.0)
        return sel
