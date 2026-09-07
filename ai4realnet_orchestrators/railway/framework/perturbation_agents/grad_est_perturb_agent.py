"""
Gradient Estimation Perturbation Agent for Flatland Railway.

Equivalent of the power-grid GradientEstimationPerturbationAgent.

Finds which observation features most affect the defender's decisions using
finite-difference gradient estimation, then applies a projected gradient
perturbation in that direction.

Works only with defenders that use observations (ObsPolicyDefender).
For ShortestPathDefender the gradient is always zero — use RandomPerturbAttacker instead.

Key differences from power grid:
  - obs.to_vect()       →  flatten_obs_dict(obs_dict)
  - obs._vectorized     →  reconstruct obs_dict from perturbed vector (approximate)
  - obs.rho             →  disruption_reward for gradient sign
"""

import copy
import logging
from typing import Dict

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent
from perturbation_agents.utils import flatten_obs_dict, corrupt_obs_dict, CORRUPTION_TYPES

logger = logging.getLogger(__name__)


class GradientEstimationPerturbationAgent(BasePerturbationAgent):
    """
    Finite-difference gradient perturbation for Flatland.

    For each step:
      1. Flatten obs_dict → x ∈ R^d
      2. For each candidate (agent, corruption_type):
           Estimate disruption_score = how much does this corruption change
           the defender's action (measured as action difference norm).
      3. Apply the highest-scoring corruption.

    Args:
        defender: A RailwayDefender instance (must use observations — ObsPolicyDefender).
        n_agents (int): Number of agents in the environment.
        n_candidates (int): How many (agent, corruption) pairs to evaluate per step.
                            Larger = better quality but slower. Default 8.
        seed (int): Random seed.
    """

    def __init__(
        self,
        defender,
        n_agents: int,
        n_candidates: int = 8,
        seed: int = 42,
    ):
        super().__init__(name="GradientEstimationPerturbationAgent", seed=seed)
        self.defender = defender
        self.n_agents = n_agents
        self.n_candidates = n_candidates

        # Build full action space
        self._actions = []
        for handle in range(n_agents):
            for ctype in CORRUPTION_TYPES:
                self._actions.append((handle, ctype))

    def perturb(self, obs_dict: Dict) -> Dict:
        """
        Apply gradient-guided perturbation.

        Samples self.n_candidates (agent, corruption) pairs, evaluates each
        by measuring the change in the defender's action, returns the one
        that causes the largest change.
        """
        # Baseline defender action on clean obs
        try:
            baseline_actions = self.defender.act(obs_dict)
            baseline_vec = np.array(
                [baseline_actions.get(h, 0) for h in sorted(obs_dict.keys())],
                dtype=float,
            )
        except Exception:
            baseline_vec = np.zeros(self.n_agents, dtype=float)

        # Sample candidates (random subset to keep it tractable)
        n_actions = len(self._actions)
        indices = self.space_prng.choice(n_actions, size=min(self.n_candidates, n_actions), replace=False)

        best_score = -np.inf
        best_action = None

        for idx in indices:
            handle, ctype = self._actions[idx]
            candidate_obs = corrupt_obs_dict(copy.deepcopy(obs_dict), handle, ctype)

            try:
                perturbed_actions = self.defender.act(candidate_obs)
                perturbed_vec = np.array(
                    [perturbed_actions.get(h, 0) for h in sorted(obs_dict.keys())],
                    dtype=float,
                )
                score = float(np.linalg.norm(perturbed_vec - baseline_vec))
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best_action = (handle, ctype)

        if best_action is None or best_score == 0:
            # No action caused any change — fall back to random blank
            handle = int(self.space_prng.randint(self.n_agents))
            best_action = (handle, "blank")

        handle, ctype = best_action
        self.last_handle = handle
        self.last_ctype = ctype
        result = corrupt_obs_dict(copy.deepcopy(obs_dict), handle, ctype)
        self.perturbation_count += 1
        return result

    def reset(self) -> None:
        super().reset()
