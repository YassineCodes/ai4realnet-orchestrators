"""
Gradient Estimation Perturbation Attacker for Flatland Railway.

Equivalent of the power-grid GEPerturbAttacker.

Wraps GradientEstimationPerturbationAgent.
Requires a defender that responds to observations (ObsPolicyDefender).
For ShortestPathDefender, gradient is always zero — use RandomPerturbAttacker.
"""

import copy
import logging

from perturbation_agents.grad_est_perturb_agent import GradientEstimationPerturbationAgent
from attack_models.BaseAttackerClass import BaseAttackerClass

logger = logging.getLogger(__name__)


class GEPerturbAttacker(BaseAttackerClass):
    """
    Gradient-estimation adversarial attacker for Flatland.

    Args:
        defender: A RailwayDefender that uses observations (ObsPolicyDefender).
        n_agents (int): Number of agents in the environment.
        n_candidates (int): Number of (agent, corruption) pairs to evaluate per step.
        seed (int): Random seed.
    """

    model_name = "GEPerturbAttacker"
    pickle_file = "ge_perturb_attacker.pkl"

    def __init__(self, defender, n_agents, n_candidates=8, seed=42):
        self.n_agents = n_agents
        self.model = GradientEstimationPerturbationAgent(
            defender=defender,
            n_agents=n_agents,
            n_candidates=n_candidates,
            seed=seed,
        )
        super().__init__(name=self.model_name)

    def load(self, path):
        pass

    def perturb_vector(self, vec):
        """
        Add scaled Gaussian noise — gradient estimation requires defender
        interaction that is unavailable at maze-obs-injection time, so we
        fall back to statistical noise scaled to the vector's own std.
        """
        import numpy as np
        v = np.array(vec, dtype=np.float64)
        std = float(np.std(v))
        std = std if std > 1e-8 else 1.0
        noise = self.model.space_prng.randn(len(v)) * std
        return v + noise

    def perturb(self, obs_dict):
        obs_t = copy.deepcopy(obs_dict)
        result = self.model.perturb(obs_t)
        self.last_handle = self.model.last_handle
        self.last_ctype = self.model.last_ctype
        return result
