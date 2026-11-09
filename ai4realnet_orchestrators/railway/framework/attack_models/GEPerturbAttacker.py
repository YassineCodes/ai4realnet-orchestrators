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
        n_candidates (int or None): Number of actions to evaluate per step. None
                            keeps the coverage this attacker had before the action
                            space gained a severity axis; see the agent docstring.
        seed (int): Random seed.
    """

    model_name = "GEPerturbAttacker"
    pickle_file = "ge_perturb_attacker.pkl"

    def __init__(self, defender, n_agents, n_candidates=None, seed=42):
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

    def perturb(self, obs_dict):
        obs_t = copy.deepcopy(obs_dict)
        result = self.model.perturb(obs_t)
        handle, ctype = self.model.last_handle, self.model.last_ctype
        severity = getattr(self.model, "last_severity", 1.0)
        self._publish_targets({handle: ctype} if handle is not None and ctype else {},
                              {handle: severity} if handle is not None and ctype else {})
        return result
