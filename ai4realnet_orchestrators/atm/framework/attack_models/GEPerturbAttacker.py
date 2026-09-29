"""
Gradient-estimation attacker (ATM).

Thin attacker wrapper over perturbation_agents/grad_est_perturb_agent.py, mirroring
power_grid/framework/attack_models/GEPerturbAttacker.py.

Author: INESC TEC
"""

import logging

from attack_models.BaseAttackerClass import BaseAttackerClass
from perturbation_agents.grad_est_perturb_agent import GradEstPerturbationAgent

logger = logging.getLogger(__name__)


class GEPerturbAttacker(BaseAttackerClass):
    """Black-box gradient estimation against the defender."""

    def __init__(self, obs_space=None, agent=None, n_candidates=8, xi=0.1, seed=3, model_name="GEPerturb", pickle_file="geperturb.pkl"):
        super().__init__(name=model_name)
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.perturbation_agent = GradEstPerturbationAgent(obs_space=obs_space, agent=agent, n_candidates=n_candidates, xi=xi, seed=seed)

    def perturb(self, obs):
        return self.perturbation_agent.perturb(obs)

    def reset(self):
        self.perturbation_agent.reset()

    def get_stats(self):
        return self.perturbation_agent.get_stats()

    @property
    def xi(self):
        """The budget this attacker enforces (SAC_5 / SAC_10 carry their own)."""
        return self.perturbation_agent.xi
