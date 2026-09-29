"""
SAC-style perturbation attacker (ATM).

Thin attacker wrapper over perturbation_agents/rl_perturb_agent.py, mirroring
power_grid/framework/attack_models/SACAttacker.py.

Author: INESC TEC
"""

import logging

from attack_models.BaseAttackerClass import BaseAttackerClass
from perturbation_agents.rl_perturb_agent import RLPerturbationAgent

logger = logging.getLogger(__name__)


class SACAttacker(BaseAttackerClass):
    """Online attacker at a scaled budget; `factor` is the percentage of the observation scale it may use (5 / 10), as in the power-grid SACAttacker."""

    def __init__(self, obs_space=None, agent=None, factor=10, seed=6, model_name=None, pickle_file=None):
        model_name = model_name or "SAC_%d" % factor
        pickle_file = pickle_file or "sac_%d.pkl" % factor
        super().__init__(name=model_name)
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.perturbation_agent = RLPerturbationAgent(obs_space=obs_space, agent=agent, factor=factor, seed=seed, epsilon=0.1, n_sample=6)

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
