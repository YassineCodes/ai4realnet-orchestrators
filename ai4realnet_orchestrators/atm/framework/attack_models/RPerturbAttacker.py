"""
Random perturbation attacker (ATM).

Thin attacker wrapper over perturbation_agents/random_perturb_agent.py, mirroring
power_grid/framework/attack_models/RPerturbAttacker.py.

Author: INESC TEC
"""

import logging

from attack_models.BaseAttackerClass import BaseAttackerClass
from perturbation_agents.random_perturb_agent import RandomPerturbationAgent

logger = logging.getLogger(__name__)


class RPerturbAttacker(BaseAttackerClass):
    """Uniform bounded noise on every observation feature."""

    def __init__(self, obs_space=None, xi=0.1, seed=1, model_name="Random", pickle_file="random.pkl"):
        super().__init__(name=model_name)
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.perturbation_agent = RandomPerturbationAgent(obs_space=obs_space, xi=xi, seed=seed)

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
