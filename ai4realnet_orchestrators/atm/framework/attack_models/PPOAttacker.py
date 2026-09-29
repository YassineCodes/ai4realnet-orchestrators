"""
PPO-style perturbation attacker (ATM).

Thin attacker wrapper over perturbation_agents/rl_perturb_agent.py, mirroring
power_grid/framework/attack_models/PPOAttacker.py.

Author: INESC TEC
"""

import logging

from attack_models.BaseAttackerClass import BaseAttackerClass
from perturbation_agents.rl_perturb_agent import RLPerturbationAgent

logger = logging.getLogger(__name__)


class PPOAttacker(BaseAttackerClass):
    """Higher-exploration online attacker (the entropy-bonus behaviour of the PPO attacker)."""

    def __init__(self, obs_space=None, agent=None, xi=0.1, seed=5, entropy_coef=0.05, model_name="PPO", pickle_file="ppo.pkl"):
        super().__init__(name=model_name)
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.perturbation_agent = RLPerturbationAgent(obs_space=obs_space, agent=agent, xi=xi, seed=seed, lr=0.05, epsilon=0.05 + entropy_coef, n_sample=4)

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
