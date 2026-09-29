"""
Lambda-PIR attacker (ATM).

Thin attacker wrapper over perturbation_agents/lambda_pir_perturbation_agent.py, mirroring
power_grid/framework/attack_models/LambdaPIRAttacker.py.

Author: INESC TEC
"""

import logging

from attack_models.BaseAttackerClass import BaseAttackerClass
from perturbation_agents.lambda_pir_perturbation_agent import LambdaPIRPerturbationAgent

logger = logging.getLogger(__name__)


class LambdaPIRAttacker(BaseAttackerClass):
    """Policy/value iteration over bounded single-feature perturbations."""

    def __init__(self, obs_space=None, agent=None, xi=0.1, lambda_param=0.7, initial_prob_policy=0.2, gradient_step_size=0.1, refinement_iterations=20, decay_schedule="exponential", seed=2, model_name="LambdaPIR", pickle_file="lambdapir.pkl"):
        super().__init__(name=model_name)
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.perturbation_agent = LambdaPIRPerturbationAgent(obs_space=obs_space, agent=agent, xi=xi, lambda_param=lambda_param, initial_prob_policy=initial_prob_policy, gradient_step_size=gradient_step_size, refinement_iterations=refinement_iterations, decay_schedule=decay_schedule, seed=seed)

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
