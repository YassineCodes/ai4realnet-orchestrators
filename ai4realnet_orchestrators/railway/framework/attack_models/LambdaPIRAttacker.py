"""
Lambda-PIR Attacker for Flatland Railway.

Equivalent of the power-grid LambdaPIRAttacker.

Wraps the Flatland-adapted LambdaPIRPerturbationAgent (Bellman / PBIR variant).
Same interface and parameters as the power-grid version — only the
observation handling differs internally.
"""

import copy
import logging
import traceback
from typing import Optional, Dict, Any

from perturbation_agents.lambda_pir_perturbation_agent import LambdaPIRPerturbationAgent
from attack_models.BaseAttackerClass import BaseAttackerClass

logger = logging.getLogger(__name__)


class LambdaPIRAttacker(BaseAttackerClass):
    """
    Lambda-PIR (Bellman-proven) adversarial attacker for Flatland.

    Hybrid policy/value iteration with Q-learning updates.
    Selects which agent to blind or corrupt, and how.

    Args:
        n_agents (int): Number of agents in the environment (REQUIRED).
        lambda_param (float): λ ∈ [0,1) lookahead depth.
        initial_prob_policy (float): Starting fraction of policy-iteration steps.
        epsilon (float): Reserved — not used for observation blanking.
        gradient_step_size (float): α for Bellman Q updates.
        refinement_iterations (int): Value-iteration refinement steps.
        decay_schedule (str): "linear", "exponential", or "constant".
        gamma (float): Discount factor for Bellman updates.
        name (str): Display name.
        seed (int): Random seed.
    """

    model_name = "LambdaPIRAttacker"
    pickle_file = "lambda_pir_attacker.pkl"

    def __init__(
        self,
        n_agents: int,
        lambda_param: float = 0.9,
        initial_prob_policy: float = 0.8,
        epsilon: float = 0.1,
        gradient_step_size: float = 0.05,
        refinement_iterations: int = 5,
        decay_schedule: str = "linear",
        gamma: float = 0.99,
        name: str = "LambdaPIRAttacker",
        seed: int = 42,
    ):
        try:
            self.model_name = name
            self.n_agents = n_agents

            self.perturbation_agent = LambdaPIRPerturbationAgent(
                n_agents=n_agents,
                lambda_param=lambda_param,
                initial_prob_policy=initial_prob_policy,
                epsilon=epsilon,
                gradient_step_size=gradient_step_size,
                refinement_iterations=refinement_iterations,
                decay_schedule=decay_schedule,
                gamma=gamma,
                seed=seed,
                name=name,
            )
            super().__init__(name=name)
            logger.info(f"{name} initialized (gamma={gamma}, n_agents={n_agents})")

        except Exception as e:
            logger.error(f"{name} init failed: {e}\n{traceback.format_exc()}")
            raise

    def load(self, path):
        """Re-create the perturbation agent (no weights to load for Lambda-PIR)."""
        pass

    def perturb(self, obs_dict):
        """
        Apply Lambda-PIR perturbation to the observation dict.

        Args:
            obs_dict (dict): {agent_handle: Node}

        Returns:
            dict: Perturbed obs_dict.
        """
        try:
            result = self.perturbation_agent.perturb(obs_dict)
            self.last_handle = self.perturbation_agent.last_handle
            self.last_ctype = self.perturbation_agent.last_ctype
            return result
        except Exception as e:
            logger.error(f"Perturbation failed: {e}")
            return copy.deepcopy(obs_dict)

    def reset(self):
        """Reset between episodes."""
        if self.perturbation_agent:
            self.perturbation_agent.reset()

    def perturb_vector(self, vec):
        """
        Negate the observation vector to disrupt the maze BC policy.

        The maze BC policy uses Tanh activations on observations with large
        dynamic range ([-1, 62]).  Amplifying types (×3) just saturate Tanh
        further to the same sign — the action does not change.  Negation
        (×−1) flips all activation signs: Tanh(−x) = −Tanh(x), forcing the
        policy into maximally different territory.

        The TreeObs Q-values select WHICH agent/field to corrupt; we ignore
        the corruption-type hint and always negate for vector obs.
        """
        import numpy as np
        from perturbation_agents.utils import corrupt_vector
        return corrupt_vector(np.array(vec, dtype=np.float64), "speed")

    def get_stats(self) -> Dict[str, Any]:
        if self.perturbation_agent:
            return self.perturbation_agent.get_stats()
        return {}
