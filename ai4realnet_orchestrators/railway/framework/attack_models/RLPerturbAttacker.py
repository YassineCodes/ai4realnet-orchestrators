"""
RL (DQN) Perturbation Attacker for Flatland Railway.

Equivalent of the power-grid RLPerturbAttacker.

Wraps RLPerturbationAgent. The DQN learns to select (agent, corruption_type)
pairs that maximise disruption to the defender.

The model trains online during evaluation and can be saved/loaded.
"""

import copy
import logging
import traceback

from perturbation_agents.rl_perturb_agent import RLPerturbationAgent
from attack_models.BaseAttackerClass import BaseAttackerClass

logger = logging.getLogger(__name__)


class RLPerturbAttacker(BaseAttackerClass):
    """
    DQN-based adversarial attacker for Flatland.

    Args:
        model_path (str or None): Path to a pre-trained DQN model.
                                  If None, starts training from scratch.
        n_agents (int): Number of agents in the environment (REQUIRED).
        obs_dim (int): Size of flatten_obs_dict(). Pass 0 to auto-detect.
        epsilon (float): Initial ε-greedy exploration rate.
        gamma (float): Discount factor.
        seed (int): Random seed.
    """

    model_name = "RLPerturbAttacker"
    pickle_file = "rl_perturb_attacker.pkl"

    def __init__(
        self,
        n_agents: int,
        model_path=None,
        obs_dim: int = 0,
        epsilon: float = 1.0,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        try:
            self.n_agents = n_agents
            self.model = RLPerturbationAgent(
                n_agents=n_agents,
                obs_dim=obs_dim,
                epsilon=epsilon,
                gamma=gamma,
                seed=seed,
            )
            if model_path is not None:
                self.model.load_model(model_path)
            super().__init__(name=self.model_name)
            logger.info(f"{self.model_name} initialized (n_agents={n_agents})")
        except Exception as e:
            logger.error(f"{self.model_name} init failed: {e}\n{traceback.format_exc()}")
            raise

    def load(self, path):
        self.model.load_model(str(path))

    def perturb(self, obs_dict):
        obs_t = copy.deepcopy(obs_dict)
        result = self.model.perturb(obs_t)
        self.last_handle = self.model.last_handle
        self.last_ctype = self.model.last_ctype
        return result

    def reset(self):
        """Reset between episodes (triggers end-of-episode buffer entry)."""
        if self.model:
            self.model.reset()

    def perturb_vector(self, vec):
        """
        Use the DQN policy to select a corruption type, apply it to the vector.
        Falls back to random blank when the network is not yet built or the
        vector dimension differs significantly.
        """
        import numpy as np
        from perturbation_agents.utils import corrupt_vector
        agent = self.model
        v = np.nan_to_num(np.array(vec, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        if agent.obs_dim > 0 and agent._q_net is not None:
            if len(v) < agent.obs_dim:
                state = np.pad(v, (0, agent.obs_dim - len(v)))
            else:
                state = v[:agent.obs_dim]
            action_idx = agent._select_action(state)
        else:
            action_idx = int(agent.space_prng.randint(agent.action_dim))
        _, ctype = agent._action_map[action_idx]
        return corrupt_vector(v, ctype)

    def save_model(self, path):
        """Save the trained DQN weights."""
        if self.model:
            self.model.save_model(path)
