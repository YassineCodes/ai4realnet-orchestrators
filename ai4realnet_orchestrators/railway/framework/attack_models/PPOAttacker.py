"""
PPO Attacker for Flatland Railway — online actor-critic version.

Equivalent of the power-grid PPOAttacker, adapted for Flatland obs_dicts.

Trains online via REINFORCE with baseline (actor-critic). No pre-trained model
required: starts random and improves over episodes.

State  : flatten_obs_dict(obs_dict)
Action : index into (agent_id × corruption_type) — n_agents * 6 discrete actions
Reward : disruption_reward(obs_before, obs_after)
Update : actor-critic policy gradient at end of each episode
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from attack_models.BaseAttackerClass import BaseAttackerClass
from perturbation_agents.utils import (
    flatten_obs_dict,
    corrupt_obs_dict,
    disruption_reward,
    CORRUPTION_TYPES,
)

logger = logging.getLogger(__name__)
_N_CORRUPTION_TYPES = len(CORRUPTION_TYPES)


class _ActorCriticNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


class PPOAttacker(BaseAttackerClass):
    """
    Online actor-critic (REINFORCE with baseline) perturbation attacker.

    Trains from scratch — no pre-trained model required.

    Args:
        n_agents (int): Number of Flatland agents.
        obs_dim (int): Flattened obs dimension (0 = auto-detect on first step).
        lr (float): Adam learning rate.
        gamma (float): Discount factor.
        entropy_coef (float): Entropy bonus weight for exploration.
        seed (int): Random seed.
        model_name (str): Display name.
        pickle_file (str): Filename for result serialisation.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int = 0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.05,
        seed: int = 42,
        model_name: str = "PPOAttacker",
        pickle_file: str = "ppo_attacker.pkl",
    ):
        super().__init__(name=model_name)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = n_agents * _N_CORRUPTION_TYPES
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.model_name = model_name
        self.pickle_file = pickle_file

        self._net: Optional[_ActorCriticNet] = None
        self._optimizer = None

        # Episode trajectory
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []

        self.perturbation_count = 0
        self.space_prng = np.random.RandomState(seed)

        self._action_map = [
            (h, CORRUPTION_TYPES[c])
            for h in range(n_agents)
            for c in range(_N_CORRUPTION_TYPES)
        ]

        if obs_dim > 0 and _TORCH_AVAILABLE:
            self._build_network(obs_dim)

    def _build_network(self, obs_dim: int):
        self._net = _ActorCriticNet(obs_dim, self.action_dim)
        self._optimizer = optim.Adam(self._net.parameters(), lr=self.lr)
        logger.info(f"PPOAttacker network built: state={obs_dim}, actions={self.action_dim}")

    def perturb(self, obs_dict: Dict) -> Dict:
        state = flatten_obs_dict(obs_dict)

        if self.obs_dim == 0 and len(state) > 0 and _TORCH_AVAILABLE:
            self.obs_dim = len(state)
            self._build_network(self.obs_dim)
        elif self.obs_dim > 0 and len(state) != self.obs_dim:
            if len(state) < self.obs_dim:
                state = np.pad(state, (0, self.obs_dim - len(state)))
            else:
                state = state[:self.obs_dim]

        action_idx = self._select_action(state)
        handle, ctype = self._action_map[action_idx]
        self.last_handle = handle
        self.last_ctype = ctype
        perturbed = corrupt_obs_dict(copy.deepcopy(obs_dict), handle, ctype)
        reward = disruption_reward(obs_dict, perturbed)

        self._states.append(state)
        self._actions.append(action_idx)
        self._rewards.append(reward)
        self.perturbation_count += 1
        return perturbed

    def _select_action(self, state: np.ndarray) -> int:
        if not _TORCH_AVAILABLE or self._net is None:
            return int(self.space_prng.randint(self.action_dim))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = self._net(s)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return int(action)

    def reset(self) -> None:
        """Update policy via REINFORCE with baseline, then clear trajectory."""
        if _TORCH_AVAILABLE and self._net is not None and len(self._rewards) > 0:
            self._update_policy()
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()

    def _update_policy(self):
        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.FloatTensor(np.array(self._states))
        actions = torch.LongTensor(self._actions)

        logits, values = self._net(states)
        values = values.squeeze(1)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
        self._optimizer.step()

    def perturb_vector(self, vec) -> np.ndarray:
        """
        Use the actor-critic policy to select a corruption type, apply it to vec.
        Falls back to random when the network is not yet built.
        """
        from perturbation_agents.utils import corrupt_vector
        v = np.nan_to_num(np.array(vec, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        if self.obs_dim > 0 and self._net is not None:
            state = np.pad(v, (0, max(0, self.obs_dim - len(v))))[:self.obs_dim]
            action_idx = self._select_action(state)
        else:
            action_idx = int(self.space_prng.randint(self.action_dim))
        _, ctype = self._action_map[action_idx]
        return corrupt_vector(v, ctype)

    def save_model(self, path) -> None:
        if not _TORCH_AVAILABLE or self._net is None:
            return
        torch.save(self._net.state_dict(), str(path))

    def load(self, path) -> None:
        if not _TORCH_AVAILABLE:
            return
        state_dict = torch.load(str(path), map_location="cpu")
        if self._net is None:
            logger.warning("Network not built yet — call perturb() once first.")
            return
        self._net.load_state_dict(state_dict)
