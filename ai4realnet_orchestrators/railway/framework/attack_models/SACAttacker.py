"""
SAC Attacker for Flatland Railway — online discrete SAC version.

Equivalent of the power-grid SACAttacker, adapted for Flatland obs_dicts.

Uses Soft Actor-Critic with discrete actions (maximum entropy RL).
Trains online from a replay buffer — no pre-trained model required.

State  : flatten_obs_dict(obs_dict)
Action : index into (agent_id × corruption_type) — n_agents * 6 discrete actions
Reward : disruption_reward(obs_before, obs_after)
Update : off-policy SAC with entropy regularisation (alpha)
"""

import copy
import logging
from collections import deque
from typing import Dict, Optional

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


class _SACNets(nn.Module if _TORCH_AVAILABLE else object):
    """Discrete SAC: separate actor and twin critics."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden = [256, 128]
        def mlp(out):
            layers = []
            in_dim = state_dim
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, out))
            return nn.Sequential(*layers)

        self.actor = mlp(action_dim)
        self.q1 = mlp(action_dim)
        self.q2 = mlp(action_dim)


class SACAttacker(BaseAttackerClass):
    """
    Online discrete Soft Actor-Critic perturbation attacker.

    Trains from scratch — no pre-trained model required.
    Maximum-entropy objective encourages diverse perturbation strategies.

    Args:
        n_agents (int): Number of Flatland agents.
        obs_dim (int): Flattened obs dimension (0 = auto-detect on first step).
        lr (float): Adam learning rate for actor and critics.
        gamma (float): Discount factor.
        alpha (float): Entropy temperature — higher = more exploration.
        batch_size (int): Replay buffer batch size.
        buffer_size (int): Replay buffer capacity.
        target_update_freq (int): Steps between hard target-network syncs.
        seed (int): Random seed.
        model_name (str): Display name.
        pickle_file (str): Filename for result serialisation.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        target_update_freq: int = 500,
        seed: int = 42,
        model_name: str = "SACAttacker",
        pickle_file: str = "sac_attacker.pkl",
    ):
        super().__init__(name=model_name)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = n_agents * _N_CORRUPTION_TYPES
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.model_name = model_name
        self.pickle_file = pickle_file

        self._net: Optional[_SACNets] = None
        self._target_net: Optional[_SACNets] = None
        self._actor_opt = None
        self._critic_opt = None

        self._buffer = deque(maxlen=buffer_size)
        self._step_count = 0
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None

        self.perturbation_count = 0
        self.space_prng = np.random.RandomState(seed)

        self._action_map = [
            (h, CORRUPTION_TYPES[c])
            for h in range(n_agents)
            for c in range(_N_CORRUPTION_TYPES)
        ]

        if obs_dim > 0 and _TORCH_AVAILABLE:
            self._build_networks(obs_dim)

    def _build_networks(self, obs_dim: int):
        self._net = _SACNets(obs_dim, self.action_dim)
        self._target_net = _SACNets(obs_dim, self.action_dim)
        self._target_net.load_state_dict(self._net.state_dict())
        self._actor_opt = optim.Adam(self._net.actor.parameters(), lr=self.lr)
        critic_params = list(self._net.q1.parameters()) + list(self._net.q2.parameters())
        self._critic_opt = optim.Adam(critic_params, lr=self.lr)
        logger.info(f"SACAttacker networks built: state={obs_dim}, actions={self.action_dim}")

    def perturb(self, obs_dict: Dict) -> Dict:
        state = flatten_obs_dict(obs_dict)

        if self.obs_dim == 0 and len(state) > 0 and _TORCH_AVAILABLE:
            self.obs_dim = len(state)
            self._build_networks(self.obs_dim)
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

        if self._prev_state is not None and self._prev_action is not None:
            self._buffer.append((self._prev_state, self._prev_action, reward, state, False))
            self._train()

        self._prev_state = state
        self._prev_action = action_idx
        self._step_count += 1
        self.perturbation_count += 1
        return perturbed

    def _select_action(self, state: np.ndarray) -> int:
        if not _TORCH_AVAILABLE or self._net is None:
            return int(self.space_prng.randint(self.action_dim))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits = self._net.actor(s)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return int(action)

    def reset(self) -> None:
        if self._prev_state is not None and self._prev_action is not None:
            self._buffer.append((self._prev_state, self._prev_action, 0.0,
                                  self._prev_state, True))
        self._prev_state = None
        self._prev_action = None

    def _train(self):
        if not _TORCH_AVAILABLE or self._net is None or len(self._buffer) < self.batch_size:
            return

        indices = self.space_prng.choice(len(self._buffer), self.batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.FloatTensor(np.array(states))
        a = torch.LongTensor(actions)
        r = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(np.array(next_states))
        d = torch.FloatTensor(dones)

        # Soft target value V(s') = E_π[Q(s',a') - α log π(a'|s')]
        with torch.no_grad():
            next_logits = self._net.actor(ns)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q = torch.min(self._target_net.q1(ns), self._target_net.q2(ns))
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            q_target = r + self.gamma * next_v * (1 - d)

        # Critic update
        q1 = self._net.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self._net.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self._critic_opt.zero_grad()
        critic_loss.backward()
        self._critic_opt.step()

        # Actor update: maximise E[Q - α log π]
        logits = self._net.actor(s)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q_vals = torch.min(self._net.q1(s), self._net.q2(s))
        actor_loss = (probs * (self.alpha * log_probs - q_vals)).sum(dim=-1).mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        if self._step_count % self.target_update_freq == 0:
            self._target_net.load_state_dict(self._net.state_dict())

    def perturb_vector(self, vec) -> np.ndarray:
        """
        Use the SAC actor to select a corruption type, apply it to vec.
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
        self._target_net.load_state_dict(self._net.state_dict())
