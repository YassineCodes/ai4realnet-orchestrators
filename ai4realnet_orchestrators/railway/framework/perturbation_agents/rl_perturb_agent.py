"""
RL (DQN) Perturbation Agent for Flatland Railway.

Equivalent of the power-grid RLPerturbationAgent.

A Deep Q-Network that learns WHICH agent to corrupt and HOW, maximising
the disruption to the defender over episodes.

State  : flatten_obs_dict(obs_dict)  — shape (n_agents * obs_per_agent,)
Action : index into (agent_id × corruption_type) — n_agents * 6 discrete actions
Reward : disruption_reward(obs_before, obs_after)

Training:
    Call train_step() each episode step when defender actions are available.
    Call save_model() / load_model() to persist.

Key differences from power grid:
  - obs.to_vect()   →  flatten_obs_dict
  - obs._vectorized →  corrupt_obs_dict (node _replace)
  - env.reset()     →  not needed; agent is episode-agnostic
"""

import copy
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from perturbation_agents.base_perturb_agent import BasePerturbationAgent
from perturbation_agents.utils import (
    flatten_obs_dict,
    corrupt_obs_dict,
    disruption_reward,
    CORRUPTION_TYPES,
)

logger = logging.getLogger(__name__)

_N_CORRUPTION_TYPES = len(CORRUPTION_TYPES)  # 6


# ──────────────────────────────────────────────────────────────────────────────
# DQN Network
# ──────────────────────────────────────────────────────────────────────────────

class _DQNNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class RLPerturbationAgent(BasePerturbationAgent):
    """
    DQN-based perturbation agent for Flatland.

    Args:
        n_agents (int): Number of Flatland agents.
        obs_dim (int): Length of flatten_obs_dict() for all agents combined.
                       If unknown, pass 0 and it will be auto-detected on first step.
        epsilon (float): Initial ε-greedy exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Per-step multiplicative decay.
        gamma (float): Discount factor for RL updates.
        lr (float): Adam learning rate.
        batch_size (int): Replay buffer batch size.
        buffer_size (int): Replay buffer capacity.
        target_update_freq (int): Steps between hard target-network updates.
        seed (int): Random seed.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int = 0,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        target_update_freq: int = 500,
        seed: int = 42,
    ):
        super().__init__(name="RLPerturbationAgent", seed=seed)

        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not installed — RLPerturbationAgent will act randomly.")

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = n_agents * _N_CORRUPTION_TYPES
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self._buffer = deque(maxlen=buffer_size)
        self._step_count = 0

        # Networks built lazily when obs_dim is known
        self._q_net: Optional[_DQNNet] = None
        self._target_net: Optional[_DQNNet] = None
        self._optimizer = None

        if obs_dim > 0 and _TORCH_AVAILABLE:
            self._build_networks(obs_dim, lr)

        # Decode action index → (handle, corruption_type)
        self._action_map = [
            (h, CORRUPTION_TYPES[c])
            for h in range(n_agents)
            for c in range(_N_CORRUPTION_TYPES)
        ]

        # State for train_step
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def perturb(self, obs_dict: Dict) -> Dict:
        """Select and apply a perturbation action."""
        state = flatten_obs_dict(obs_dict)

        # Auto-detect obs_dim on first call
        if self.obs_dim == 0 and len(state) > 0 and _TORCH_AVAILABLE:
            self.obs_dim = len(state)
            self._build_networks(self.obs_dim, lr=1e-3)
        elif self.obs_dim > 0 and len(state) != self.obs_dim:
            # node_to_vector is recursive → variable size; pad/truncate to fixed dim
            if len(state) < self.obs_dim:
                state = np.pad(state, (0, self.obs_dim - len(state)))
            else:
                state = state[:self.obs_dim]

        action_idx = self._select_action(state)
        handle, ctype = self._action_map[action_idx]
        self.last_handle = handle
        self.last_ctype = ctype
        perturbed = corrupt_obs_dict(copy.deepcopy(obs_dict), handle, ctype)

        # Compute reward and store transition (no next_state yet — deferred)
        reward = disruption_reward(obs_dict, perturbed)
        if self._prev_state is not None and self._prev_action is not None:
            self._buffer.append((self._prev_state, self._prev_action, reward, state, False))
            self._train()

        self._prev_state = state
        self._prev_action = action_idx
        self._step_count += 1
        self.perturbation_count += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return perturbed

    def reset(self) -> None:
        super().reset()
        # Mark end of episode in buffer
        if self._prev_state is not None and self._prev_action is not None:
            self._buffer.append((self._prev_state, self._prev_action, 0.0,
                                  self._prev_state, True))
        self._prev_state = None
        self._prev_action = None

    def save_model(self, path) -> None:
        """Save Q-network weights."""
        if not _TORCH_AVAILABLE or self._q_net is None:
            logger.warning("Cannot save: torch unavailable or network not built.")
            return
        import torch
        torch.save(self._q_net.state_dict(), str(path))
        logger.info(f"RLPerturbationAgent saved to {path}")

    def load_model(self, path) -> None:
        """Load Q-network weights."""
        if not _TORCH_AVAILABLE:
            logger.warning("Cannot load: torch not available.")
            return
        import torch
        if self._q_net is None:
            logger.warning("Network not yet built — call perturb() once first to auto-detect obs_dim.")
            return
        self._q_net.load_state_dict(torch.load(str(path), map_location="cpu"))
        self._target_net.load_state_dict(self._q_net.state_dict())
        self.epsilon = self.epsilon_min  # evaluation mode
        logger.info(f"RLPerturbationAgent loaded from {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_networks(self, obs_dim: int, lr: float):
        import torch.optim as optim
        self._q_net = _DQNNet(obs_dim, self.action_dim)
        self._target_net = _DQNNet(obs_dim, self.action_dim)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optimizer = optim.Adam(self._q_net.parameters(), lr=lr)
        logger.info(f"DQN built: state={obs_dim}, actions={self.action_dim}")

    def _select_action(self, state: np.ndarray) -> int:
        if self.space_prng.random() < self.epsilon or self._q_net is None:
            return int(self.space_prng.randint(self.action_dim))
        import torch
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q = self._q_net(s)
            return int(q.argmax().item())

    def _train(self):
        if not _TORCH_AVAILABLE or self._q_net is None:
            return
        if len(self._buffer) < self.batch_size:
            return

        import torch

        indices = self.space_prng.choice(len(self._buffer), self.batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.FloatTensor(np.array(states))
        a = torch.LongTensor(actions)
        r = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(np.array(next_states))
        d = torch.FloatTensor(dones)

        # Current Q values
        q_curr = self._q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            q_next = self._target_net(ns).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = torch.nn.functional.mse_loss(q_curr, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Hard target update
        if self._step_count % self.target_update_freq == 0:
            self._target_net.load_state_dict(self._q_net.state_dict())
