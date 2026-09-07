"""
Lambda-PIR Perturbation Agent for Flatland Railway (Bellman / PBIR version).

Equivalent of the power-grid LambdaPIRPerturbationAgent, adapted for
Flatland's multi-agent tree observations.

Algorithm (unchanged from power grid):
    Q(s,a) ← (1-α)*Q(s,a) + α*[R(s,a) + γ*V(s')]

Differences from power grid:
  - Observation: {agent_id: Node} instead of grid2op obs vector
  - Action space: (agent_id, corruption_type) pairs instead of obs index mutations
  - Reward: disruption via conflict/target/malfunction signals (not rho overload)
  - No obs.rho, no obs._vectorized, no grid2op imports
"""

import copy
import logging
from typing import Dict, Optional, Any, List, Tuple

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent
from perturbation_agents.utils import (
    flatten_obs_dict,
    corrupt_obs_dict,
    corrupt_node,
    disruption_reward,
    CORRUPTION_TYPES,
)

logger = logging.getLogger(__name__)


class LambdaPIRPerturbationAgent(BasePerturbationAgent):
    """
    Lambda-PIR with Bellman updates for Flatland.

    Hybrid attacker switching between:
      - Policy iteration (k-armed bandit on Q-values)
      - Value iteration (lookahead refinement)

    Action space: cross-product of (agent_handle × corruption_type).
    Reward: disruption_reward() — penalises conflict proximity, target confusion,
            and phantom malfunctions.

    Args:
        n_agents (int): Number of Flatland agents.
        lambda_param (float): Lookahead depth λ ∈ [0,1).
        initial_prob_policy (float): Starting fraction of policy-iteration steps.
        epsilon (float): Perturbation magnitude (not used for blanking, reserved).
        gradient_step_size (float): α for Bellman updates.
        refinement_iterations (int): Value-iteration refinement steps.
        decay_schedule (str): "linear", "exponential", or "constant".
        gamma (float): Bellman discount factor.
        seed (int): Random seed.
    """

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
        seed: int = 42,
        name: str = "LambdaPIRPerturbationAgent",
    ):
        super().__init__(name=name, seed=seed)
        self.n_agents = n_agents
        self.lambda_param = lambda_param
        self.initial_prob_policy = initial_prob_policy
        self.current_prob_policy = initial_prob_policy
        self.epsilon = epsilon
        self.gradient_step_size = gradient_step_size
        self.refinement_iterations = refinement_iterations
        self.decay_schedule = decay_schedule
        self.gamma = gamma

        # Build action space: (do_nothing) + (handle × corruption_type)
        self.possible_actions: List[Tuple] = [("do_nothing", None, None)]
        for handle in range(n_agents):
            for ctype in CORRUPTION_TYPES:
                self.possible_actions.append(("corrupt", handle, ctype))

        self.action_values = np.zeros(len(self.possible_actions))

        # Bellman state tracking
        self.prev_obs: Optional[Dict] = None
        self.prev_action_idx: Optional[int] = None
        self.prev_reward: Optional[float] = None

        self.iteration_count = 0
        self.policy_updates = 0
        self.value_updates = 0
        self.action_history: List[int] = []

        logger.info(f"Initialized {name} with {len(self.possible_actions)} actions "
                    f"(gamma={gamma}, n_agents={n_agents})")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def perturb(self, obs_dict: Dict) -> Dict:
        """Apply Lambda-PIR perturbation to the multi-agent observation dict."""
        try:
            # Bellman update from previous transition
            if self.prev_obs is not None and self.prev_action_idx is not None:
                self._bellman_update(self.prev_action_idx, self.prev_reward, obs_dict)

            # Select action
            prob_policy = self._probability_schedule(self.iteration_count)
            if self.space_prng.random() < prob_policy:
                action_idx = self._policy_step(obs_dict)
                self.policy_updates += 1
            else:
                action_idx = self._value_step(obs_dict)
                self.value_updates += 1

            # Apply action
            obs_perturbed = self._apply_action(obs_dict, action_idx)
            _atype, _h, _c = self.possible_actions[action_idx]
            self.last_handle = _h
            self.last_ctype = _c

            # Immediate reward
            reward = disruption_reward(obs_dict, obs_perturbed)

            # Store for next Bellman update
            self.prev_obs = copy.deepcopy(obs_dict)
            self.prev_action_idx = action_idx
            self.prev_reward = reward

            self.action_history.append(action_idx)
            self.iteration_count += 1
            self.perturbation_count += 1

            return obs_perturbed

        except Exception as e:
            logger.error(f"Perturbation failed: {e}")
            return copy.deepcopy(obs_dict)

    def reset(self) -> None:
        super().reset()
        self.prev_obs = None
        self.prev_action_idx = None
        self.prev_reward = None
        self.iteration_count = 0
        self.action_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.iteration_count,
            "policy_updates": self.policy_updates,
            "value_updates": self.value_updates,
            "current_prob_policy": self.current_prob_policy,
            "gamma": self.gamma,
            "mean_q_value": float(np.mean(self.action_values)),
            "bellman_updates": True,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bellman_update(self, prev_action_idx: int, reward: float, current_obs: Dict):
        """Q(s,a) ← (1-α)*Q(s,a) + α*[R(s,a) + γ*V(s')]"""
        next_v = self._state_value(current_obs)
        target = reward + self.gamma * next_v
        alpha = self.gradient_step_size
        self.action_values[prev_action_idx] = (
            (1 - alpha) * self.action_values[prev_action_idx] + alpha * target
        )

    def _state_value(self, obs_dict: Dict) -> float:
        """V(s) = max_a Q(s,a) over a small candidate set."""
        candidates = [0]  # do_nothing
        # Sample a few actions from each type
        step = max(1, len(self.possible_actions) // 10)
        candidates += list(range(1, len(self.possible_actions), step))
        best = max(
            self.action_values[i] + self._heuristic_value(obs_dict, i)
            for i in candidates
            if i < len(self.action_values)
        )
        return best

    def _heuristic_value(self, obs_dict: Dict, action_idx: int) -> float:
        """Domain heuristic bonus for an action."""
        action_type, handle, ctype = self.possible_actions[action_idx]
        if action_type == "do_nothing":
            return -10.0  # discourage doing nothing
        if handle is None or handle not in obs_dict:
            return 0.0
        node = obs_dict[handle]
        if node is None:
            return 0.0

        def safe(attr):
            v = getattr(node, attr, 0.0)
            return 0.0 if not np.isfinite(v) else float(v)

        score = 0.0
        if ctype == "conflict":
            # More valuable when conflict is already close
            dc = safe("dist_potential_conflict")
            score += max(0.0, 10.0 - dc)
        elif ctype == "target":
            # More valuable when train is close to target (corrupt to confuse)
            dt = safe("dist_min_to_target")
            score += max(0.0, 50.0 - dt) * 0.2
        elif ctype == "blank":
            # Always high value — full blindness is always disruptive
            score += 5.0
        return score

    def _policy_step(self, obs_dict: Dict) -> int:
        """Greedy over Q + heuristic, with ε-exploration."""
        if self.space_prng.random() < 0.1:
            return int(self.space_prng.randint(len(self.possible_actions)))
        scores = [
            self.action_values[i] + self._heuristic_value(obs_dict, i)
            for i in range(len(self.possible_actions))
        ]
        return int(np.argmax(scores))

    def _value_step(self, obs_dict: Dict) -> int:
        """Value iteration with refinement starting from policy step."""
        best_idx = self._policy_step(obs_dict)
        best_val = self.action_values[best_idx] + self._heuristic_value(obs_dict, best_idx)

        for _ in range(self.refinement_iterations):
            candidate = int(self.space_prng.randint(len(self.possible_actions)))
            val = self.action_values[candidate] + self._heuristic_value(obs_dict, candidate)
            if val > best_val:
                best_val = val
                best_idx = candidate

        return best_idx

    def _apply_action(self, obs_dict: Dict, action_idx: int) -> Dict:
        """Apply the selected action to the observation dict."""
        action_type, handle, ctype = self.possible_actions[action_idx]
        if action_type == "do_nothing":
            return copy.deepcopy(obs_dict)
        return corrupt_obs_dict(copy.deepcopy(obs_dict), handle, ctype)

    def _probability_schedule(self, iteration: int) -> float:
        if self.decay_schedule == "linear":
            factor = 1.0 / (1.0 + 0.01 * iteration)
        elif self.decay_schedule == "exponential":
            factor = np.exp(-0.01 * iteration)
        else:
            factor = 1.0
        self.current_prob_policy = float(
            np.clip(self.initial_prob_policy * factor, 0.1, 1.0)
        )
        return self.current_prob_policy
