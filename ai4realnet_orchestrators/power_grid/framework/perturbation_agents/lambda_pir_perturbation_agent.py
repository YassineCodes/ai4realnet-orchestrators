"""
Lambda-PIR Perturbation Agent with Missing/Large/Adversarial Actions

Uses Bellman-based value updates (PBIR) for convergence guarantees:
Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R(s,a) + gamma * V(s')]

Where:
- R(s,a) is the immediate reward (disruption caused)
- gamma is the discount factor
- V(s') = max_a' Q(s', a') is the value of the next state

This enables Theorem 3 (Convergence with Randomization) to apply.
"""

import copy
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
import grid2op
from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)


class LambdaPIRPerturbationAgent(BasePerturbationAgent):
    """
    Lambda-PIR Agent with Missing/Large/Adversarial perturbations.

    Uses the same action space as RL agent:
    - Missing values (set to 0)
    - Large values (set to 999999)
    - Adversarial examples

    Lambda-PIR decides WHICH type and WHERE to apply based on:
    - Policy iteration: Quick decisions from learned patterns
    - Value iteration: Refined decisions through gradient search
    - Bellman updates: Proper V(s') bootstrapping for convergence
    """

    def __init__(self,
             obs_space: grid2op.Observation.ObservationSpace,
             agent,
             policy_model: Optional[Any] = None,
             lambda_param: float = 0.9,
             initial_prob_policy: float = 0.8,
             epsilon: float = 0.1,
             gradient_step_size: float = 0.05,
             refinement_iterations: int = 5,
             decay_schedule: str = "linear",
             gamma: float = 0.99,
             name: str = "LambdaPIRPerturbationAgent",
             save_dir: str = "",
             debug: bool = True,
             use_gpu: bool = True):
        """
        Initialize Lambda-PIR with Bellman updates.

        Args:
            obs_space: Grid2Op observation space
            agent: Target defender agent
            policy_model: Optional pre-trained PPO/SAC model
            lambda_param: lambda in [0,1) for lookahead depth
            initial_prob_policy: Starting probability of policy iteration
            epsilon: Maximum perturbation magnitude
            gradient_step_size: Learning rate for value updates (alpha)
            refinement_iterations: Number of refinement steps
            decay_schedule: "linear", "exponential", or "constant"
            gamma: Discount factor for Bellman updates (key for convergence)
            name: Agent identifier
            save_dir: Directory for saving history
            debug: Enable debug logging
            use_gpu: Enable GPU acceleration
        """
        super().__init__(obs_space, name=name)

        self.agent = agent
        self.policy_model = policy_model

        # Lambda-PIR parameters
        self.lambda_param = lambda_param
        self.initial_prob_policy = initial_prob_policy
        self.current_prob_policy = initial_prob_policy
        self.epsilon = epsilon
        self.gradient_step_size = gradient_step_size
        self.refinement_iterations = refinement_iterations
        self.decay_schedule = decay_schedule
        self.gamma = gamma  # Bellman discount factor
        self.debug = debug

        # Initialize with default action space
        self.possible_actions = [("do_nothing", 0)]
        self.missing_indices = []
        self.large_indices = []
        self.adv_indices = []
        self.attr_start_idx = {}

        # Tracking
        self.iteration_count = 0
        self.policy_updates = 0
        self.value_updates = 0
        self.action_history = []

        # Q-value estimates (updated via Bellman equation)
        self.action_values = np.zeros(len(self.possible_actions))

        # State for Bellman updates
        self.prev_obs = None
        self.prev_action_idx = None
        self.prev_reward = None

        logger.info(f"Initialized {name} with Bellman updates (gamma={gamma})")

    def _build_action_space(self, sample_obs):
        """Build the discrete action space matching RL agent."""
        self.possible_actions = []
        self.possible_actions.append(("do_nothing", 0))

        obs_vector = sample_obs.to_vect()

        # Map attribute start indices
        self.attr_start_idx = {}
        current_idx = 0
        for attr in ["year", "month", "day", "hour_of_day", "minute_of_hour",
                    "day_of_week", "gen_p", "gen_q", "gen_v", "load_p",
                    "load_q", "load_v", "p_or", "q_or", "v_or", "a_or",
                    "p_ex", "q_ex", "v_ex", "a_ex", "rho"]:
            if hasattr(sample_obs, attr):
                attr_array = getattr(sample_obs, attr)
                if isinstance(attr_array, np.ndarray):
                    self.attr_start_idx[attr] = current_idx
                    current_idx += len(attr_array)

        # Select key indices for perturbation (rho and power flows)
        rho_start = self.attr_start_idx.get("rho", 0)
        rho_end = rho_start + len(sample_obs.rho)
        p_or_start = self.attr_start_idx.get("p_or", 0)
        p_or_end = p_or_start + len(sample_obs.p_or)
        critical_indices = list(range(rho_start, rho_end)) + list(range(p_or_start, p_or_end))

        # Missing values
        self.missing_indices = []
        for idx in critical_indices:
            self.possible_actions.append(("missing", idx))
            self.missing_indices.append(len(self.possible_actions) - 1)

        # Large values
        self.large_indices = []
        for idx in critical_indices:
            self.possible_actions.append(("large", idx))
            self.large_indices.append(len(self.possible_actions) - 1)

        # Adversarial examples
        self.adv_indices = []
        if hasattr(self.agent, 'action_space'):
            n_agent_actions = min(20, getattr(self.agent.action_space, 'n', 20))
            for target_action in range(n_agent_actions):
                self.possible_actions.append(("adv_exmpl", target_action))
                self.adv_indices.append(len(self.possible_actions) - 1)

        logger.info(f"Built action space with {len(self.possible_actions)} actions")

    def _ensure_action_space_built(self, obs):
        """Build action space on first observation if not already built."""
        if len(self.possible_actions) == 1:
            try:
                self._build_action_space(obs)
                self.action_values = np.zeros(len(self.possible_actions))
                logger.info(f"Built action space with {len(self.possible_actions)} actions")
            except Exception as e:
                logger.error(f"Failed to build action space: {e}")

    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """Apply Lambda-PIR perturbation with Bellman updates."""
        try:
            self._ensure_action_space_built(obs)

            if len(self.possible_actions) == 0:
                logger.error("No actions available")
                return obs.copy()

            # Bellman update from previous transition
            if self.prev_obs is not None and self.prev_action_idx is not None:
                self._bellman_update(self.prev_obs, self.prev_action_idx,
                                    self.prev_reward, obs)

            obs_perturbed = copy.deepcopy(obs)

            # Get probability schedule
            prob_policy = self._get_probability_schedule(self.iteration_count)
            use_policy = self.space_prng.random() < prob_policy

            if use_policy:
                action_idx = self._policy_iteration_step(obs)
                self.policy_updates += 1
            else:
                action_idx = self._value_iteration_step(obs)
                self.value_updates += 1

            # Apply the selected action
            action_type, action_param = self.possible_actions[action_idx]
            obs_perturbed = self._apply_action(obs_perturbed, action_type, action_param)

            # Compute immediate reward
            reward = self._compute_reward(obs, obs_perturbed)

            # Store for next Bellman update
            self.prev_obs = copy.deepcopy(obs)
            self.prev_action_idx = action_idx
            self.prev_reward = reward

            self.action_history.append(action_idx)

            if self.debug and self.iteration_count % 50 == 0:
                logger.debug(f"[ITER {self.iteration_count}] action={action_type}({action_param}) "
                        f"policy={use_policy} p_k={prob_policy:.3f} reward={reward:.2f}")

            self.iteration_count += 1
            self.perturbation_count += 1

            return obs_perturbed

        except Exception as e:
            logger.error(f"Perturbation failed: {e}")
            return obs.copy()

    def _bellman_update(self, prev_obs, prev_action_idx: int, reward: float, current_obs):
        """
        Perform proper Bellman update for convergence guarantees.

        Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R(s,a) + gamma * V(s')]
        """
        try:
            # V(s') = max_a' Q(s', a')
            next_state_value = self._compute_state_value(current_obs)

            # Bellman target: R + gamma * V(s')
            bellman_target = reward + self.gamma * next_state_value

            # Update Q(s,a)
            alpha = self.gradient_step_size
            self.action_values[prev_action_idx] = (
                (1 - alpha) * self.action_values[prev_action_idx] +
                alpha * bellman_target
            )

        except Exception as e:
            logger.debug(f"Bellman update failed: {e}")

    def _compute_state_value(self, obs) -> float:
        """Compute V(s) = max_a Q(s,a)."""
        try:
            max_value = float('-inf')

            actions_to_evaluate = [0]  # do_nothing
            if self.missing_indices:
                actions_to_evaluate.extend(self.missing_indices[:5])
            if self.large_indices:
                actions_to_evaluate.extend(self.large_indices[:5])
            if self.adv_indices:
                actions_to_evaluate.extend(self.adv_indices[:3])

            for action_idx in actions_to_evaluate:
                if action_idx < len(self.possible_actions):
                    value = self._evaluate_action(obs, action_idx)
                    total_value = value + self.action_values[action_idx]
                    max_value = max(max_value, total_value)

            return max_value if max_value > float('-inf') else 0.0

        except Exception as e:
            logger.debug(f"State value computation failed: {e}")
            return 0.0

    def _compute_reward(self, obs_before, obs_after) -> float:
        """Compute immediate reward R(s,a)."""
        reward = 0.0

        try:
            if hasattr(obs_after, 'rho') and hasattr(obs_before, 'rho'):
                max_rho_after = np.max(obs_after.rho)
                max_rho_before = np.max(obs_before.rho)
                reward += (max_rho_after - max_rho_before) * 10

                critical_before = np.sum(obs_before.rho > 0.95)
                critical_after = np.sum(obs_after.rho > 0.95)
                reward += (critical_after - critical_before) * 50
        except Exception as e:
            logger.debug(f"Reward computation failed: {e}")

        return reward

    def _policy_iteration_step(self, obs) -> int:
        """Policy iteration with state-dependent heuristics."""
        if len(self.possible_actions) == 0:
            return 0

        if self.space_prng.random() < 0.1:  # 10% exploration
            return self.space_prng.randint(0, len(self.possible_actions))

        scores = self.action_values.copy()

        # Add heuristic bonuses (state-dependent)
        obs_vector = obs.to_vect()

        for i, (action_type, idx) in enumerate(self.possible_actions):
            if action_type == "large" and idx < len(obs_vector):
                if "rho" in self.attr_start_idx:
                    rho_start = self.attr_start_idx["rho"]
                    if rho_start <= idx < rho_start + len(obs.rho):
                        rho_idx = idx - rho_start
                        current_load = obs.rho[rho_idx]
                        scores[i] += current_load * 10

        if len(scores) == 0:
            return 0

        return np.argmax(scores)

    def _value_iteration_step(self, obs) -> int:
        """Value iteration with refinement."""
        best_action = self._policy_iteration_step(obs)
        best_value = self._evaluate_action(obs, best_action) + self.action_values[best_action]

        for _ in range(self.refinement_iterations):
            candidates = []
            action_type, _ = self.possible_actions[best_action]

            if action_type == "missing":
                candidates = self.missing_indices[:5]
            elif action_type == "large":
                candidates = self.large_indices[:5]
            elif action_type == "adv_exmpl":
                candidates = self.adv_indices[:5]

            for candidate_idx in candidates:
                heuristic_value = self._evaluate_action(obs, candidate_idx)
                total_value = heuristic_value + self.action_values[candidate_idx]
                if total_value > best_value:
                    best_value = total_value
                    best_action = candidate_idx

        return best_action

    def _evaluate_action(self, obs, action_idx: int) -> float:
        """Evaluate action using domain heuristics."""
        try:
            obs_test = copy.deepcopy(obs)
            action_type, action_param = self.possible_actions[action_idx]
            obs_test = self._apply_action(obs_test, action_type, action_param)

            score = 0.0

            if hasattr(obs_test, 'rho'):
                max_rho = np.max(obs_test.rho)
                score += max_rho * 10
                critical_lines = np.sum(obs_test.rho > 0.95)
                score += critical_lines * 50

            if action_type == "do_nothing":
                score -= 100

            return score

        except Exception as e:
            logger.debug(f"Action evaluation failed: {e}")
            return 0.0

    def _apply_action(self, obs, action_type: str, action_param: int):
        """Apply action to observation."""
        obs.to_vect()

        if action_type == "do_nothing":
            return obs
        elif action_type == "missing":
            obs._vectorized[action_param] = 0
            self._update_obs_attribute(obs, action_param, 0)
        elif action_type == "large":
            obs._vectorized[action_param] = 999999
            self._update_obs_attribute(obs, action_param, 999999)
        elif action_type == "adv_exmpl":
            obs_vector = obs.to_vect()
            noise = np.random.randn(*obs_vector.shape) * self.epsilon
            obs._vectorized = obs_vector + noise
            if hasattr(obs, 'rho'):
                rho_start = self.attr_start_idx.get("rho", 0)
                rho_end = rho_start + len(obs.rho)
                obs.rho = obs._vectorized[rho_start:rho_end]

        return obs

    def _update_obs_attribute(self, obs, idx: int, value: float):
        """Update attribute after modifying vectorized form."""
        for attr_name, start_idx in self.attr_start_idx.items():
            if hasattr(obs, attr_name):
                attr_array = getattr(obs, attr_name)
                end_idx = start_idx + len(attr_array)
                if start_idx <= idx < end_idx:
                    attr_idx = idx - start_idx
                    attr_array[attr_idx] = value
                    setattr(obs, attr_name, attr_array)
                    break

    def _get_probability_schedule(self, iteration: int) -> float:
        """Get probability of using policy iteration."""
        if self.decay_schedule == "linear":
            decay_factor = 1.0 / (1.0 + 0.01 * iteration)
        elif self.decay_schedule == "exponential":
            decay_factor = np.exp(-0.01 * iteration)
        else:
            decay_factor = 1.0

        self.current_prob_policy = self.initial_prob_policy * decay_factor
        return np.clip(self.current_prob_policy, 0.1, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        action_counts = {"do_nothing": 0, "missing": 0, "large": 0, "adv_exmpl": 0}

        for action_idx in self.action_history[-100:]:
            if action_idx < len(self.possible_actions):
                action_type, _ = self.possible_actions[action_idx]
                action_counts[action_type] += 1

        return {
            "total_iterations": self.iteration_count,
            "policy_updates": self.policy_updates,
            "value_updates": self.value_updates,
            "current_prob_policy": self.current_prob_policy,
            "action_counts": action_counts,
            "gamma": self.gamma,
            "bellman_updates": True,
            "mean_q_value": np.mean(self.action_values) if len(self.action_values) > 0 else 0.0,
        }

    def reset(self):
        """Reset between episodes."""
        self.iteration_count = 0
        self.action_history = []
        self.prev_obs = None
        self.prev_action_idx = None
        self.prev_reward = None
        super().reset()
