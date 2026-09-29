"""
Lambda-PIR Perturbation Agent for the ATM domain (BlueSky-Gym observations).

Same agent as power_grid/framework/perturbation_agents/lambda_pir_perturbation_agent.py -
same class name, same API (_build_action_space / perturb / _policy_iteration_step /
_value_iteration_step / _evaluate_action / _apply_action / get_stats / reset) - with the
action space rebuilt over the BlueSky-Gym Dict observation instead of the grid2op vector,
exactly as the railway domain rebuilt it over Flatland tree observations.

ONE DELIBERATE DIFFERENCE from the power-grid file, which the evaluation depends on:
the perturbation is bounded by an explicit L-infinity budget xi. The power-grid agent
writes a sentinel (999999 for "large", 0 for "missing") into the observation. Against an
ATM policy that is not a robustness measurement: it is an impossible sensor reading, and
what it measures is the absence of input validation. Here "large"/"missing" mean
"+xi" / "-xi" on the chosen feature, so every KPI describes a bounded, physically
plausible sensor error, and the realised maximum is reported by get_stats().

Author: INESC TEC
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)

# Perturbation types, as in the power-grid agent's action space.
PERTURBATION_TYPES = ("do_nothing", "missing", "large")


class LambdaPIRPerturbationAgent(BasePerturbationAgent):
    """
    Lambda-PIR: alternates policy iteration (exploit the learned action values) and value
    iteration (refine by evaluating candidates), with a decaying probability schedule.

    Args:
        obs_space: Observation space of the BlueSky-Gym env (unused; kept for API parity).
        agent: The defender being attacked; queried for actions (black box).
        xi: L-infinity budget per observation feature.
        lambda_param, initial_prob_policy, gradient_step_size, refinement_iterations,
        decay_schedule: as in the power-grid agent.
    """

    def __init__(self,
                 obs_space=None,
                 agent=None,
                 xi: float = 0.1,
                 lambda_param: float = 0.7,
                 initial_prob_policy: float = 0.2,
                 gradient_step_size: float = 0.1,
                 refinement_iterations: int = 20,
                 decay_schedule: str = "exponential",
                 name: str = "LambdaPIRPerturbationAgent",
                 seed: int = 2,
                 debug: bool = False):
        # BasePerturbationAgent takes (name, seed, **kwargs); obs_space is kept for API parity
        super().__init__(name=name, seed=seed)

        self.agent = agent
        self.xi = float(xi)
        self.lambda_param = float(lambda_param)
        self.initial_prob_policy = float(initial_prob_policy)
        self.current_prob_policy = float(initial_prob_policy)
        self.gradient_step_size = float(gradient_step_size)
        self.refinement_iterations = int(refinement_iterations)
        self.decay_schedule = decay_schedule
        self.debug = debug

        self._rng = np.random.default_rng(seed)
        self.possible_actions = [("do_nothing", None, 0.0)]
        self.action_values = np.zeros(1, dtype=np.float64)

        self.iteration_count = 0
        self.policy_updates = 0
        self.value_updates = 0
        self.action_history = []
        self._max_perturb = {}

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def _build_action_space(self, sample_obs):
        """
        One action per (feature, perturbation type).

        Indices are resolved by KEY and position within that key's array, never by a
        hand-built offset table. The power-grid agent's offset table was wrong by six
        slots for years, silently attacking different features than it reported.
        """
        self.possible_actions = [("do_nothing", None, 0.0)]
        for key in sorted(sample_obs.keys()):
            size = int(np.asarray(sample_obs[key]).size)
            for i in range(size):
                self.possible_actions.append(("missing", (key, i), -self.xi))
                self.possible_actions.append(("large", (key, i), +self.xi))
        self.action_values = np.zeros(len(self.possible_actions), dtype=np.float64)
        logger.info(f"Built action space with {len(self.possible_actions)} actions")

    def _ensure_action_space_built(self, obs):
        if len(self.possible_actions) <= 1:
            self._build_action_space(obs)

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    def perturb(self, obs):
        """Select one action (policy or value iteration) and apply it to the observation."""
        try:
            self._ensure_action_space_built(obs)

            prob_policy = self._get_probability_schedule(self.iteration_count)
            if self._rng.random() < prob_policy:
                action_idx = self._policy_iteration_step(obs)
                self.policy_updates += 1
            else:
                action_idx = self._value_iteration_step(obs)
                self.value_updates += 1

            value = self._evaluate_action(obs, action_idx)
            self._update_action_value(action_idx, value)

            self.action_history.append(action_idx)
            self.iteration_count += 1
            return self._apply_action(obs, action_idx, record=True)

        except Exception as e:
            logger.error(f"Perturbation failed: {e}")
            return {k: np.array(v, dtype=np.float64, copy=True) for k, v in obs.items()}

    def _policy_iteration_step(self, obs) -> int:
        """Exploit the learned action values, with a little exploration."""
        if self._rng.random() < 0.1:
            return int(self._rng.integers(len(self.possible_actions)))
        return int(np.argmax(self.action_values))

    def _value_iteration_step(self, obs) -> int:
        """Refine: evaluate a sample of candidates and keep the best."""
        n = min(self.refinement_iterations, len(self.possible_actions))
        candidates = self._rng.choice(len(self.possible_actions), size=n, replace=False)
        best_idx, best_value = int(candidates[0]), -np.inf
        for idx in candidates:
            idx = int(idx)
            value = self._evaluate_action(obs, idx) + self.lambda_param * self.action_values[idx]
            if value > best_value:
                best_idx, best_value = idx, value
        return best_idx

    def _evaluate_action(self, obs, action_idx: int) -> float:
        """How far this perturbation moves the defender's action (black-box query)."""
        if self.agent is None:
            return 0.0
        try:
            base = np.asarray(self.agent.act(obs), dtype=np.float64).ravel()
            probe = np.asarray(self.agent.act(self._apply_action(obs, action_idx, record=False)),
                               dtype=np.float64).ravel()
            return float(np.linalg.norm(probe - base))
        except Exception as e:
            logger.debug(f"Action evaluation failed: {e}")
            return 0.0

    def _apply_action(self, obs, action_idx: int, record: bool = True):
        """Apply one action to a copy of the observation."""
        out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in obs.items()}
        action_type, target, delta = self.possible_actions[action_idx]
        if action_type == "do_nothing" or target is None:
            return out
        key, pos = target
        self._update_obs_attribute(out, key, pos, delta, record=record)
        return out

    def _update_obs_attribute(self, obs, key, pos, delta, record=True):
        """Write the bounded change into one feature, and record what was realised."""
        delta = float(np.clip(delta, -self.xi, self.xi))
        obs[key].reshape(-1)[pos] += delta
        if record:
            m = abs(delta)
            if m > self._max_perturb.get(key, 0.0):
                self._max_perturb[key] = m

    def _update_action_value(self, action_idx: int, value: float):
        """Incremental update of the action value (power-grid agent's rule)."""
        a = self.gradient_step_size
        self.action_values[action_idx] = (1 - a) * self.action_values[action_idx] + a * value

    def _get_probability_schedule(self, iteration: int) -> float:
        if self.decay_schedule == "linear":
            decay = 1.0 / (1.0 + 0.01 * iteration)
        elif self.decay_schedule == "exponential":
            decay = float(np.exp(-0.01 * iteration))
        else:
            decay = 1.0
        self.current_prob_policy = self.initial_prob_policy * decay
        return float(np.clip(self.current_prob_policy, 0.1, 1.0))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        counts = {t: 0 for t in PERTURBATION_TYPES}
        for idx in self.action_history:
            if idx < len(self.possible_actions):
                counts[self.possible_actions[idx][0]] += 1
        return {
            "total_iterations": self.iteration_count,
            "policy_updates": self.policy_updates,
            "value_updates": self.value_updates,
            "current_prob_policy": self.current_prob_policy,
            "action_counts": counts,
            "mean_q_value": float(np.mean(self.action_values)) if self.action_values.size else 0.0,
            "xi": self.xi,
            "max_perturb_per_key": dict(self._max_perturb),
        }

    def reset(self):
        """Between episodes: drop the per-episode history, keep the learned values."""
        self.iteration_count = 0
        self.action_history = []
