"""
Evaluation wrapper around a BlueSky-Gym environment (ATM domain).

Exposes the same interface as the power-grid Environment and the railway
FlatlandEnvironment, so evaluation_framework/result_getter.py and metrics.py are shared
unchanged:

    reset(seed) -> flat observation vector
    step()      -> (obs, perturbation, act, act_unperturbed, reward, done)
    do_nothing_action(), get_similarity_score(a, b), last_performance

The BlueSky-Gym observation is a gymnasium Dict of float arrays (single aircraft):
    destination_waypoint_distance / _cos_drift / _sin_drift   (1,)
    restricted_area_radius / _distance                        (NUM_OBSTACLES,)
    cos_difference_restricted_area_pos / sin_...              (NUM_OBSTACLES,)
Actions are continuous: Box(-1, 1, (2,)) = (heading change, speed change).

Author: INESC TEC
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Distance to the destination waypoint, as the env reports it in the observation.
# Used for the per-step performance signal (see _compute_performance).
_DISTANCE_KEY = "destination_waypoint_distance"


class BlueSkyGymEnvironment:
    """
    Args:
        env:      A gymnasium env created with gym.make("StaticObstacleEnv-v0", ...).
        defender: An ATM defender exposing act(obs_dict) -> np.ndarray (see defenders.py).
        attacker: Attacker instance or None. result_getter sets `env.attacker = ...`.
    """

    def __init__(self, env, defender, attacker=None):
        self.env = env
        self.defender = defender
        self.attacker = attacker

        self._current_obs = {}
        self._keys = None            # fixed key order, established on the first observation
        self._initial_distance = None
        self.last_performance = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self._current_obs = obs
        if self._keys is None:
            self._keys = sorted(obs.keys())

        if self.attacker is not None and hasattr(self.attacker, "reset"):
            self.attacker.reset()
        if self.defender is not None and hasattr(self.defender, "reset"):
            self.defender.reset()

        # Baseline for the progress signal. Guard against a zero start distance so the
        # performance curve cannot divide by zero on a degenerate episode.
        d0 = float(np.ravel(obs[_DISTANCE_KEY])[0]) if _DISTANCE_KEY in obs else 0.0
        self._initial_distance = abs(d0) if abs(d0) > 1e-9 else None
        self.last_performance = self._compute_performance()

        return self._flatten_obs(obs)

    def step(self):
        """
        One environment step: perturb -> defender acts -> env advances.

        Returns:
            obs (np.ndarray), perturbation (np.ndarray), act (np.ndarray),
            act_unperturbed (np.ndarray), reward (float), done (bool)
        """
        clean_obs = self._current_obs

        if self.attacker is not None:
            perturbed_obs = self.attacker.perturb(clean_obs)
            perturbation = self._flatten_obs(perturbed_obs) - self._flatten_obs(clean_obs)
        else:
            perturbed_obs = clean_obs
            perturbation = np.zeros(self._obs_size(), dtype=np.float64)

        act = np.asarray(self.defender.act(perturbed_obs), dtype=np.float64).ravel()

        # Counterfactual action on the clean observation. The SB3 policy is stateless, so
        # a second call cannot disturb the rollout (unlike the railway/power-grid
        # defenders, whose act() advances internal state).
        if self.attacker is not None:
            act_unperturbed = np.asarray(self.defender.act(clean_obs), dtype=np.float64).ravel()
        else:
            act_unperturbed = act.copy()

        obs, reward, terminated, truncated, _ = self.env.step(act)
        self._current_obs = obs
        done = bool(terminated or truncated)

        self.last_performance = self._compute_performance()

        return self._flatten_obs(obs), perturbation, act, act_unperturbed, float(reward), done

    def do_nothing_action(self):
        """Neither heading nor speed change."""
        return np.zeros(2, dtype=np.float64)

    def get_similarity_score(self, act1, act2):
        """
        Similarity between two continuous action vectors, in [0, 1].

        The railway/power-grid domains use discrete actions and score equality. ATM
        actions are continuous in [-1, 1]^2, so equality would be 0 almost surely and
        KPI-SF-071 would report maximum severity for an arbitrarily small nudge. Instead
        the distance is normalised by the largest distance the action space allows,
        giving 1.0 for identical actions and 0.0 for opposite corners.
        """
        a1 = np.asarray(act1, dtype=np.float64).ravel()
        a2 = np.asarray(act2, dtype=np.float64).ravel()
        if a1.size == 0 or a1.shape != a2.shape:
            return 1.0
        max_distance = 2.0 * np.sqrt(a1.size)       # each component spans [-1, 1]
        return float(np.clip(1.0 - np.linalg.norm(a1 - a2) / max_distance, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_performance(self):
        """
        Progress towards the destination waypoint, in [0, 1]; higher is better.

        The resilience KPIs (AF-074, DF-075, RF-076) compare a performance curve against
        the unperturbed baseline, so the signal must vary over the episode. The ATM reward
        is dominated by sparse events (+1 on reaching the waypoint, -5 per intrusion), so
        progress towards the waypoint is used instead: it moves every step and falls
        behind the baseline exactly when an attack diverts the aircraft.
        """
        if not self._current_obs or self._initial_distance is None:
            return 0.0
        d = float(np.ravel(self._current_obs.get(_DISTANCE_KEY, [0.0]))[0])
        return float(np.clip(1.0 - abs(d) / self._initial_distance, 0.0, 1.0))

    def _obs_size(self):
        if self._keys is None or not self._current_obs:
            return 0
        return int(sum(np.ravel(self._current_obs[k]).size for k in self._keys))

    def _flatten_obs(self, obs_dict):
        """Concatenate the Dict observation into one vector, in a fixed key order."""
        if not obs_dict:
            return np.zeros(self._obs_size(), dtype=np.float64)
        keys = self._keys or sorted(obs_dict.keys())
        return np.concatenate([np.ravel(np.asarray(obs_dict[k], dtype=np.float64)) for k in keys])
