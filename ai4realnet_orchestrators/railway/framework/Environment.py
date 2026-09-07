"""
Flatland Environment wrapper for the AI4REALNET robustness/resilience framework.

Provides the same interface as the power-grid Environment so that
evaluation_framework/result_getter.py and metrics.py work unchanged:

    env.reset(seed)           -> flat obs vector (np.ndarray)
    env.step()                -> (obs, perturbation, act, act_unperturbed, reward, done)
    env.do_nothing_action()   -> np.ndarray of zeros, one per agent
    env.get_similarity_score  -> callable(act1, act2) -> float in [0, 1]
    env.attacker              -> set externally by result_getter
"""

import logging

import numpy as np


logger = logging.getLogger(__name__)

_NODE_FIELDS = [
    "dist_own_target_encountered",
    "dist_other_target_encountered",
    "dist_other_agent_encountered",
    "dist_potential_conflict",
    "dist_unusable_switch",
    "dist_to_next_branch",
    "dist_min_to_target",
    "num_agents_same_direction",
    "num_agents_opposite_direction",
    "num_agents_malfunctioning",
    "speed_min_fractional",
    "num_agents_ready_to_depart",
]
_N_NODE_FEATURES = len(_NODE_FIELDS)
_CHILD_DIRECTIONS = ["L", "F", "R", "B"]


def _node_to_vector(node):
    """
    Recursively flatten a Flatland TreeObs Node to a 1-D numpy array.

    Missing children are not included — the resulting vector size depends on
    the actual tree depth returned by the environment. All inf/-inf values
    are clipped to ±1e6 so cosine similarity arithmetic stays finite.
    """
    if node is None:
        return np.array([], dtype=np.float64)

    features = np.array(
        [getattr(node, f, 0.0) for f in _NODE_FIELDS], dtype=np.float64
    )
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    child_parts = []
    childs = node.childs if node.childs else {}
    for direction in _CHILD_DIRECTIONS:
        child = childs.get(direction)
        # Flatland 4.x uses -inf float sentinels for empty branches
        if child is not None and not isinstance(child, (float, int)):
            child_parts.append(_node_to_vector(child))

    if child_parts:
        return np.concatenate([features] + child_parts)
    return features


class FlatlandEnvironment:
    """
    Evaluation wrapper around a Flatland RailEnv.

    Args:
        env:      A Flatland RailEnv instance (already configured with obs builder).
        defender: A RailwayDefender instance (see defenders.py).
        attacker: Attacker instance or None. Can also be set via `env.attacker = ...`
                  from result_getter, matching the power-grid pattern.
    """

    def __init__(self, env, defender, attacker=None):
        # Some defenders own the environment rather than merely reading it. A
        # maze-flatland policy, for instance, must be driven through its own maze env,
        # which internally advances a RailEnv. In that case the defender's RailEnv is the
        # single source of truth: stepping a separate RailEnv here would evaluate the
        # policy against a different map from the one it observes.
        self.drives_env = bool(getattr(defender, "drives_env", False))
        if self.drives_env:
            # The defender's env wins unconditionally. Callers build a RailEnv for the
            # ordinary case; honouring it here would reintroduce the two-environment bug
            # this mode exists to prevent.
            if env is not None:
                logger.debug("defender drives the environment; ignoring the RailEnv passed in")
            env = defender.rail_env
        self.env = env
        self.defender = defender
        self.attacker = attacker

        self._current_obs = {}
        self._single_obs_size = None  # lazily determined from first real observation

        # Per-step performance signal for the resilience KPIs (AF-074, DF-075, RF-076).
        # Flatland's reward is sparse and terminal -- exactly 0.0 at every step with a
        # single penalty at the end -- so a reward *curve* carries no shape to detect
        # degradation or restoration in. The fraction of trains that have arrived does:
        # it is monotone non-decreasing, varies through the episode, and drops behind the
        # baseline exactly when an attack delays trains. See last_performance.
        self.last_performance = 0.0

    # ------------------------------------------------------------------
    # Public interface (same as power-grid Environment)
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """
        Reset the environment and return a flat observation vector.

        Args:
            seed (int or None): Random seed for reproducible episode generation.

        Returns:
            np.ndarray: Flat concatenated observation for all agents.
        """
        if self.drives_env:
            # The defender resets its own env and hands back tree observations computed
            # from that same env, so observations, actions and rewards all refer to one map.
            obs_dict = self.defender.reset(seed=seed)
            self.env = self.defender.rail_env
            self._current_obs = obs_dict if obs_dict else {}
        else:
            obs_dict, _ = self.env.reset(
                regenerate_rail=True,
                regenerate_schedule=True,
                random_seed=seed,
            )
            self._current_obs = obs_dict if obs_dict else {}

        if self.attacker is not None and hasattr(self.attacker, "reset"):
            self.attacker.reset()

        if (not self.drives_env) and self.defender is not None and hasattr(self.defender, "reset"):
            self.defender.reset(seed=seed)

        # Register attacker with state-based defenders (e.g. MazeFlatlandDefender)
        # so they can inject perturbations into their own internal observations.
        if hasattr(self.defender, "set_attacker"):
            self.defender.set_attacker(self.attacker)

        if self._single_obs_size is None:
            self._single_obs_size = self._measure_obs_size(self._current_obs)

        self.last_performance = self._compute_performance()

        return self._flatten_obs(self._current_obs)

    def step(self):
        """
        Run one environment step.

        Returns:
            obs (np.ndarray):             Flat observation after the step.
            perturbation (np.ndarray):    Difference vector (perturbed - clean) passed to the defender.
            act (np.ndarray):             Actions taken (one int per agent).
            act_unperturbed (np.ndarray): Actions the defender would have taken on clean obs.
            reward (float):               Sum of all agents' rewards.
            done (bool):                  True when the episode ends.
        """
        n_agents = self.env.get_num_agents()
        obs_size = (self._single_obs_size or 0) * n_agents

        clean_obs = self._current_obs

        # Apply perturbation
        if self.attacker is not None:
            perturbed_obs = self.attacker.perturb(clean_obs)
            perturbation = self._flatten_obs(perturbed_obs) - self._flatten_obs(clean_obs)
        else:
            perturbed_obs = clean_obs
            perturbation = np.zeros(obs_size, dtype=np.float64)

        if self.drives_env:
            # One flat step inside the defender's env. It returns the actions actually
            # taken, the counterfactual actions on clean observations, and the reward and
            # done flag read from the very RailEnv the policy acted on.
            action_dict, clean_action_dict, reward, done = self.defender.act_and_step(
                perturbed_obs, clean_obs
            )
            self._current_obs = self.defender.tree_obs()
            # Must be refreshed here too: this branch returns early, and the resilience
            # KPIs and KPI-RF-078 are computed from this signal. Leaving it frozen made a
            # working agent look like it delivered no trains at all.
            self.last_performance = self._compute_performance()
            act_vec = self._flatten_actions(action_dict)
            act_unperturbed_vec = (self._flatten_actions(clean_action_dict)
                                   if self.attacker is not None else act_vec.copy())
            return (self._flatten_obs(self._current_obs), perturbation,
                    act_vec, act_unperturbed_vec, float(reward), bool(done))

        # Defender acts on perturbed observations
        action_dict = self.defender.act(perturbed_obs)
        act_vec = self._flatten_actions(action_dict)

        # Defender acts on clean observations for robustness comparison.
        # Use act_comparison() if available — state-based defenders (e.g. MazeFlatland)
        # override this to return cached actions without advancing internal state,
        # preventing artificial action differences from double-stepping.
        if self.attacker is not None:
            if hasattr(self.defender, "act_comparison"):
                action_dict_clean = self.defender.act_comparison(clean_obs)
            else:
                action_dict_clean = self.defender.act(clean_obs)
            act_unperturbed_vec = self._flatten_actions(action_dict_clean)
        else:
            act_unperturbed_vec = act_vec.copy()

        # Step the Flatland environment
        self._current_obs, rewards_dict, dones_dict, _ = self.env.step(action_dict)

        reward = float(sum(rewards_dict.values())) if rewards_dict else 0.0
        done = dones_dict.get("__all__", False) if dones_dict else True

        # Recorded by result_getter alongside the reward; see last_performance.
        self.last_performance = self._compute_performance()

        obs_vec = self._flatten_obs(self._current_obs)

        return obs_vec, perturbation, act_vec, act_unperturbed_vec, reward, done

    def _compute_performance(self):
        """
        Fraction of trains that have arrived at their target, in [0, 1].

        This is the per-step performance signal the resilience KPIs are computed on.
        Higher is better, matching the "reward curve" semantics those KPIs assume, and
        unlike the raw Flatland reward it actually varies over the episode.

        Falls back to the done-flag count on Flatland versions without TrainState.
        """
        agents = getattr(self.env, "agents", None)
        if not agents:
            return 0.0
        try:
            from flatland.envs.step_utils.states import TrainState
            n_done = sum(1 for a in agents if a.state == TrainState.DONE)
        except Exception:
            n_done = sum(1 for a in agents if getattr(a, "position", None) is None
                         and getattr(a, "old_position", None) is not None)
        return n_done / len(agents)

    def do_nothing_action(self):
        """Return the do-nothing action vector (all zeros = RailEnvActions.DO_NOTHING)."""
        # number_of_agents is set in RailEnv.__init__ and is valid before reset();
        # get_num_agents() only returns non-zero after the first reset() call.
        n = getattr(self.env, "number_of_agents", None) or self.env.get_num_agents()
        return np.zeros(n, dtype=int)

    def get_similarity_score(self, act1, act2):
        """
        Compute similarity between two action vectors.

        Returns the fraction of agents with the same action (in [0, 1]).
        """
        if len(act1) == 0:
            return 1.0
        return float(np.mean(act1 == act2))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs_dict):
        """Concatenate all agents' tree observations into one vector."""
        if not obs_dict:
            n = self.env.get_num_agents()
            size = (self._single_obs_size or 0) * n
            return np.zeros(size, dtype=np.float64)

        vectors = []
        for handle in sorted(obs_dict.keys()):
            obs = obs_dict[handle]
            vec = _node_to_vector(obs)
            # Pad or truncate to the reference size established on first real obs
            if self._single_obs_size is not None:
                if len(vec) < self._single_obs_size:
                    vec = np.pad(vec, (0, self._single_obs_size - len(vec)))
                elif len(vec) > self._single_obs_size:
                    vec = vec[: self._single_obs_size]
            vectors.append(vec)

        return np.concatenate(vectors) if vectors else np.array([], dtype=np.float64)

    def _flatten_actions(self, action_dict):
        """Convert {handle: action} to a fixed-length numpy array."""
        handles = sorted(self.env.get_agent_handles())
        return np.array(
            [action_dict.get(h, 0) for h in handles], dtype=int
        )

    def _measure_obs_size(self, obs_dict):
        """Return the number of features in a single agent's tree observation."""
        for obs in obs_dict.values():
            if obs is not None:
                return len(_node_to_vector(obs))
        return _N_NODE_FEATURES  # minimal fallback: root node only
