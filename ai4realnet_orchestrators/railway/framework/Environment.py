"""
Flatland Environment wrapper for the AI4REALNET robustness/resilience framework.

Provides the same interface as the power-grid Environment so that
evaluation_framework/result_getter.py and metrics.py work unchanged:

    env.reset(seed)           -> flat obs vector (np.ndarray)
    env.step()                -> (obs, perturbation, act, act_unperturbed, reward, done)
    env.do_nothing_action()   -> np.ndarray of zeros, one per agent
    env.get_similarity_score  -> callable(act1, act2) -> float in [0, 1]
    env.attacker              -> set externally by result_getter

IMPORTANT -- perturbation semantics for a maze-flatland submission
------------------------------------------------------------------
The maze-flatland policy never reads the TreeObs that step() builds and that the
attacker's perturb() is called on. Its input is the maze observation produced by
GraphDirectionalObservationConversion: an encoding of maze-flatland's own
TreeSwitchObsRailEnv (computed on the same RailEnv), plus train features taken from the
FlatlandMazeState and the action mask.

What follows from that:

  - A perturbation applied only to the TreeObs cannot change this policy's action. An
    attacker that acts solely through the TreeObs -- for instance a query-based attacker
    scoring candidates through defender.act() -- is measuring structural immunity, not
    robustness. Those KPIs must be reported as NOT APPLICABLE.
  - The attacker's decision is therefore re-applied to the maze observation. The
    attacker chooses once per step, in perturb(), and publishes the choice; the defender
    only applies it (see MazeFlatlandDefender.act_and_step). This does change actions.
  - The perturbation recorded for a step is the delta actually applied to the maze
    observation the policy read, not the TreeObs delta, so that KPI-VF-073 correlates
    action changes against features the policy really saw. An all-zero record for a
    whole attacker means the attack never reached the policy.
  - The resilience KPIs are arrival-based and are affected whenever the attack changes
    train dynamics, independently of the above.

Two book-keeping series exist for the same reason. `last_env_time` carries the RailEnv
step counter, because one framework step is one decision point and may span several
timesteps; `last_queried_mask` marks the trains whose policy was actually queried,
because a skipped train has no action of its own.
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

        # RailEnv step counter after the most recent step, recorded per step by
        # result_getter. One framework step is one decision point; with the maze
        # defender's SubStepSkippingWrapper a decision point can span several timesteps,
        # and it spans a different number in the baseline and the perturbed run once
        # their trajectories diverge. The resilience KPIs are aligned on this clock
        # rather than on the decision index.
        self.last_env_time = 0

        # Boolean per agent handle: was this agent's policy queried in the most recent
        # step. A skipped or finished train never reaches the policy, so it has no
        # action of its own; including it in the action comparison would score it as
        # agreeing with the baseline and as having taken no action.
        self.last_queried_mask = None

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
            perturbation (np.ndarray):    The perturbation that was actually APPLIED to the
                                          observation the policy read. For a defender that
                                          drives its own env this is the delta in that
                                          policy's own observation format, not the TreeObs
                                          delta: the two are different representations, and
                                          correlating action changes against a delta the
                                          policy never received measures nothing.
            act (np.ndarray):             Actions taken (one int per agent).
            act_unperturbed (np.ndarray): Actions the defender would have taken on clean obs.
            reward (float):               Sum of all agents' rewards.
            done (bool):                  True when the episode ends.
        """
        n_agents = self.env.get_num_agents()
        obs_size = (self._single_obs_size or 0) * n_agents

        clean_obs = self._current_obs

        # Apply perturbation. This is the attacker's ONE decision per step: it selects
        # the targets and corruption types, advances its own learning state, and
        # publishes the choice in attacker.last_targets for the defender to apply.
        # Calling perturb() a second time per step drew an independent sample, so the
        # recorded perturbation belonged to a different draw than the applied one.
        if self.attacker is not None:
            perturbed_obs = self.attacker.perturb(clean_obs)
            perturbation = self._flatten_obs(perturbed_obs) - self._flatten_obs(clean_obs)
        else:
            perturbed_obs = clean_obs
            perturbation = np.zeros(obs_size, dtype=np.float64)

        if self.drives_env:
            # One flat step inside the defender's env. It returns the actions actually
            # taken, the counterfactual actions on clean observations, the reward and
            # done flag read from the very RailEnv the policy acted on, and the
            # perturbation it applied to the policy's own observation.
            action_dict, clean_action_dict, reward, done, applied = self.defender.act_and_step(
                perturbed_obs, clean_obs
            )
            # Record what the policy received, not what the attacker did to the TreeObs.
            # The TreeObs delta above is the attacker's INTENT in a representation this
            # policy never reads; reporting it as the perturbation made the vulnerability
            # KPI correlate action changes with a perturbation that was never applied.
            if applied is not None:
                perturbation = np.asarray(applied, dtype=np.float64)
            self._current_obs = self.defender.tree_obs()
            # Must be refreshed here too: this branch returns early, and the resilience
            # KPIs and KPI-RF-078 are computed from this signal. Leaving it frozen made a
            # working agent look like it delivered no trains at all.
            self.last_performance = self._compute_performance()
            # The defender knows which trains it served and what the RailEnv clock
            # reads; neither can be recovered from the action dict alone, because a
            # skipped train and a train that chose DO_NOTHING are indistinguishable
            # once the actions are flattened.
            self.last_env_time = self._read_env_time()
            self.last_queried_mask = self._queried_mask(
                getattr(self.defender, "last_queried_handles", None)
            )
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
        # On this path every agent is queried every step and one step is one timestep,
        # so the two series are the trivial ones. They are still recorded, so the
        # downstream metrics take the same code path in both modes.
        self.last_env_time = self._read_env_time()
        self.last_queried_mask = self._queried_mask(None)

        obs_vec = self._flatten_obs(self._current_obs)

        return obs_vec, perturbation, act_vec, act_unperturbed_vec, reward, done

    def _read_env_time(self):
        """
        RailEnv step counter, falling back to the previous value plus one.

        The fallback keeps the series strictly increasing on a Flatland build that does
        not expose _elapsed_steps, in which case it degenerates to the decision index
        and the alignment below becomes a no-op rather than an error.
        """
        if self.drives_env:
            t = getattr(self.defender, "last_env_time", None)
            if t is not None:
                return int(t)
        t = getattr(self.env, "_elapsed_steps", None)
        if t is not None:
            return int(t)
        return int(getattr(self, "last_env_time", 0) or 0) + 1

    def _queried_mask(self, handles):
        """
        Boolean array over sorted agent handles: was this agent's policy queried.

        `handles` of None means "every agent was queried", which is the case for every
        defender that does not drive its own environment.
        """
        all_handles = sorted(self.env.get_agent_handles())
        if handles is None:
            return np.ones(len(all_handles), dtype=bool)
        served = set(handles)
        return np.array([h in served for h in all_handles], dtype=bool)

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
