"""
Defender wrappers for Flatland railway agents.

Normalises the two incompatible act_many() signatures found in Flatland:
  - ShortestPathPolicy.act_many(handles, [env, env, ...])  (passes env per agent)
  - Observation-based act_many(handles, [obs, obs, ...])   (passes obs per agent)

All wrappers expose a single method:
    act(obs_dict: dict) -> action_dict: dict
"""


def _action_to_int(action):
    """
    Coerce a Flatland action to a plain int.

    Flatland's RailEnvActions is a fastenum.Enum (NOT an IntEnum), so policies
    such as ShortestPathPolicy return enum members that numpy cannot cast to
    int. Observation-based RL policies usually return plain ints already.
    """
    value = getattr(action, "value", action)
    return int(value)


class RailwayDefender:
    """Abstract base for all railway defender wrappers."""

    def act(self, obs_dict):
        """
        Choose actions for all active agents.

        Args:
            obs_dict (dict): {agent_handle: TreeObs Node}

        Returns:
            dict: {agent_handle: int action}
        """
        raise NotImplementedError


class ShortestPathDefender(RailwayDefender):
    """
    Wraps Flatland's ShortestPathPolicy.

    ShortestPathPolicy.act_many() needs the env object per agent slot,
    not tree observations. This wrapper holds a reference to the env so
    callers do not need to know the difference.

    Args:
        env: The RailEnv instance.
        policy: Optional pre-built ShortestPathPolicy. If None, one is created.
    """

    def __init__(self, env, policy=None):
        self._policy_factory = None
        if policy is None:
            from flatland.envs.rail_env_policies import ShortestPathPolicy
            self._policy_factory = ShortestPathPolicy
            policy = ShortestPathPolicy()
        self._policy = policy
        self._env = env

    def reset(self, seed=None):
        """
        Drop the policy's per-episode path cache.

        ShortestPathPolicy memoises `_shortest_paths` per agent handle and has
        no reset() of its own, so a policy reused after env.reset(regenerate_rail=True)
        would follow paths computed for the previous map. FlatlandEnvironment.reset()
        calls this hook before each episode.
        """
        if self._policy_factory is not None:
            self._policy = self._policy_factory()
        elif hasattr(self._policy, "reset"):
            self._policy.reset()
        elif hasattr(self._policy, "_shortest_paths"):
            self._policy._shortest_paths.clear()

    def act(self, obs_dict):
        handles = sorted(obs_dict.keys())
        actions = self._policy.act_many(handles, [self._env] * len(handles))
        return {h: _action_to_int(a) for h, a in actions.items()}


class ObsPolicyDefender(RailwayDefender):
    """
    Wraps any observation-based policy that has act_many(handles, obs_list).

    Suitable for trained RL agents (PPO, DQN, etc.) that take a list of
    individual tree observations as input.

    Args:
        policy: Policy object with act_many(handles, obs_list) method.
    """

    def __init__(self, policy):
        self._policy = policy

    def reset(self, seed=None):
        """Forward the per-episode reset to the wrapped policy, if it has one."""
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def act(self, obs_dict):
        handles = sorted(obs_dict.keys())
        obs_list = [obs_dict[h] for h in handles]
        actions = self._policy.act_many(handles, obs_list)
        return {h: _action_to_int(a) for h, a in actions.items()}


class DoNothingDefender(RailwayDefender):
    """Baseline defender: all agents always do nothing (action 0)."""

    def act(self, obs_dict):
        return {handle: 0 for handle in obs_dict}
