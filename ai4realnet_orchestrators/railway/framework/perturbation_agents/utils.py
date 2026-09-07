"""
Shared observation utilities for all Flatland perturbation agents.

Flatland observations are {agent_handle: Node} where Node is a namedtuple.
These utilities handle:
  - Flattening obs_dict to a 1D numpy vector (for model input)
  - Corrupting specific fields of a Node (namedtuple._replace)
  - Computing attacker reward (disruption caused)

The 12 root-level tree features per agent, in order:
  dist_own_target_encountered, dist_other_target_encountered,
  dist_other_agent_encountered, dist_potential_conflict,
  dist_unusable_switch, dist_to_next_branch, dist_min_to_target,
  num_agents_same_direction, num_agents_opposite_direction,
  num_agents_malfunctioning, speed_min_fractional,
  num_agents_ready_to_depart
"""

import numpy as np

NODE_FIELDS = [
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
N_NODE_FIELDS = len(NODE_FIELDS)
CHILD_DIRECTIONS = ["L", "F", "R", "B"]

# Perturbation types (index = action id)
CORRUPTION_TYPES = ["blank", "conflict", "target", "speed", "direction", "malfunction"]


def node_to_vector(node):
    """Recursively flatten a Flatland TreeObs Node to a 1D numpy vector."""
    if node is None:
        return np.array([], dtype=np.float64)

    features = np.array(
        [getattr(node, f, 0.0) for f in NODE_FIELDS], dtype=np.float64
    )
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    child_parts = []
    childs = getattr(node, "childs", {}) or {}
    for d in CHILD_DIRECTIONS:
        child = childs.get(d)
        # Flatland 4.x uses -inf float sentinels for empty branches
        if child is not None and not isinstance(child, (float, int)):
            child_parts.append(node_to_vector(child))

    return np.concatenate([features] + child_parts) if child_parts else features


def flatten_obs_dict(obs_dict):
    """Flatten {handle: Node} to a 1D numpy vector, handles sorted ascending."""
    if not obs_dict:
        return np.array([], dtype=np.float64)
    parts = [node_to_vector(obs_dict[h]) for h in sorted(obs_dict.keys())]
    return np.concatenate(parts)


def corrupt_node(node, corruption_type):
    """
    Return a new Node (via _replace) with a specific adversarial corruption applied.

    corruption_type:
        "blank"       - all features set to -inf, childs emptied
        "conflict"    - dist_potential_conflict → 0 (phantom imminent collision)
        "target"      - dist_min_to_target → 1e6 (train appears lost)
        "speed"       - speed_min_fractional → 0 (phantom blockage)
        "direction"   - num_agents_opposite_direction → 100 (phantom head-ons)
        "malfunction" - num_agents_malfunctioning → 100 (phantom failures)
    """
    if node is None:
        return node

    if corruption_type == "blank":
        kwargs = {f: -np.inf for f in NODE_FIELDS}
        kwargs["childs"] = {}
        return node._replace(**kwargs)

    mapping = {
        "conflict":    {"dist_potential_conflict": 0.0},
        "target":      {"dist_min_to_target": 1e6},
        "speed":       {"speed_min_fractional": 0.0},
        "direction":   {"num_agents_opposite_direction": 100.0},
        "malfunction": {"num_agents_malfunctioning": 100.0},
    }
    if corruption_type in mapping:
        return node._replace(**mapping[corruption_type])
    return node


def corrupt_obs_dict(obs_dict, handle, corruption_type):
    """Apply a single-agent corruption, returning a new obs_dict dict."""
    new_obs = dict(obs_dict)
    if handle in new_obs and new_obs[handle] is not None:
        new_obs[handle] = corrupt_node(new_obs[handle], corruption_type)
    return new_obs


def disruption_reward(obs_before, obs_after):
    """
    Compute attacker reward = how much disruption the perturbation caused.

    Rewarded for:
      - Decreasing dist_potential_conflict  (conflict feels closer)
      - Increasing dist_min_to_target       (train appears further from goal)
      - Increasing num_agents_malfunctioning (phantom failures reported)
    """
    reward = 0.0
    for handle in obs_before:
        nb = obs_before[handle]
        na = obs_after.get(handle)
        if nb is None or na is None:
            continue

        def safe(node, attr):
            v = getattr(node, attr, 0.0)
            return 0.0 if (v is None or not np.isfinite(v)) else float(v)

        # Conflict: lower is worse for the defender
        dc_b = safe(nb, "dist_potential_conflict")
        dc_a = safe(na, "dist_potential_conflict")
        reward += max(0.0, dc_b - dc_a)

        # Target distance: higher means train is confused
        dt_b = safe(nb, "dist_min_to_target")
        dt_a = safe(na, "dist_min_to_target")
        reward += max(0.0, dt_a - dt_b) * 0.1

        # Malfunctions: phantom failures
        m_b = safe(nb, "num_agents_malfunctioning")
        m_a = safe(na, "num_agents_malfunctioning")
        reward += max(0.0, m_a - m_b) * 5.0

    return reward


def corrupt_vector(vec: np.ndarray, ctype: str) -> np.ndarray:
    """
    Apply a format-agnostic corruption to any flat observation vector.

    Used by perturb_vector() when injecting perturbations into non-TreeObs
    formats (e.g. maze-rl internal graph observations).  No assumption is made
    about the vector's semantic layout — corruptions are statistical.

    corruption_type:
        "blank"                      - zero the entire vector (full blindness)
        "conflict"/"target"/"malfunction" - amplify values ×3 (phantom danger)
        "speed"/"direction"          - negate values (reversed perception)
    """
    v = vec.copy().astype(np.float64)
    if len(v) == 0:
        return v
    if ctype == "blank":
        v[:] = 0.0
    elif ctype in ("conflict", "target", "malfunction"):
        v *= 3.0
    elif ctype in ("speed", "direction"):
        v *= -1.0
    return v
