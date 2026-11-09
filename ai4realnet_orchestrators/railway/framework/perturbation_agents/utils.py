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

# Corruption applied to a flat observation vector when an attacker names a target but
# no corruption type. Negation is used rather than an amplifying type because the maze
# BC policy's Tanh activations saturate under amplification (same sign, so the action
# does not change), while Tanh(-x) = -Tanh(x) does move the policy.
DEFAULT_VECTOR_CTYPE = "speed"


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
    if ctype not in CORRUPTION_TYPES:
        # A silently unrecognised type returns the vector unchanged, which scores
        # the agent as robust against an attack that was never applied. Fail loudly.
        raise ValueError(
            f"corrupt_vector: unknown corruption type {ctype!r}; "
            f"expected one of {CORRUPTION_TYPES}"
        )
    v = vec.copy().astype(np.float64)
    if len(v) == 0:
        return v

    # Prefer the named fields when this vector IS a maze observation, so the corruption
    # type keeps the meaning it has on a TreeObs. Without this, "conflict", "target" and
    # "malfunction" all became "multiply the whole vector by 3", which was measured to
    # change the BC policy's action in 0 of 25 episodes -- recorded as an attack, but not
    # one the policy could respond to.
    semantic = _semantic_corruption(v, ctype)
    if semantic is not None:
        return semantic

    if ctype == "blank":
        v[:] = 0.0
    elif ctype in ("conflict", "target", "malfunction"):
        v *= 3.0
    else:  # "speed", "direction"
        v *= -1.0
    return v


def _semantic_corruption(v: np.ndarray, ctype: str):
    """
    Write the corruption into the maze observation fields it is actually about.

    Returns the corrupted vector, or None when this is not a maze observation of the
    expected width, when the layout cannot be read, or when the type has no field
    mapping ("blank" means total blindness, which is the whole vector by definition).
    """
    if ctype == "blank":
        return None
    try:
        from perturbation_agents.maze_obs_layout import layout_size, semantic_indices
    except ImportError:  # pragma: no cover - module always ships beside this one
        return None

    size = layout_size()
    # Only when this vector is one maze observation. A different width means a TreeObs
    # vector or some other format, where the statistical corruption is the right thing.
    if size is None or len(v) != size:
        return None

    resolved = semantic_indices(ctype)
    if resolved is None:
        return None
    idx, value, flag_idx = resolved

    out = v.copy()
    if value is None:
        # "target": the goal should look as distant as anything this observation can
        # express. Taken from the observation itself rather than a fixed constant, so it
        # stays inside the range the policy was trained on and needs no map-size
        # assumption.
        finite = out[np.isfinite(out)]
        value = float(np.max(np.abs(finite))) if finite.size else 1.0
    out[idx] = value
    if flag_idx.size:
        out[flag_idx] = 1.0
    return out
