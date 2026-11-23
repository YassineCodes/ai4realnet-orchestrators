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

import logging

import numpy as np

logger = logging.getLogger(__name__)

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


def corrupt_node(node, corruption_type, severity=1.0):
    """
    Return a new Node (via _replace) with a specific adversarial corruption applied.

    corruption_type:
        "blank"       - all features set to -inf, childs emptied
        "conflict"    - dist_potential_conflict → 0 (phantom imminent collision)
        "target"      - dist_min_to_target → 1e6 (train appears lost)
        "speed"       - speed_min_fractional → 0 (phantom blockage)
        "direction"   - num_agents_opposite_direction → 100 (phantom head-ons)
        "malfunction" - num_agents_malfunctioning → 100 (phantom failures)

    severity scales how far past the plausible value the claim is pushed:
    `written = clean + severity * (plausible - clean)`, so 1.0 writes exactly the value
    above and larger values leave the range the simulator can produce. See
    perturbation_agents.attack_space for why that axis exists. "blank" ignores it.
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
    if corruption_type not in mapping:
        return node

    fields = mapping[corruption_type]
    if severity != 1.0:
        scaled = {}
        for field, plausible in fields.items():
            clean = getattr(node, field, 0.0)
            clean = 0.0 if (clean is None or not np.isfinite(clean)) else float(clean)
            scaled[field] = clean + severity * (plausible - clean)
        fields = scaled
    return node._replace(**fields)


def corrupt_obs_dict(obs_dict, handle, corruption_type, severity=1.0):
    """Apply a single-agent corruption, returning a new obs_dict dict."""
    new_obs = dict(obs_dict)
    if handle in new_obs and new_obs[handle] is not None:
        new_obs[handle] = corrupt_node(new_obs[handle], corruption_type, severity)
    return new_obs


_WARNED_NO_QUERY = set()


def query_disruption_reward(defender, obs_dict, ctype, severity, clean_probs=None):
    """
    Query-based attacker reward: how far a corruption moved the DEFENDER's decision.

        reward = p_clean[a*] - p_corrupted[a*],   a* = arg max p_clean

    i.e. how much confidence the policy loses in the action it wanted. This is power
    grid's attacker objective, which minimises the probability assigned to the preferred
    action, expressed for this domain.

    WHY THIS REPLACES disruption_reward()
    -------------------------------------
    disruption_reward() compares two TreeObs dicts and never consults the policy. It pays
    for making three hand-picked fields look worse, whether or not the agent's decision
    changes at all, and in this domain the policy does not even read the TreeObs: it reads
    the maze observation, so a perturbation scored there cannot reach it. Measured
    consequence: the share of an attacker's choices falling in the class the surrogate
    pays for but the policy ignores predicted measured effectiveness at Pearson -0.980,
    so the surrogate and not the algorithm explained the whole attacker ranking. Three of
    the four attackers that used it scored significantly WORSE than random noise.

    WHY A PROBABILITY AND NOT AN ACTION CHANGE
    ------------------------------------------
    Two reasons. One train acts per step, so "did the action change" is a single bit per
    step and a learned attacker cannot descend it. And KPI-FF-070 and KPI-SF-071 measure
    action change, so rewarding it would make the attacker optimise the metric it is
    being scored by. A probability drop is dense and is not one of the ten KPIs.

    THREAT MODEL
    ------------
    This makes the attacker query-based rather than black-box: it asks the defender what
    it would do. That is a declared adaptation, and it is the model GEPerturb already
    uses, which is the one attacker in this suite that beats the random baseline.

    Returns:
        float reward, or None when the defender cannot be queried. None means "no
        signal" and the caller must fall back, never read it as "no disruption".
    """
    probs_fn = getattr(defender, "action_probs", None)
    corrupt_fn = getattr(defender, "action_probs_with_corruption", None)
    if probs_fn is None or corrupt_fn is None:
        return None

    if clean_probs is None:
        clean_probs = probs_fn(obs_dict)
    if clean_probs is None:
        return None

    corrupted = corrupt_fn(obs_dict, ctype, severity)
    if corrupted is None or len(corrupted) != len(clean_probs):
        return None

    preferred = int(np.argmax(clean_probs))
    return float(clean_probs[preferred] - corrupted[preferred])


def warn_black_box_fallback(name):
    """Say once, per attacker, that it is scoring without querying the defender."""
    if name not in _WARNED_NO_QUERY:
        _WARNED_NO_QUERY.add(name)
        logger.warning(
            "%s has no queryable defender, so it is falling back to the observation-only "
            "disruption_reward(). That surrogate is what made the learned attackers score "
            "worse than random noise; the resulting KPI values are not comparable with a "
            "query-based run.", name)


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


def corrupt_vector(vec: np.ndarray, ctype: str, severity: float = 1.0,
                   bound=None) -> np.ndarray:
    """
    Apply a corruption to a flat observation vector.

    Used by perturb_vector() when injecting perturbations into non-TreeObs formats
    (e.g. the maze graph observation). When the vector IS a maze observation, the
    corruption is written into the fields its type is about; for any other width no
    semantic layout is known and the corruption is statistical.

    corruption_type:
        "blank"                      - zero the entire vector (full blindness)
        "conflict"/"target"/"malfunction" - amplify values ×3 (phantom danger)
        "speed"/"direction"          - negate values (reversed perception)

    Args:
        vec: the clean observation vector.
        ctype: corruption type, one of CORRUPTION_TYPES.
        severity: how far past the plausible value to push the claim. 1.0 writes the
            plausible value itself; larger values leave the range the simulator can
            produce, which is the axis the policy is actually sensitive to. Ignored for
            "blank" and on the statistical path, which have no per-field target value.
        bound: optional (low, high) of the observation space, applied after the
            corruption so the result stays a value the space can hold. None means no
            clip.

    Returns:
        A corrupted copy of vec.
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
    semantic = _semantic_corruption(v, ctype, severity)
    if semantic is not None:
        return _clip_to_space(semantic, bound)

    if ctype == "blank":
        v[:] = 0.0
    elif ctype in ("conflict", "target", "malfunction"):
        v *= 3.0
    else:  # "speed", "direction"
        v *= -1.0
    return _clip_to_space(v, bound)


def _clip_to_space(v, bound):
    """Clip into the observation space's declared range; a bound of None means no clip."""
    if bound is None:
        return v
    low, high = bound
    return np.clip(v, low, high)


def clip_to_budget(delta, clean, xi):
    """
    Enforce a perturbation budget on an already-computed delta.

    The other domains in this benchmark bound the attacker: the power-grid adversarial
    env clips its perturbation to +/- epsilon (0.1 by default) before adding it to the
    observation. Railway had no bound, so the attacks that work here are ones no bounded
    attacker could play, and a robustness number from this domain was not comparable with
    a bounded one.

    The bound is relative rather than absolute because these observations are not
    normalised: fields range from a -1 "absent" sentinel to distances in the tens, so one
    absolute epsilon would be most of a flag's range and a rounding error on a distance.
    Each element may move by xi times its own clean magnitude, with a floor of 1.0 so a
    field that reads exactly 0 can still be perturbed by xi.

    Args:
        delta: the (perturbed - clean) vector.
        clean: the clean observation the delta applies to.
        xi: budget as a fraction of each element's magnitude. None means unbounded.

    Returns:
        The delta, clipped elementwise. Unchanged when xi is None.
    """
    if xi is None:
        return delta
    limit = float(xi) * np.maximum(np.abs(np.asarray(clean, dtype=np.float64)), 1.0)
    return np.clip(delta, -limit, limit)


def _semantic_corruption(v: np.ndarray, ctype: str, severity: float = 1.0):
    """
    Write the corruption into the maze observation fields it is actually about.

    `severity` scales the deviation from the clean value, so severity 1.0 writes the
    plausible value for this corruption type and larger severities push the same claim
    past what the simulator produces. The scaling is signed, so a claim whose plausible
    value sits at a boundary (a conflict at distance 0) overshoots into the numbers the
    policy is actually sensitive to, while one pointing at the "absent" sentinel
    saturates there -- an asymmetry of the representation, not of the attack.

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
    out[idx] = out[idx] + severity * (value - out[idx])
    if flag_idx.size:
        out[flag_idx] = out[flag_idx] + severity * (1.0 - out[flag_idx])
    return out
