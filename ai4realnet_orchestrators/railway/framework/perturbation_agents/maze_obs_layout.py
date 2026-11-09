"""
Semantic field layout of the maze-flatland observation.

Why this exists
---------------
The attackers choose a corruption TYPE whose meaning is a named observation field:
"conflict" means "claim a conflict is imminent", "target" means "claim the train is far
from its goal", and so on. `corrupt_node()` honours that meaning on a Flatland TreeObs,
because a TreeObs is a namedtuple whose fields can be addressed by name.

The policy under evaluation does not read a TreeObs. It reads the maze-flatland graph
observation: one flat float vector per train. `corrupt_vector()` was written to be
"format-agnostic -- corruptions are statistical", so on that vector all six corruption
types collapsed to three whole-vector operations (zero, x3, x-1). Measured against the
BC policy, the x3 class -- which is where "conflict", "target" and "malfunction" landed --
changed the policy's action in 0 of 25 episodes, i.e. it was indistinguishable from no
attack at all, while still being recorded as a perturbation. An attacker that chose any
of those three was scored as having attacked when nothing reached the decision.

maze-flatland names every element of that vector itself, via
`GraphDirectionalObservationConversion.convert_to_dict()`. This module asks it for that
layout at runtime and resolves each corruption type to the indices it should actually
touch, so the attacker's choice keeps its meaning in the representation the policy reads.
Nothing here hardcodes the vector's size or field order: both come from maze-flatland, so
a change on that side is picked up rather than silently mismatched.
"""

import logging
import re
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Which observation fields each corruption type is about. The patterns are matched
# against maze-flatland's own field names, so they follow its naming rather than a copy
# of its layout: "L--", "F--", "R--" prefix the left/forward/right branches, and a second
# segment ("L-F--") is the branch beyond that one.
#
# Each entry is (regex over field names, value to write). The value mirrors the intent of
# corrupt_node() for the same corruption type:
#   conflict    - a conflict is imminent              -> conflict distance 0
#   target      - the train has lost its goal         -> target distance maximal
#   speed       - the train believes it cannot move   -> speed 0
#   direction   - phantom oncoming traffic            -> nearest other train at distance 0
#   malfunction - phantom blockage ahead              -> deadlock flag set
# "blank" is not listed: it means total blindness, which is the whole vector, not a field.
CTYPE_FIELDS = {
    "conflict":    (r"potential_conflict_dist$", 0.0),
    "target":      (r"target_dist$", None),          # None -> that field's space maximum
    "speed":       (r"^train_speed$", 0.0),
    "direction":   (r"(other_train_dist|conflicting_train_distance)$", 0.0),
    "malfunction": (r"deadlock$", 1.0),
}

# Fields set to 1.0 alongside "direction", so phantom oncoming traffic is also reported as
# blocking rather than merely being nearby.
DIRECTION_FLAG_FIELDS = r"obstructed$"


@lru_cache(maxsize=1)
def field_index_map():
    """
    {field name: index} for the maze observation vector, as maze-flatland defines it.

    Returns None when maze-flatland is not importable or its layout cannot be read, so
    callers can fall back rather than crash on a non-maze observation format.
    """
    try:
        from maze_flatland.space_interfaces.observation_conversion.graph_based_directional import (  # noqa: E501
            GraphDirectionalObservationConversion,
        )
    except ImportError:
        logger.warning(
            "maze-flatland is not importable; maze observation fields cannot be resolved "
            "and corruption types will fall back to whole-vector operations, which are "
            "not semantically distinct."
        )
        return None

    # convert_to_dict() zips its own key list against the vector it is handed, so handing
    # it the index of each position recovers {name: index} without assuming the length.
    probe_len = 4096
    try:
        mapping = GraphDirectionalObservationConversion.convert_to_dict(
            {"observation": np.arange(probe_len)}
        )
    except Exception as exc:  # pragma: no cover - depends on the installed maze version
        logger.warning("could not read the maze observation layout: %s", exc)
        return None

    # zip() stops at the shorter sequence, so every key that came back was matched to an
    # index below probe_len; a key list longer than that would be silently truncated.
    out = {k: int(v) for k, v in mapping.items() if np.isscalar(v) or np.ndim(v) == 0}
    if not out:
        return None
    if max(out.values()) >= probe_len - 1:
        logger.warning("maze observation layout exceeds the probe length; not using it")
        return None
    return out


@lru_cache(maxsize=1)
def layout_size():
    """Number of named positions in the maze observation, or None if unknown."""
    m = field_index_map()
    return None if m is None else len(m)


@lru_cache(maxsize=64)
def semantic_indices(ctype):
    """
    Indices the given corruption type should write, and the value to write.

    Returns (indices, value, flag_indices) or None when the type has no field mapping
    (notably "blank", which is whole-vector by definition) or the layout is unknown.
    `value` of None means "use the per-field maximum", resolved by the caller against the
    observation space.
    """
    if ctype not in CTYPE_FIELDS:
        return None
    mapping = field_index_map()
    if mapping is None:
        return None

    pattern, value = CTYPE_FIELDS[ctype]
    idx = sorted(i for name, i in mapping.items() if re.search(pattern, name))
    if not idx:
        logger.warning("no maze observation field matches %r for corruption %r",
                       pattern, ctype)
        return None

    flags = ()
    if ctype == "direction":
        flags = tuple(sorted(i for name, i in mapping.items()
                             if re.search(DIRECTION_FLAG_FIELDS, name)))

    return np.asarray(idx, dtype=int), value, np.asarray(flags, dtype=int)
