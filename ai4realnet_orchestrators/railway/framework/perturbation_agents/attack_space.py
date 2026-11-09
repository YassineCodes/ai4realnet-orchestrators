"""
The attackers' action space: which train, which false claim, and how extreme.

Why severity exists
-------------------
Until now an attack action was (which train) x (which corruption type) -- 19 actions for
three trains. Measured against the reference BC policy, that space does not contain the
attack that works. A corruption confined to the fields its type is about does essentially
nothing while the value it writes stays inside the range the simulator produces:
potential_conflict_dist never exceeds 20 in a clean rollout, and writing 2, 5 or 20 into
it changes the policy's action 0.000, 0.001 and 0.005 of the time. Writing 500 changes it
0.176 of the time, and 1369 changes it 0.252 of the time.

So the effective axis is how far outside the legitimate range the written number is, and
that axis was not expressible: every attacker was searching a space in which the one
effective dimension was absent. This module adds it as a third factor, so an attacker
chooses (train, type, severity) and can learn the magnitude instead of being fixed at the
plausible value.

Severity 1.0 is the plausible lie -- exactly the value corrupt_node() writes for that
corruption type on a TreeObs, which is what the two-factor space was permanently stuck
at. Higher severities push the same claim further past what the simulator could produce,
up to the observation space's own declared bound.

"blank" takes no severity: it means total blindness, which is the whole vector set to
zero, and there is no "more blind". Enumerating severities for it would only add
duplicate arms for a learner to waste samples on.
"""

from perturbation_agents.utils import CORRUPTION_TYPES

# Multipliers on the deviation from the clean value. 1.0 reproduces the plausible lie, so
# the two-factor action space is the severity-1.0 slice of this one and earlier results
# stay interpretable. The upper end is past the declared observation bound for every
# field, which the corruption clips to, so the top level means "as far out of range as
# this observation space allows" without naming a map size here.
SEVERITY_LEVELS = (1.0, 10.0, 100.0, 1000.0)

# The one type that ignores severity, because it is defined on the whole vector.
WHOLE_VECTOR_TYPES = ("blank",)


def attack_actions(n_agents, severity_levels=SEVERITY_LEVELS):
    """
    Enumerate the attack action space as a list of (handle, ctype, severity).

    Deterministic order, so an action index means the same thing across attackers and
    across runs. Index 0 is train 0's "blank".

    Args:
        n_agents (int): Number of trains.
        severity_levels (tuple): Severity multipliers to offer per non-blank type.

    Returns:
        list[tuple]: (handle, corruption_type, severity).
    """
    out = []
    for handle in range(n_agents):
        for ctype in CORRUPTION_TYPES:
            if ctype in WHOLE_VECTOR_TYPES:
                out.append((handle, ctype, 1.0))
            else:
                out.extend((handle, ctype, float(s)) for s in severity_levels)
    return out


def n_attack_actions(n_agents, severity_levels=SEVERITY_LEVELS):
    """Size of the action space attack_actions() enumerates."""
    return len(attack_actions(n_agents, severity_levels))
