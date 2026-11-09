"""
The severity axis and the perturbation budget.

Why these exist
---------------
An attack action used to be (which train) x (which corruption type). Measured against
the reference BC policy, that space does not contain the attack that works: a corruption
confined to the fields its type is about does essentially nothing while the value it
writes stays inside the range the simulator produces, and only becomes effective well
outside it. So the one effective dimension was not expressible, and every attacker was
searching a space without it. These tests pin:

  1. Severity is a real third factor, and severity 1.0 reproduces exactly the old
     two-factor action space, so earlier results remain the severity-1.0 slice of this
     one.
  2. "blank" takes no severity -- it is defined on the whole vector, and duplicate arms
     would only waste a learner's samples.
  3. The severity an attacker chooses reaches the observation the policy reads. A
     severity discarded on the way is the same class of defect as a discarded corruption
     type: the attacker is scored on a choice the policy never saw.
  4. A corruption cannot write a value the observation space could not hold.
  5. With a budget set, the delta the policy receives respects it -- and the recorded
     delta is the clipped one, so the KPI reports the perturbation that was applied.
  6. An attacker written against the old two-factor perturb_vector(vec, ctype) still
     works, and plays the plausible value.

The maze policy and maze env are faked, as in test_attack_application.py: the contract
is about the framework's plumbing.
"""

import os
import sys

import numpy as np

FRAMEWORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from attack_models.BaseAttackerClass import BaseAttackerClass  # noqa: E402
from perturbation_agents.attack_space import (  # noqa: E402
    SEVERITY_LEVELS,
    attack_actions,
    n_attack_actions,
)
from perturbation_agents.utils import (  # noqa: E402
    CORRUPTION_TYPES,
    clip_to_budget,
    corrupt_vector,
)
from evaluation_framework.test_attack_application import (  # noqa: E402
    N_TRAINS,
    OBS_KEY,
    OBS_LEN,
    FakeMazeEnv,
    FakePolicy,
    _step,
)


# ----------------------------------------------------------------------------------
# 1-2. The action space
# ----------------------------------------------------------------------------------

def test_severity_one_slice_is_exactly_the_old_action_space():
    """The two-factor space is recoverable, so old results stay interpretable."""
    actions = attack_actions(N_TRAINS)
    plausible = [(h, c) for h, c, s in actions if s == 1.0]
    assert sorted(plausible) == sorted(
        (h, c) for h in range(N_TRAINS) for c in CORRUPTION_TYPES
    )


def test_severity_adds_arms_without_duplicating_blank():
    actions = attack_actions(N_TRAINS)
    assert len(actions) == n_attack_actions(N_TRAINS)
    # Exactly one blank per train: blank is whole-vector, so severity does not apply.
    blanks = [a for a in actions if a[1] == "blank"]
    assert len(blanks) == N_TRAINS
    assert all(s == 1.0 for _, _, s in blanks)
    # Every other type is offered at every severity.
    for ctype in CORRUPTION_TYPES:
        if ctype == "blank":
            continue
        offered = sorted(s for h, c, s in actions if c == ctype and h == 0)
        assert offered == sorted(float(s) for s in SEVERITY_LEVELS)


def test_action_index_is_stable():
    """An index must mean the same action across attackers and across runs."""
    assert attack_actions(N_TRAINS) == attack_actions(N_TRAINS)
    assert attack_actions(2) == attack_actions(3)[: n_attack_actions(2)]


# ----------------------------------------------------------------------------------
# 3-4. Severity on the vector
# ----------------------------------------------------------------------------------

def test_severity_scales_the_deviation_from_the_clean_value():
    from perturbation_agents.maze_obs_layout import layout_size, semantic_indices

    size = layout_size()
    if size is None or semantic_indices("conflict") is None:
        import pytest
        pytest.skip("maze-flatland observation layout unavailable")

    idx, plausible, _flags = semantic_indices("conflict")
    clean = np.full(size, -1.0)

    at_one = corrupt_vector(clean, "conflict", severity=1.0)
    assert np.allclose(at_one[idx], plausible), "severity 1.0 must write the plausible value"

    # written = clean + severity * (plausible - clean), so each step out is linear in
    # severity and leaves the range the simulator produces.
    for severity in (10.0, 100.0):
        out = corrupt_vector(clean, "conflict", severity=severity)
        expected = -1.0 + severity * (plausible - -1.0)
        assert np.allclose(out[idx], expected)
        assert abs(out[idx][0]) > abs(at_one[idx][0]), "higher severity must move further"


def test_corruption_cannot_leave_the_declared_observation_space():
    from perturbation_agents.maze_obs_layout import layout_size, semantic_indices

    size = layout_size()
    if size is None or semantic_indices("conflict") is None:
        import pytest
        pytest.skip("maze-flatland observation layout unavailable")

    clean = np.full(size, -1.0)
    bound = (-1.0, 1369.0)
    out = corrupt_vector(clean, "conflict", severity=10_000.0, bound=bound)
    assert out.min() >= bound[0] and out.max() <= bound[1]


def test_severity_is_ignored_where_it_has_no_meaning():
    """blank is the whole vector zeroed, at any severity."""
    clean = np.arange(1.0, OBS_LEN + 1.0)
    for severity in (1.0, 1000.0):
        assert np.allclose(corrupt_vector(clean, "blank", severity=severity), 0.0)


# ----------------------------------------------------------------------------------
# 5. The budget
# ----------------------------------------------------------------------------------

def test_clip_to_budget_is_relative_to_each_element():
    clean = np.array([0.0, 1.0, 20.0, -1.0])
    delta = np.array([100.0, 100.0, 100.0, -100.0])
    out = clip_to_budget(delta, clean, xi=0.1)
    # xi * max(|clean|, 1): the floor of 1.0 keeps a field reading exactly 0 perturbable.
    assert np.allclose(out, [0.1, 0.1, 2.0, -0.1])


def test_no_budget_leaves_the_delta_untouched():
    delta = np.array([5.0, -7.0])
    assert np.allclose(clip_to_budget(delta, np.ones(2), xi=None), delta)


# ----------------------------------------------------------------------------------
# 6. End to end through the defender
# ----------------------------------------------------------------------------------

class _BoundedFakeMazeEnv(FakeMazeEnv):
    """FakeMazeEnv that declares an observation space, as the real maze env does."""

    class _Box:
        def __init__(self, low, high):
            self.low, self.high = low, high

    class _Dict:
        def __init__(self, spaces):
            self.spaces = spaces

    @property
    def observation_space(self):
        return self._Dict({OBS_KEY: self._Box(-1.0, 8.0)})


class SeverityAttacker(BaseAttackerClass):
    """Publishes a fixed (ctype, severity) and records what reaches perturb_vector."""

    def __init__(self, handle, ctype, severity):
        super().__init__(name="SeverityAttacker")
        self._decision = (handle, ctype, severity)
        self.seen = []

    def perturb(self, obs):
        handle, ctype, severity = self._decision
        self._publish_targets({handle: ctype}, {handle: severity})
        return obs

    def perturb_vector(self, vec, ctype=None, severity=None, bound=None):
        self.seen.append((ctype, severity, bound))
        return super().perturb_vector(vec, ctype, severity, bound)


class TwoFactorAttacker(BaseAttackerClass):
    """An attacker written before severity existed. Must still work."""

    def __init__(self, handle, ctype):
        super().__init__(name="TwoFactorAttacker")
        self._decision = (handle, ctype)
        self.seen = []

    def perturb(self, obs):
        handle, ctype = self._decision
        self._publish_targets({handle: ctype})
        return obs

    def perturb_vector(self, vec, ctype=None):
        self.seen.append(ctype)
        return super().perturb_vector(vec, ctype)


def _make_defender(attacker, xi=None, env=None):
    from defenders_maze import MazeFlatlandDefender

    maze_env = env if env is not None else FakeMazeEnv()
    defender = MazeFlatlandDefender(
        maze_env=maze_env, policy=FakePolicy(), n_trains=N_TRAINS,
        attack_mode="obs_perturb", xi=xi,
    )
    defender._maze_obs = maze_env.obs()
    defender.set_attacker(attacker)
    return defender


def test_the_chosen_severity_reaches_the_observation_the_policy_reads():
    attacker = SeverityAttacker(handle=0, ctype="conflict", severity=100.0)
    defender = _make_defender(attacker)
    _step(defender, attacker)
    assert attacker.seen, "perturb_vector was never called; no attack was applied"
    ctypes = [c for c, _s, _b in attacker.seen]
    severities = [s for _c, s, _b in attacker.seen]
    assert set(ctypes) == {"conflict"}
    assert set(severities) == {100.0}, (
        "the defender applied a different severity than the attacker chose"
    )


def test_a_two_factor_attacker_still_works_and_plays_the_plausible_value():
    attacker = TwoFactorAttacker(handle=0, ctype="conflict")
    defender = _make_defender(attacker)
    _step(defender, attacker)
    assert attacker.seen == ["conflict"]
    assert attacker.last_severity == 1.0


def test_the_budget_bounds_the_delta_that_is_applied_and_recorded():
    attacker = SeverityAttacker(handle=0, ctype="conflict", severity=1000.0)
    xi = 0.1
    defender = _make_defender(attacker, xi=xi)
    *_rest, applied = _step(defender, attacker)

    clean = FakeMazeEnv().obs()[OBS_KEY]
    limit = xi * np.maximum(np.abs(clean), 1.0)
    nonzero = applied[np.abs(applied) > 0]
    assert nonzero.size, "a bounded attack must still perturb something"
    assert np.max(np.abs(applied)) <= np.max(limit) + 1e-9, (
        "the applied perturbation exceeded the budget"
    )


def test_an_unbounded_attack_exceeds_what_the_budget_would_allow():
    """Guards the budget test above against passing because nothing was applied."""
    attacker = SeverityAttacker(handle=0, ctype="conflict", severity=1000.0)
    defender = _make_defender(attacker, xi=None)
    *_rest, applied = _step(defender, attacker)

    clean = FakeMazeEnv().obs()[OBS_KEY]
    assert np.max(np.abs(applied)) > 0.1 * np.max(np.abs(clean))


def test_the_declared_space_bound_is_read_from_the_env():
    attacker = SeverityAttacker(handle=0, ctype="conflict", severity=1000.0)
    defender = _make_defender(attacker, env=_BoundedFakeMazeEnv())
    _step(defender, attacker)
    bounds = [b for _c, _s, b in attacker.seen]
    assert bounds and all(b == (-1.0, 8.0) for b in bounds), (
        "the corruption was not given the observation space the env declares"
    )


# ----------------------------------------------------------------------------------
# 7. A query-based attacker must score against the observation the policy reads
# ----------------------------------------------------------------------------------

def test_act_with_corruption_scores_against_the_policys_own_observation():
    """
    The defect this guards: GEPerturbAttacker scored candidates by corrupting the
    TreeObs and calling act(). The maze policy does not read the TreeObs, so every
    candidate scored identically, the best score was always 0, and the attacker fell
    through to its random fallback on every step -- its search carried no information.
    """
    defender = _make_defender(attacker=None)
    obs_dict = {h: None for h in range(N_TRAINS)}

    clean = defender.act(obs_dict)
    # FakePolicy picks on the sign of the observation sum; blanking flips it.
    corrupted = defender.act_with_corruption(obs_dict, "blank")
    assert clean != corrupted, (
        "a corruption that changes the policy's action scored as no change"
    )


def test_act_with_corruption_does_not_advance_the_env():
    """One query per candidate: stepping here would race the env ahead of the framework."""
    from defenders_maze import MazeFlatlandDefender

    maze_env = FakeMazeEnv()
    defender = MazeFlatlandDefender(maze_env=maze_env, policy=FakePolicy(),
                                    n_trains=N_TRAINS, attack_mode="obs_perturb")
    defender._maze_obs = maze_env.obs()
    obs_dict = {h: None for h in range(N_TRAINS)}

    before = (maze_env._cursor, maze_env._rail_env._elapsed_steps)
    for _ in range(10):
        defender.act_with_corruption(obs_dict, "conflict", 1000.0)
    assert (maze_env._cursor, maze_env._rail_env._elapsed_steps) == before


def test_the_gradient_estimation_attacker_uses_the_corruption_aware_scorer():
    from perturbation_agents.grad_est_perturb_agent import (
        GradientEstimationPerturbationAgent,
    )

    class _Defender:
        def __init__(self):
            self.act_calls = 0
            self.scorer_calls = []

        def act(self, obs_dict):
            self.act_calls += 1
            return {h: 0 for h in obs_dict}

        def act_with_corruption(self, obs_dict, ctype, severity=1.0):
            self.scorer_calls.append((ctype, severity))
            # Report a changed action, so the search has something to prefer and does
            # not reach the random fallback.
            return {h: 1 for h in obs_dict}

    defender = _Defender()
    agent = GradientEstimationPerturbationAgent(defender=defender, n_agents=N_TRAINS)
    agent.perturb({h: None for h in range(N_TRAINS)})

    assert defender.scorer_calls, "the attacker never scored against the maze observation"
    assert len(defender.scorer_calls) == agent.n_candidates
    # The candidate it settles on must be one it actually scored.
    assert (agent.last_ctype, agent.last_severity) in defender.scorer_calls
