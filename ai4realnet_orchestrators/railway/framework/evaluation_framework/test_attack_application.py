"""
Attack decision/application contract for the railway framework.

These tests pin the behaviour reviewed by the maze-flatland developers:

  1. The attacker decides ONCE per environment step. act_and_step() must not call
     perturb() again -- a second call drew an independent sample, so the recorded
     perturbation belonged to a different draw than the applied one, halved the
     exploration schedule, and made half of a learning attacker's bootstrapped
     updates see s' = s.
  2. The corruption TYPE the attacker chose is the type applied. Discarding it
     reduced a learning attacker's contribution to "which train", so its learned
     field selection never reached the policy.
  3. Breadth is what the attacker chose. An attacker that blinds one train must be
     applied to one train, not to all of them.
  4. The perturbation that is RECORDED is the perturbation that was APPLIED.
  5. An attack that never reaches the policy is observable as an all-zero applied
     perturbation, so the KPI can be reported as not applicable.

The maze policy and maze env are faked: the contract is about the framework's
plumbing, and faking them keeps the test runnable without the BC checkpoint.
"""

import os
import sys

import numpy as np
import pytest

FRAMEWORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from attack_models.BaseAttackerClass import BaseAttackerClass  # noqa: E402
from perturbation_agents.utils import DEFAULT_VECTOR_CTYPE  # noqa: E402

N_TRAINS = 3
OBS_KEY = "observation"
OBS_LEN = 4


class _ActorId:
    def __init__(self, agent_id):
        self.agent_id = agent_id


class FakeRailEnv:
    def __init__(self):
        self._elapsed_steps = 0
        self.rewards_dict = {h: 0.0 for h in range(N_TRAINS)}
        self.dones = {"__all__": False}
        self.agents = []
        self.distance_map = None

    def get_num_agents(self):
        return N_TRAINS


class FakeMazeEnv:
    """Serves one train per sub-step and advances a fake RailEnv after the last."""

    def __init__(self, n_trains=N_TRAINS):
        self.n_trains = n_trains
        self._cursor = 0
        self.core_env = self
        self._rail_env = FakeRailEnv()

    def actor_id(self):
        return _ActorId(self._cursor)

    def obs(self):
        return {
            OBS_KEY: np.arange(1.0, OBS_LEN + 1.0) + self._cursor,
            "train_move_mask": np.ones(5, dtype=np.int8),
        }

    def step(self, action):
        self._cursor += 1
        if self._cursor >= self.n_trains:
            self._cursor = 0
            self._rail_env._elapsed_steps += 1
        return self.obs(), 0.0, False, {}


class FakePolicy:
    """Deterministic policy: the action follows the sign of the observation sum."""

    def compute_action(self, observation, maze_state, env, actor_id, deterministic):
        total = float(np.sum(observation[OBS_KEY]))
        return {"train_move": 1 if total > 0 else 2}


class CountingAttacker(BaseAttackerClass):
    """Records every perturb() call and publishes a fixed decision."""

    def __init__(self, targets, name="CountingAttacker"):
        super().__init__(name=name)
        self._targets = dict(targets)
        self.perturb_calls = 0
        self.perturb_vector_calls = []

    def perturb(self, obs):
        self.perturb_calls += 1
        self._publish_targets(self._targets)
        return obs

    def perturb_vector(self, vec, ctype=None):
        self.perturb_vector_calls.append(ctype)
        return super().perturb_vector(vec, ctype)


def _make_defender(attacker, attack_mode="obs_perturb"):
    from defenders_maze import MazeFlatlandDefender

    maze_env = FakeMazeEnv()
    defender = MazeFlatlandDefender(
        maze_env=maze_env, policy=FakePolicy(), n_trains=N_TRAINS,
        attack_mode=attack_mode,
    )
    defender._maze_obs = maze_env.obs()
    defender.set_attacker(attacker)
    return defender


def _step(defender, attacker):
    """One flat step, driven the way FlatlandEnvironment.step() drives it."""
    clean_obs = {h: None for h in range(N_TRAINS)}
    if attacker is not None:
        attacker.perturb(clean_obs)
    return defender.act_and_step(clean_obs, clean_obs)


# ---------------------------------------------------------------- 1. one decision
def test_act_and_step_does_not_call_perturb_again():
    attacker = CountingAttacker({0: "blank"})
    defender = _make_defender(attacker)
    _step(defender, attacker)
    assert attacker.perturb_calls == 1, (
        "perturb() ran %d times for one env step; the attack decision must be made "
        "exactly once per step" % attacker.perturb_calls
    )


# ------------------------------------------------------------- 2. ctype is honoured
@pytest.mark.parametrize("ctype", ["blank", "conflict", "target", "speed",
                                   "direction", "malfunction"])
def test_applied_corruption_is_the_chosen_corruption(ctype):
    attacker = CountingAttacker({1: ctype})
    defender = _make_defender(attacker)
    _step(defender, attacker)
    assert attacker.perturb_vector_calls, "the chosen attack was never applied"
    assert set(attacker.perturb_vector_calls) == {ctype}, (
        "attacker chose %r but %r was applied"
        % (ctype, set(attacker.perturb_vector_calls))
    )


def test_distinct_ctypes_produce_distinct_applied_perturbations():
    """A discarded ctype makes every corruption type look identical downstream."""
    applied = {}
    for ctype in ("blank", "conflict", "speed"):
        attacker = CountingAttacker({0: ctype})
        defender = _make_defender(attacker)
        applied[ctype] = _step(defender, attacker)[4]
    assert not np.array_equal(applied["blank"], applied["conflict"])
    assert not np.array_equal(applied["conflict"], applied["speed"])
    assert not np.array_equal(applied["blank"], applied["speed"])


# ------------------------------------------------------------------- 3. breadth
def _per_train(applied):
    width = len(applied) // N_TRAINS
    return [applied[i * width:(i + 1) * width] for i in range(N_TRAINS)]


def test_only_the_chosen_train_is_attacked():
    attacker = CountingAttacker({1: "blank"})
    defender = _make_defender(attacker)
    per_train = _per_train(_step(defender, attacker)[4])
    assert np.any(per_train[1] != 0.0), "the chosen train was not attacked"
    assert np.all(per_train[0] == 0.0), "train 0 was attacked but was not chosen"
    assert np.all(per_train[2] == 0.0), "train 2 was attacked but was not chosen"


def test_two_chosen_trains_are_both_attacked():
    attacker = CountingAttacker({0: "blank", 2: "blank"})
    defender = _make_defender(attacker)
    per_train = _per_train(_step(defender, attacker)[4])
    assert np.any(per_train[0] != 0.0) and np.any(per_train[2] != 0.0)
    assert np.all(per_train[1] == 0.0)


def test_an_empty_published_decision_attacks_nothing():
    """
    A random attacker whose blindness roll came up empty corrupted no train. Treating
    that as "no information" and corrupting all three would hand the baseline a breadth
    no targeted attacker gets, which is how a random baseline out-scored a learned one.
    """
    attacker = CountingAttacker({})
    defender = _make_defender(attacker)
    applied = _step(defender, attacker)[4]
    assert np.all(applied == 0.0), "an empty decision was applied as an attack"
    assert attacker.perturb_vector_calls == []


def test_attacker_that_publishes_nothing_falls_back_to_every_train():
    """Documented fallback for a legacy attacker: breadth is wrong, silence is worse."""

    class LegacyAttacker(CountingAttacker):
        """Pre-contract attacker: never publishes, exposes no handle."""

        def perturb(self, obs):
            self.perturb_calls += 1
            return obs

    attacker = LegacyAttacker({})
    attacker.last_targets = None
    defender = _make_defender(attacker)
    per_train = _per_train(_step(defender, attacker)[4])
    for part in per_train:
        assert np.any(part != 0.0)
    assert set(attacker.perturb_vector_calls) == {DEFAULT_VECTOR_CTYPE}


def test_legacy_single_target_attacker_is_still_honoured():
    """An attacker that sets only last_handle/last_ctype must not get blanket breadth."""

    class LegacySingleTarget(CountingAttacker):
        def perturb(self, obs):
            self.perturb_calls += 1
            self.last_handle = 2
            self.last_ctype = "conflict"
            return obs

    attacker = LegacySingleTarget({})
    attacker.last_targets = None
    defender = _make_defender(attacker)
    per_train = _per_train(_step(defender, attacker)[4])
    assert np.any(per_train[2] != 0.0)
    assert np.all(per_train[0] == 0.0) and np.all(per_train[1] == 0.0)
    assert set(attacker.perturb_vector_calls) == {"conflict"}


# -------------------------------------------------------- 4. recorded == applied
def test_recorded_perturbation_equals_the_observation_delta():
    attacker = CountingAttacker({0: "speed"})
    defender = _make_defender(attacker)
    clean = defender._maze_obs[OBS_KEY].astype(np.float64).copy()
    applied = _step(defender, attacker)[4]
    width = len(applied) // N_TRAINS
    np.testing.assert_allclose(applied[:width], -clean - clean)


def test_recorded_perturbation_length_is_stable_across_steps():
    attacker = CountingAttacker({0: "blank"})
    defender = _make_defender(attacker)
    lengths = {len(_step(defender, attacker)[4]) for _ in range(4)}
    assert len(lengths) == 1, "perturbation length varied across steps: %r" % lengths


# ------------------------------------------- 5. an unreachable attack is observable
def test_unreached_policy_yields_an_all_zero_applied_perturbation():
    """state_inject never touches the policy observation -> zero, i.e. not applicable."""
    attacker = CountingAttacker({0: "malfunction"})
    defender = _make_defender(attacker, attack_mode="state_inject")
    applied = _step(defender, attacker)[4]
    assert np.all(applied == 0.0)
    assert attacker.perturb_vector_calls == []


# ----------------------------------------- application must stay side-effect free
def test_perturb_vector_is_deterministic_and_stateless():
    attacker = CountingAttacker({0: "speed"})
    vec = np.arange(1.0, 6.0)
    first = attacker.perturb_vector(vec.copy(), "speed")
    second = attacker.perturb_vector(vec.copy(), "speed")
    np.testing.assert_array_equal(first, second)


def test_unknown_corruption_type_is_rejected():
    """A silent no-op here scores the agent as robust against nothing."""
    from perturbation_agents.utils import corrupt_vector

    with pytest.raises(ValueError):
        corrupt_vector(np.arange(3.0), "not-a-corruption")
    with pytest.raises(ValueError):
        corrupt_vector(np.arange(3.0), None)
