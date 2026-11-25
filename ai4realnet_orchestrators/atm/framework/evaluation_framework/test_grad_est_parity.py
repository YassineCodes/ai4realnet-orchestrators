"""
The ATM gradient-estimation attacker must actually estimate a gradient.

It is the power-grid attacker's algorithm -- central-difference gradient estimation plus
projected gradient descent -- with power grid's hyper-parameters (fd_delta 0.01,
step_size 0.05, n_iter 5) and the objective adapted to a continuous action space:
maximise ||pi(obs + delta) - pi(obs)||.

What it replaced was random search over 8 budgeted candidates, which shared only the
name. Measured against a trained v1 SAC policy at the same xi = 0.1 budget, the ported
attacker moves the action 4.33x further per step, using 6 batched policy queries.

Runs with a stub defender; no bluesky_gym, gymnasium or stable-baselines3 needed.
"""
import os
import sys

import numpy as np
import pytest

FRAMEWORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FRAMEWORK not in sys.path:
    sys.path.insert(0, FRAMEWORK)

from perturbation_agents import grad_est_perturb_agent as ge_mod  # noqa: E402
from perturbation_agents.grad_est_perturb_agent import GradEstPerturbationAgent  # noqa: E402


def _obs():
    """Three keys, 6 scalars: small enough to reason about, shaped like a Dict obs."""
    return {"a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0, 5.0]),
            "c": np.array([6.0])}


class LinearDefender:
    """
    Action is a fixed linear function of the flattened observation.

    With a linear policy the attack objective has a constant gradient, so the exact
    optimum inside the L-infinity ball is known: push every feature to the budget in the
    direction of its own weight. That makes the PGD result checkable rather than merely
    plausible.
    """

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=np.float64)
        self.calls = 0
        self.batch_calls = 0

    @staticmethod
    def _flat(o):
        return np.concatenate([np.asarray(o[k], dtype=np.float64).ravel()
                               for k in sorted(o)])

    def act(self, o):
        self.calls += 1
        return np.array([float(self.w @ self._flat(o)), 0.0])

    def act_batch(self, os_):
        self.batch_calls += 1
        return np.stack([self.act(o) for o in os_])


class ConstantDefender:
    """Ignores the observation entirely — the straight-line reference's behaviour."""

    def __init__(self):
        self.calls = 0

    def act(self, o):
        self.calls += 1
        return np.zeros(2)

    def act_batch(self, os_):
        return np.zeros((len(os_), 2))


def _delta(obs, out):
    return np.concatenate([np.asarray(out[k] - obs[k]).ravel() for k in sorted(obs)])


# ---------------------------------------------------------------------------
# the parameters are the power-grid ones
# ---------------------------------------------------------------------------

def test_parameters_match_the_power_grid_attacker():
    assert ge_mod.FD_DELTA == 0.01
    assert ge_mod.STEP_SIZE == 0.05
    assert ge_mod.N_ITER == 5
    a = GradEstPerturbationAgent(agent=None)
    assert (a.fd_delta, a.step_size, a.n_iter) == (0.01, 0.05, 5)


# ---------------------------------------------------------------------------
# it estimates a gradient and follows it
# ---------------------------------------------------------------------------

def test_pgd_finds_the_known_optimum_against_a_linear_policy():
    """
    For the linear policy pi(x) = (w.x, 0) with w.x > 0, the clean action points along
    +e0, so the objective J = -(w.x + w.delta) is maximised by driving w.delta as
    negative as the budget allows: delta_i = -xi * sign(w_i). Five steps of 0.05 reach
    the 0.1 budget, so PGD must land exactly on that corner.
    """
    w = np.array([1.0, -2.0, 0.5, -0.25, 3.0, -1.0])
    d = LinearDefender(w)
    agent = GradEstPerturbationAgent(agent=d, xi=0.1, seed=0)
    obs = _obs()
    assert float(w @ d._flat(obs)) > 0, "this test assumes a positive clean action"
    got = _delta(obs, agent.perturb(obs))
    assert np.allclose(got, -0.1 * np.sign(w), atol=1e-9), got
    assert agent.get_stats()["action_counts"]["zero_gradient_fallback"] == 0


def test_the_step_drives_the_action_away_from_what_the_agent_wanted():
    """
    The attack reduces the projection of the action onto the clean action. On a linear
    policy that means the action's magnitude falls towards, and past, zero.
    """
    w = np.array([1.0, -2.0, 0.5, -0.25, 3.0, -1.0])
    d = LinearDefender(w)
    obs = _obs()
    clean = d.act(obs)[0]
    out = GradEstPerturbationAgent(agent=d, xi=0.1, seed=0).perturb(obs)
    attacked = d.act(out)[0]
    assert attacked < clean, f"action moved from {clean:.4f} to {attacked:.4f}"


def test_the_symmetric_objective_would_have_produced_no_gradient():
    """
    Guards the choice of objective. ||pi(obs+delta) - pi(obs)|| is symmetric about
    delta = 0, so its central difference there is exactly zero for every feature and PGD
    could never start. This asserts that property directly, so nobody re-introduces it.
    """
    w = np.array([1.0, -2.0, 0.5, -0.25, 3.0, -1.0])
    d = LinearDefender(w)
    obs = _obs()
    clean = d.act(obs)
    h = ge_mod.FD_DELTA
    flat = d._flat(obs)
    for i in range(len(w)):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += h
        minus[i] -= h
        def act_of(v):
            o = {"a": v[0:2], "b": v[2:5], "c": v[5:6]}
            return d.act(o)
        j_plus = np.linalg.norm(act_of(plus) - clean)
        j_minus = np.linalg.norm(act_of(minus) - clean)
        assert (j_plus - j_minus) == pytest.approx(0.0, abs=1e-12)


def test_the_finite_differences_are_batched_one_call_per_iteration():
    """
    Estimating the gradient over n scalars needs 2n probes. They must go out as one
    batched call per PGD iteration; 2n separate queries per iteration would make the
    attacker unusably slow on a 48-scalar observation.
    """
    d = LinearDefender(np.ones(6))
    agent = GradEstPerturbationAgent(agent=d, xi=0.1, seed=0)
    agent.perturb(_obs())
    assert d.batch_calls == ge_mod.N_ITER
    # one clean query plus one batched query per iteration
    assert agent.get_stats()["action_counts"]["queries"] == 1 + ge_mod.N_ITER


def test_it_beats_random_search_at_the_same_budget():
    """
    The attacker it replaced sampled 8 budgeted candidates and kept the best. Judged on
    the same objective -- how far the action is driven away from what the agent wanted --
    following the gradient must do better at the same budget.
    """
    w = np.array([1.0, -2.0, 0.5, -0.25, 3.0, -1.0])
    obs = _obs()

    ge = GradEstPerturbationAgent(agent=LinearDefender(w), xi=0.1, seed=0)
    ge_reduction = -float(w @ _delta(obs, ge.perturb(obs)))

    rng = np.random.default_rng(0)
    best_random = max(-float(w @ rng.uniform(-0.1, 0.1, 6)) for _ in range(8))

    assert ge_reduction > best_random, (
        f"gradient {ge_reduction:.4f} did not beat random search {best_random:.4f}"
    )


# ---------------------------------------------------------------------------
# the budget, and the degenerate cases
# ---------------------------------------------------------------------------

def test_the_budget_bounds_every_feature():
    for xi in (0.01, 0.1, 0.5):
        d = LinearDefender(np.array([5.0, -5.0, 5.0, -5.0, 5.0, -5.0]))
        obs = _obs()
        out = GradEstPerturbationAgent(agent=d, xi=xi, seed=1).perturb(obs)
        assert np.all(np.abs(_delta(obs, out)) <= xi + 1e-12)


def test_the_realised_maximum_is_reported_for_the_budget_check():
    d = LinearDefender(np.ones(6))
    agent = GradEstPerturbationAgent(agent=d, xi=0.1, seed=1)
    agent.perturb(_obs())
    realised = agent.get_stats()["max_perturb_per_key"]
    assert set(realised) == {"a", "b", "c"}
    assert max(realised.values()) <= 0.1 + 1e-12


def test_a_policy_that_ignores_its_observation_still_gets_a_real_perturbation():
    """
    A constant policy gives a zero gradient. Proposing nothing would record an all-zero
    perturbation, which the pipeline reads as "the attack never reached the policy" --
    the wrong reading for an agent that is structurally immune. A random budgeted
    perturbation is applied instead, so the unchanged action is evidence about the agent.
    """
    agent = GradEstPerturbationAgent(agent=ConstantDefender(), xi=0.1, seed=2)
    obs = _obs()
    out = agent.perturb(obs)
    assert np.any(_delta(obs, out) != 0.0)
    assert agent.get_stats()["action_counts"]["zero_gradient_fallback"] == 1


def test_no_defender_falls_back_to_a_budgeted_random_perturbation():
    agent = GradEstPerturbationAgent(agent=None, xi=0.1, seed=2)
    obs = _obs()
    out = agent.perturb(obs)
    assert np.all(np.abs(_delta(obs, out)) <= 0.1 + 1e-12)
    assert np.any(_delta(obs, out) != 0.0)


def test_a_defender_without_a_batched_path_degrades_instead_of_issuing_2n_queries():
    class NoBatch(LinearDefender):
        act_batch = None

    d = NoBatch(np.ones(6))
    agent = GradEstPerturbationAgent(agent=d, xi=0.1, seed=2)
    obs = _obs()
    out = agent.perturb(obs)
    assert np.any(_delta(obs, out) != 0.0)          # still attacks
    assert d.calls == 1                              # only the clean query


def test_a_failing_defender_does_not_abort_the_episode():
    class Broken:
        def act(self, o):
            raise RuntimeError("policy exploded")

        def act_batch(self, os_):
            raise RuntimeError("policy exploded")

    agent = GradEstPerturbationAgent(agent=Broken(), xi=0.1, seed=2)
    obs = _obs()
    out = agent.perturb(obs)
    assert np.all(np.abs(_delta(obs, out)) <= 0.1 + 1e-12)


# ---------------------------------------------------------------------------
# the observation contract
# ---------------------------------------------------------------------------

def test_shapes_and_keys_are_preserved_and_the_input_is_not_mutated():
    d = LinearDefender(np.ones(6))
    obs = _obs()
    before = {k: np.array(v, copy=True) for k, v in obs.items()}
    out = GradEstPerturbationAgent(agent=d, xi=0.1, seed=3).perturb(obs)
    assert sorted(out) == sorted(obs)
    for k in obs:
        assert np.asarray(out[k]).shape == np.asarray(obs[k]).shape
        assert np.array_equal(obs[k], before[k]), "perturb() mutated its input"


def test_no_feature_name_is_hardcoded():
    """A task with more feature blocks must be attacked without a change here."""
    for n_keys in (3, 7, 12):
        obs = {f"k{i}": np.ones(4) for i in range(n_keys)}
        d = LinearDefender(np.ones(4 * n_keys))
        out = GradEstPerturbationAgent(agent=d, xi=0.1, seed=4).perturb(obs)
        assert sorted(out) == sorted(obs)
        assert np.all(np.abs(_delta(obs, out)) <= 0.1 + 1e-12)


def test_same_seed_same_attack():
    w = np.array([1.0, -2.0, 0.5, -0.25, 3.0, -1.0])
    def run():
        return _delta(_obs(), GradEstPerturbationAgent(
            agent=LinearDefender(w), xi=0.1, seed=42).perturb(_obs()))
    assert np.array_equal(run(), run())
