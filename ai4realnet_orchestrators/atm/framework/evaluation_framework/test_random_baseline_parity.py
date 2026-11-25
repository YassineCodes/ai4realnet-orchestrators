"""
The ATM random baseline must be the power-grid random baseline.

Every learned attacker is reported as a multiple of this control, so the control has to
be the shipped baseline rather than a local invention or the multiples mean nothing. These
tests pin the four parameters and the three structural properties carried over from
power_grid/framework/perturbation_agents/random_perturb_agent.py:

    sparse      one scalar per new perturbation, not every feature
    occasional  a new perturbation is started on ~prob_perturb of steps, not every step
    persistent  it lasts round(exponential(mean_perturb_duration_steps)) steps
    lossy       with probability p_dropout the factor is exactly 0.0, a lost reading

Runs without bluesky_gym, gymnasium or stable-baselines3 installed.
"""
import os
import sys

import numpy as np
import pytest

FRAMEWORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FRAMEWORK not in sys.path:
    sys.path.insert(0, FRAMEWORK)

from perturbation_agents import random_perturb_agent as rpa_mod  # noqa: E402
from perturbation_agents.random_perturb_agent import RandomPerturbationAgent  # noqa: E402


def _obs(n_keys=12, width=5):
    """A stand-in for the AI4REALNET v1 observation: 12 keys, mostly 5 wide."""
    return {f"feature_{i}": np.full(width, 10.0, dtype=np.float64) for i in range(n_keys)}


def _changed(a, b):
    """Indices, as (key, position), where two observations differ."""
    out = []
    for k in a:
        d = np.asarray(b[k]).ravel() - np.asarray(a[k]).ravel()
        out += [(k, int(i)) for i in np.flatnonzero(d)]
    return out


# ---------------------------------------------------------------------------
# the parameters are the power-grid ones
# ---------------------------------------------------------------------------

def test_parameters_match_the_power_grid_baseline():
    assert rpa_mod.PROB_PERTURB == 0.4
    assert rpa_mod.SIGMA == 0.15
    assert rpa_mod.P_DROPOUT == 0.2
    assert rpa_mod.MEAN_PERTURB_DURATION_STEPS == 6.0

    a = RandomPerturbationAgent(xi=None, seed=0)
    assert a.prob_perturb == rpa_mod.PROB_PERTURB
    assert a.sigma == rpa_mod.SIGMA
    assert a.p_dropout == rpa_mod.P_DROPOUT
    assert a.mean_perturb_duration_steps == rpa_mod.MEAN_PERTURB_DURATION_STEPS


# ---------------------------------------------------------------------------
# sparse, occasional, persistent, lossy
# ---------------------------------------------------------------------------

def test_a_new_perturbation_touches_one_scalar_not_the_whole_observation():
    """
    The previous implementation drew noise on every feature at every step, which is what
    saturated KPI-VF-073 and KPI-FF-070 at 1.0 for every attacker.
    """
    agent = RandomPerturbationAgent(xi=None, seed=3)
    obs = _obs()
    out = agent.perturb(obs)
    touched = _changed(obs, out)
    assert len(touched) <= 1, f"a single step corrupted {len(touched)} scalars"


def test_most_steps_start_no_new_perturbation():
    """prob_perturb=0.4, so over many steps clearly fewer than all steps start one."""
    agent = RandomPerturbationAgent(xi=None, seed=5)
    for _ in range(400):
        agent.perturb(_obs())
    started = agent.get_stats()["action_counts"]["new_perturbation"]
    assert 0.25 < started / 400 < 0.55, f"started on {started}/400 steps"


def test_a_perturbation_persists_across_steps():
    """
    Persistence is the property that makes this a sensor-failure model rather than
    per-step noise: the same scalar stays wrong for several steps.
    """
    agent = RandomPerturbationAgent(xi=None, seed=11)
    seen = []
    for _ in range(60):
        obs = _obs()
        out = agent.perturb(obs)
        seen.append(_changed(obs, out))
    # At least one step must have carried more than one active perturbation, which can
    # only happen if an earlier one was still alive when a new one started.
    assert max(len(s) for s in seen) > 1, "no perturbation ever outlived its own step"


def test_a_lost_reading_zeroes_the_scalar():
    """p_dropout fires a factor of exactly 0.0 -- the lost-reading case."""
    agent = RandomPerturbationAgent(xi=None, seed=1, prob_perturb=1.0, p_dropout=1.0)
    obs = _obs()
    out = agent.perturb(obs)
    touched = _changed(obs, out)
    assert len(touched) == 1
    key, idx = touched[0]
    assert np.asarray(out[key]).ravel()[idx] == 0.0


def test_the_factor_is_multiplicative_and_log_normal_in_scale():
    """Without dropout the scalar is scaled, so it stays the same sign and order."""
    agent = RandomPerturbationAgent(xi=None, seed=7, prob_perturb=1.0, p_dropout=0.0)
    ratios = []
    for _ in range(200):
        obs = _obs()
        out = agent.perturb(obs)
        for key, idx in _changed(obs, out):
            ratios.append(np.asarray(out[key]).ravel()[idx] / 10.0)
    assert ratios, "nothing was perturbed"
    assert all(r > 0 for r in ratios), "a multiplicative log-normal factor is positive"
    # log of a lognormal(0, 0.15) sample: mean ~0, and well inside +-1 for any sane draw.
    assert abs(float(np.mean(np.log(ratios)))) < 0.2


# ---------------------------------------------------------------------------
# the budget is a declared restriction on top, not a different adversary
# ---------------------------------------------------------------------------

def test_the_budget_bounds_every_feature():
    agent = RandomPerturbationAgent(xi=0.1, seed=2, prob_perturb=1.0, p_dropout=1.0)
    for _ in range(50):
        obs = _obs()
        out = agent.perturb(obs)
        for k in obs:
            delta = np.abs(np.asarray(out[k]) - np.asarray(obs[k]))
            assert np.all(delta <= 0.1 + 1e-12), f"{k} moved by {delta.max()} > xi"


def test_the_realised_maximum_is_what_the_policy_received():
    """
    get_stats() is what the runner checks against the declared threat model, so it has
    to report the clipped delta, not the one that was proposed before clipping.
    """
    agent = RandomPerturbationAgent(xi=0.1, seed=4, prob_perturb=1.0, p_dropout=1.0)
    for _ in range(30):
        agent.perturb(_obs())
    realised = agent.get_stats()["max_perturb_per_key"]
    assert realised, "no perturbation was recorded"
    assert max(realised.values()) <= 0.1 + 1e-12


def test_without_a_budget_the_power_grid_behaviour_is_reproduced():
    """xi=None must leave the multiplicative factor untouched, as power grid does."""
    agent = RandomPerturbationAgent(xi=None, seed=9, prob_perturb=1.0, p_dropout=1.0)
    obs = _obs()
    out = agent.perturb(obs)
    key, idx = _changed(obs, out)[0]
    # A dropout on a feature of magnitude 10 is a delta of 10, far outside any budget.
    assert abs(np.asarray(out[key]).ravel()[idx] - 10.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# reproducibility and the observation contract
# ---------------------------------------------------------------------------

def test_same_seed_same_attack():
    """The paired design needs the control to repeat exactly for a given seed."""
    def run():
        a = RandomPerturbationAgent(xi=0.1, seed=42)
        return [_changed(o, a.perturb(o)) for o in (_obs() for _ in range(40))]
    assert run() == run()


def test_reset_clears_persistence_so_episodes_are_independent():
    agent = RandomPerturbationAgent(xi=None, seed=13, prob_perturb=1.0)
    for _ in range(20):
        agent.perturb(_obs())
    assert agent.current_perturbations, "expected active perturbations before reset"
    agent.reset()
    assert agent.current_perturbations == []


def test_no_feature_name_is_hardcoded():
    """
    The v0 task has 7 observation keys and the v1 task 12. The agent must perturb
    whatever the environment declares, so that switching task needs no change here.
    """
    for n_keys in (7, 12, 3):
        agent = RandomPerturbationAgent(xi=0.1, seed=6, prob_perturb=1.0)
        obs = {f"k{i}": np.ones(4) for i in range(n_keys)}
        out = agent.perturb(obs)
        assert set(out.keys()) == set(obs.keys())


def test_shapes_are_preserved():
    agent = RandomPerturbationAgent(xi=0.1, seed=8, prob_perturb=1.0)
    obs = {"a": np.ones(1), "b": np.ones(5), "c": np.ones(10)}
    out = agent.perturb(obs)
    for k in obs:
        assert np.asarray(out[k]).shape == np.asarray(obs[k]).shape


def test_the_clean_observation_is_not_modified_in_place():
    agent = RandomPerturbationAgent(xi=None, seed=10, prob_perturb=1.0, p_dropout=1.0)
    obs = _obs()
    before = {k: np.array(v, copy=True) for k, v in obs.items()}
    agent.perturb(obs)
    for k in obs:
        assert np.array_equal(obs[k], before[k]), "perturb() mutated its input"
