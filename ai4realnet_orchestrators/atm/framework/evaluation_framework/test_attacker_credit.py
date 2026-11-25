"""
Tests for what the online perturbation agent learns from.

RLPerturbationAgent is the single class behind four of this domain's attackers: PPO,
SAC_5, SAC_10 and RLPerturb differ only in epsilon, n_sample, lr and the budget. So what
it credits decides four of the suite's seven rows.

It keeps a value per (feature, index): how much perturbing there moves the defender. The
value used to be credited as

    ||act(obs_t) - act(obs_{t-1})||        attached to step t-1's slots

which is the change in the defender's action between consecutive steps, not the effect of
a perturbation. The defender's action moves anyway as the aircraft fly, so the signal
mixed "my noise moved the policy" with "the world moved on", and attached the result to
the previous step's choice. Measured on a defender whose action also drifts by 1.0 per
step while the attack is worth 0.1, the learned value read 0.953 instead of 0.096: ten
times the real effect, almost all of it the world.

It is now the same-step counterfactual, ||act(perturbed) - act(clean)||, which is what
this domain's LambdaPIR already used.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perturbation_agents.rl_perturb_agent import RLPerturbationAgent  # noqa: E402

XI = 0.1


class OneFeatureMatters:
    """Action depends only on ('a', 0). Every other slot is a decoy."""

    def __init__(self, key="a", pos=0):
        self.key, self.pos = key, pos
        self.queries = 0

    def act(self, obs):
        self.queries += 1
        return np.array([float(obs[self.key].reshape(-1)[self.pos]), 0.0])


class DriftingWorld(OneFeatureMatters):
    """
    As above, but the defender's action also drifts from step to step on its own.

    The drift advances once per STEP, driven by the test loop, not once per act() call.
    That matters: the counterfactual is only meaningful because a defender query is
    side-effect-free, which both defenders in this domain are (SB3Defender.act is a
    model.predict, StraightLineDefender.act returns zeros). A query that advanced the
    world would put a drift step *inside* the counterfactual and break it. This fixture
    drifts per step so that a regression of that kind fails here rather than silently
    inflating every learned value.
    """

    def __init__(self, drift=1.0, **kw):
        super().__init__(**kw)
        self.drift, self.t = drift, 0.0

    def step(self):
        self.t += self.drift

    def act(self, obs):
        self.queries += 1
        return np.array([float(obs[self.key].reshape(-1)[self.pos]) + self.t, 0.0])


def _obs():
    return {"a": np.zeros(2), "b": np.zeros(2)}


def _run(defender, steps=30, **kw):
    agent = RLPerturbationAgent(agent=defender, xi=XI, seed=0, **kw)
    for _ in range(steps):
        if hasattr(defender, "step"):
            defender.step()
        agent.perturb({k: v.copy() for k, v in _obs().items()})
    return agent


def test_the_defender_query_is_side_effect_free():
    """
    The precondition the counterfactual rests on.

    If a defender query mutated anything, the second of the two queries per step would
    see a different world and the measured "effect of the perturbation" would include
    whatever the first query changed. Both defenders in this domain are pure, and this
    pins that: a defender whose act() advanced the environment would corrupt every
    query-based attacker's episodes without failing any other test.
    """
    from defenders import StraightLineDefender

    d = StraightLineDefender()
    obs = _obs()
    first = d.act(obs)
    second = d.act(obs)
    assert np.array_equal(first, second)
    assert all(np.allclose(obs[k], 0.0) for k in obs), "the query mutated the observation"


def test_the_learned_value_is_the_true_causal_effect():
    agent = _run(OneFeatureMatters(), epsilon=0.0, n_sample=1)
    learned = [v for v in agent._values.values() if abs(v) > 1e-9]
    assert learned, "nothing was credited at all"
    # The defender's action moves by exactly the perturbation, so the value must
    # converge on xi and not on some multiple of it.
    assert max(learned) == pytest.approx(XI, rel=0.1)


def test_a_drifting_world_does_not_inflate_the_value():
    # The defect: the old signal read ~10x here, almost all of it the world's drift.
    quiet = _run(OneFeatureMatters(), epsilon=0.0, n_sample=1)
    drifting = _run(DriftingWorld(drift=1.0), epsilon=0.0, n_sample=1)
    q = max(v for v in quiet._values.values())
    d = max(v for v in drifting._values.values())
    assert d == pytest.approx(q, rel=0.05), (d, q)
    assert d < 2 * XI, f"value {d} is inflated by world drift, not caused by the attack"


def test_only_the_feature_that_moves_the_defender_gains_value():
    agent = _run(OneFeatureMatters(key="b", pos=1), steps=120, epsilon=0.3, n_sample=1)
    credited = {k: v for k, v in agent._values.items() if v > 1e-9}
    assert credited, "nothing was credited"
    best = max(credited, key=credited.get)
    assert best == ("b", 1), (best, credited)


def test_a_decoy_feature_is_credited_nothing():
    agent = _run(OneFeatureMatters(key="a", pos=0), steps=120, epsilon=0.3, n_sample=1)
    assert agent._values.get(("b", 0), 0.0) == pytest.approx(0.0, abs=1e-9)
    assert agent._values.get(("b", 1), 0.0) == pytest.approx(0.0, abs=1e-9)


def test_credit_lands_on_the_step_that_earned_it_not_the_one_before():
    # With one slot per step and no exploration, a correctly attributed value is already
    # at the causal slot after a single step. Delayed credit leaves it empty.
    agent = RLPerturbationAgent(agent=OneFeatureMatters(), xi=XI, seed=0,
                                epsilon=0.0, n_sample=1)
    agent.perturb(_obs())
    assert any(abs(v) > 1e-9 for v in agent._values.values()), \
        "nothing credited after the first step, so credit is still deferred"


def test_credit_is_attributed_per_slot_not_shared_across_the_bundle():
    """
    The defect that made fixing the timing alone change nothing.

    n_sample slots are perturbed together. All of them used to be credited with one
    shared scalar, so a slot that moved the policy and the passengers perturbed beside it
    came out with the same value: measured on this fixture, causal 0.0995 against a top
    passenger of 0.0981, a 1.4% margin, which is noise. Values inside a chosen set
    therefore never separated, the incumbent set stayed top-n for ever, and old and new
    timing signals chose identical slots at 120 of 120 steps.

    Each chosen slot is now probed alone, so a passenger earns nothing.
    """
    d = OneFeatureMatters(key="b", pos=1)
    agent = _run(d, steps=60, epsilon=0.3, n_sample=3)
    causal = agent._values.get(("b", 1), 0.0)
    passengers = [v for k, v in agent._values.items() if k != ("b", 1)]
    assert causal > 0.5 * XI, f"the causal slot was not learned at all ({causal})"
    top = max(passengers) if passengers else 0.0
    # A margin, not a strict inequality: shared credit passed a strict ">" by 1.4%.
    assert top < 0.2 * causal, f"passenger {top} is comparable to causal {causal}"


def test_the_probes_are_not_applied_to_the_returned_observation():
    # Probing must not perturb anything extra: exactly n_sample slots may move, each by
    # at most xi, or the attack no longer matches the declared budget.
    d = OneFeatureMatters()
    agent = RLPerturbationAgent(agent=d, xi=XI, seed=0, epsilon=0.0, n_sample=2)
    out = agent.perturb(_obs())
    flat = np.concatenate([out[k].reshape(-1) for k in sorted(out)])
    moved = np.flatnonzero(np.abs(flat) > 1e-12)
    assert len(moved) == 2, f"{len(moved)} slots moved, expected n_sample=2"
    assert np.all(np.abs(flat[moved]) <= XI + 1e-12)


def test_the_query_budget_is_one_plus_n_sample_per_step():
    d = OneFeatureMatters()
    agent = RLPerturbationAgent(agent=d, xi=XI, seed=0, epsilon=0.0, n_sample=3)
    agent.perturb(_obs())
    assert d.queries == 4, d.queries          # one clean + one per chosen slot


def test_it_costs_two_defender_queries_per_step():
    # With n_sample=1 the per-slot budget is one clean query plus one probe.
    d = OneFeatureMatters()
    agent = RLPerturbationAgent(agent=d, xi=XI, seed=0, epsilon=0.0, n_sample=1)
    agent.perturb(_obs())
    assert d.queries == 2, d.queries


def test_without_a_defender_it_perturbs_nothing_and_learns_nothing():
    agent = RLPerturbationAgent(agent=None, xi=XI, seed=0)
    out = agent.perturb(_obs())
    assert all(np.allclose(out[k], 0.0) for k in out)
    assert agent._values == {}


def test_a_defender_that_raises_does_not_take_the_run_down():
    class Broken:
        def act(self, obs):
            raise RuntimeError("policy exploded")

    agent = RLPerturbationAgent(agent=Broken(), xi=XI, seed=0)
    out = agent.perturb(_obs())            # must not raise
    assert set(out) == {"a", "b"}
