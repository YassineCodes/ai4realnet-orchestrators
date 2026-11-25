"""
A reported KPI has to be re-derivable, so the attackers' randomness has to be pinned.

Every attacker already took a `seed` and used it for a numpy RandomState, but nothing
seeded torch -- and the learned attackers build torch networks, so their weight
initialisation came from torch's global, unseeded generator. Two consecutive runs of the
same evaluation, same code and same environment seeds, differed on nine of the ten KPIs
(KPI-FF-070 by 24%).

These tests pin the helper that fixes it. The end-to-end check is in the README: with
seeding in place, two consecutive 4-episode runs are bit-identical on all ten KPIs.
"""
import os
import sys

import numpy as np
import pytest

FRAMEWORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FRAMEWORK not in sys.path:
    sys.path.insert(0, FRAMEWORK)

from perturbation_agents.determinism import seed_attacker_rngs  # noqa: E402

torch = pytest.importorskip("torch", reason="the learned attackers' RNG is torch's")


def test_torch_sampling_repeats_after_the_same_seed():
    seed_attacker_rngs(7)
    a = torch.randn(64)
    seed_attacker_rngs(7)
    b = torch.randn(64)
    assert torch.equal(a, b), "torch was not seeded"


def test_network_initialisation_repeats_after_the_same_seed():
    """
    This is the case that actually broke the evaluation: a learned attacker's network
    weights, drawn at construction time from torch's global generator.
    """
    def build():
        seed_attacker_rngs(3)
        net = torch.nn.Sequential(
            torch.nn.Linear(144, 64), torch.nn.Tanh(), torch.nn.Linear(64, 63)
        )
        return torch.cat([p.detach().flatten() for p in net.parameters()])

    assert torch.equal(build(), build())


def test_different_seeds_give_different_draws():
    """Guards against a helper that pins everything to one constant."""
    seed_attacker_rngs(1)
    a = torch.randn(32)
    seed_attacker_rngs(2)
    b = torch.randn(32)
    assert not torch.equal(a, b)


def test_numpy_global_generator_is_seeded_too():
    """Some attacker code draws from np.random directly rather than its own RandomState."""
    seed_attacker_rngs(11)
    a = np.random.rand(16)
    seed_attacker_rngs(11)
    b = np.random.rand(16)
    assert np.array_equal(a, b)


def test_python_random_is_seeded_too():
    import random
    seed_attacker_rngs(13)
    a = [random.random() for _ in range(8)]
    seed_attacker_rngs(13)
    b = [random.random() for _ in range(8)]
    assert a == b


def test_seeding_is_independent_of_how_many_attackers_came_before():
    """
    The runner seeds immediately before constructing each attacker, so an attacker's
    draws must depend only on its own seed -- not on how many were built first, and not
    on whether another attacker consumed randomness in between.
    """
    seed_attacker_rngs(5)
    expected = torch.randn(16)

    seed_attacker_rngs(99)
    torch.randn(1000)            # another attacker burning randomness
    seed_attacker_rngs(5)
    assert torch.equal(torch.randn(16), expected)


def test_a_float_seed_is_accepted_as_an_integer():
    seed_attacker_rngs(4)
    a = torch.randn(8)
    seed_attacker_rngs(4.0)
    assert torch.equal(torch.randn(8), a)
