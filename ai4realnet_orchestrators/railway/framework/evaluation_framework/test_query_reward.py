"""
Tests for the query-based attacker reward.

The four learned attackers used to be rewarded by disruption_reward(obs_before,
obs_after): a comparison of two observations that never consults the policy. It paid for
making three hand-picked fields look worse whether or not the agent's decision changed,
and in this domain the policy does not even read those observations, it reads the maze
observation. Measured consequence: three of the four scored significantly WORSE than
random noise, and the share of an attacker's choices in the class the surrogate paid for
but the policy ignores predicted effectiveness at Pearson -0.980.

The reward is now the drop in the probability the policy assigns to the action it
preferred, which is power grid's attacker objective. This is a declared adaptation of the
threat model from black-box to query-based.

These tests pin the three things that can silently break it:
  1. an ineffective corruption must score 0, not a positive surrogate value;
  2. a defender that cannot answer must fall back, not be read as "no disruption";
  3. every learned attacker must actually route through it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perturbation_agents.utils import (  # noqa: E402
    disruption_reward, query_disruption_reward)


class _Defender:
    """Returns the probability vectors it is told to, and counts its queries."""

    def __init__(self, clean, corrupted):
        self._clean = clean
        self._corrupted = corrupted
        self.queries = 0

    def action_probs(self, obs_dict):
        self.queries += 1
        return None if self._clean is None else np.asarray(self._clean, dtype=float)

    def action_probs_with_corruption(self, obs_dict, ctype, severity):
        self.queries += 1
        return None if self._corrupted is None else np.asarray(self._corrupted, dtype=float)


def test_reward_is_the_lost_confidence_in_the_preferred_action():
    d = _Defender([0.7, 0.2, 0.1], [0.3, 0.5, 0.2])
    assert query_disruption_reward(d, {}, "speed", 1.0) == pytest.approx(0.4)


def test_a_corruption_the_policy_ignores_scores_zero():
    # The whole point. The old surrogate paid for this case; it must now score nothing.
    d = _Defender([0.7, 0.2, 0.1], [0.7, 0.2, 0.1])
    assert query_disruption_reward(d, {}, "speed", 1.0) == pytest.approx(0.0)


def test_a_corruption_that_makes_the_policy_more_certain_scores_negative():
    # Signed on purpose: an attack that reinforces the decision is worse than useless,
    # and a learned attacker has to be able to see that.
    d = _Defender([0.5, 0.3, 0.2], [0.9, 0.05, 0.05])
    assert query_disruption_reward(d, {}, "speed", 1.0) < 0.0


def test_the_preferred_action_is_taken_from_the_clean_distribution():
    # arg max clean is index 1; the reward must track THAT action, not the new arg max.
    d = _Defender([0.2, 0.6, 0.2], [0.5, 0.1, 0.4])
    assert query_disruption_reward(d, {}, "speed", 1.0) == pytest.approx(0.5)


def test_a_defender_that_cannot_answer_returns_none_rather_than_zero():
    # None means "no signal, fall back". Returning 0.0 here would tell the attacker the
    # corruption was useless, which is a different and wrong claim.
    assert query_disruption_reward(_Defender(None, [0.3]), {}, "speed", 1.0) is None
    assert query_disruption_reward(_Defender([0.7, 0.3], None), {}, "speed", 1.0) is None


def test_a_defender_without_the_query_api_returns_none():
    class Old:
        def act(self, obs_dict):
            return {}
    assert query_disruption_reward(Old(), {}, "speed", 1.0) is None
    assert query_disruption_reward(None, {}, "speed", 1.0) is None


def test_mismatched_distribution_lengths_return_none():
    d = _Defender([0.7, 0.2, 0.1], [0.5, 0.5])
    assert query_disruption_reward(d, {}, "speed", 1.0) is None


def test_it_queries_the_defender_twice_per_call_unless_given_the_clean_probs():
    d = _Defender([0.7, 0.3], [0.4, 0.6])
    query_disruption_reward(d, {}, "speed", 1.0)
    assert d.queries == 2
    d2 = _Defender([0.7, 0.3], [0.4, 0.6])
    query_disruption_reward(d2, {}, "speed", 1.0, clean_probs=np.array([0.7, 0.3]))
    assert d2.queries == 1


def test_the_old_surrogate_still_exists_as_the_declared_fallback():
    # Kept deliberately: a defender that cannot be queried must still produce a number.
    assert callable(disruption_reward)


@pytest.mark.parametrize("make", [
    pytest.param(lambda d: _rl(d), id="RLPerturb"),
    pytest.param(lambda d: _pir(d), id="LambdaPIR"),
    pytest.param(lambda d: _ppo(d), id="PPO"),
    pytest.param(lambda d: _sac(d), id="SAC"),
])
def test_every_learned_attacker_routes_through_the_query_reward(make):
    d = _Defender([0.9, 0.1], [0.1, 0.9])
    agent = make(d)
    assert agent.defender is d
    assert agent._attack_reward({}, {}, "speed", 1.0) == pytest.approx(0.8)


def _rl(d):
    from perturbation_agents.rl_perturb_agent import RLPerturbationAgent
    return RLPerturbationAgent(n_agents=3, defender=d)


def _pir(d):
    from perturbation_agents.lambda_pir_perturbation_agent import LambdaPIRPerturbationAgent
    return LambdaPIRPerturbationAgent(n_agents=3, defender=d)


def _ppo(d):
    from attack_models.PPOAttacker import PPOAttacker
    return PPOAttacker(n_agents=3, defender=d)


def _sac(d):
    from attack_models.SACAttacker import SACAttacker
    return SACAttacker(n_agents=3, defender=d)


@pytest.mark.parametrize("make", [_rl, _pir, _ppo, _sac])
def test_without_a_defender_each_attacker_falls_back_instead_of_failing(make):
    agent = make(None)
    assert agent.defender is None
    # The surrogate on two empty observations is 0.0; the point is that it returns a
    # number rather than raising or reporting the query path's None.
    assert agent._attack_reward({}, {}, "speed", 1.0) == pytest.approx(0.0)
