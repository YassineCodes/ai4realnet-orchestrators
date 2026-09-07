"""
Tests for KPI-RF-078 (reward per action).

Run with pytest, or directly:
    python test_reward_per_action.py

The three cases called out when the KPI was specified are T1, T2 and T3:
  T1  all steps are actions        => ave_reward_per_action == ave_reward_per_step
  T2  zero actions                 => NaN, never 0 and never a divide-by-zero
  T3  two episodes of very different length => pooled value, not a mean of ratios
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_framework.metrics import metrics, REWARD_PER_ACTION_TARGET_RATIO  # noqa: E402

DO_NOTHING = np.array([0.0, 0.0, 0.0])
ACTION = np.array([1.0, 0.0, 0.0])
OBS_DIM = 3


def _similarity(act1, act2):
    """Stub for similarity_score_fn; irrelevant to these tests."""
    return 1.0


def _build(actions, rewards, base_actions=None, base_rewards=None):
    """
    Build the (perturbed, unperturbed) data dict pair that metrics() consumes.

    Args:
        actions:      list of episodes, each a list of action vectors (perturbed rollout)
        rewards:      list of episodes, each a list of per-step rewards (perturbed rollout)
        base_actions: same for the baseline rollout; defaults to the perturbed one
        base_rewards: same for the baseline rollout; defaults to the perturbed one
    """
    base_actions = actions if base_actions is None else base_actions
    base_rewards = rewards if base_rewards is None else base_rewards

    def observations(eps):
        return [[np.ones(OBS_DIM) for _ in ep] for ep in eps]

    def perturbations(eps):
        return [[np.zeros(OBS_DIM) for _ in ep] for ep in eps]

    perturbed = {
        "observations": observations(actions),
        "perturbations": perturbations(actions),
        "actions": actions,
        "actions_unperturbed": actions,
        "rewards": rewards,
    }
    unperturbed = {
        "observations": observations(base_actions),
        "perturbations": perturbations(base_actions),
        "actions": base_actions,
        "actions_unperturbed": base_actions,
        "rewards": base_rewards,
    }
    return perturbed, unperturbed


def _metrics(actions, rewards, base_actions=None, base_rewards=None, **kwargs):
    perturbed, unperturbed = _build(actions, rewards, base_actions, base_rewards)
    return metrics(perturbed, unperturbed, DO_NOTHING, _similarity, **kwargs)


# ---------------------------------------------------------------------------
# T1 - regression guard: when every step is an action the new metric must
#      collapse onto the existing per-step one.
# ---------------------------------------------------------------------------

def test_all_steps_are_actions_equals_ave_reward_per_step():
    m = _metrics([[ACTION] * 4], [[1.0, 2.0, 3.0, 4.0]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 4
    assert np.isclose(row["ave_reward_per_action"], row["ave_reward_per_step"])
    # and the pre-existing column is untouched
    assert np.isclose(row["ave_reward_per_step"], 2.5)


def test_identical_baseline_gives_unit_ratio():
    m = _metrics([[ACTION] * 4], [[1.0, 2.0, 3.0, 4.0]])
    assert np.isclose(m.metrics_robustness.iloc[0]["reward_per_action_ratio"], 1.0)
    assert np.isclose(m.reward_per_action["reward_per_action_ratio"], 1.0)


# ---------------------------------------------------------------------------
# T2 - zero actions is undefined, not zero.
# ---------------------------------------------------------------------------

def test_zero_actions_yields_nan_not_zero():
    m = _metrics([[DO_NOTHING] * 3], [[1.0, 1.0, 1.0]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 0
    assert np.isnan(row["ave_reward_per_action"])
    assert np.isnan(row["reward_per_action_ratio"])
    # the per-step metric is still perfectly well defined
    assert np.isclose(row["ave_reward_per_step"], 1.0)


# ---------------------------------------------------------------------------
# T3 - pooled aggregation, not a mean of per-episode ratios.
# ---------------------------------------------------------------------------

def test_aggregation_is_pooled_not_mean_of_episodes():
    # ep0: 1 action  earning 10  -> 10.0 per action
    # ep1: 9 actions earning  9  ->  1.0 per action
    # pooled = 19 / 10 = 1.9   ;   mean of per-episode values = 5.5
    m = _metrics([[ACTION], [ACTION] * 9], [[10.0], [1.0] * 9])

    pooled = m.reward_per_action["ave_reward_per_action"]
    mean_of_episodes = m.metrics_robustness["ave_reward_per_action"].mean()

    assert np.isclose(pooled, 1.9)
    assert np.isclose(mean_of_episodes, 5.5)
    assert not np.isclose(pooled, mean_of_episodes)
    assert m.reward_per_action["n_actions"] == 10


# ---------------------------------------------------------------------------
# NaN masking, ratio, thresholds, column order, recovery classification
# ---------------------------------------------------------------------------

def test_nan_rewards_drop_the_same_steps_from_both_sides():
    # 4 action steps, but two carry a NaN reward: both the sum and the count
    # must ignore exactly those two.
    m = _metrics([[ACTION] * 4], [[2.0, np.nan, 2.0, np.nan]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 2
    assert np.isclose(row["ave_reward_per_action"], 2.0)


def test_ratio_against_a_different_baseline_rollout():
    # perturbed: 4 actions earning 4 -> 1.0 per action
    # baseline : 2 actions earning 4 -> 2.0 per action  => ratio 0.5
    m = _metrics(
        [[ACTION] * 4], [[1.0] * 4],
        base_actions=[[ACTION, ACTION, DO_NOTHING, DO_NOTHING]],
        base_rewards=[[2.0, 2.0, 0.0, 0.0]],
    )
    row = m.metrics_robustness.iloc[0]

    assert np.isclose(row["ave_reward_per_action_unperturbed"], 2.0)
    assert np.isclose(row["reward_per_action_ratio"], 0.5)


def test_target_threshold_is_applied_and_surfaced():
    m = _metrics(
        [[ACTION] * 4], [[1.0] * 4],
        base_actions=[[ACTION, ACTION, DO_NOTHING, DO_NOTHING]],
        base_rewards=[[2.0, 2.0, 0.0, 0.0]],
    )
    assert np.isclose(m.reward_per_action["target_ratio"], REWARD_PER_ACTION_TARGET_RATIO)
    assert bool(m.reward_per_action["meets_target"]) is False  # 0.5 < 0.90


def test_existing_columns_keep_their_positions():
    # test_results/ CSVs are indexed positionally, so the original seven columns
    # must stay first and in order.
    m = _metrics([[ACTION] * 2], [[1.0, 1.0]])
    assert list(m.metrics_robustness.columns[:7]) == [
        "episode", "n_steps_with_act", "n_actions_changed", "similarity_score",
        "total_reward", "n_steps", "ave_reward_per_step",
    ]


def test_recovery_actions_are_counted_but_also_reported_separately():
    actions, rewards = [[ACTION] * 3], [[1.0, 1.0, 1.0]]

    without = _metrics(actions, rewards)
    assert np.isnan(without.metrics_robustness.iloc[0]["n_actions_excl_recovery"])

    with_fn = _metrics(
        actions, rewards,
        recovery_action_fn=lambda act: bool(np.array_equal(act, ACTION)),
    )
    row = with_fn.metrics_robustness.iloc[0]
    assert row["n_actions"] == 3               # recovery actions still count
    assert row["n_actions_excl_recovery"] == 0  # and are reported separately


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
