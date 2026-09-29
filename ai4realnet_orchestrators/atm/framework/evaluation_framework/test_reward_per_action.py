"""
Tests for KPI-RF-078 (reward per action) in the railway domain.

Run with pytest, or directly:
    python test_reward_per_action.py

Mirrors power_grid/framework/evaluation_framework/test_reward_per_action.py, plus the
cases that only exist here because railway is multi-agent:

  * an action is one TRAIN acting, summed across trains -- not one per timestep
  * RailEnvActions.STOP_MOVING (4) counts as an action; only DO_NOTHING (0) does not
  * a perfect Flatland baseline scores exactly 0, which leaves the ratio undefined
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_framework.metrics import metrics, REWARD_PER_ACTION_TARGET_RATIO  # noqa: E402

N_TRAINS = 3
DO_NOTHING = np.zeros(N_TRAINS, dtype=int)          # RailEnvActions.DO_NOTHING per train

MOVE_FORWARD = 2
STOP_MOVING = 4

ALL_THREE_MOVE = np.array([MOVE_FORWARD] * 3)
ONE_MOVES = np.array([MOVE_FORWARD, 0, 0])
ALL_THREE_STOP = np.array([STOP_MOVING] * 3)
OBS_DIM = 4


def _similarity(act1, act2):
    return float(np.mean(act1 == act2))


def _build(actions, rewards, base_actions=None, base_rewards=None):
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
# The multi-agent counting decision
# ---------------------------------------------------------------------------

def test_actions_are_counted_per_train_not_per_timestep():
    # 2 timesteps, all 3 trains acting each time => 6 actions, not 2.
    m = _metrics([[ALL_THREE_MOVE, ALL_THREE_MOVE]], [[-1.0, -1.0]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 6
    # the "any train acted" reading would have given 2 -- guard against regressing to it
    assert row["n_actions"] != 2


def test_idle_trains_do_not_count():
    # 2 timesteps, only one train acting each time => 2 actions.
    m = _metrics([[ONE_MOVES, ONE_MOVES]], [[-1.0, -1.0]])
    assert m.metrics_robustness.iloc[0]["n_actions"] == 2


def test_stop_moving_counts_as_an_action():
    # Holding every train at a signal is a dispatching decision, not an absence of one.
    m = _metrics([[ALL_THREE_STOP]], [[-1.0]])
    assert m.metrics_robustness.iloc[0]["n_actions"] == 3


def test_all_do_nothing_yields_nan_not_zero():
    m = _metrics([[DO_NOTHING, DO_NOTHING]], [[-1.0, -1.0]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 0
    assert np.isnan(row["ave_reward_per_action"])
    assert np.isnan(row["reward_per_action_ratio"])
    assert np.isclose(row["ave_reward_per_step"], -1.0)


# ---------------------------------------------------------------------------
# Same guarantees as power grid
# ---------------------------------------------------------------------------

def test_one_action_per_step_matches_ave_reward_per_step():
    # With exactly one train acting per step the per-action and per-step figures agree.
    m = _metrics([[ONE_MOVES] * 4], [[-1.0, -2.0, -3.0, -4.0]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 4
    assert np.isclose(row["ave_reward_per_action"], row["ave_reward_per_step"])
    assert np.isclose(row["ave_reward_per_step"], -2.5)


def test_aggregation_is_pooled_not_mean_of_episodes():
    # ep0: 1 train-action earning -10 -> -10.0 per action
    # ep1: 9 train-actions earning -9 ->  -1.0 per action
    # pooled = -19 / 10 = -1.9  ;  mean of per-episode values = -5.5
    m = _metrics(
        [[ONE_MOVES], [ALL_THREE_MOVE] * 3],
        [[-10.0], [-3.0, -3.0, -3.0]],
    )
    pooled = m.reward_per_action["ave_reward_per_action"]
    mean_of_episodes = m.metrics_robustness["ave_reward_per_action"].mean()

    assert m.reward_per_action["n_actions"] == 10   # 1 + 9
    assert np.isclose(pooled, -1.9)
    assert np.isclose(mean_of_episodes, -5.5)
    assert not np.isclose(pooled, mean_of_episodes)


def test_nan_rewards_drop_the_same_steps_from_both_sides():
    m = _metrics([[ONE_MOVES] * 4], [[-2.0, np.nan, -2.0, np.nan]])
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 2
    assert np.isclose(row["ave_reward_per_action"], -2.0)


def test_ratio_against_a_different_baseline_rollout():
    # perturbed: 6 train-actions for -12 -> -2.0 per action
    # baseline : 2 train-actions for -2  -> -1.0 per action  => ratio 2.0
    # (>1 means the attack made each intervention cost more, since rewards are penalties)
    m = _metrics(
        [[ALL_THREE_MOVE, ALL_THREE_MOVE]], [[-6.0, -6.0]],
        base_actions=[[ONE_MOVES, ONE_MOVES]], base_rewards=[[-1.0, -1.0]],
    )
    row = m.metrics_robustness.iloc[0]

    assert np.isclose(row["ave_reward_per_action_unperturbed"], -1.0)
    assert np.isclose(row["reward_per_action_ratio"], 2.0)


def test_existing_columns_keep_their_positions():
    m = _metrics([[ONE_MOVES] * 2], [[-1.0, -1.0]])
    assert list(m.metrics_robustness.columns[:7]) == [
        "episode", "n_steps_with_act", "n_actions_changed", "similarity_score",
        "total_reward", "n_steps", "ave_reward_per_step",
    ]


def test_column_names_match_power_grid():
    m = _metrics([[ONE_MOVES] * 2], [[-1.0, -1.0]])
    for column in ["n_actions", "n_actions_excl_recovery", "ave_reward_per_action",
                   "n_actions_unperturbed", "ave_reward_per_action_unperturbed",
                   "reward_per_action_ratio"]:
        assert column in m.metrics_robustness.columns
    for key in ["ave_reward_per_action", "reward_per_action_ratio",
                "target_ratio", "meets_target"]:
        assert key in m.reward_per_action.index


# ---------------------------------------------------------------------------
# The Flatland-specific trap
# ---------------------------------------------------------------------------

def test_perfect_flatland_baseline_leaves_the_ratio_undefined():
    """
    Flatland's reward is a per-step penalty that sums to exactly 0 when every train
    arrives. A perfect baseline therefore has a reward-per-action of 0 and the ratio is
    undefined -- NaN, and emphatically not 0, which would read as a failing agent.
    """
    m = _metrics(
        [[ALL_THREE_MOVE]], [[-3.0]],
        base_actions=[[ALL_THREE_MOVE]], base_rewards=[[0.0]],
    )
    row = m.metrics_robustness.iloc[0]

    assert np.isclose(row["ave_reward_per_action_unperturbed"], 0.0)
    assert np.isnan(row["reward_per_action_ratio"])
    assert np.isnan(m.reward_per_action["reward_per_action_ratio"])
    assert bool(m.reward_per_action["meets_target"]) is False


def test_legacy_pickle_without_baseline_actions_degrades_to_nan():
    perturbed, unperturbed = _build([[ONE_MOVES] * 2], [[-1.0, -1.0]])
    del unperturbed["actions"]          # pickle written before KPI-RF-078 existed

    m = metrics(perturbed, unperturbed, DO_NOTHING, _similarity)
    row = m.metrics_robustness.iloc[0]

    assert np.isnan(row["n_actions_unperturbed"])
    assert np.isnan(row["reward_per_action_ratio"])
    assert row["n_actions"] == 2                       # perturbed side still measured
    assert np.isclose(row["ave_reward_per_step"], -1.0)  # pre-existing column intact


def test_recovery_column_is_nan_without_a_classifier():
    m = _metrics([[ALL_THREE_MOVE]], [[-3.0]])
    assert np.isnan(m.metrics_robustness.iloc[0]["n_actions_excl_recovery"])


def test_recovery_classifier_applies_per_train():
    m = _metrics(
        [[np.array([MOVE_FORWARD, STOP_MOVING, MOVE_FORWARD])]], [[-3.0]],
        recovery_action_fn=lambda train_action: int(train_action) == STOP_MOVING,
    )
    row = m.metrics_robustness.iloc[0]

    assert row["n_actions"] == 3                 # all three still count
    assert row["n_actions_excl_recovery"] == 2   # the STOP_MOVING one is excluded


def test_target_threshold_is_surfaced():
    m = _metrics([[ONE_MOVES] * 2], [[-1.0, -1.0]])
    assert np.isclose(m.reward_per_action["target_ratio"], REWARD_PER_ACTION_TARGET_RATIO)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
