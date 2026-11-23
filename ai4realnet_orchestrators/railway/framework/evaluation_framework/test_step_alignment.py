"""
Tests for the two book-keeping series the maze defender records, and for what the
metrics do with them.

Both exist because one recorded step is one DECISION POINT, not one timestep:
maze-flatland's SubStepSkippingWrapper keeps skipping trains that have a single legal
option, across RailEnv steps as well.

  queried  -- which trains were actually asked at this step. A skipped or finished train
              carries a placeholder in the action vector; counting it scores it as
              agreeing with the baseline and as having taken no action.
  env_time -- the RailEnv step counter. The baseline and the perturbed run cover a
              different number of timesteps per decision point once their trajectories
              diverge, so the resilience curves have to be compared at equal env time.

Runs without torch, flatland or maze installed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_framework.metrics import metrics  # noqa: E402


DN = np.zeros(3, dtype=int)


def _fraction_equal(a, b):
    """The railway similarity function: fraction of compared agents that agree."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return 1.0
    return float(np.mean(a == b))


def _bare_metrics():
    """A metrics instance whose __init__ work is bypassed; only the methods are tested."""
    m = metrics.__new__(metrics)
    m.similarity_score_fn = _fraction_equal
    m.recovery_action_fn = None
    return m


# ---------------------------------------------------------------------------
# queried: the action comparison must ignore trains that were never asked
# ---------------------------------------------------------------------------

def test_unqueried_train_does_not_count_as_agreeing():
    """
    Train 2 is never asked, so both rollouts record 0 for it. Train 0 changed its
    action. Over the queried trains only, half the comparison disagrees; counting the
    placeholder as well would report two thirds agreement.
    """
    m = _bare_metrics()
    perturbed = [np.array([3, 1, 0])]
    clean = [np.array([2, 1, 0])]
    queried = [np.array([True, True, False])]

    *_, changed, sim_masked, _ = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN, queried=queried)
    *_, _, sim_full, _ = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN, queried=None)

    assert changed == 1
    assert sim_masked == pytest.approx(0.5)
    assert sim_full == pytest.approx(2.0 / 3.0)
    assert sim_masked < sim_full, "the placeholder was inflating the similarity score"


def test_a_step_where_only_unqueried_trains_differ_is_not_a_changed_action():
    """
    If the only disagreement sits on a train that was never asked, the policy did not
    change its mind and the step must not be counted as a changed action.
    """
    m = _bare_metrics()
    perturbed = [np.array([1, 1, 4])]
    clean = [np.array([1, 1, 0])]
    queried = [np.array([True, True, False])]

    *_, changed_masked, _, _ = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN, queried=queried)
    *_, changed_full, _, _ = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN, queried=None)

    assert changed_masked == 0
    assert changed_full == 1


def test_reward_per_action_counts_decisions_not_vector_slots():
    """
    KPI-RF-078's denominator must count interventions the agent actually decided on.
    Train 1 acted but was never asked -- its move was forced by a single legal option --
    so only train 0's action counts.
    """
    m = _bare_metrics()
    actions = [np.array([2, 3, 0])]
    rewards = [0.0]
    queried = [np.array([True, False, False])]

    masked = m.get_reward_per_action_single_ep(actions, rewards, DN,
                                              performance=[1.0], queried=queried)
    full = m.get_reward_per_action_single_ep(actions, rewards, DN,
                                            performance=[1.0], queried=None)

    assert masked["n_actions"] == 1
    assert full["n_actions"] == 2
    # Same delivered value over fewer interventions, so the ratio is higher.
    assert masked["ave_reward_per_action"] > full["ave_reward_per_action"]


def test_stop_moving_still_counts_for_a_queried_train():
    """Holding a train at a signal is a decision; masking must not swallow it."""
    m = _bare_metrics()
    out = m.get_reward_per_action_single_ep(
        [np.array([4, 0, 0])], [0.0], DN,
        performance=[1.0], queried=[np.array([True, False, False])])
    assert out["n_actions"] == 1


# ---------------------------------------------------------------------------
# queried: a defective mask must never silently shrink the comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    pytest.param([np.array([False, False, False])], id="all-false"),
    pytest.param([np.array([True, True])], id="wrong-width"),
    pytest.param([], id="series-too-short"),
])
def test_an_unusable_mask_falls_back_to_the_full_vector(bad):
    m = _bare_metrics()
    perturbed = [np.array([3, 1, 0])]
    clean = [np.array([2, 1, 0])]

    *_, sim_bad, _ = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN, queried=bad)
    *_, sim_none, _ = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN, queried=None)
    assert sim_bad == pytest.approx(sim_none)


def test_no_mask_reproduces_the_previous_behaviour_exactly():
    """Every single-agent domain, and every non-maze defender, must be unaffected."""
    m = _bare_metrics()
    perturbed = [np.array([1, 2, 3]), np.array([0, 0, 0])]
    clean = [np.array([1, 0, 3]), np.array([0, 0, 1])]
    full = m.get_robustness_metrics_single_ep([], [], clean, perturbed, DN)
    all_true = m.get_robustness_metrics_single_ep(
        [], [], clean, perturbed, DN,
        queried=[np.ones(3, dtype=bool), np.ones(3, dtype=bool)])
    assert full[2:] == all_true[2:]


# ---------------------------------------------------------------------------
# env_time: the resilience curves must be compared at equal environment time
# ---------------------------------------------------------------------------

def test_forward_fill_holds_the_last_known_value():
    """
    Decision points complete at t=2, 4 and 6. At t=3 the newest completed decision is
    the one from t=2, so the value is still 0.0; 0.5 only becomes true at t=4. Reading
    ahead instead would credit the arrival to t=3, before it happened.
    """
    grid = np.arange(1, 7, dtype=float)
    out = metrics._align_on_env_time([0.0, 0.5, 1.0], [2, 4, 6], grid)
    assert list(out) == [0.0, 0.0, 0.0, 0.5, 0.5, 1.0]


def test_alignment_removes_a_gap_that_is_only_an_artefact_of_step_compression():
    """
    The same physical episode recorded at two different decision/timestep ratios: the
    baseline needed 2 decision points for 4 timesteps, the perturbed run needed 4.
    Both deliver the same fraction of trains at the same env times, so the gap -- and
    therefore every resilience KPI -- must be zero. Compared by decision index it is not.
    """
    m = _bare_metrics()
    base_perf, base_t = [0.5, 1.0], [2, 4]
    pert_perf, pert_t = [0.0, 0.5, 0.5, 1.0], [1, 2, 3, 4]

    degr, rest, _, _, _, area = m.get_resilience_metrics_performance_single_ep(
        base_perf, pert_perf, env_time_unperturbed=base_t, env_time_perturbed=pert_t)
    assert area[0] == pytest.approx(0.0)
    assert degr[0] == 0 and rest[0] == 0

    # Without the clock the first index pairs 0.5 against 0.0 and invents a degradation.
    _, _, _, _, _, area_idx = m.get_resilience_metrics_performance_single_ep(
        base_perf, pert_perf)
    assert area_idx[0] > 0


def test_alignment_keeps_a_real_delay_visible():
    """A genuine delay in env time must still register once aligned."""
    m = _bare_metrics()
    base_perf, base_t = [0.5, 1.0, 1.0, 1.0], [1, 2, 3, 4]
    pert_perf, pert_t = [0.0, 0.0, 0.5, 1.0], [1, 2, 3, 4]

    degr, _, _, _, _, area = m.get_resilience_metrics_performance_single_ep(
        base_perf, pert_perf, env_time_unperturbed=base_t, env_time_perturbed=pert_t)
    assert area[0] > 0
    assert degr[0] >= 1


def test_a_non_monotone_clock_falls_back_to_the_decision_index():
    """An unreadable clock must degrade to the old behaviour, not raise."""
    out = metrics._align_on_env_time([0.0, 1.0, 2.0], [5, 1, 9], np.arange(1, 4.0))
    assert list(out) == [0.0, 1.0, 2.0]


def test_a_nan_sample_drops_its_clock_entry_with_it():
    """
    The performance curve is NaN-filtered. If the clock were merely truncated to the
    filtered curve's length, every value after an interior NaN would be paired with the
    wrong timestep. Here the baseline's middle sample is missing; both runs still
    describe the same physical episode, so the gap must stay zero.
    """
    m = _bare_metrics()
    base_perf = [0.0, float("nan"), 0.5, 1.0]
    base_t = [1, 2, 3, 4]
    pert_perf = [0.0, 0.0, 0.5, 1.0]
    pert_t = [1, 2, 3, 4]

    _, _, _, _, _, area = m.get_resilience_metrics_performance_single_ep(
        base_perf, pert_perf, env_time_unperturbed=base_t, env_time_perturbed=pert_t)
    assert area[0] == pytest.approx(0.0)


def test_drop_missing_keeps_values_paired_with_their_own_timesteps():
    vals, times = metrics._drop_missing([1.0, float("nan"), 3.0], [10, 20, 30])
    assert list(vals) == [1.0, 3.0]
    assert list(times) == [10.0, 30.0]


def test_drop_missing_without_a_clock_returns_none():
    vals, times = metrics._drop_missing([1.0, float("nan"), 3.0], None)
    assert list(vals) == [1.0, 3.0]
    assert times is None


def test_missing_clock_reproduces_the_index_based_result():
    m = _bare_metrics()
    base, pert = [0.0, 0.5, 1.0], [0.0, 0.0, 0.5]
    with_none = m.get_resilience_metrics_performance_single_ep(base, pert)
    explicit_none = m.get_resilience_metrics_performance_single_ep(
        base, pert, env_time_unperturbed=None, env_time_perturbed=None)
    assert with_none == explicit_none
