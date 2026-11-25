"""
Pins the DIRECTION of KPI-DF-075 (degradation_time) and KPI-RF-076 (restoration_time).

These two KPIs are titled "Time until performance degrades" and "Time to restore
performance", which reads as though a larger value were better. The implementation
measures the opposite thing: the SPAN of the degradation, from the gap opening to its
widest point, and 0 when no gap ever opens. So a perfectly resilient agent scores 0 and
a higher value is a longer, deeper degradation.

The documentation had it backwards ("Good Value: High") with nothing pinning it, so these
tests assert the direction directly. If someone changes the detector so that a robust
agent scores high, these fail rather than the deliverable quietly inverting a conclusion.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_framework.metrics import metrics  # noqa: E402


def _resilience(base, pert):
    m = metrics.__new__(metrics)
    d, r, *_ = metrics.get_resilience_metrics_performance_single_ep(
        m, np.asarray(base, dtype=float), np.asarray(pert, dtype=float))
    return d[0], r[0]


def test_an_agent_that_never_falls_behind_scores_zero_degradation():
    # The most resilient outcome there is. It must be the LOW end of the scale.
    curve = [0.0, 0.25, 0.5, 0.75, 1.0]
    degradation, restoration = _resilience(curve, curve)
    assert degradation == 0
    assert restoration == 0


def test_an_agent_that_does_better_than_the_baseline_also_scores_zero():
    base = [0.0, 0.25, 0.5, 0.5, 0.5]
    pert = [0.0, 0.50, 0.75, 1.0, 1.0]
    degradation, _ = _resilience(base, pert)
    assert degradation == 0


def test_a_longer_degradation_scores_higher_not_lower():
    # Identical depth, different duration: the slower slide must score HIGHER.
    base = [1.0] * 10
    quick = [1.0, 0.0] + [1.0] * 8                     # gap opens and peaks at once
    slow = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] + [1.0] * 4  # five steps to the worst point
    d_quick, _ = _resilience(base, quick)
    d_slow, _ = _resilience(base, slow)
    assert d_slow > d_quick, (d_slow, d_quick)


def test_an_unrecovered_gap_is_censored_at_the_episode_end_not_scored_zero():
    # A gap that never closes is the WORST case, so restoration_time must not come back
    # as 0, which is what the "never recovered" branch exists to prevent.
    base = [1.0] * 6
    pert = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _, restoration = _resilience(base, pert)
    assert restoration > 0


def test_the_two_kpis_rank_a_weak_and_a_strong_attack_in_that_order():
    # Both KPIs measure a span, not a latency: a stronger attack degrades for longer
    # and takes longer to recover, so both must read HIGHER for it. Pin that ordering,
    # because the titles ("degradation time", "restorative time") read the other way.
    base = [1.0] * 12
    weak = [1.0, 0.9] + [1.0] * 10
    strong = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    d_weak, r_weak = _resilience(base, weak)
    d_strong, r_strong = _resilience(base, strong)
    assert d_strong > d_weak
    assert r_strong > r_weak
