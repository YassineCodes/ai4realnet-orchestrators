"""
Random Perturbation Agent for the ATM domain.

THE SAME ADVERSARY AS POWER GRID
--------------------------------
This is the power-grid random baseline
(power_grid/framework/perturbation_agents/random_perturb_agent.py), expressed over a
BlueSky-Gym Dict observation instead of a grid2op observation. Same algorithm, same
parameters, so a robustness number measured here answers the same question as the
power-grid one:

  1. every perturbation already active is re-applied and its remaining duration
     decremented; expired ones are dropped,
  2. with probability `prob_perturb` a NEW perturbation is started: one feature group is
     drawn, one scalar inside it is drawn, and a multiplicative factor is drawn -- with
     probability `p_dropout` the factor is exactly 0.0, modelling a lost reading, and
     otherwise it is log-normal with sigma `sigma`,
  3. the new perturbation persists for round(exponential(`mean_perturb_duration_steps`))
     steps.

The previous ATM implementation drew independent uniform noise on EVERY feature at EVERY
step. That is a different and far broader adversary, and it saturated two KPIs: with all
48 features moving every step, KPI-VF-073 (vulnerability) was 1.0000 and KPI-FF-070
(frequency of changed output) was 1.0000 for every attacker, so neither could rank them.
A sparse, persistent perturbation is also the realistic sensor-failure model the KPI was
specified against.

ONE DELIBERATE GENERALISATION
-----------------------------
Power grid draws its component group with fixed weights (generation 25%, load 25%, line
flow 50%) because it has exactly those three. An ATM observation declares its own
feature groups -- 12 of them on the AI4REALNET v1 task, 7 on v0 -- so the group is drawn
uniformly over whatever the environment declares. Nothing here names a feature.

THE BUDGET IS APPLIED ON TOP, AND DECLARED
------------------------------------------
Power grid's baseline is unbounded multiplicative noise. ATM declares an L-infinity
budget xi per feature, so the proposed perturbation is clipped to it and the realised
maximum is reported by get_stats() for checking against the declared threat model.
With xi=None the agent reproduces the power-grid baseline exactly; with xi set it is
that same adversary restricted to the declared budget. The clip is the only difference
between the two domains' baselines, and it is reported with the result.

Author: INESC TEC
"""

import logging

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)

# Carried over unchanged from the power-grid baseline so the two are the same adversary.
PROB_PERTURB = 0.4                  # chance per step of starting a new perturbation
SIGMA = 0.15                        # log-normal sigma of the multiplicative factor
P_DROPOUT = 0.2                     # chance the factor is 0.0 (a lost reading)
MEAN_PERTURB_DURATION_STEPS = 6.0   # mean persistence, exponential


class RandomPerturbationAgent(BasePerturbationAgent):
    """
    The mandatory random control: no feedback from the agent under test.

    Every learned attacker is reported as a multiple of this one over the same episodes,
    so it has to be the shipped baseline rather than a local invention; the parameters
    below are pinned by evaluation_framework/test_random_baseline_parity.py.
    """

    def __init__(self, obs_space=None, xi=0.1, seed=1,
                 name="RandomPerturbationAgent",
                 prob_perturb=PROB_PERTURB, sigma=SIGMA, p_dropout=P_DROPOUT,
                 mean_perturb_duration_steps=MEAN_PERTURB_DURATION_STEPS):
        # BasePerturbationAgent takes (name, seed, **kwargs); obs_space is kept for API
        # parity with the power-grid signature and is not needed here, because the
        # groups are read from each observation.
        super().__init__(name=name, seed=seed)
        self.xi = None if xi is None else float(xi)
        self.prob_perturb = float(prob_perturb)
        self.sigma = float(sigma)
        self.p_dropout = float(p_dropout)
        self.mean_perturb_duration_steps = float(mean_perturb_duration_steps)

        # [key, index, factor, remaining_steps] -- same record as power grid's
        # current_perturbations, so the two can be compared step for step.
        self.current_perturbations = []

        self._max_perturb = {}
        self._counts = {"new_perturbation": 0, "dropout": 0, "steps": 0}

    # ------------------------------------------------------------------
    # The attack
    # ------------------------------------------------------------------

    def perturb(self, obs):
        clean = self._copy(obs)
        out = self._copy(obs)

        self._apply_persistent_perturbations(out)

        if self.space_prng.random() <= self.prob_perturb:
            self._apply_new_perturbation(out)

        self._enforce_budget(clean, out)
        self._counts["steps"] += 1
        self.perturbation_count += 1
        return out

    def _apply_persistent_perturbations(self, obs):
        """Re-apply the active perturbations, age them, and drop the expired ones."""
        if not self.current_perturbations:
            return
        expired = []
        for i, (key, idx, factor, remaining) in enumerate(self.current_perturbations):
            if remaining <= 0:
                expired.insert(0, i)
                continue
            self._apply_factor(obs, key, idx, factor)
            self.current_perturbations[i][3] -= 1
        for i in expired:
            self.current_perturbations.pop(i)

    def _apply_new_perturbation(self, obs):
        """Draw one scalar of one feature group and corrupt it, with persistence."""
        keys = self._keys(obs)
        if not keys:
            return
        key = keys[int(self.space_prng.randint(0, len(keys)))]
        n = int(np.asarray(obs[key]).size)
        if n == 0:
            return
        idx = int(self.space_prng.randint(0, n))

        if self.space_prng.random() < self.p_dropout:
            factor = 0.0                      # the reading is lost
            self._counts["dropout"] += 1
        else:
            factor = float(self.space_prng.lognormal(mean=0.0, sigma=self.sigma))

        duration = float(np.round(
            self.space_prng.exponential(self.mean_perturb_duration_steps)))

        self._apply_factor(obs, key, idx, factor)
        self.current_perturbations.append([key, idx, factor, duration])
        self._counts["new_perturbation"] += 1

    @staticmethod
    def _apply_factor(obs, key, idx, factor):
        if key not in obs:
            return
        flat = np.asarray(obs[key]).ravel()
        if idx < flat.size:
            flat[idx] *= factor
            obs[key] = flat.reshape(np.asarray(obs[key]).shape)

    def _enforce_budget(self, clean, out):
        """
        Clip to the declared L-infinity budget and record the realised maximum.

        The realised maximum is recorded AFTER clipping, so what get_stats() reports is
        what the policy actually received rather than what was proposed.
        """
        for key in self._keys(out):
            delta = np.asarray(out[key], dtype=np.float64) - np.asarray(clean[key], dtype=np.float64)
            if self.xi is not None:
                delta = np.clip(delta, -self.xi, self.xi)
                out[key] = np.asarray(clean[key], dtype=np.float64) + delta
            m = float(np.max(np.abs(delta))) if np.size(delta) else 0.0
            if m > self._max_perturb.get(key, 0.0):
                self._max_perturb[key] = m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keys(obs):
        return sorted(obs.keys())

    @staticmethod
    def _copy(obs):
        return {k: np.array(v, dtype=np.float64, copy=True) for k, v in obs.items()}

    def get_stats(self):
        return {
            "xi": self.xi,
            "max_perturb_per_key": dict(self._max_perturb),
            "action_counts": dict(self._counts),
            "active_perturbations": len(self.current_perturbations),
            "prob_perturb": self.prob_perturb,
            "mean_duration": self.mean_perturb_duration_steps,
            "sigma": self.sigma,
            "mean_q_value": 0.0,
        }

    def reset(self):
        """Clear persistence so episodes stay independent and seed-reproducible."""
        self.current_perturbations = []
        super().reset()
