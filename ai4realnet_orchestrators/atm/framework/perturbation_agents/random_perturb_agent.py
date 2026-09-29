"""
Random Perturbation Agent for the ATM domain.

ATM (BlueSky-Gym) counterpart of power_grid/framework/perturbation_agents/random_perturb_agent.py,
rebuilt over the Dict observation. Every perturbation is bounded by an L-infinity budget
xi per feature, and the realised maximum is reported by get_stats(), so a KPI value can be
checked against the threat model it claims.

Author: INESC TEC
"""

import logging

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)


class RandomPerturbationAgent(BasePerturbationAgent):
    """Uniform noise within the budget on every feature: the minimum-intelligence baseline."""

    def __init__(self, obs_space=None, xi=0.1, seed=1, name="RandomPerturbationAgent"):
        # BasePerturbationAgent takes (name, seed, **kwargs); obs_space is kept for API parity
        super().__init__(name=name, seed=seed)
        self.xi = float(xi)
        self._rng = np.random.default_rng(seed)
        self._max_perturb = {}
        self._counts = {"random_noise": 0}

    def perturb(self, obs):
        out = self._copy(obs)
        for k in self._keys(out):
            d = self._rng.uniform(-self.xi, self.xi, out[k].shape)
            out[k] += d
            self._record(k, d)
        self._counts["random_noise"] += 1
        return out

    def _keys(self, obs):
        return sorted(obs.keys())

    def _copy(self, obs):
        return {k: np.array(v, dtype=np.float64, copy=True) for k, v in obs.items()}

    def _record(self, key, delta):
        m = float(np.max(np.abs(delta))) if np.size(delta) else 0.0
        if m > self._max_perturb.get(key, 0.0):
            self._max_perturb[key] = m

    def get_stats(self):
        return {"xi": self.xi, "max_perturb_per_key": dict(self._max_perturb),
                "action_counts": dict(self._counts), "mean_q_value": self._mean_q()}

    def _mean_q(self):
        return 0.0

    def reset(self):
        pass
