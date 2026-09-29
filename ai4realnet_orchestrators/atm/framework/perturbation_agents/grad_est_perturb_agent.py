"""
Gradient-Estimation Perturbation Agent for the ATM domain.

ATM (BlueSky-Gym) counterpart of power_grid/framework/perturbation_agents/grad_est_perturb_agent.py,
rebuilt over the Dict observation. Every perturbation is bounded by an L-infinity budget
xi per feature, and the realised maximum is reported by get_stats(), so a KPI value can be
checked against the threat model it claims.

Author: INESC TEC
"""

import logging

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)


class GradEstPerturbationAgent(BasePerturbationAgent):
    """
    Black-box gradient estimation: sample budgeted candidates, keep the one whose action is
    furthest from the action on the clean observation. Costs n_candidates + 1 policy queries.
    """

    def __init__(self, obs_space=None, agent=None, n_candidates=8, xi=0.1, seed=3,
                 name="GradEstPerturbationAgent"):
        # BasePerturbationAgent takes (name, seed, **kwargs); obs_space is kept for API parity
        super().__init__(name=name, seed=seed)
        self.agent = agent
        self.n_candidates = int(n_candidates)
        self.xi = float(xi)
        self._rng = np.random.default_rng(seed)
        self._max_perturb = {}
        self._counts = {"gradient_est": 0}

    def _sample(self, obs):
        return {k: self._rng.uniform(-self.xi, self.xi, np.asarray(obs[k]).shape)
                for k in self._keys(obs)}

    def _with(self, obs, deltas):
        out = self._copy(obs)
        for k, d in deltas.items():
            out[k] += d
        return out

    def perturb(self, obs):
        self._counts["gradient_est"] += 1
        if self.agent is None:
            return self._with(obs, self._sample(obs))
        try:
            base = np.asarray(self.agent.act(obs), dtype=np.float64).ravel()
        except Exception as e:
            logger.error("[GradEst] defender query failed: %s" % e)
            return self._with(obs, self._sample(obs))

        best, best_dist = None, -np.inf
        for _ in range(self.n_candidates):
            deltas = self._sample(obs)
            try:
                action = np.asarray(self.agent.act(self._with(obs, deltas)), dtype=np.float64).ravel()
            except Exception:
                continue
            dist = float(np.linalg.norm(action - base))
            if dist > best_dist:
                best, best_dist = deltas, dist

        deltas = best if best is not None else self._sample(obs)
        for k, d in deltas.items():
            self._record(k, d)
        return self._with(obs, deltas)

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
