"""
Reinforcement-Learning Perturbation Agent for the ATM domain.

ATM (BlueSky-Gym) counterpart of power_grid/framework/perturbation_agents/rl_perturb_agent.py,
rebuilt over the Dict observation. Every perturbation is bounded by an L-infinity budget
xi per feature, and the realised maximum is reported by get_stats(), so a KPI value can be
checked against the threat model it claims.

Author: INESC TEC
"""

import logging

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)


class RLPerturbationAgent(BasePerturbationAgent):
    """
    Online value-based perturbation agent.

    No pre-trained attacker checkpoint exists for ATM (power_grid loads SAC.zip / PPO.zip /
    trained_rlpa_0.pth), so this learns during evaluation, as the railway attackers do: it
    keeps a value per (feature, index) - how much budgeted noise there moved the defender's
    action - and picks epsilon-greedily from it.

    `factor` scales the budget as a percentage of the observation scale, mirroring the
    power-grid SACAttacker's `factor` (5 / 10) used for SAC_5 / SAC_10.
    """

    def __init__(self, obs_space=None, agent=None, xi=0.1, factor=None, seed=4, lr=0.1,
                 gamma=0.99, epsilon=0.2, n_sample=4, name="RLPerturbationAgent"):
        # BasePerturbationAgent takes (name, seed, **kwargs); obs_space is kept for API parity
        super().__init__(name=name, seed=seed)
        self.agent = agent
        self.xi = float(factor) / 100.0 if factor is not None else float(xi)
        self.lr, self.gamma, self.epsilon = float(lr), float(gamma), float(epsilon)
        self.n_sample = int(n_sample)
        self._rng = np.random.default_rng(seed)
        self._values, self._last = {}, None
        self._max_perturb = {}
        self._counts = {"rl_perturb": 0}

    def _slots(self, obs):
        return [(k, i) for k in self._keys(obs) for i in range(int(np.asarray(obs[k]).size))]

    def _choose(self, slots):
        n = min(self.n_sample, len(slots))
        if self._rng.random() < self.epsilon:
            return [slots[int(i)] for i in self._rng.choice(len(slots), size=n, replace=False)]
        return sorted(slots, key=lambda s: self._values.get(s, 0.0), reverse=True)[:n]

    def perturb(self, obs):
        self._counts["rl_perturb"] += 1
        slots = self._slots(obs)
        out = self._copy(obs)
        if not slots or self.agent is None:
            return out
        try:
            base = np.asarray(self.agent.act(obs), dtype=np.float64).ravel()
        except Exception as e:
            logger.error("[RLPerturb] defender query failed: %s" % e)
            return out

        # Credit the previous step's choice with the effect it actually produced.
        if self._last is not None:
            prev_slots, prev_base = self._last
            effect = float(np.linalg.norm(base - prev_base))
            for s in prev_slots:
                self._values[s] = (1 - self.lr) * self._values.get(s, 0.0) + self.lr * effect

        chosen = self._choose(slots)
        for key, pos in chosen:
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            out[key].reshape(-1)[pos] += sign * self.xi
            self._record(key, np.array([sign * self.xi]))
        self._last = (chosen, base)
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
        return float(np.mean(list(self._values.values()))) if self._values else 0.0

    def reset(self):
        # Values persist (the agent keeps learning); the pending credit refers to a
        # finished episode, so it is dropped.
        self._last = None
