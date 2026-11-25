"""
Gradient-Estimation Perturbation Agent for the ATM domain.

THE SAME ATTACK AS POWER GRID
-----------------------------
This is the power-grid gradient-estimation attacker
(power_grid/framework/perturbation_agents/grad_est_perturb_agent.py) over a BlueSky-Gym
Dict observation: **finite-difference gradient estimation followed by projected gradient
descent**, with power grid's hyper-parameters unchanged.

    delta      = 0.01   finite-difference step
    step_size  = 0.05   PGD step
    n_iter     = 5      PGD iterations
    epsilon    = xi     the declared budget the perturbation is projected onto

Each iteration estimates the gradient of the attack objective with respect to every
observation scalar by central differences, takes a sign step, and projects back into the
budget. The 2*n finite differences go out as ONE batched policy call, exactly as power
grid batches them through its model, so an iteration costs one query rather than 2*n.

The previous ATM implementation was **not** gradient estimation: it drew `n_candidates`
(8) uniform samples inside the budget and kept whichever moved the action most. That is
random search with a query-based selection rule — a weaker attacker that shares only the
name, and one whose cost does not grow with the observation size because it never looks
at individual features.

WHAT THE OBJECTIVE HAD TO BECOME, AND WHY
-----------------------------------------
Power grid minimises the probability the policy assigns to the action it wanted, which
needs a discrete action distribution. ATM's action space is `Box(-1, 1, (2,))`: there is
no action index and no probability to push down.

The obvious substitute -- maximise `|| pi(obs + delta) - pi(obs) ||` -- does not work,
and the reason is worth recording because it is silent. That objective is symmetric
about `delta = 0` and minimised there, so its central difference along every feature is
`(J(+h) - J(-h)) / 2h = 0` exactly. The estimated gradient vanishes on the first
iteration, PGD has nothing to follow, and the attacker quietly degrades to the random
perturbation it falls back on. It appears to work against a real policy only because
floating-point asymmetry leaves a little noise to climb.

The objective used instead is directional, and is the faithful analogue of power grid's:
drive the action AWAY from the one the agent wanted, by minimising its projection onto
that action.

    a_hat    = pi(obs) / || pi(obs) ||
    J(delta) = - a_hat . pi(obs + delta)                maximised

`J` is not symmetric about `delta = 0` and its gradient there is non-zero, so the
estimate is informative from the first iteration. For a discrete action space, reducing
this projection is reducing the preference for the chosen action, which is what power
grid minimises. This is a domain adaptation of the objective, not of the algorithm, and
it is the one difference from power grid here.

THE BUDGET
----------
Power grid projects onto the looser of a relative and an absolute epsilon
(`min(x(1-e), x-e)` to `max(x(1+e), x+e)`). ATM declares, and verifies at run time, an
absolute L-infinity budget xi per feature, so that is what the projection uses. The
difference is one of the per-domain budget forms already declared with every result
rather than a new one.

A policy whose action does not depend on the observation at all (the straight-line
reference) yields an identically zero gradient, so PGD would propose no perturbation and
the KPI would be reported as "the attack never reached the policy". That is the wrong
reading for an agent that is structurally immune, so a zero gradient falls back to a
random budgeted perturbation: the attack is then genuinely applied, and the unchanged
action is evidence about the agent.

Author: INESC TEC
"""

import logging

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)

# Carried over unchanged from the power-grid attacker.
FD_DELTA = 0.01      # finite-difference step
STEP_SIZE = 0.05     # PGD step
N_ITER = 5           # PGD iterations


class GradEstPerturbationAgent(BasePerturbationAgent):
    """Finite-difference gradient estimation plus projected gradient descent."""

    def __init__(self, obs_space=None, agent=None, xi=0.1, seed=3,
                 fd_delta=FD_DELTA, step_size=STEP_SIZE, n_iter=N_ITER,
                 n_candidates=8, name="GradEstPerturbationAgent"):
        # obs_space is kept for API parity with the power-grid signature; the feature
        # layout is read from each observation instead.
        super().__init__(name=name, seed=seed)
        self.agent = agent
        self.xi = float(xi)
        self.fd_delta = float(fd_delta)
        self.step_size = float(step_size)
        self.n_iter = int(n_iter)
        # Only used by the random fallback, and kept so the previous constructor call
        # still works.
        self.n_candidates = int(n_candidates)
        self._rng = np.random.default_rng(seed)
        self._max_perturb = {}
        self._counts = {"gradient_est": 0, "zero_gradient_fallback": 0, "queries": 0}

    # ------------------------------------------------------------------
    # The attack
    # ------------------------------------------------------------------

    def perturb(self, obs):
        self._counts["gradient_est"] += 1
        self.perturbation_count += 1

        keys = self._keys(obs)
        sizes = [int(np.asarray(obs[k]).size) for k in keys]
        n = int(sum(sizes))
        if self.agent is None or n == 0:
            return self._apply(obs, self._random_delta(obs))

        try:
            clean_action = np.asarray(self.agent.act(obs), dtype=np.float64).ravel()
            self._counts["queries"] += 1
        except Exception as exc:
            logger.error(f"[GradEst] defender query failed: {exc}")
            return self._apply(obs, self._random_delta(obs))

        # The direction the attack pushes against: the clean action itself. A clean
        # action of (almost) zero carries no direction, so the first axis is used -- the
        # objective is then "move this control away from neutral", which is still a
        # well-posed, non-symmetric objective.
        norm = float(np.linalg.norm(clean_action))
        if norm > 1e-12:
            direction = clean_action / norm
        else:
            direction = np.zeros_like(clean_action)
            if direction.size:
                direction[0] = 1.0

        flat = self._flatten(obs, keys)
        delta = np.zeros(n, dtype=np.float64)

        for _ in range(self.n_iter):
            grad = self._estimate_gradient(obs, keys, sizes, flat, delta, direction)
            if grad is None:
                return self._apply(obs, self._random_delta(obs))
            if not np.any(grad):
                # The objective does not respond to any feature: either the policy
                # ignores its observation, or the finite-difference step is below its
                # resolution. Either way PGD has nothing to follow.
                self._counts["zero_gradient_fallback"] += 1
                return self._apply(obs, self._random_delta(obs))
            # Ascend J, which is minus the projection onto the clean action: the step
            # drives the policy away from what it wanted. Power grid descends, because
            # its objective is stated as the probability of that action rather than as
            # its negation.
            delta = np.clip(delta + self.step_size * np.sign(grad), -self.xi, self.xi)

        return self._apply(obs, self._unflatten(delta, keys, sizes))

    def _estimate_gradient(self, obs, keys, sizes, flat, delta, direction):
        """
        Central-difference gradient of J with respect to each observation scalar.

        J(delta) = -direction . pi(obs + delta). See the module docstring for why the
        objective is this projection and not the symmetric action deviation.

        All 2*n probes go out in a single batched call; without a batched path on the
        defender the attacker degrades to the random fallback rather than issuing 2*n
        separate queries per iteration.
        """
        n = len(delta)
        probes = np.repeat((flat + delta)[None, :], 2 * n, axis=0)
        rows = np.arange(n)
        probes[rows, rows] += self.fd_delta
        probes[rows + n, rows] -= self.fd_delta

        obs_batch = [self._unflatten_obs(obs, probes[i], keys, sizes) for i in range(2 * n)]
        act_batch = getattr(self.agent, "act_batch", None)
        if act_batch is None:
            return None
        try:
            actions = np.asarray(act_batch(obs_batch), dtype=np.float64)
            self._counts["queries"] += 1
        except Exception as exc:
            logger.error(f"[GradEst] batched defender query failed: {exc}")
            return None
        if actions.shape[0] != 2 * n:
            logger.error(
                f"[GradEst] batched query returned {actions.shape[0]} actions for "
                f"{2 * n} probes"
            )
            return None

        j = -(actions @ direction)
        return (j[:n] - j[n:]) / (2.0 * self.fd_delta)

    # ------------------------------------------------------------------
    # Observation plumbing: no feature name appears here
    # ------------------------------------------------------------------

    @staticmethod
    def _keys(obs):
        return sorted(obs.keys())

    @staticmethod
    def _flatten(obs, keys):
        return np.concatenate([np.asarray(obs[k], dtype=np.float64).ravel() for k in keys])

    @staticmethod
    def _unflatten(flat, keys, sizes):
        out, i = {}, 0
        for k, s in zip(keys, sizes):
            out[k] = flat[i:i + s]
            i += s
        return out

    def _unflatten_obs(self, obs, flat, keys, sizes):
        """Rebuild a Dict observation from a flat vector, preserving each key's shape."""
        out, i = {}, 0
        for k, s in zip(keys, sizes):
            out[k] = flat[i:i + s].reshape(np.asarray(obs[k]).shape)
            i += s
        return out

    def _random_delta(self, obs):
        return {k: self._rng.uniform(-self.xi, self.xi, np.asarray(obs[k]).shape)
                for k in self._keys(obs)}

    def _apply(self, obs, deltas):
        out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in obs.items()}
        for k, d in deltas.items():
            d = np.clip(np.asarray(d, dtype=np.float64), -self.xi, self.xi)
            out[k] = out[k] + d.reshape(out[k].shape)
            self._record(k, d)
        return out

    def _record(self, key, delta):
        m = float(np.max(np.abs(delta))) if np.size(delta) else 0.0
        if m > self._max_perturb.get(key, 0.0):
            self._max_perturb[key] = m

    def get_stats(self):
        return {
            "xi": self.xi,
            "max_perturb_per_key": dict(self._max_perturb),
            "action_counts": dict(self._counts),
            "fd_delta": self.fd_delta,
            "step_size": self.step_size,
            "n_iter": self.n_iter,
            "mean_q_value": 0.0,
        }

    def reset(self):
        super().reset()
