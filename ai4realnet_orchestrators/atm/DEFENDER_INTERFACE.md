# Evaluating a submitted ATM control agent

The robustness and resilience framework (KPI-DF-069, FF-070, SF-071, SF-072, VF-073,
AF-074, DF-075, RF-076, SF-077, RF-078) evaluates a control agent by attacking its
observation with seven attackers and measuring how its decisions and its task
performance change. This note is the contract a submitted agent has to satisfy.

The agent currently bundled as `framework/agent.zip` is a reference policy, not a
submission: it exists so the pipeline runs end to end. Published KPI values describe
whichever agent was passed in, and are re-measured per submission.

## The contract

```python
class ATMDefender:
    name = "ATMDefender"

    def act(self, obs_dict):      # -> np.ndarray, shape (2,), values in [-1, 1]
        raise NotImplementedError

    def reset(self):              # called once per episode; optional
        pass
```

A Stable-Baselines3 checkpoint needs no code: pass it as `--model <agent>.zip` and
`SB3Defender` loads it. SAC, TD3, PPO and DDPG are supported, and the algorithm is
detected by trying each in turn, because the checkpoints do not record which one wrote
them. The policy is loaded with `env=None` on CPU and queried with
`predict(obs, deterministic=True)`.

Anything else is one subclass of `ATMDefender` implementing `act`.

## The requirement that is easy to miss

**`act()` must be side-effect-free.** Calling it must not advance any state that a
later call depends on.

Five of the seven attackers are query based. The gradient-estimation attacker evaluates
2 queries per observation scalar per iteration (96 for this observation) as central
differences; the four variants of the reinforcement-learning perturbation attacker (PPO,
SAC_5, SAC_10, RLPerturb) probe each candidate slot alone against the clean observation
to credit it with its own effect. These are counterfactual queries: they ask what the
agent would do, and must not be experienced by it.

An agent that carries state advancing on every call -- a recurrent hidden state, a
running filter or estimator, an internal step counter, a stateful conflict-detection or
safety layer -- is corrupted by those probes, and every query-based attacker's episodes
become meaningless while still producing well-formed numbers.
`test_the_defender_query_is_side_effect_free` pins this, so a stateful agent fails
loudly rather than scoring plausibly. This is a declared adaptation of the threat model
from black box to query based, not a hidden assumption.

`act_batch(obs_dicts) -> (n, action_dim)` is optional. Implementing it answers a whole
finite-difference batch in one forward pass; the gradient-estimation attacker falls back
to one call per observation when it is absent, which is correct but slower.

## Observation and action

`StaticObstacleCREnv-v1`, read from the environment itself: a `gymnasium` `Dict` of 12
groups totalling 48 scalars, for a single controlled aircraft.

| key | shape |
|-----|-------|
| `destination_waypoint_distance` | `(1,)` |
| `destination_waypoint_cos_drift` | `(1,)` |
| `destination_waypoint_sin_drift` | `(1,)` |
| `restricted_area_radius` | `(5,)` |
| `restricted_area_distance` | `(5,)` |
| `cos_difference_restricted_area_pos` | `(5,)` |
| `sin_difference_restricted_area_pos` | `(5,)` |
| `intruder_distance` | `(5,)` |
| `intruder_cos_difference_pos` | `(5,)` |
| `intruder_sin_difference_pos` | `(5,)` |
| `intruder_x_difference_speed` | `(5,)` |
| `intruder_y_difference_speed` | `(5,)` |

Action: `Box(-1.0, 1.0, (2,), float64)` = (heading change, speed change). The task is
horizontal only; altitude is neither observed nor commanded.

`destination_waypoint_distance` is the one key the framework names, as the per-step
performance signal. Everything else -- perturbation, action comparison, flattening --
iterates whatever keys the configured environment declares. An observation without that
key is rejected rather than scored as zero.

## The environment a result is tied to

Install the AI4REALNET bluesky-gym fork at upstream commit `539ff884` or later; see the
install note in `README_robustness_resilience_kpi_069_077.md` for why the commit matters
and not only the environment id. Each run records the SHA-256 of the environment module
the registry resolved to, so a submission measured against different task code can be
told apart rather than compared.

## Running it

```
python test_local_robustness_resilience_kpi_069_077.py \
    --model <agent>.zip --episodes 200
```

With no `--model` the `StraightLineDefender` reference is used and the values describe
that baseline, labelled as such. Attacker draws are seeded per attacker, so two runs of
the same agent are bit-identical on all ten KPIs.
