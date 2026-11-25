# Robustness & Resilience KPIs for ATM (KPI-DF-069, KPI-FF-070, KPI-SF-071, KPI-SF-072, KPI-VF-073, KPI-AF-074, KPI-DF-075, KPI-RF-076, KPI-SF-077, KPI-RF-078)

## Overview

This module implements **10 KPIs** for evaluating AI agent robustness and resilience in air
traffic management scenarios. The evaluation uses multiple adversarial attack strategies to
stress-test defender agents.

**Author:** INESC TEC
**Test Runner:** `test_runner_robustness_resilience_kpi_069_077.py`
**Local runner:** `test_local_robustness_resilience_kpi_069_077.py`

Structure mirrors `power_grid/test_runner_robustness_resilience_kpi_069_077.py`: one
`ATMRobustnessTestRunner` runs a single evaluation against every attacker, computes all metrics,
and returns the one selected by `test_id` as `{"primary": value}`.

---

## Implemented KPIs

### Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-DF-069 | `4819e8f6-a2d4-497f-9b61-fc90883a0dfb` | `reward_drop_percent` | Percentage decrease in reward [0-100] |
| KPI-FF-070 | `f0f94fb1-2aef-44f6-ba80-5b8320725fb0` | `action_change_freq` | Proportion of timesteps with changed actions [0-1] |
| KPI-SF-071 | `02bfbe09-6e9b-4243-a376-1a51b1beef19` | `severity_of_change` | Severity of action changes [0-1, higher=worse] |
| KPI-SF-072 | `c466661d-12dc-4d1e-81a4-1db1623e3cc1` | `n_steps_survived` | Number of timesteps before failure |
| KPI-VF-073 | `5cfc7e4d-024b-4dd1-82a5-c3d9bf25ba50` | `perturb_vulnerability` | Proportion of features vulnerable to attack [0-1] |
| KPI-RF-078 | `885cab0d-d4fc-4d93-95db-243870506405` | `reward_per_action_ratio` | Delivered value per action, as a fraction of the unperturbed baseline [0-1] |

### Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-AF-074 | `5372decd-6a2a-4c50-bf7a-cd57cfebe3de` | `area_between_curves` | Integrated performance degradation |
| KPI-DF-075 | `2baef867-c1f2-4b6e-b13c-0eac9463c2fa` | `degradation_time` | Time until performance degrades. Computed as the span from the performance gap opening to its widest point, and 0 when no gap ever opens, so a higher value is a longer and deeper degradation |
| KPI-RF-076 | `7b15a7b3-2413-4953-b91a-24f5c0c5b6da` | `restoration_time` | Time to restore performance. Computed as the span from the widest point until the gap closes, censored at the episode end when it never closes |
| KPI-SF-077 | `e3fb76a2-2121-4889-adf2-b60ca29c5c71` | `state_similarity` | Cosine similarity to unperturbed states [-1 to 1] |

---

## Environment

[BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym) `StaticObstacleCREnv-v1`,
the AI4REALNET air traffic task: one aircraft must reach a destination waypoint while
avoiding both restricted areas and other aircraft.

- **Observation** — `gymnasium.spaces.Dict`, 12 keys, 48 scalars:
  `destination_waypoint_distance`, `destination_waypoint_cos_drift`,
  `destination_waypoint_sin_drift` (1 each); `restricted_area_radius`,
  `restricted_area_distance`, `cos_difference_restricted_area_pos`,
  `sin_difference_restricted_area_pos` (`NUM_OBSTACLES` = 5 each); and
  `intruder_distance`, `intruder_cos_difference_pos`, `intruder_sin_difference_pos`,
  `intruder_x_difference_speed`, `intruder_y_difference_speed`
  (`NUM_INTRUDERS` = 5 each).
- **Action** — `Box(-1, 1, (2,))`: heading change and speed change.
- **Reward** — one scalar: `REACH_REWARD` (+5) on reaching the waypoint, a progress
  reward weighted by `PROGRESS_REWARD_WEIGHT` (0.5),
  `RESTRICTED_AREA_INTRUSION_PENALTY` (-5) per restricted-area intrusion and
  `AC_INTRUSION_PENALTY` (-5) per loss of separation against another aircraft.

**The environment id is configuration, not a constant of the KPIs.** `ENV_ID` in the
test runner defaults to the task above and is overridable with
`AI4REALNET_ATM_ENV_ID`. Everything that depends on the environment is read from it at
run time: the observation keys and shapes, the action space, and the reward constants
that set KPI-DF-069's scale (`env_reward_constants()` resolves them through gymnasium's
registry, so a task without other aircraft is not normalised by a penalty it cannot
award). No feature name appears anywhere in the attack path.

The earlier `StaticObstacleEnv-v0` remains selectable and is a **different task**, not a
variant: 7 observation keys and 43 scalars, 10 restricted areas, no other aircraft,
`REACH_REWARD` of +1, and a drift penalty in place of the progress reward. An agent
trained on one cannot be evaluated on the other, and KPI values from the two are not
comparable.

> **Note on installing the environment.** `StaticObstacleCREnv-v1` exists only in the
> AI4REALNET bluesky-gym fork, not in the `bluesky-gym` release on PyPI.
>
> Install it at upstream commit `539ff884` or later, which is where every figure in
> this document was measured. The commit this project was originally pinned to,
> `e91749ba20`, has `ALTITUDE = 350  # In FL` passed to `bs.traf.cre(acalt=...)`, which
> BlueSky reads as metres: aircraft were created at 350 m rather than FL350. `539ff884`
> corrects it to `35000 ft * 0.3048 = 10668 m`. The constant is not cosmetic, because
> BlueSky derives true airspeed from calibrated airspeed and altitude at aircraft
> creation: measured true airspeed moves 65.7%, episodes are 36% shorter, and no seed
> reproduces its previous return. A checkout on an earlier commit is measuring a
> different task, which the module SHA-256 recorded below will show.
>
> The fork declares `requires-python = ">=3.10"`, but
> `bluesky_gym/envs/common/screen_dummy.py` does `from warnings import deprecated`,
> which is Python 3.13+ (PEP 702), so importing any env module fails on 3.10 to 3.12.
> `framework/_bluesky_compat.py` provides the missing attribute before `bluesky_gym` is
> imported, without modifying the fork. It is a no-op on 3.13 and can be deleted once
> the fork is fixed; the fix there is a version guard or
> `from typing_extensions import deprecated`. Its `pyproject.toml` also pins
> `numpy==2.3.1`, which itself requires Python 3.11 or newer, while the same file
> declares the floor as 3.10, so that floor cannot resolve.

## What's in `atm/framework/`

| path | purpose |
|------|---------|
| `Environment.py` | `BlueSkyGymEnvironment`, exposing the interface `result_getter` and `metrics` expect |
| `defenders.py` | `SB3Defender` (SAC/TD3/PPO/DDPG), `StraightLineDefender` |
| `agent.zip` | trained reference agent |
| `attack_models/` | seven attackers on a common `BaseAgent` / `BaseAttackerClass` |
| `perturbation_agents/` | the perturbation strategies behind them |
| `evaluation_framework/` | `metrics`, `result_getter`, `plots_and_tables` |

`evaluation_framework/` and `utility/` were taken from the railway domain. They are
**not** currently identical to it: the railway copies have since received the attack
application, corruption-semantics, severity-axis and step-alignment work, and the
differences are tracked in the parity note below.

## Reference agent

`framework/agent.zip` is a Stable-Baselines3 SAC policy (`MultiInputPolicy`) trained for
2,000,000 timesteps (seed 42, learning rate 3e-4, `learning_starts` 10,000,
`max_episode_steps` 200), following the hyper-parameters of BlueSky-Gym's own `main.py`.

It is trained on `StaticObstacleCREnv-v1`, the default task, and scores 6.58 ± 6.57 over
10 fresh episodes, with a mean episode length of 50.8. `training_metadata.json` beside it
records the requested and completed timestep counts, the seed, the wall time and the
model's SHA-256, so a reported KPI can be tied to the exact weights that produced it.

It replaces a checkpoint trained on `StaticObstacleEnv-v0` (24,440 episodes; -0.42 ± 2.46)
which **could not run the default task at all**: v1's observation carries 12 named groups
and 48 scalars against v0's 7 and 43, so the feature extractor rejected it. To reproduce
the superseded v0 numbers, set `AI4REALNET_ATM_ENV_ID=StaticObstacleEnv-v0` and pass a v0
policy with `--model`.

The simulator is the bottleneck, and what that costs depends on the host: measured on
`StaticObstacleCREnv-v1` with a random policy and no learning at all, the environment runs
at **105.7 steps per second** on the machine that trained the shipped policy, so 2,000,000
timesteps is about 5.3 hours of pure simulation and neither a GPU nor a hyper-parameter
change touches that. The shipped policy was trained single-process (`DummyVecEnv`) and
took **10.9 hours** of wall clock for its 2,000,000 steps, an average of 50.9 steps per
second: about half the wall clock is the simulator and half is the SAC update and the
policy inference around it.

`train_atm_agent_parallel.py` in the research tree runs the same recipe over
`SubprocVecEnv` with 7 workers, as bluesky-gym's own `main_parallelised.py` does for this
environment, for hosts where the simulator costs more than it does here. It sets
`gradient_steps` to the worker count so the update-to-data ratio matches the
single-process recipe; left at the default, parallel SAC would perform `n_envs` times
fewer updates per sample, which is a different algorithm rather than a faster one.

Without a submission the evaluation falls back to `StraightLineDefender` and labels the
result as describing that reference, so a fallback value can never be read as an
agent's score.

## Submissions

Accepted as a Stable-Baselines3 `.zip`, or an archive containing one. The algorithm is detected
from the checkpoint when not stated. Without a submission the evaluation falls back to
`StraightLineDefender`, whose values describe that reference and are not an evaluation of a
submitted agent.

## ATM-specific metric definitions

The perturbation and metric code follows power_grid, with three domain-specific choices, each
marked in the source:

- **Bounded perturbations.** Every attacker enforces an explicit L-infinity budget ξ per
  observation feature (default 0.1; SAC_5 uses 0.05), and the realised maximum perturbation is
  reported by `get_stats()["max_perturb_per_key"]` and checked against the declared budget at run
  time. `missing` / `large` mean -ξ / +ξ rather than a sentinel value.
- **`get_similarity_score`** normalises the distance between two continuous action vectors by the
  largest distance the action space allows. Equality-based scoring would report maximum severity
  for an arbitrarily small change, because ATM actions are continuous.
- **Performance signal.** AF-074, DF-075 and RF-076 are computed from progress towards the
  destination waypoint, recorded per step by `BlueSkyGymEnvironment`. The ATM reward is dominated
  by sparse events (+1 on arrival, -5 per intrusion), so a reward curve carries no shape to detect
  degradation in.

`reward_drop_percent` is normalised by the worst reward reachable over the same number of steps,
as in the railway domain, because the ATM reward can be near zero or negative.

## Attackers

Same suite as power_grid: `GEPerturb`, `LambdaPIR`, `Random`, `PPO`, `SAC_10`, `SAC_5`,
`RLPerturb`. SAC_5 / SAC_10 differ in budget (5% / 10% of the observation scale), as power_grid's
`factor` does. No pre-trained attacker checkpoints exist for ATM, so `PPO`, `SAC_*` and
`RLPerturb` learn online during evaluation, as the railway attackers do.

### Parity with the power-grid attackers

The attackers in this domain are meant to be the same perturbation agents the rest of
the framework uses, not local inventions. Where each one stands against
`power_grid/framework/perturbation_agents/`:

| Attacker | State |
|----------|-------|
| `Random` | **Same adversary.** The power-grid algorithm and all four of its parameters (`prob_perturb` 0.4, `sigma` 0.15, `p_dropout` 0.2, `mean_perturb_duration_steps` 6.0), expressed over the Dict observation. Pinned by `evaluation_framework/test_random_baseline_parity.py`. |
| `GEPerturb` | **Same attack.** Power grid's central-difference gradient estimation plus projected gradient descent, with its hyper-parameters (`fd_delta` 0.01, `step_size` 0.05, `n_iter` 5); the 2·n finite differences go out as one batched policy call per iteration. The objective is adapted to a continuous action space (see below). Pinned by `evaluation_framework/test_grad_est_parity.py`. |
| `LambdaPIR`, `PPO`, `SAC_*`, `RLPerturb` | **Reimplementations, not ports.** The algorithm families match, and the hyper-parameters have not been reconciled against the originals line by line. |

`Random` matters most of the three states because it is the mandatory control every
learned attacker is measured against, and because its previous form was actively
destroying two KPIs. It drew independent uniform noise on **every** feature at **every**
step, so with 48 features always moving, `perturb_vulnerability` (KPI-VF-073) and
`action_change_freq` (KPI-FF-070) were both 1.0000 for every attacker and could not rank
them. The shipped baseline perturbs one scalar on roughly 40% of steps and lets it persist,
which is also the sensor-failure model these KPIs were specified against. Over the
200-episode campaign KPI-VF-073 reads 0.3279 for the baseline rather than a saturated
1.0000, and KPI-FF-070 0.9270, so both now rank the attackers against it.

The budget is the one deliberate difference and is declared with every result: the
original baseline is unbounded multiplicative noise, while ATM clips each feature to the
L-infinity budget ξ. With `xi=None` this agent reproduces the original baseline exactly;
with ξ set it is that baseline restricted to the declared threat model.

**`GEPerturb` was not estimating a gradient either.** It drew 8 uniform candidates inside
the budget and kept whichever moved the action most, which is random search with a
query-based selection rule. It now runs power grid's algorithm, and against the v1 policy
at the same ξ = 0.1 it moves the action **3.57× further per step** (0.2596 against 0.0727
over 40 steps), with the estimated gradient informative on every step. That comparison was
measured on the superseded environment code and the agent trained against it, because its
before arm is the random-search attacker this one replaced; it is a statement about the
two search rules, which the altitude correction does not bear on.

One objective had to change, and the reason is a trap worth knowing. Power grid minimises
the probability the policy assigns to the action it wanted, which needs a discrete action
distribution; ATM's action space is `Box(-1, 1, (2,))`. The obvious substitute --
maximise `||π(obs+δ) − π(obs)||` -- is **silently broken**: it is symmetric about δ = 0
and minimised there, so its central difference along every feature is exactly zero, the
estimated gradient vanishes on the first iteration, and the attacker degrades to its
random fallback while still looking like it works. The objective used instead is
directional and is the faithful analogue: minimise the projection of the action onto the
clean action, driving the policy away from what it wanted. A regression test asserts the
symmetric objective's gradient is zero so the trap cannot be re-introduced.

### Reproducibility

Every attacker takes a `seed` and used it for a numpy `RandomState`. **Nothing seeded
torch**, and the learned attackers build torch networks, so their weight initialisation
came from torch's global unseeded generator and a reported KPI value could not be
re-derived. `framework/perturbation_agents/determinism.py` seeds torch (CPU and CUDA),
numpy's global generator and Python's `random`, and the runner calls it immediately
before constructing each attacker with that attacker's own declared seed, so an
attacker's draws depend on its seed alone and not on how many were built before it.

Two consecutive runs of the same evaluation against the same policy are now
bit-identical on all ten KPIs. That was re-measured on the environment code above as
two independent 200-episode campaigns over all seven attackers: every KPI value and
every per-phase step count agrees exactly, and only wall-clock time differs.

The policy is only half of what a number has to be tied to. `StaticObstacleCREnv-v1`
exists only in the AI4REALNET bluesky-gym fork, which installs from a working directory
and reports a placeholder distribution version (`bluesky-gym 0.0.0+ai4realnet`), so the
environment id and that version together identify nothing: the same id can resolve to
different task code on another machine or at another date. The run header therefore
records the module gymnasium's registry resolves the id to, that module's path and its
SHA-256, printed by `env_provenance()` beside the reward constants read from the same
module. `evaluation_framework/test_env_provenance.py` pins it.

**Every number in this document was measured on environment code
`static_obstacle_cr_env.py` SHA-256 `ce3f7b2e30822908d632d188dee0c675b787afb09d271c1c6557ff6db17ba544`.**
A checkout whose hash differs is running a different task, and the KPI values here do not
describe it. Re-run the campaign rather than comparing against this table.

Bit-identity holds for repeated runs on one machine with one set of library versions and
is not a cross-platform guarantee. **The platform comparison in the rest of this section
was measured on the superseded environment code** (`static_obstacle_cr_env.py` SHA-256
`1d5881121583e0ccccb7c82fefdf2958f5e1dcdfb764996d3fe4eb90ea59d145`) and
has not been re-measured on the code the table above was produced with, because it needs a
second platform rather than a second run. It is kept because it is a statement about the
simulator's floating-point sensitivity and about which KPI orderings survive a platform
change, neither of which the altitude correction bears on; its absolute figures belong to
the superseded task. That much was measured rather than assumed: the whole campaign was
run twice, on Linux and on Windows, from the same model SHA-256 and the same seeds. No
KPI reproduces to the last digit, because the simulator amplifies small
floating-point differences into different episode lengths -- the unperturbed baseline
alone comes out at 14786 steps against 14800.

What survives the platform change is the ordering, for most KPIs but not all. The two the
conclusions here rest on are stable: **KPI-VF-073 and KPI-SF-071 rank all seven attackers
identically on both platforms**, disagreeing by 1.0% and 1.2% on average. KPI-DF-075 and
KPI-RF-078 also hold their order. Five do not, and for three of those the swap is between
values that were tied to begin with: KPI-SF-072 (0.6 steps apart), KPI-SF-077 (0.0007
apart) and KPI-RF-076 (an exact tie on one platform). Two are real exceptions.
**KPI-DF-069** spans only 0.00% to 1.85%, so a 0.09-point shift is a large relative one.
**KPI-AF-074** disagrees by 11% on average and reverses its bottom two attackers even
though they sit 34% apart, so a single attacker's area figure should be treated as
platform-dependent.

Quote a KPI value together with the environment that produced it.

### Pipeline properties this domain already holds

Four pipeline corrections and a severity axis were applied elsewhere in the framework.
None is outstanding here, and the reason differs per item, so each is recorded rather
than left looking like open work.

1. **The attack decision is taken once per step, and the perturbation recorded is the one
   applied.** Already structural in this domain. `Environment.step()` calls
   `attacker.perturb()` exactly once, and records the perturbation as
   `flatten(perturbed_obs) - flatten(clean_obs)` — measured from the observation actually
   handed to the defender rather than reported by the attacker. The recorded and applied
   perturbation cannot diverge, because the record is derived from the application.

2. **Corruption types resolve against the observation the policy reads.** Not applicable.
   The ATM attacks are bounded additive perturbations on named observation slots under
   the ξ budget; there is no corruption-type indirection to resolve.

3. **The attack action space carries a severity axis.** Not applicable, for the same
   reason: magnitude is already the ξ budget, which is bounded per feature and set at the
   command line, so a separate severity axis would be a second name for it.

4. **A query-based attacker scores its candidates against the policy's own observation.**
   Satisfied, and there is no representation split here to get wrong: the defender
   consumes the same named Dict observation the attacker perturbs. GEPerturb takes its
   finite differences through the defender, and the RLPerturb family probes each slot
   with `agent.act(probe_obs)` against the clean observation at the same step.

The step-alignment machinery (`env_time` / `queried` series) is deliberately **not**
carried into this domain: one aircraft is queried at every step and one recorded step is
one timestep, so the mask is all-true and the alignment is the identity. Adding it would
introduce a code path that can never differ from the current one.

## Measured results

200 episodes per attacker on `StaticObstacleCREnv-v1`, every attacker on the same seeds,
so the rows are paired. Agent: the shipped `framework/agent.zip`, the 2,000,000-step SAC
policy. Unperturbed baseline: 9448 steps over 200 episodes, 47.2 steps per episode.
Perturbation budget xi = 0.1 (L-infinity, per observation feature), except `SAC_5` at
0.05, which is what its name records.

| Attacker | VF-073 vuln | SF-071 severity | FF-070 freq | AF-074 area | SF-072 steps | DF-069 drop % | DF-075 degr | RF-076 rest | SF-077 state sim | RF-078 rpa |
|----------|------------:|----------------:|------------:|------------:|-------------:|--------------:|------------:|------------:|-----------------:|-----------:|
| GEPerturb | **0.9982** | **0.6232** | 1.0000 | **5674.94** | **83.1** | **1.66** | 26.6 | 1.4 | **0.5275** | **0.3464** |
| SAC_10 | 0.5482 | 0.1643 | 1.0000 | 2872.93 | 47.4 | 0.00 | 18.5 | 1.8 | 0.5653 | 0.9934 |
| PPO | 0.4262 | 0.1490 | 1.0000 | 2882.04 | 45.5 | 0.02 | 16.6 | 1.9 | 0.5682 | 1.0000 |
| RLPerturb | 0.5811 | 0.1436 | 1.0000 | 2527.37 | 45.9 | 0.00 | 15.4 | 2.3 | 0.5682 | 1.0000 |
| LambdaPIR | 0.1783 | 0.1276 | 0.9998 | 2386.55 | 45.4 | 0.00 | 14.3 | 1.5 | 0.5658 | 1.0000 |
| SAC_5 | 0.5760 | 0.0969 | 1.0000 | 2392.55 | 49.8 | 0.00 | 18.1 | 1.6 | 0.5717 | 0.9842 |
| *Random (the baseline)* | *0.3279* | *0.0380* | *0.9270* | *3129.11* | *47.6* | *0.00* | *17.8* | *2.5* | *0.5657* | *1.0000* |

Suite aggregate, the arithmetic mean over the seven attackers: KPI-DF-069 0.2400,
KPI-FF-070 0.9895, KPI-SF-071 0.1918, KPI-SF-072 52.1036, KPI-VF-073 0.5194, KPI-AF-074
3123.6415, KPI-DF-075 18.2014, KPI-RF-076 1.8543, KPI-SF-077 0.5618, KPI-RF-078 0.9034.

### Reading

**Every attacker exceeds the random baseline on severity**, KPI-SF-071, which is the
robustness KPI that discriminates in this domain: 0.6232 for GEPerturb and 0.0969 to
0.1643 for the rest, against 0.0380 for random noise — a factor of 2.6 to 4.3 for the
learned attackers and 16 for GEPerturb.

**KPI-AF-074 does not reproduce that separation**, and is not a second opinion on it. Only
GEPerturb (5674.94) exceeds the baseline's 3129.11, and all five of the other learned
attackers fall below it, from SAC_10's 2872.93 down to LambdaPIR's 2386.55, so on area
alone every attacker but one looks weaker than the noise it beats by a factor of 2.6 to
4.3 on severity. It is also the least stable of the ten KPIs
across platforms, as Reproducibility above records. The area integrates a reward
difference over episodes whose length the attacker itself changes, so it mixes how much
reward was lost with how long the episode ran. KPI-SF-071 is the KPI to rank attackers by
in this domain.

**Only GEPerturb degrades the agent's task performance.** It is the one attacker that
moves KPI-RF-078 materially off 1.0000 (0.3464, so the agent earns around a third of its
baseline reward per action, where SAC_10 and SAC_5 shift it only to 0.9934 and 0.9842
and the rest leave it at 1.0000), the one that lengthens episodes materially (83.1 steps
against a 47.2-step baseline, the others within 6% of it), and the one that moves
KPI-SF-077 below the pack. The learned attackers reliably change the commanded action,
but this agent still completes its task under them. That is a robustness result about
the agent, not a defect in the attackers: at xi = 0.1 it is robust to the learned
attacks and not to gradient estimation.

**KPI-FF-070 saturates in this domain and should not be used to rank attackers here.**
The action space is continuous, so a changed action is the normal case rather than the
informative one. It is not a numerical artefact: measured over every step of the campaign,
the largest per-component change between the perturbed and clean action has a median of
1.6774 for GEPerturb on an action space spanning 2.0 — 84% of the full command range — and
0.3040 for SAC_10, 0.2644 for PPO. Steps on which the action does not move at all are
vanishingly rare for the learned attackers: one in 4536 for LambdaPIR, and none at all for
GEPerturb, PPO, SAC_5, SAC_10 or RLPerturb. The random baseline reads 0.9270 rather than
1.0000 because on 696 of its 9528 steps — 7.30% — it perturbs nothing at all, by its
dropout and duration parameters.

Those counts also check the KPI itself: one minus the fraction of unchanged steps
reproduces the reported KPI-FF-070 to four decimals for all seven attackers, so the metric
measures what its name says and the saturation is a property of the action space rather
than of the implementation. `evaluation_framework/action_change_report.py` recomputes
every figure in this paragraph from the archives the campaign already writes, so they can
be re-derived rather than taken on trust. Severity (KPI-SF-071) carries the information
that frequency carries in a discrete action space.

**KPI-DF-069 has little resolution on this task**, reading 0.00% to 1.66%. The reward gap
is normalised by the worst reward reachable over the same number of steps rather than by
the unperturbed total, because the ATM reward mixes a +5 waypoint reward with -5
intrusion penalties and a per-step drift penalty, so the baseline total can be near zero
or negative and a ratio to it is undefined or explosive. That normaliser is deliberately
conservative, which keeps the KPI inside [0, 100] for any baseline at the cost of
compressing this task's range. Normalising by the observed spread instead would restore
resolution but would stop being comparable across agents, so it is left as it is and read
alongside KPI-AF-074 and KPI-RF-078, which do separate the attackers.

### Effect of the per-slot attacker credit

**The figures in this section were measured on the superseded environment code**
(`static_obstacle_cr_env.py` SHA-256 `1d5881121583e0ccccb7c82fefdf2958f5e1dcdfb764996d3fe4eb90ea59d145`)
and on the agent trained against it, because the before arm needs the shared-credit rule
that the change replaced. They are kept because the comparison is a statement about the
credit signal, whose direction does not depend on the altitude constant, and they are not
comparable with the table above cell by cell.

`RLPerturbationAgent` backs four of the seven attackers, so the same 200-episode campaign
was run a second time with its previous shared-credit rule to measure what the change is
worth. Both arms use the same model SHA-256, the same seeds and the same machine. The other
three rows of the suite do not run this class, so the change cannot reach them: GEPerturb
and LambdaPIR are omitted, and the random baseline is the 0.0344 that the `vs random`
column divides by.
The unperturbed baseline came out at 14786 steps in both arms, so the two tables are
paired on an identical reference rather than on two runs that merely agree closely.

| Attacker | SF-071 before | after | change | vs random | VF-073 before | after |
|----------|--------------:|------:|-------:|----------:|--------------:|------:|
| SAC_10 | 0.0706 | 0.1470 | +108% | 2.05x → **4.27x** | 0.6856 | 0.6372 |
| PPO | 0.0538 | 0.1366 | +154% | 1.56x → **3.97x** | 0.5461 | 0.5045 |
| RLPerturb | 0.0561 | 0.1331 | +137% | 1.63x → **3.87x** | 0.7098 | 0.6746 |
| SAC_5 | 0.0407 | 0.0906 | +123% | 1.18x → **2.63x** | 0.6876 | 0.6589 |

Severity at least doubles for every one of the four, and all four now clear the random
baseline by a factor of 2.6 or more, where SAC_5 previously sat 18% above it and the rest
between 1.6x and 2.1x. That is the measurement the change was made for: under shared
credit these attackers were barely separable from noise on the KPI that discriminates in
this domain.

KPI-VF-073 falls for all four, by 4% to 8%. That is the expected direction rather than a
regression: KPI-VF-073 measures the proportion of features the attacker finds worth
attacking, and per-slot credit makes the attacker selective rather than broad. Breadth is
not strength here — LambdaPIR attacks the fewest features of any attacker in the suite
(KPI-VF-073 0.2056, against SAC_5's 0.6589) and still scores higher on severity than SAC_5
does.

KPI-AF-074 moves in both directions (SAC_10 969.4 → 1725.3, RLPerturb 1533.1 → 2700.3,
SAC_5 1745.4 → 2146.4, PPO 1973.3 → 1406.3) and is not read as evidence either way, for
the reasons given under Reading above.

The applied attack is unchanged by the fix: across the campaign both arms perturb the same
number of slots per step (4 for PPO and RLPerturb, 6 for the two SAC budgets) under the
same budget ceiling. Only the choice of slots changes.

### Phase timings

200 episodes each: unperturbed 96.0s, GEPerturb 364.2s, LambdaPIR 288.0s, Random 102.0s,
PPO 114.6s, SAC_10 126.4s, SAC_5 132.3s, RLPerturb 115.9s, 22m19s in total.

Those are wall-clock times and are **not** a cost comparison between attackers: they
depend on the load on the machine and on how long the attacker makes its own episodes.
This campaign ran on an unloaded machine, where LambdaPIR is the most expensive attacker
per step at 0.032 s/step and GEPerturb costs 0.022, so GEPerturb's larger total is its
longer episodes (16618 steps against 9072) rather than a higher cost per step. An earlier
campaign on the same host ran its GEPerturb phase under heavy contention at 0.212 s/step
against 0.031 for the phase that followed, a factor of seven from scheduling alone, which
is why these figures are not quoted as attacker costs. What is robust is that both of them
cost about two to two and a half times what the RL family does, because they spend the
most defender queries per step: the four RLPerturb-backed attackers run at 0.013 s/step
against the unperturbed baseline's 0.010.

Re-running the same command against the same `--save-dir` reloads all eight archives and
recomputes the KPIs without stepping an environment, and reproduces every figure above
exactly.

## Running locally

```
# stable-baselines3 2.x is REQUIRED, not optional; see the note below
pip install "stable-baselines3>=2.0"
# StaticObstacleCREnv-v1 is only in the AI4REALNET fork, not the PyPI release:
pip install -e <path-to-the-ai4realnet-bluesky-gym-fork>
# --model must be a policy trained on the configured task (see Reference agent)
python test_local_robustness_resilience_kpi_069_077.py --model <v1-policy>.zip --episodes 50     --save-dir results/
```

Options: `--episodes`, `--xi`, `--attackers`, `--kpi`, `--save-dir`, `--verbose`.

Pass `--save-dir` for anything whose numbers will be quoted. Without it the per-episode
archives, plots and the HTML report go to a temporary directory that is deleted when the
run ends, so only the printed aggregates survive and no per-attacker breakdown can be
recovered afterwards.

`--save-dir` also makes a campaign **restartable**. Each phase, the unperturbed baseline
and then one per attacker, writes its own archive, and a phase whose archive is already
present is loaded rather than re-run. A run that dies partway therefore resumes at the
phase it was in and loses at most that one phase, which matters because a full suite is
hours long.

The archive is reused only when it holds the episode count the new run asks for; a
mismatch re-runs that phase and says so (`[REDO]`). Without that check a save directory
left over from a 50-episode run would be reused wholesale by a 200-episode run, and the
KPIs would be computed with a baseline of one length against perturbed phases of another
with nothing reporting it. **The check is on length, not on identity: point a new agent
at a directory holding another agent's archives and they will be reused.** Use a fresh
directory per agent.

### stable-baselines3 2.x is required

On stable-baselines3 1.x, loading a v1 policy fails with

```
NotImplementedError: Dict('cos_difference_restricted_area_pos': Box(...), ...)
observation space is not supported
```

which reads as though the environment were wrong. It is not: the observation is a Dict,
1.x does not support a Dict observation space for SAC/TD3/DDPG, and a checkpoint saved by
2.x additionally carries an `lr_schedule` that 1.x cannot deserialise (`Can't get
attribute 'FloatSchedule'`). Both disappear on 2.x. Check with
`python -c "import stable_baselines3 as s; print(s.__version__)"` before reporting an
environment problem.
