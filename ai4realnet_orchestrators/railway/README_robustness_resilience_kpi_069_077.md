# Robustness & Resilience KPIs for Railway (KPI-DF-069, KPI-FF-070, KPI-SF-071, KPI-SF-072, KPI-VF-073, KPI-AF-074, KPI-DF-075, KPI-RF-076, KPI-SF-077, KPI-RF-078)

## Overview

This module implements **10 KPIs** for evaluating AI agent robustness and resilience in Flatland railway traffic management scenarios. The evaluation uses multiple adversarial attack strategies to stress-test defender agents.

**Author:** INESC TEC
**Test Runner:** `test_runner_robustness_resilience_kpi_069_077.py`
**Local runner:** `run_local_robustness_resilience_kpi_069_077.py`

---

## Implemented KPIs

### Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-DF-069 | `a94c858e-4bc3-4d67-bd78-5c81506e39f7` | `reward_drop_percent` | Percentage decrease in reward [0-100] |
| KPI-FF-070 | `5abadf6b-991c-4d37-810f-f77bb71d490d` | `action_change_freq` | Proportion of timesteps with changed actions [0-1] |
| KPI-SF-071 | `dce32e78-e827-4994-a0a2-06feee2528cc` | `severity_of_change` | Severity of action changes [0-1, higher=worse] |
| KPI-SF-072 | `e5206c56-75a0-41fa-9db3-bec66359337e` | `n_steps_survived` | Number of timesteps before failure |
| KPI-VF-073 | `0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc` | `perturb_vulnerability` | Proportion of features vulnerable to attack [0-1] |
| KPI-RF-078 | `8ebc88f0-896c-4910-8997-a44d107e7eb7` | `reward_per_action_ratio` | Delivered value per action, as a fraction of the unperturbed baseline [0-1] |

### Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-AF-074 | `707a1a4e-7073-432b-94fc-af4a5ee9f07d` | `area_between_curves` | Integrated performance degradation |
| KPI-DF-075 | `2c4be118-6108-43b3-b09f-a4bee842167a` | `degradation_time` | Time until performance degrades |
| KPI-RF-076 | `2cac54e0-aaf3-4f22-8307-f23878c432f0` | `restoration_time` | Time to restore performance |
| KPI-SF-077 | `d432299f-dbee-46ba-9e15-77954086440a` | `state_similarity` | Cosine similarity to unperturbed states [-1 to 1] |

---

## Attack Strategies

The evaluation runs the defender agent against **6 different attacker types**:

| Attacker | Description | Pre-trained model |
|----------|-------------|-------------------|
| **LambdaPIR** | Lambda policy/value iteration with Bellman-bootstrapped refinement | no, learns online |
| **SAC** | Discrete soft actor-critic adversary | no, learns online |
| **RLPerturb** | Deep Q-learning perturbation agent | no, learns online |
| **Random** | Random perturbation baseline | no |
| **PPO** | Actor-critic adversary | no, learns online |
| **GEPerturb** | Gradient estimation; scores candidates by querying the defender | no |

Unlike the power grid domain, the railway attackers train from scratch during the evaluation, so no `trained_models/` folder is required.

### Measured results

200 episodes against the bundled reference agent (`framework/agent.zip`), on the fixed
pipeline. Every attacker is evaluated on the same scenarios (`env.reset(seed=ep)` for
`ep` in `0..199`), so the rows are paired. The unperturbed baseline delivers all three
trains in 168 of 200 episodes, mean arrival 0.875.

| Attacker | `action_change_freq` | `perturb_vulnerability` | `reward_per_action_ratio` | `degradation_time` | `area_between_curves` |
|----------|---------------------|-------------------------|---------------------------|--------------------|-----------------------|
| Random | 0.339 | 0.304 | 0.433 | 3.82 | 5148 |
| GEPerturb | 0.332 | 0.296 | 0.496 | 3.81 | 5070 |
| RLPerturb | 0.187 | 0.213 | 0.755 | 3.00 | 3593 |
| PPO | 0.135 | 0.078 | 0.842 | 1.08 | 1533 |
| SAC | 0.048 | 0.013 | 0.938 | 0.34 | 460 |
| LambdaPIR | 0.008 | 0.005 | 0.974 | 0.41 | 496 |

Operational effect of the same runs, the fraction of trains that actually arrived:

| Attacker | mean arrival | change vs unperturbed | all-3-trains episodes | trains attacked per perturbed step | mean abs applied delta |
|----------|--------------|-----------------------|-----------------------|------------------------------------|------------------------|
| Random | 0.467 | -0.408 | 38/200 | 1.15 | 3.86 |
| GEPerturb | 0.498 | -0.377 | 41/200 | 1.00 | 3.85 |
| RLPerturb | 0.730 | -0.145 | 96/200 | 1.00 | 7.19 |
| PPO | 0.787 | -0.088 | 126/200 | 1.00 | 8.31 |
| SAC | 0.852 | -0.023 | 157/200 | 1.00 | 7.34 |
| LambdaPIR | 0.855 | -0.020 | 161/200 | 1.00 | 7.88 |
| *unperturbed* | 0.875 | n/a | 168/200 | n/a | n/a |

Aggregated across all six attackers:

| KPI | Value |
|-----|-------|
| `perturb_vulnerability` | 0.1516 |
| `n_steps_survived` | 27.8775 |
| `severity_of_change` | 0.3383 |
| `reward_drop_percent` | 0.2118 |
| `action_change_freq` | 0.1748 |
| `area_between_curves` | 2716.8298 |
| `degradation_time` | 2.0758 |
| `restoration_time` | 0.6108 |
| `state_similarity` | 0.9721 |
| `reward_per_action_ratio` | 0.7397 |

#### What the fix changed, and the result it produced

The 50-episode table below is kept for comparison only. It was produced while
`act_and_step()` made a second, independent attack decision per step, discarded the
attacker's chosen corruption type, applied the attack to every train whenever the
attacker published no target, and recorded the TreeObs delta instead of the delta
applied to the policy's own observation.

| Attacker | `action_change_freq` (50 ep, pre-fix) | `action_change_freq` (200 ep, fixed) |
|----------|---------------------------------------|--------------------------------------|
| LambdaPIR | 0.841 | 0.008 |
| SAC | 0.339 | 0.048 |
| RLPerturb | 0.336 | 0.187 |
| Random | 0.275 | 0.339 |
| PPO | 0.233 | 0.135 |
| GEPerturb | 0.070 | 0.332 |

The ordering inverts. Pre-fix, LambdaPIR led by 2.5x over the random baseline; on the
fixed pipeline it is the weakest of the six, and **every learned attacker scores below
the random baseline**. Paired per-episode Wilcoxon signed-rank against Random over the
same 200 seeds: LambdaPIR z = -12.26, p = 1.4e-34; SAC z = -11.36, p = 6.3e-30;
PPO z = -9.87, p = 5.8e-23; RLPerturb z = -10.06, p = 8.2e-24. GEPerturb is the only
attacker statistically indistinguishable from Random (z = -1.02, p = 0.31) — consistent
with its candidate scoring being degenerate, since it selects `blank` on every step and
`blank` is also what the random baseline applies.

Breadth is now matched, which it was not before: Random attacks 1.15 trains per
perturbed step against 1.00 for every targeted attacker, and all six perturb a similar
share of steps (47-54%). The pre-fix random baseline was applied to all three trains on
every step, so its old 0.275 was earned with roughly 3x the breadth of any
targeted attacker.

Two independent measurements agree on the ordering, so this is not an artifact of one
metric: `action_change_freq` and the drop in trains delivered rank the six attackers
identically (Spearman +1.000, Pearson +0.977).

#### The attacker objective is anti-correlated with harm

The fixed run makes the reward-function problem measurable rather than arguable.
`disruption_reward()` scores distortion of the TreeObs, and the attackers that maximise
it succeed at exactly that and at nothing else:

- LambdaPIR produces the **largest** applied perturbation of all six (mean absolute
  delta 7.88) and the **smallest** behavioural effect (0.008 action changes, -0.020
  arrival).
- Random produces one of the smallest perturbations (3.86) and by far the largest effect
  (0.339, -0.408 arrival, all three trains delivered in 38 of 200 episodes against 168
  unperturbed).
- Across the six attackers, perturbation magnitude is **negatively** rank-correlated
  with operational harm (Spearman -0.771).

The mechanism: the learned attackers scale or negate fields (`speed`, `direction`), which
moves the observation far in L2 but in directions the behaviour-cloned policy is
insensitive to. Blanking a field that the policy depends on is small in L2 and
behaviourally decisive. Optimising TreeObs distortion therefore selects against
effectiveness, which is why a random baseline beats every learned attacker here.

This is a result about the objective, not about the domain: it motivates rewarding the
attacker on the defender's own outcome. The counterfactual clean action is already
computed every step (`act_and_step` returns it), so that signal is available at no extra
cost. Replacing the reward changes what LambdaPIR *is* and breaks comparability with the
submitted Smart Grid work, so it belongs in an additional attacker variant rather than in
LambdaPIR itself.

#### Known limitation of the recorded observation

`perturbations` now records the delta applied to the maze observation the policy reads
(258 values, 86 per train). `observations` still records the flattened TreeObs (144
values, 48 per train) — a representation the BC policy never consumes. So
`perturb_vulnerability` (KPI-VF-073) is computed on what the policy received, but
`state_similarity` (KPI-SF-077) and the per-episode cosine-similarity curves are still
computed on the TreeObs. Those two are a residual instance of the representation mismatch
and should not be read as "how far the policy's input moved".

### What this benchmark actually measures

Measured against the reference BC policy, by writing a chosen value into a chosen group
of maze observation fields and recording how often the policy's action changes
(12 episodes per cell):

| field group written | = -1 | = 0 | = 1 | = 1369 |
|---------------------|------|-----|-----|--------|
| `potential_conflict_dist` (3 fields) | 0.000 | 0.000 | 0.000 | **0.252** |
| `target_dist` (12) | 0.017 | 0.016 | 0.016 | **0.161** |
| `train_speed` (1) | 0.000 | 0.000 | 0.000 | **0.095** |
| `other_train_dist` / `conflicting_train_distance` (6) | 0.000 | 0.001 | 0.002 | **0.245** |
| `deadlock` (12) | 0.001 | 0.000 | 0.002 | **0.189** |
| `obstructed` (3) | 0.000 | 0.000 | 0.001 | **0.189** |
| `unusable_switch_dist` (3) | 0.001 | 0.001 | 0.001 | **0.249** |
| *all 86 fields set to 0, which is what the random baseline does* | | **0.249** | | |

No semantically meaningful value changes the policy's behaviour. Claiming a conflict is
imminent (0 or 1, where -1 is the "none" sentinel) does nothing. Claiming a conflict at
distance 1369 — further than the map is wide, a value the field can never legitimately
hold — does everything. And it does not matter which field carries the magnitude:
`potential_conflict_dist = 1369` (0.252), `unusable_switch_dist = 1369` (0.249) and
blanking all 86 fields (0.249) are the same attack as far as the policy is concerned.

The dose-response confirms it. `potential_conflict_dist` never exceeds 20 in a clean
rollout:

| value written | 2 | 5 | 20 (its real maximum) | 50 | 100 | 500 | 1369 |
|---------------|---|---|------------------------|----|-----|-----|------|
| action change | 0.000 | 0.001 | 0.005 | 0.018 | 0.042 | 0.176 | 0.252 |

Inside the range the field actually takes, the attack is worth 0.005. It needs 25-70x the
field's legitimate maximum to work.

So these KPIs measure **the policy's sensitivity to out-of-range numerical magnitude**,
not its robustness to plausible-but-false observations. Three consequences:

1. **There is almost nothing for a learned attacker to learn.** The attackers' action
   space is (which train) x (which corruption type) — 19 actions for 3 trains. It has no
   magnitude axis, so the only effective attack direction cannot be expressed in it.
   Every attacker is searching a space in which the effective dimension is absent, which
   is why none of them beats a baseline that happens to produce a large out-of-range push.
2. **`train_speed` is unattackable by construction.** It holds exactly one distinct value
   (1.0) across 285 clean observations, because training used
   `speed_ratio_map={1.0: 1.0}`. The policy never saw it vary and cannot have learned to
   use it.
3. **Under a perturbation budget, all of these attacks score zero.** ATM clips at
   xi = 0.1; railway has no budget, and the table above shows these attacks live entirely
   outside any plausible one. A cross-domain robustness claim between the two domains is
   not comparing like with like until railway is budgeted.

### Corruption types must be resolved against the observation the policy reads

`corrupt_node()` honours a corruption type's meaning on a TreeObs, because a TreeObs has
named fields. The policy reads the maze graph observation instead: one flat vector per
train. `corrupt_vector()` was written to be format-agnostic — "corruptions are
statistical" — so on that vector all six types collapsed to three whole-vector
operations:

| type | whole-vector operation | measured action change |
|------|------------------------|------------------------|
| `blank` | x0 | 0.254 |
| `conflict`, `target`, `malfunction` | x3 | **0.000000** |
| `speed`, `direction` | x-1 | 0.263 |

The x3 class changed the policy's action in **zero** of 25 episodes while still recording
a non-zero perturbation, so an attacker that chose one of those three was scored as having
attacked when nothing reached the decision. `conflict`, `target` and `malfunction` were
identical to each other to six decimals, as were `speed` and `direction`: six choices,
three outcomes.

`disruption_reward()` pays for exactly the three inert ones. It rewards movement in
`dist_potential_conflict` (-> `conflict`), `dist_min_to_target` (-> `target`) and
`num_agents_malfunctioning` (-> `malfunction`), and pays nothing for `blank`, `speed` or
`direction`. The payout ordering — `target` ~ 1e5, `malfunction` ~ 5e2, `conflict` ~ 0 —
predicts what each attacker converged on (LambdaPIR: `target` 86%, `conflict` 11%). The
share of an attacker's choices falling in the inert class predicts its measured
effectiveness at Pearson -0.980, which accounts for the whole attacker ranking.

`perturbation_agents/maze_obs_layout.py` now resolves each corruption type to the maze
observation fields it is about, using maze-flatland's own
`GraphDirectionalObservationConversion.convert_to_dict()` key list at runtime. Neither the
vector width nor the field order is hardcoded here, so a change on the maze-flatland side
is picked up rather than silently mismatched. `target` went from 0.000 to 0.118 action
change, and all six types are now distinct.

This does not make the learned attackers win, and it was never going to: it restores the
semantic axis, while the table above shows the effective axis is magnitude. With it in
place LambdaPIR moves from completely ineffective (-0.020 arrival) to substantial
(-0.213), and remains behind the random baseline (-0.407) on a paired per-episode
comparison over the same seeds (0.273 vs 0.338 action change, Wilcoxon z = -1.86,
p = 0.064).

> The 200-episode table above predates this change: it was measured with the whole-vector
> corruption, so the three inert corruption types were still inert in it.

### When a robustness score is not a robustness score

Only agents that derive their actions from the observation can be scored meaningfully.
Every attacker perturbs the observation, so a policy reading environment state directly
(`ShortestPathDefender`) is unaffected by construction.

An unchanged action is evidence of robustness **only if the attack was actually applied
to the observation the policy read**. Two cases where it was not:

- **The perturbation never reached the policy.** `MazeFlatlandDefender`'s BC policy reads
  the maze observation (`GraphDirectionalObservationConversion` + `train_move_mask`, built
  from `FlatlandMazeState`), not the TreeObs the framework perturbs. In `obs_perturb` mode
  the attacker's chosen corruption is re-applied to that maze observation, so the attack
  does reach the policy and the KPI is valid. Where it cannot reach the policy at all, the
  applied perturbation is identically zero; `result_getter` records this per attacker as
  `attack_reached_policy=False` and prints a warning. **Report those KPIs as NOT
  APPLICABLE, not as high robustness.**
- **The attacker chose blind.** A query-based attacker such as GEPerturb scores candidate
  perturbations through `defender.act()`, which answers from the maze observation and
  ignores the TreeObs it is handed — so every candidate scores identically and its
  selection carries no gradient information. The attack it publishes *is* applied, but it
  was not informed. That is a limitation of the attacker against this defender, not
  robustness of the agent.

### Attack decision vs. attack application

The attacker decides once per environment step, in `perturb()`, and publishes the decision
as `attacker.last_targets = {handle: corruption_type}`. `MazeFlatlandDefender.act_and_step()`
only *applies* that decision; it never calls `perturb()` again and never re-chooses. The
perturbation reported for a step is the delta actually applied to the policy's own
observation, returned by `act_and_step()`, not the TreeObs delta.

An attacker that publishes `{}` corrupted nothing that step and nothing is applied. An
attacker that publishes nothing at all (`None`) leaves the defender no choice but to
corrupt every train, which is logged as a warning because it is not breadth-matched
against targeted attackers.

---

## Configuration

Default settings in `MultiAttackerRobustnessTestRunner`:

```python
ATTACKER_TYPES    = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC", "RLPerturb"]
NUM_EPISODES      = 50
BASELINE_DEFENDER = None      # no default; an evaluation without an agent fails loudly

# Flatland environment (matches the reference agent's training setup)
N_TRAINS   = 3
MAP_WIDTH  = 37
MAP_HEIGHT = 37
N_CITIES   = 2
TREE_DEPTH = 2
```

---

## Framework Structure

```
railway/
├── __init__.py
├── orchestrator.py                                   # Celery orchestrator
├── orchestrator_definitions.py                       # KPI UUID -> TestRunner wiring
├── test_runner_robustness_resilience_kpi_069_077.py  # These KPIs implementation
├── run_local_robustness_resilience_kpi_069_077.py   # Local runner (no FAB)
├── README_robustness_resilience_kpi_069_077.md       # This documentation
└── framework/
    ├── agent.zip                                     # Reference agent (see below)
    ├── Environment.py                                # FlatlandEnvironment wrapper
    ├── defenders.py                                  # Defender wrappers
    ├── defenders_maze.py                             # maze-flatland policy adapter
    ├── example_evaluation.py                         # Minimal usage example
    ├── attack_models/                                # Attacker implementations
    │   ├── BaseAgent.py
    │   ├── BaseAttackerClass.py
    │   ├── GEPerturbAttacker.py
    │   ├── LambdaPIRAttacker.py
    │   ├── PPOAttacker.py
    │   ├── RandomPerturbAttacker.py
    │   ├── RLPerturbAttacker.py
    │   └── SACAttacker.py
    ├── perturbation_agents/                          # Perturbation strategies
    ├── evaluation_framework/                         # Metrics computation
    │   ├── metrics.py
    │   ├── plots_and_tables.py
    │   └── result_getter.py
    └── utility/                                      # HTML report helpers
```

---

## Submission Format

Agents are wrapped in a `RailwayDefender` (see `framework/defenders.py`):

| Class | Wraps |
|-------|-------|
| `MazeFlatlandDefender` | a maze-flatland policy; drives its own maze environment |
| `ObsPolicyDefender` | any policy exposing `act_many(handles, obs_list) -> {handle: action}` |
| `ShortestPathDefender` | Flatland's `ShortestPathPolicy` (reads the env, not the observation) |
| `DoNothingDefender` | baseline that takes no action |

Two submission formats are accepted:

1. **Pickle file** (`.pkl`) containing either a `RailwayDefender` instance, used as-is, or a policy object exposing `act_many(handles, obs_list)`, which is wrapped in an `ObsPolicyDefender`.

2. **ZIP file** (`.zip`) containing either
   ```
   state_dict-epoch_N.pt        # maze-flatland checkpoint
   spaces_config.pkl
   ```
   which is wrapped in a `MazeFlatlandDefender`, or
   ```
   agent.pkl                    # or policy.pkl
   ```
   which is treated as case 1.

**Docker submissions are not supported for these KPIs.** The other railway KPIs (`KPI-PF-026`, `KPI-NF-045`) drive containerised policies via `flatland-trajectory-generate-from-policy`, which replays a finished trajectory. These KPIs perturb the agent's observation at every step and therefore require in-process inference.

### Reference agent (`framework/agent.zip`)

A trained maze-flatland behavioural-cloning policy ships with the framework, following the same convention as `power_grid/framework/agent.zip`:

```
state_dict-epoch_200.pt    trained policy weights
spaces_config.pkl          observation/action space config saved with the run
training_config.yaml       the run's hydra config
LICENSE.maze-flatland      MIT licence, (c) 2025 EnliteAI GmbH
```

`build_maze_env()` reproduces the environment this policy was trained in, which `training_config.yaml` documents: 3 trains, 37x37, 2 cities, malfunction rate 0, `DirectionalAC`, `GraphDirectionalObservationConversion`, `FlatlandMaskingWrapper` with `LogicMaskBuilder(mask_out_dead_ends=True, disable_stop_on_switches=False)`, and `SubStepSkippingWrapper(do_skipping_in_reset=True)`.

`MazeFlatlandDefender` sets `drives_env = True`: the maze environment internally advances a `RailEnv`, so `FlatlandEnvironment` drives each transition through `act_and_step()` and reads tree observations from that same `RailEnv`. The environment is seeded per episode so that every submission is scored on the same scenarios.

---

## Usage

### Local

```bash
cd ai4realnet_orchestrators/railway

# The bundled reference agent, 5 episodes, report KPI-VF-073
python run_local_robustness_resilience_kpi_069_077.py --agent framework/agent.zip

# Any KPI, more episodes, print all 10 values from the single run
python run_local_robustness_resilience_kpi_069_077.py --kpi KPI-FF-070 --episodes 50 --all

# Your own submission
python run_local_robustness_resilience_kpi_069_077.py --agent path/to/agent.pkl --all
```

A 50-episode run against the reference agent takes roughly 5 minutes, followed by the per-episode plots (`episodes x attackers x 2` SVG files), which dominate the wall time.

### Via the orchestrator

The orchestrator routes submissions to the appropriate KPI test automatically:

```python
# Example: Evaluating KPI-VF-073 (Vulnerability)
result = run_scenario(
    scenario_id="47a11418-fad5-4d55-a637-6b90a8351500",
    submission_id="your-submission-uuid"
)
# Returns: {"primary": 0.08}  # 8% vulnerability
```

---

## Metrics Interpretation

| Metric | Good Value | Bad Value | Interpretation |
|--------|------------|-----------|----------------|
| `perturb_vulnerability` | Low (0.0) | High (1.0) | Lower = more robust |
| `n_steps_survived` | High | Low (0) | Higher = more robust |
| `severity_of_change` | Low (0.0) | High (1.0) | Lower = more stable |
| `reward_drop_percent` | Low (0%) | High (100%) | Lower = maintains performance |
| `action_change_freq` | Low (0.0) | High (1.0) | Lower = more consistent |
| `area_between_curves` | Low | High | Lower = faster recovery |
| `degradation_time` | High | Low | Higher = slower degradation |
| `restoration_time` | Low | High | Lower = faster recovery |
| `state_similarity` | High (1.0) | Low (-1.0) | Higher = maintains state |
| `reward_per_action_ratio` | High (1.0) | Low (0.0) | Higher = value per intervention preserved |

---

## Railway-specific metric definitions

Three metrics are defined differently from the power grid implementation, because Flatland's reward is sparse and non-positive: it is `0.0` at every step with a single terminal penalty, and exactly `0` for an agent that delivers every train. Each divergence is marked `DIVERGENCE FROM POWER GRID` in the source.

**`reward_drop_percent` (KPI-DF-069)** is normalised by the worst reward reachable over the same number of steps, `N_TRAINS * total_steps`, rather than by the unperturbed total, which is `0` for a successful agent.

**`severity_of_change` (KPI-SF-071)** is divided by the number of changed actions. `metrics.similarity_score` accumulates a sum over changed steps rather than a mean, so the unnormalised value leaves the declared `[0, 1]` range once an attacker changes more than one action.

**The resilience KPIs (AF-074, DF-075, RF-076) and KPI-RF-078** are computed from the fraction of trains that have arrived, recorded per step by `FlatlandEnvironment`, rather than from the reward. The arrivals curve varies through the episode and falls behind the baseline when an attack delays a train, whereas a sparse terminal reward carries no such shape. `get_resilience_metrics_performance_single_ep()` measures the gap between the baseline and perturbed arrivals curves directly: degradation time runs from the gap opening to its widest point, restoration time from that point until it closes (censored at the end of the episode), and the area is the integral of the gap.

`reward_per_action_ratio` is capped at 1.0 for scoring: an attack that suppresses the agent's interventions without reducing trains delivered would otherwise report a value above 1. The uncapped value remains available on `metrics.reward_per_action["reward_per_action_ratio"]`.

Pickles written before the performance signal existed fall back to the reward-curve behaviour, so earlier results still load.

---

## Contact

For questions about this KPI implementation, contact INESC TEC.
