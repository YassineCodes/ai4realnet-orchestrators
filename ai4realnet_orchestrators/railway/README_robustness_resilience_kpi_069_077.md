# Robustness & Resilience KPIs for Railway (KPI-DF-069, KPI-FF-070, KPI-SF-071, KPI-SF-072, KPI-VF-073, KPI-AF-074, KPI-DF-075, KPI-RF-076, KPI-SF-077, KPI-RF-078)

## Overview

This module implements **10 KPIs** for evaluating AI agent robustness and resilience in Flatland railway traffic management scenarios. The evaluation uses multiple adversarial attack strategies to stress-test defender agents.

**Author:** INESC TEC
**Test Runner:** `test_runner_robustness_resilience_kpi_069_077.py`
**Local runner:** `test_local_robustness_resilience_kpi_069_077.py`

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

50 episodes against the bundled reference agent (`framework/agent.zip`). The unperturbed baseline delivers all three trains in 45 of 50 episodes, mean arrival 0.93.

| Attacker | `action_change_freq` | `perturb_vulnerability` | `reward_per_action_ratio` | `degradation_time` | `area_between_curves` |
|----------|---------------------|-------------------------|---------------------------|--------------------|-----------------------|
| LambdaPIR | 0.841 | 0.049 | 0.361 | 4.40 | 5091 |
| SAC | 0.339 | 0.036 | 0.557 | 3.88 | 4378 |
| RLPerturb | 0.336 | 0.122 | 0.597 | 2.82 | 4303 |
| Random | 0.275 | 0.156 | 0.446 | 3.34 | 4845 |
| PPO | 0.233 | 0.042 | 0.663 | 3.08 | 4049 |
| GEPerturb | 0.070 | 0.056 | 0.860 | 1.50 | 2363 |

Aggregated across all six attackers:

| KPI | Value |
|-----|-------|
| `perturb_vulnerability` | 0.0770 |
| `n_steps_survived` | 28.93 |
| `severity_of_change` | 0.3392 |
| `reward_drop_percent` | 0.1469 |
| `action_change_freq` | 0.3489 |
| `area_between_curves` | 4171.53 |
| `degradation_time` | 3.17 |
| `restoration_time` | 0.9733 |
| `state_similarity` | 0.9539 |
| `reward_per_action_ratio` | 0.5808 |

Only agents that derive their actions from the observation can be scored meaningfully. Every attacker perturbs the observation, so a policy reading environment state directly (`ShortestPathDefender`) is unaffected by construction. The same applies to query-based attackers such as GEPerturb against a defender that consumes a different observation format: its score reflects that structural property rather than the agent's robustness.

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
├── test_local_robustness_resilience_kpi_069_077.py   # Local runner (no FAB)
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
python test_local_robustness_resilience_kpi_069_077.py --agent framework/agent.zip

# Any KPI, more episodes, print all 10 values from the single run
python test_local_robustness_resilience_kpi_069_077.py --kpi KPI-FF-070 --episodes 50 --all

# Your own submission
python test_local_robustness_resilience_kpi_069_077.py --agent path/to/agent.pkl --all
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
