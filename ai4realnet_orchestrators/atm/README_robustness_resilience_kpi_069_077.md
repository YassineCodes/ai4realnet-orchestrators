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
| KPI-DF-075 | `2baef867-c1f2-4b6e-b13c-0eac9463c2fa` | `degradation_time` | Time until performance degrades |
| KPI-RF-076 | `7b15a7b3-2413-4953-b91a-24f5c0c5b6da` | `restoration_time` | Time to restore performance |
| KPI-SF-077 | `e3fb76a2-2121-4889-adf2-b60ca29c5c71` | `state_similarity` | Cosine similarity to unperturbed states [-1 to 1] |

---

## Environment

[BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym) `StaticObstacleEnv-v0`, the
single-agent ATM control task: one aircraft must reach a destination waypoint while avoiding
restricted areas.

- **Observation** — `gymnasium.spaces.Dict`: `destination_waypoint_distance`,
  `destination_waypoint_cos_drift`, `destination_waypoint_sin_drift` (1 each),
  `restricted_area_radius`, `restricted_area_distance`,
  `cos_difference_restricted_area_pos`, `sin_difference_restricted_area_pos`
  (`NUM_OBSTACLES` = 10 each).
- **Action** — `Box(-1, 1, (2,))`: heading change and speed change.
- **Reward** — one scalar: `REACH_REWARD` (+1) on reaching the waypoint,
  `DRIFT_PENALTY` (-0.01) per unit drift, `RESTRICTED_AREA_INTRUSION_PENALTY` (-5) per
  intrusion.

## What's in `atm/framework/`

| path | purpose |
|------|---------|
| `Environment.py` | `BlueSkyGymEnvironment`, exposing the interface `result_getter` and `metrics` expect |
| `defenders.py` | `SB3Defender` (SAC/TD3/PPO/DDPG), `StraightLineDefender` |
| `agent.zip` | trained reference agent |
| `attack_models/` | seven attackers on a common `BaseAgent` / `BaseAttackerClass` |
| `perturbation_agents/` | the perturbation strategies behind them |
| `evaluation_framework/` | `metrics`, `result_getter`, `plots_and_tables` |

`evaluation_framework/` and `utility/` are byte-identical to the railway domain's copies.

## Reference agent

`framework/agent.zip` is a Stable-Baselines3 SAC policy (`MultiInputPolicy`) trained on
`StaticObstacleEnv-v0` for 2,000,000 timesteps (24,440 episodes, seed 42, learning rate 3e-4,
`learning_starts` 10,000, `max_episode_steps` 200), following the hyper-parameters of
BlueSky-Gym's own `main.py`. Evaluated over 10 fresh episodes it returns -0.42 ± 2.46.

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

## Measured results

50 episodes per attacker against the bundled reference agent (baseline: 4394 steps over 50
episodes):

| Attacker | `action_change_freq` | `perturb_vulnerability` | `reward_per_action_ratio` | area |
|----------|---------------------|-------------------------|---------------------------|------|
| GEPerturb | 1.0000 | 0.9994 | 1.0000 | 706.8 |
| LambdaPIR | 1.0000 | 0.3768 | 1.0000 | 4846.4 |
| Random | 1.0000 | 1.0000 | 1.0000 | 4749.6 |
| PPO | 1.0000 | 0.6216 | 0.9991 | 6954.2 |
| SAC_10 | 1.0000 | 0.7787 | 0.9536 | 5493.5 |
| SAC_5 | 1.0000 | 0.7585 | 1.0000 | 2580.1 |
| RLPerturb | 1.0000 | 0.7911 | 0.9804 | 3542.7 |

Aggregated: drop-off in reward 0.181, action change frequency 1.000, severity 0.095, steps
survived 77.4, vulnerability 0.761, area 4124.8, degradation time 27.3, restoration time 3.09,
state similarity 0.377, reward-per-action ratio 0.990.

`action_change_freq` is 1.0000 for every attacker: the action space is continuous, so any
non-zero perturbation changes the action vector at every step.

## Running locally

```
pip install bluesky-gym stable-baselines3
python test_local_robustness_resilience_kpi_069_077.py --model framework/agent.zip --episodes 50
```

Options: `--episodes`, `--xi`, `--attackers`, `--kpi`, `--verbose`.
