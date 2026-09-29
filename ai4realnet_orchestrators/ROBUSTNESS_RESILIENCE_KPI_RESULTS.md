# Robustness & Resilience KPIs — measured results

KPI-DF-069, FF-070, SF-071, SF-072, VF-073, AF-074, DF-075, RF-076, SF-077 and RF-078,
measured in the three domains against their reference agents.

**Run configuration**

| | power_grid | railway | atm |
|---|---|---|---|
| Environment | Grid2Op, ICAPS 2021 | Flatland, 3 trains | BlueSky-Gym, `StaticObstacleEnv-v0` |
| Reference agent | `framework/agent.zip` (CurriculumAgent) | `framework/agent.zip` (maze-flatland BC policy) | `framework/agent.zip` (SAC, 2 000 000 steps) |
| Episodes per attacker | 50 | 50 | 50 |
| Attackers | 7 | 6 | 7 |
| Rollouts (incl. unperturbed baseline) | 400 | 350 | 400 |

Each attacker is evaluated on the same 50 episodes as the unperturbed baseline. Reported KPI
values are the mean across attackers, as defined in
`power_grid/test_runner_robustness_resilience_kpi_069_077.py`.

---

## Aggregated KPI values

| KPI | power_grid | railway | atm |
|-----|-----------:|--------:|----:|
| DF-069 Drop-off in reward | 0.0000 | 0.1612 | 0.1813 |
| FF-070 Frequency changed output | 1.7773 | 0.3690 | 1.0000 |
| SF-071 Severity of changed output | −0.4826 | 0.3392 | 0.0947 |
| SF-072 Steps survived | 656.81 | 29.46 | 77.43 |
| VF-073 Vulnerability to perturbation | 0.2244 | 0.0697 | 0.7609 |
| AF-074 Area between curves | 0.0000 | 4232.07 | 4124.78 |
| DF-075 Degradation time | 0.0000 | 3.25 | 27.32 |
| RF-076 Restorative time | 0.0000 | 0.97 | 3.09 |
| SF-077 Similarity to unperturbed state | 0.9923 | 0.9522 | 0.3773 |
| RF-078 Reward per action | — | 0.5372 | 0.9904 |

---

## power_grid — per attacker (50 episodes)

| Attacker | Vulnerability | Steps survived | Severity | Reward drop % | Action change freq | Area | Degradation | Restoration | State similarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GEPerturb | 0.2383 | 809.0 | −0.1900 | 0.00 | 0.2948 | 0.0000 | 0.0 | 0.0 | 0.9952 |
| LambdaPIR | 0.6125 | 161.2 | −3.4262 | 0.00 | 0.9891 | 0.0000 | 0.0 | 0.0 | 0.9844 |
| RPerturb | 0.0006 | 887.0 | 0.8547 | 0.00 | 0.1922 | 0.0000 | 0.0 | 0.0 | 0.9942 |
| PPO | 0.0282 | 854.8 | 1.0000 | 0.00 | 8.6651 | 0.0000 | 0.0 | 0.0 | 0.9959 |
| SAC_10 | 0.0826 | 652.2 | −0.2336 | 0.00 | 0.8352 | 0.0000 | 0.0 | 0.0 | 0.9926 |
| SAC_5 | 0.0315 | 931.8 | 0.5743 | 0.00 | 0.6846 | 0.0000 | 0.0 | 0.0 | 0.9945 |
| RLPerturb | 0.5773 | 301.8 | −1.9573 | 0.00 | 0.7798 | 0.0000 | 0.0 | 0.0 | 0.9891 |

`severity_of_change` is computed as `1 − similarity_score`, where `similarity_score` is the
per-episode **sum** of per-step similarities; values fall below 0 when more than one action
changes per episode. `action_change_freq` is `n_actions_changed / n_steps_with_act`; the two
counters differ in unit for PPO. `reward_drop_percent` is returned as 0.0 when the unperturbed
total reward is not strictly positive.

## railway — per attacker (50 episodes)

Reference agent delivers all three trains in 45 of 50 unperturbed episodes.

| Attacker | Vulnerability | Steps survived | Severity | Reward drop % | Action change freq | Area | Degradation | Restoration | State similarity | Reward/action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GEPerturb | 0.0558 | 23.6 | 0.3333 | 0.00 | 0.0696 | 2362.76 | 1.5 | 0.6 | 0.9724 | 0.8599 |
| LambdaPIR | 0.0491 | 37.6 | 0.3333 | 0.00 | 0.8414 | 5091.36 | 4.4 | 1.0 | 0.9417 | 0.3613 |
| Random | 0.1563 | 28.0 | 0.3688 | 0.14 | 0.2750 | 4845.24 | 3.3 | 1.2 | 0.9473 | 0.4461 |
| PPO | 0.0069 | 28.6 | 0.3333 | 0.11 | 0.2603 | 3767.20 | 3.3 | 1.1 | 0.9541 | 0.5839 |
| SAC | 0.0204 | 28.6 | 0.3333 | 0.03 | 0.3718 | 4178.34 | 3.4 | 0.9 | 0.9511 | 0.5022 |
| RLPerturb | 0.1295 | 30.5 | 0.3333 | 0.68 | 0.3961 | 5147.54 | 3.6 | 1.1 | 0.9467 | 0.4700 |

## atm — per attacker (50 episodes)

Unperturbed baseline: 4394 steps over 50 episodes. Perturbations are bounded by an
L-infinity budget ξ = 0.1 per observation feature (SAC_5: ξ = 0.05); the realised maximum
perturbation is recorded per run and matched the budget in every case.

| Attacker | Vulnerability | Steps survived | Severity | Reward drop % | Action change freq | Area | Degradation | Restoration | State similarity | Reward/action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GEPerturb | 0.9994 | 68.8 | 0.2320 | 0.27 | 1.0000 | 706.83 | 23.5 | 2.9 | 0.3524 | 1.0000 |
| LambdaPIR | 0.3768 | 83.7 | 0.0895 | 0.08 | 1.0000 | 4846.39 | 26.4 | 5.6 | 0.3890 | 1.0000 |
| Random | 1.0000 | 76.9 | 0.1126 | 0.12 | 1.0000 | 4749.55 | 31.7 | 2.2 | 0.3804 | 1.0000 |
| PPO | 0.6216 | 77.0 | 0.0580 | 0.13 | 1.0000 | 6954.24 | 31.3 | 3.0 | 0.3545 | 0.9991 |
| SAC_10 | 0.7787 | 82.0 | 0.0764 | 0.23 | 1.0000 | 5493.54 | 32.1 | 2.7 | 0.4147 | 0.9536 |
| SAC_5 | 0.7585 | 76.8 | 0.0388 | 0.24 | 1.0000 | 2580.14 | 23.3 | 1.7 | 0.3761 | 1.0000 |
| RLPerturb | 0.7911 | 76.6 | 0.0557 | 0.19 | 1.0000 | 3542.74 | 23.0 | 3.6 | 0.3742 | 0.9804 |

Episode length per attacker, against the 4394-step baseline: GEPerturb 3442, RLPerturb 3832,
SAC_5 3842, Random 3847, PPO 3849, SAC_10 4100, LambdaPIR 4187.

`action_change_freq` is 1.0000 for every attacker: the action space is continuous
(`Box(-1, 1, (2,))`), so any non-zero perturbation changes the action vector at every step.

---

## Cross-domain observations

- **LambdaPIR produces the highest action-change frequency against both discrete-action
  defenders**: 0.9891 (power_grid) and 0.8414 (railway), in both cases the highest of the
  suite. It perturbs one observation feature per step.
- **GEPerturb is the lowest on railway (0.0696) yet still produces area 2362.8**, and on ATM it
  reaches the highest vulnerability (0.9994) with the lowest area (706.8).
- **Reported KPI values are means across attackers.** For railway the reported action-change
  frequency is 0.3690 while the per-attacker maximum is 0.8414; for power_grid the reported
  value is 1.7773 while five of seven attackers are below 1.0.
- **SF-077 separates the domains**: 0.99 (power_grid), 0.95 (railway), 0.38 (ATM), reflecting
  observation dimensionality and the fraction of features perturbed.

## Reproducing

```
# power_grid / atm
python run_all_kpis_local.py --episodes 50
python atm/test_local_robustness_resilience_kpi_069_077.py --model <agent.zip> --episodes 50

# railway
python railway/test_local_robustness_resilience_kpi_069_077.py \
    --episodes 50 --all --agent framework/agent.zip
```
