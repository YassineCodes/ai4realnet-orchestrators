# FAB Power Grid Orchestrator - Deployment Guide

## Overview

This orchestrator evaluates **defender agents** against multiple **adversarial attackers** to compute **9 robustness/resilience KPIs** for the AI4RealNet FAB platform.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10.18 | **EXACT version required** |
| OS | Ubuntu 22.04/24.04 | Linux recommended |
| Disk Space | 15 GB | For all dependencies |
| RAM | 16 GB | Recommended |
| GPU | Optional | CUDA 11.8+ for GPU acceleration |

## Quick Start

```bash
# 1. Ensure Python 3.10.18
python --version  # Should show 3.10.18

# 2. Navigate to framework directory
cd /path/to/ai4realnet-orchestrators/ai4realnet_orchestrators/power_grid/framework

# 3. Run setup script
python setup_fab_power_grid.py

# 4. Test locally
python test_local.py

# 5. Run production orchestrator
python -m ai4realnet_orchestrators.power_grid.orchestrator
```

## Directory Structure

```
power_grid/
├── power_grid_test_runner.py     # Main test runner (9 KPIs)
├── orchestrator.py               # FAB RabbitMQ orchestrator
├── test_local.py                 # Local testing script
├── setup_fab_power_grid.py       # Environment setup script
├── requirements.txt              # Python dependencies
│
└── framework/                    # Robustness/Resilience Framework
    │
    ├── attack_models/            # Attacker implementations
    │   ├── Environment.py        # Grid2Op wrapper
    │   ├── SACAttacker.py        # SAC-based attacker
    │   ├── PPOAttacker.py        # PPO-based attacker
    │   ├── RLPerturbAttacker.py  # RL perturbation agent
    │   ├── GEPerturbAttacker.py  # Gradient estimation 
    │   ├── RPerturbAttacker.py   # Random perturbation
    │   └── LambdaPIRAttacker.py  # Lambda-PIR hybrid
    │
    ├── evaluation_framework/     # Metrics computation
    │   ├── result_getter.py      # Episode runner
    │   └── metrics.py            # KPI calculations
    │
    ├── modified_curriculum_classes/  # Agent loading
    │   ├── baseline.py           # CurriculumAgent class
    │   ├── my_agent.py           # TensorFlow model wrapper
    │   ├── obs_converter.py      # Observation conversion
    │   └── utilities.py          # Helper functions
    │
    ├── perturbation_agents/      # Perturbation logic
    │
    └── trained_models/           # Pre-trained attacker models
        ├── SAC.zip               # SAC model (used by SAC_5, SAC_10, LambdaPIR)
        ├── PPO.zip               # PPO model
        └── RLPerturbAgent/
            ├── trained_rlpa_0.pth
            └── trained_rlpa_target_net_0.pth
```

## KPIs Implemented

### Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)

| KPI ID | Name | Metric | Range |
|--------|------|--------|-------|
| b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920 | KPI-VF-073 | Vulnerability to perturbation | [0-1] |
| a121d8bd-1943-41ba-b3a7-472a0154f8f9 | KPI-SF-072 | Steps survived | count |
| 3d033ec6-942a-4b03-b26e-f8152ba48022 | KPI-SF-071 | Severity of change | [0-1] |
| 1cbb7783-47b4-4289-9abf-27939da69a2f | KPI-DF-069 | Drop-off in reward | [0-100%] |
| acaf712a-c06c-4a04-a00f-0e7feeefb60c | KPI-FF-070 | Frequency changed output | [0-1] |

### Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)

| KPI ID | Name | Metric | Range |
|--------|------|--------|-------|
| 534f5a1f-7115-48a5-b58c-4deb044d425d | KPI-AF-074 | Area between curves | area units |
| 04a23bfc-fc44-4ec4-a732-c29214130a83 | KPI-DF-075 | Degradation time | steps |
| 225aaee8-7c7f-4faf-810b-407b551e9f2a | KPI-RF-076 | Restorative time | steps |
| 7fe4210f-1253-411c-ba03-49d8b37c71fa | KPI-SF-077 | State similarity | [-1 to 1] |

## Attackers

| Attacker | Type | Description |
|----------|------|-------------|
| SAC_5 | Learned | SAC model with factor=5 (moderate) |
| SAC_10 | Learned | SAC model with factor=10 (aggressive) |
| PPO | Learned | PPO-based attacker |
| GEPerturb | Learned | Gradient Estimation based attacker |
| RLPerturb | Learned | RL perturbation agent |
| Random | Random | Random perturbations (prob=0.6) |
| LambdaPIR | Hybrid | Lambda-PIR with gradient refinement |

**Note:** GEPerturb is disabled due to TensorFlow SavedModel compatibility issues.

## Configuration

Edit `power_grid_test_runner.py`:

```python
class MultiAttackerRobustnessTestRunner(TestRunner):
    # Evaluation configuration
    ATTACKER_TYPES = ["SAC_5", "SAC_10", "PPO", "GEPerturb", "RLPerturb", "Random", "LambdaPIR"]
    NUM_EPISODES = 30        # 30 for production, 2 for testing
    ENV_NAME = "l2rpn_case14_sandbox"  # or "l2rpn_icaps_2021"
```

## RabbitMQ Configuration

Set environment variables before running:

```bash
export FAB_RABBITMQ_HOST=ai4realnet-rabbitmq.flatland.cloud
export FAB_RABBITMQ_PORT=5672
export FAB_RABBITMQ_USER=<your_username>
export FAB_RABBITMQ_PASS=<your_password>
```

## Installation Steps

The `setup_fab_power_grid.py` script installs:

1. **PyTorch 2.8.0 + torch-geometric 2.6.1**
2. **requirements.txt** (Grid2Op 1.10.5, LightSim2Grid, PandaPower, etc.)
3. **TensorFlow 2.12.1**
4. **Ray 2.5.1 + Gymnasium + Stable-Baselines3**
5. **typing-extensions** (upgraded to fix conflicts)
6. **NNI 2.10.1**
7. **Pika** (RabbitMQ client)

## Troubleshooting

### Common Issues

1. **llvmlite/numba conflicts**
   ```bash
   pip install --force-reinstall --no-deps llvmlite numba --break-system-packages
   ```

2. **TensorFlow model loading errors**
   - Check that `my_agent.py` calls model with only `observations` parameter
   - No `timestep` or other extra arguments

3. **Resilience metrics column errors**
   - Different attackers produce different column names (`area` vs `area_per_1000_steps`)
   - `_aggregate_metrics()` handles this automatically

4. **Import errors**
   - Ensure `framework/` is in `sys.path`
   - Check that `power_grid_test_runner.py` uses `FRAMEWORK_PATH` correctly

### Debug Logging

Enable debug logging in `test_local.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Suppress noisy loggers
logging.getLogger("pandapower").setLevel(logging.ERROR)
logging.getLogger("grid2op").setLevel(logging.WARNING)
```

## Example Output

```
============================================================
ATTACKER: SAC_5
============================================================
  Robustness Metrics:
    - Vulnerability:        0.2622
    - Steps Survived:       816.4
    - Severity of Change:   0.0023
    - Reward Drop (%):      52.28
    - Action Change Freq:   0.1215
  Resilience Metrics:
    - Area Between Curves:  21.49
    - Degradation Time:     15.8
    - Restoration Time:     202.8
    - State Similarity:     0.9977

============================================================
AGGREGATED METRICS (Average across 6 attackers)
============================================================
  perturb_vulnerability: 0.2622
  n_steps_survived: 816.4167
  severity_of_change: 1.0000
  reward_drop_percent: 52.2830
  action_change_freq: 12.1520
  area_between_curves: 21.4876
  degradation_time: 15.8333
  restoration_time: 202.8333
  state_similarity: 0.9977
```

## Contact

For issues, contact the INESC TEC@AI4RealNet team.
