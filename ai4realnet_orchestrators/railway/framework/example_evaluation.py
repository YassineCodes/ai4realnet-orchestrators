"""
Example: evaluate a Flatland railway agent with all available attackers.

Run from the framework/ directory:
    python example_evaluation.py

Covers all six attackers used by the robustness/resilience KPIs. None of them
needs a pre-trained model: the learning attackers train online during evaluation.

  - RandomPerturbAttacker  (random sensor blackout)
  - LambdaPIRAttacker      (Bellman-proven hybrid PIR/VIR)
  - GEPerturbAttacker      (gradient estimation, queries the defender)
  - RLPerturbAttacker      (DQN)
  - PPOAttacker            (actor-critic)
  - SACAttacker            (discrete soft actor-critic)

The defender must READ THE OBSERVATION for any of this to mean anything. These
attackers perturb the observation, so a policy that queries the RailEnv directly
(ShortestPathDefender) or reads its own internal state (MazeFlatlandDefender) is
immune by construction and will score a perfect but uninformative result.
"""

import os
import tempfile

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv

from Environment import FlatlandEnvironment
from defenders import ObsPolicyDefender
from attack_models.RandomPerturbAttacker import RandomPerturbAttacker
from attack_models.LambdaPIRAttacker import LambdaPIRAttacker
from attack_models.RLPerturbAttacker import RLPerturbAttacker
from attack_models.GEPerturbAttacker import GEPerturbAttacker
from attack_models.PPOAttacker import PPOAttacker
from attack_models.SACAttacker import SACAttacker
from evaluation_framework.result_getter import result_getter

N_AGENTS = 4
N_EPISODES = 5

# ── 1. Build the Flatland environment ──────────────────────────────────────────
obs_builder = TreeObsForRailEnv(max_depth=2)

rail_env = RailEnv(
    width=25,
    height=25,
    rail_generator=sparse_rail_generator(max_num_cities=3, seed=42),
    line_generator=sparse_line_generator(),
    number_of_agents=N_AGENTS,
    obs_builder_object=obs_builder,
)

# ── 2. Wrap with the framework ─────────────────────────────────────────────────
# Wrap any policy exposing act_many(handles, obs_list); see README.
# defender = ObsPolicyDefender(my_policy)
from defenders_maze import load_maze_policy, build_maze_env, MazeFlatlandDefender
maze_env = build_maze_env(n_trains=N_AGENTS)
policy = load_maze_policy(CHECKPOINT, SPACES_CONFIG)
defender = MazeFlatlandDefender(maze_env, policy, n_trains=N_AGENTS)
env = FlatlandEnvironment(rail_env, defender)

# ── 3. Define attackers ────────────────────────────────────────────────────────
attackers = [
    # Random observation blackouts at different rates
    RandomPerturbAttacker(perturbation_rate=0.05, seed=0),
    RandomPerturbAttacker(perturbation_rate=0.20, seed=1),
    RandomPerturbAttacker(perturbation_rate=0.50, seed=2),

    # Lambda-PIR: directed hybrid attacker (trains online, no model needed)
    LambdaPIRAttacker(n_agents=N_AGENTS, gamma=0.99, seed=10),
    LambdaPIRAttacker(n_agents=N_AGENTS, gamma=0.99, lambda_param=0.95, seed=11),

    # DQN attacker: learns online during evaluation
    RLPerturbAttacker(n_agents=N_AGENTS, epsilon=1.0, seed=20),

    # Gradient estimation: queries the defender to pick the worst perturbation
    GEPerturbAttacker(defender=defender, n_agents=N_AGENTS, n_candidates=8, seed=30),

    # Actor-critic attackers: also train online, no pre-trained model needed
    PPOAttacker(n_agents=N_AGENTS, gamma=0.99, entropy_coef=0.05, seed=40),
    SACAttacker(n_agents=N_AGENTS, gamma=0.99, alpha=0.2, seed=50),
]

# ── 4. Run evaluation (all KPIs: robustness + resilience) ─────────────────────
rg = result_getter(
    env=env,
    defender=defender,
    n_episodes=N_EPISODES,
    # Write outside the package so the framework folder stays free of results
    save_folder=os.path.join(tempfile.gettempdir(), "railway_eval"),
    attackers=attackers,
)

metrics_list = rg.calculate_metrics()

# ── 5. Print summary ───────────────────────────────────────────────────────────
for m in metrics_list:
    print(f"\n{'='*60}")
    print(f"Attacker: {m.model_name}")
    print("\nRobustness:")
    print(m.metrics_robustness.to_string(index=False))
    print("\nResilience (reward-based):")
    print(m.metrics_resilience.to_string())
