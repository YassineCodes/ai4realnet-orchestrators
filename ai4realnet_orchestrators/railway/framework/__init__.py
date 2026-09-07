"""
Railway robustness/resilience framework.

The modules here import each other by bare name (`from Environment import ...`), which
assumes this directory is on sys.path -- that is how every caller uses the framework
(the test runners insert it explicitly). Ensure it here as well so the package is also
importable the ordinary way, e.g. as
`ai4realnet_orchestrators.railway.framework`, which is what pytest does when it walks
up the chain of __init__.py files. This is a no-op when the path is already set.
"""

import os as _os
import sys as _sys

_FRAMEWORK_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _FRAMEWORK_DIR not in _sys.path:
    _sys.path.insert(0, _FRAMEWORK_DIR)

from Environment import FlatlandEnvironment
from defenders import (
    RailwayDefender,
    ShortestPathDefender,
    ObsPolicyDefender,
    DoNothingDefender,
)
# Maze-flatland BC defender (optional — requires maze-rl + trained checkpoint)
try:
    from defenders_maze import MazeFlatlandDefender, load_maze_policy, build_maze_env
except ImportError:
    pass

# Base classes
from attack_models.BaseAgent import BaseAgent
from attack_models.BaseAttackerClass import BaseAttackerClass

# Attackers — mirror of power-grid attack_models
from attack_models.RandomPerturbAttacker import RandomPerturbAttacker
from attack_models.LambdaPIRAttacker import LambdaPIRAttacker
from attack_models.GEPerturbAttacker import GEPerturbAttacker
from attack_models.RLPerturbAttacker import RLPerturbAttacker
# PPOAttacker and SACAttacker require stable-baselines3 + a trained Flatland model
# from attack_models.PPOAttacker import PPOAttacker
# from attack_models.SACAttacker import SACAttacker

# Perturbation agents
from perturbation_agents.base_perturb_agent import BasePerturbationAgent, NullPerturbationAgent
from perturbation_agents.random_perturb_agent import RandomPerturbationAgent
from perturbation_agents.lambda_pir_perturbation_agent import LambdaPIRPerturbationAgent
from perturbation_agents.grad_est_perturb_agent import GradientEstimationPerturbationAgent
from perturbation_agents.rl_perturb_agent import RLPerturbationAgent

# Evaluation
from evaluation_framework.result_getter import result_getter
from evaluation_framework.metrics import metrics

__all__ = [
    # Environment
    "FlatlandEnvironment",
    # Defenders
    "RailwayDefender", "ShortestPathDefender", "ObsPolicyDefender", "DoNothingDefender",
    "MazeFlatlandDefender", "load_maze_policy", "build_maze_env",
    # Attack models
    "BaseAgent", "BaseAttackerClass",
    "RandomPerturbAttacker", "LambdaPIRAttacker", "GEPerturbAttacker", "RLPerturbAttacker",
    # Perturbation agents
    "BasePerturbationAgent", "NullPerturbationAgent",
    "RandomPerturbationAgent", "LambdaPIRPerturbationAgent",
    "GradientEstimationPerturbationAgent", "RLPerturbationAgent",
    # Evaluation
    "result_getter", "metrics",
]
