"""
ATM defenders for the robustness/resilience framework.

A defender exposes `act(obs_dict) -> np.ndarray` over the BlueSky-Gym Dict observation
and returns a continuous action in [-1, 1]^2 (heading change, speed change).

SB3Defender wraps a Stable-Baselines3 policy, which is what the AI4REALNET ATM agents
are (SAC/TD3/PPO/DDPG with MultiInputPolicy). StraightLineDefender is the do-nothing
reference used when no submission is supplied.

Author: INESC TEC
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Algorithms the ATM deployment plugin supports (READ_ME of ai4realnet_deploy_RL_tools_batch).
_ALGOS = ("SAC", "TD3", "PPO", "DDPG")


class ATMDefender:
    """Interface every ATM defender implements."""

    name = "ATMDefender"

    def act(self, obs_dict):
        raise NotImplementedError

    def reset(self):
        pass


class SB3Defender(ATMDefender):
    """
    A trained Stable-Baselines3 policy.

    Args:
        model_path: Path to the SB3 .zip (with or without the extension).
        algorithm:  One of SAC/TD3/PPO/DDPG. When None, each is tried in turn, because
                    the shipped ATM checkpoints do not record which algorithm wrote them
                    outside their own metadata.
        deterministic: Use the deterministic policy output (as the deployment plugin does:
                    `model.predict(obs, deterministic=True)`).
    """

    def __init__(self, model_path, algorithm=None, deterministic=True):
        import stable_baselines3 as sb3

        self.deterministic = deterministic
        candidates = [algorithm] if algorithm else list(_ALGOS)
        errors = []
        self.model = None
        for algo_name in candidates:
            try:
                algo = getattr(sb3, algo_name)
            except AttributeError:
                errors.append(f"{algo_name}: not available in stable_baselines3")
                continue
            try:
                # env=None: the policy is only queried for actions here; the evaluation
                # environment is driven by BlueSkyGymEnvironment.
                self.model = algo.load(model_path, env=None, device="cpu")
                self.algorithm = algo_name
                break
            except Exception as e:  # wrong algorithm for this checkpoint, or a load error
                errors.append(f"{algo_name}: {type(e).__name__}: {e}")
        if self.model is None:
            raise RuntimeError(f"Could not load an SB3 policy from {model_path}. Tried: " + "; ".join(errors))

        self.name = f"SB3Defender({self.algorithm})"
        logger.info(f"Loaded {self.algorithm} policy from {model_path}")

    def act(self, obs_dict):
        action, _ = self.model.predict(obs_dict, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float64).ravel()


class StraightLineDefender(ATMDefender):
    """
    Reference agent: never changes heading or speed.

    Used when no submission is given, so the KPI pipeline can run end to end. Its values
    describe this baseline, not a submitted agent, and must not be reported as an
    evaluation of one.
    """

    name = "StraightLineDefender"

    def act(self, obs_dict):
        return np.zeros(2, dtype=np.float64)
