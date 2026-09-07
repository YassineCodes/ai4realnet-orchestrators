"""
Random Perturbation Agent for Flatland.

Equivalent of the power-grid RandomPerturbationAgent.

Implements the same "sensor blackout" mechanism as Flatland's built-in
perturbation_tree_observation_builder_wrapper: agents are randomly made
blind for a Poisson-sampled duration. Additionally supports targeted
single-feature corruptions (conflict, target distance, speed, direction).
"""

import copy
from typing import Dict, Optional

import numpy as np

from perturbation_agents.base_perturb_agent import BasePerturbationAgent
from perturbation_agents.utils import corrupt_node, CORRUPTION_TYPES


class RandomPerturbationAgent(BasePerturbationAgent):
    """
    Random observation perturbation for Flatland.

    Two modes:
      - "blank" (default): full sensor blackout — entire obs set to -inf.
      - "feature": random corruption of a single field per agent.

    Blindness is persistent for `duration` steps sampled from
    Uniform[min_duration, max_duration].

    Args:
        perturbation_rate (float): Per-agent probability of triggering an event.
        min_duration (int): Minimum blind/corrupt steps per event.
        max_duration (int): Maximum blind/corrupt steps per event.
        mode (str): "blank" or "feature".
        seed (int): Random seed.
    """

    def __init__(
        self,
        perturbation_rate: float = 0.1,
        min_duration: int = 1,
        max_duration: int = 3,
        mode: str = "blank",
        seed: int = 42,
    ):
        super().__init__(name="RandomPerturbationAgent", seed=seed)
        self.perturbation_rate = perturbation_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.mode = mode
        self._blind_counters: Dict[int, int] = {}
        self._corruption_type: Dict[int, str] = {}

    def perturb(self, obs_dict: Dict) -> Dict:
        perturbed = copy.deepcopy(obs_dict)

        for handle in obs_dict:
            # Decrement ongoing blindness
            if self._blind_counters.get(handle, 0) > 0:
                self._blind_counters[handle] -= 1
            elif self.space_prng.random() < self.perturbation_rate:
                duration = self.space_prng.randint(self.min_duration, self.max_duration + 1)
                self._blind_counters[handle] = duration
                if self.mode == "feature":
                    # Pick a random corruption type (excluding blank)
                    feature_types = [c for c in CORRUPTION_TYPES if c != "blank"]
                    self._corruption_type[handle] = feature_types[
                        self.space_prng.randint(len(feature_types))
                    ]
                else:
                    self._corruption_type[handle] = "blank"

            if self._blind_counters.get(handle, 0) > 0:
                ctype = self._corruption_type.get(handle, "blank")
                if perturbed[handle] is not None:
                    perturbed[handle] = corrupt_node(perturbed[handle], ctype)

        self.perturbation_count += 1
        return perturbed

    def reset(self) -> None:
        super().reset()
        self._blind_counters.clear()
        self._corruption_type.clear()
