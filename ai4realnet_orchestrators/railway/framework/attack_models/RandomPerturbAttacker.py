"""
Random Perturbation Attacker for Flatland railway environments.

Mimics the perturbation_tree_observation_builder_wrapper mechanism:
agents are randomly made "blind" for a duration sampled from a Poisson-like
process. A blind agent receives an all-(-inf) observation, meaning it cannot
perceive the track around it.
"""

import copy
import numpy as np
from numpy.random import RandomState

from attack_models.BaseAttackerClass import BaseAttackerClass


def _blank_node():
    """Return a blank node that signals blindness to observation-based policies."""
    try:
        from flatland.core.env_observation_builder import Node
        return Node(
            dist_own_target_encountered=-np.inf,
            dist_other_target_encountered=-np.inf,
            dist_other_agent_encountered=-np.inf,
            dist_potential_conflict=-np.inf,
            dist_unusable_switch=-np.inf,
            dist_to_next_branch=-np.inf,
            dist_min_to_target=-np.inf,
            num_agents_same_direction=-np.inf,
            num_agents_opposite_direction=-np.inf,
            num_agents_malfunctioning=-np.inf,
            speed_min_fractional=-np.inf,
            num_agents_ready_to_depart=-np.inf,
            childs={},
        )
    except (ImportError, TypeError):
        # Fallback: return None (Environment handles None obs as zeros)
        return None


class RandomPerturbAttacker(BaseAttackerClass):
    """
    Random observation-blanking attacker for Flatland.

    With probability `perturbation_rate` per step, an agent's entire tree
    observation is replaced with -inf values for `duration` steps, simulating
    a sensor failure or communication blackout.

    Args:
        perturbation_rate (float): Per-agent probability of triggering blindness each step.
        min_duration (int): Minimum number of consecutive blind steps.
        max_duration (int): Maximum number of consecutive blind steps.
        seed (int): Random seed for reproducibility.
    """

    model_name = "RandomPerturbAttacker"
    pickle_file = "random_perturb_attacker.pkl"

    def __init__(self, perturbation_rate=0.1, min_duration=1, max_duration=3, seed=42):
        self.perturbation_rate = perturbation_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self._rng = RandomState(seed)
        self._blind_counters = {}
        super().__init__(name=self.model_name)

    def load(self, path):
        pass

    def perturb(self, obs_dict):
        """
        Apply random observation blanking to a subset of agents.

        Args:
            obs_dict (dict): {agent_handle: TreeObs Node}

        Returns:
            dict: Perturbed observation dict (deep copy with some agents blanked).
        """
        perturbed = copy.deepcopy(obs_dict)

        for handle in obs_dict:
            # Decrement existing blindness counter
            if self._blind_counters.get(handle, 0) > 0:
                self._blind_counters[handle] -= 1
            elif self._rng.random() < self.perturbation_rate:
                # Trigger new blindness event
                duration = self._rng.randint(self.min_duration, self.max_duration + 1)
                self._blind_counters[handle] = duration

            if self._blind_counters.get(handle, 0) > 0:
                perturbed[handle] = _blank_node()

        return perturbed

    def perturb_vector(self, vec):
        """
        Blank the entire vector with probability perturbation_rate (whole-sensor failure).
        Otherwise return a copy unchanged.
        """
        import numpy as np
        if self._rng.random() < self.perturbation_rate:
            return np.zeros(len(vec), dtype=np.float64)
        return np.array(vec, dtype=np.float64)

    def reset(self):
        """Reset blindness state between episodes."""
        self._blind_counters = {}
