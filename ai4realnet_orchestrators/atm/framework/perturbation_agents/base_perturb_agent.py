"""
Base Perturbation Agent for ATM (BlueSky-Gym) Robustness Testing.

Domain-agnostic equivalent of the power-grid BasePerturbationAgent.
No grid2op imports — uses numpy RandomState instead of RandomObject.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


class BasePerturbationAgent(ABC):
    """
    Abstract base class for all ATM perturbation agents.

    All agents receive the BlueSky-Gym Dict observation -- {feature name: np.ndarray},
    the keys and shapes being whichever the configured environment declares -- and
    return a perturbed copy of it. Subclasses must implement perturb().

    Agents iterate the keys the observation actually carries and never name them, so a
    task that adds feature blocks (the AI4REALNET v1 task adds five intruder-aircraft
    blocks over the earlier v0 task) is perturbed correctly without a change here.

    Attributes:
        name (str): Identifier for this agent.
        perturbation_count (int): Cumulative number of perturb() calls.
        reset_count (int): Cumulative number of reset() calls.
        space_prng (numpy.RandomState): Seeded random state for reproducibility.
    """

    def __init__(self, name: Optional[str] = None, seed: int = 42, **kwargs):
        self.name = name or self.__class__.__name__
        self.perturbation_count = 0
        self.reset_count = 0
        self.space_prng = np.random.RandomState(seed)
        self.config = kwargs
        self.last_handle = None
        self.last_ctype = None
        logger.debug(f"Initialized {self.name}")

    @abstractmethod
    def perturb(self, obs_dict: Dict) -> Dict:
        """
        Apply perturbation to the observation dict.

        Args:
            obs_dict: {agent_handle: TreeObs Node} from the Flatland env.

        Returns:
            Perturbed copy of obs_dict (same structure, modified values).
        """
        self.perturbation_count += 1
        return obs_dict

    def reset(self) -> None:
        """Reset state at the beginning of a new episode."""
        self.reset_count += 1
        self.last_handle = None
        self.last_ctype = None
        logger.debug(f"{self.name}: reset #{self.reset_count}")

    def seed(self, seed: int) -> None:
        """Re-seed the random state."""
        self.space_prng = np.random.RandomState(seed)

    def get_perturbation_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "perturbation_count": self.perturbation_count,
            "reset_count": self.reset_count,
        }

    def __str__(self):
        return f"{self.name}(perturbations={self.perturbation_count})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class NullPerturbationAgent(BasePerturbationAgent):
    """No-op perturbation agent. Returns observations unchanged."""

    def perturb(self, obs_dict: Dict) -> Dict:
        super().perturb(obs_dict)
        return obs_dict
