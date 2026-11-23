"""
Base Perturbation Agent for Flatland Railway Robustness Testing.

Domain-agnostic equivalent of the power-grid BasePerturbationAgent.
No grid2op imports — uses numpy RandomState instead of RandomObject.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

import numpy as np

from .utils import (
    disruption_reward,
    query_disruption_reward,
    warn_black_box_fallback,
)

logger = logging.getLogger(__name__)


class BasePerturbationAgent(ABC):
    """
    Abstract base class for all Flatland perturbation agents.

    All agents receive {agent_handle: TreeObs Node} and return a perturbed
    copy of that dict.  Subclasses must implement perturb().

    Attributes:
        name (str): Identifier for this agent.
        perturbation_count (int): Cumulative number of perturb() calls.
        reset_count (int): Cumulative number of reset() calls.
        space_prng (numpy.RandomState): Seeded random state for reproducibility.
    """

    def __init__(self, name: Optional[str] = None, seed: int = 42, defender=None,
                 **kwargs):
        self.name = name or self.__class__.__name__
        # The defender this attacker may query when scoring its own choices. None keeps
        # the black-box behaviour, which measurably does not work; see _attack_reward.
        self.defender = defender
        self.perturbation_count = 0
        self.reset_count = 0
        self.space_prng = np.random.RandomState(seed)
        self.config = kwargs
        self.last_handle = None
        self.last_ctype = None
        # How far past the plausible value the chosen claim is pushed; 1.0 is the
        # plausible value itself. See perturbation_agents.attack_space.
        self.last_severity = 1.0
        logger.debug(f"Initialized {self.name}")

    def _attack_reward(self, obs_dict: Dict, perturbed: Dict, ctype, severity) -> float:
        """
        Reward for the perturbation this agent just chose.

        Query-based when a defender is available: how much confidence the policy loses in
        the action it preferred, which is power grid's attacker objective. Otherwise the
        observation-only surrogate, with one warning, because that surrogate is why three
        of the four learned attackers scored worse than random noise.

        A None from the query path means the defender could not answer, not that the
        corruption did nothing, so it falls back rather than reporting zero disruption.
        """
        if self.defender is not None:
            reward = query_disruption_reward(self.defender, obs_dict, ctype, severity)
            if reward is not None:
                return reward
        warn_black_box_fallback(self.name)
        return disruption_reward(obs_dict, perturbed)

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
        self.last_severity = 1.0
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
