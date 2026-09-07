from attack_models.BaseAgent import BaseAgent


class BaseAttackerClass(BaseAgent):
    """Base class for all Flatland railway attackers."""

    def __init__(self, name=""):
        super().__init__(name)
        self.last_handle = None
        self.last_ctype = None

    def load(self, path):
        pass

    def perturb(self, obs):
        """
        Apply perturbation to observation dict.

        Args:
            obs (dict): {agent_handle: TreeObs Node} from Flatland env.

        Returns:
            dict: Perturbed observation dict with same structure.
        """
        return obs

    def perturb_vector(self, vec):
        """
        Apply perturbation to a generic 1-D numpy observation vector.

        Called by MazeFlatlandDefender to inject perturbations directly into
        the maze policy's internal observations (which are numpy arrays, not
        TreeObs Nodes).  Subclasses override to implement their strategy.

        Args:
            vec (np.ndarray): Flat observation vector of any length.

        Returns:
            np.ndarray: Corrupted copy of vec.
        """
        import numpy as np
        return np.array(vec, dtype=np.float64)
