from attack_models.BaseAgent import BaseAgent


class BaseAttackerClass(BaseAgent):
    """
    Base class for all Flatland railway attackers.

    Attack decision vs. attack application
    --------------------------------------
    An attacker makes its decision exactly ONCE per environment step, inside
    perturb(). perturb() must publish that decision in `last_targets`:

        {agent_handle: corruption_type}

    for every handle it actually corrupted. `last_handle` / `last_ctype` remain
    available as a single-target convenience view.

    An EMPTY dict and None mean different things, and conflating them corrupts the
    experiment: {} is a published decision to corrupt nothing this step (a random
    attacker whose blindness roll came up empty), and must apply nothing. None means
    the attacker never published a decision at all, and leaves the defender no option
    but to corrupt every train -- which hands that attacker a breadth no targeted
    attacker gets. Always publish, including the empty case.

    perturb_vector() then only APPLIES an already-made decision to a different
    observation format (the maze policy's dense vector). It must be a pure
    function of (vec, ctype): no RNG draw, no network query, no state update.
    Re-deciding there would mean the perturbation whose magnitude is recorded is
    not the perturbation the policy actually received, and -- for the learning
    attackers -- that the learned selection is discarded at the point of use.
    """

    def __init__(self, name=""):
        super().__init__(name)
        self.last_handle = None
        self.last_ctype = None
        # {handle: ctype} actually corrupted by the most recent perturb() call;
        # {} means "corrupt nothing", None means "no decision published".
        self.last_targets = None

    def load(self, path):
        pass

    def _publish_targets(self, targets):
        """
        Record which handles this perturb() call corrupted, and how.

        Args:
            targets (dict): {agent_handle: corruption_type}. Empty when the call
                corrupted nothing.
        """
        self.last_targets = dict(targets)
        if targets:
            handle, ctype = next(iter(targets.items()))
            self.last_handle = handle
            self.last_ctype = ctype
        else:
            self.last_handle = None
            self.last_ctype = None

    def perturb(self, obs):
        """
        Apply perturbation to observation dict.

        Args:
            obs (dict): {agent_handle: TreeObs Node} from Flatland env.

        Returns:
            dict: Perturbed observation dict with same structure.
        """
        self._publish_targets({})
        return obs

    def perturb_vector(self, vec, ctype=None):
        """
        Apply an ALREADY-CHOSEN corruption to a generic 1-D observation vector.

        Called by MazeFlatlandDefender to inject the perturbation into the maze
        policy's own observations (dense numpy arrays, not TreeObs Nodes). The
        corruption type comes from the decision perturb() already made, so this
        is deterministic and carries no learning state.

        Args:
            vec (np.ndarray): Flat observation vector of any length.
            ctype (str or None): Corruption type to apply. Defaults to the type
                chosen by the last perturb() call, and to utils.DEFAULT_VECTOR_CTYPE
                when the attacker published no type at all.

        Returns:
            np.ndarray: Corrupted copy of vec.
        """
        import numpy as np
        from perturbation_agents.utils import DEFAULT_VECTOR_CTYPE, corrupt_vector
        resolved = ctype or self.last_ctype or DEFAULT_VECTOR_CTYPE
        return corrupt_vector(np.array(vec, dtype=np.float64), resolved)
