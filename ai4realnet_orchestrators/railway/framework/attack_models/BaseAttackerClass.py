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

    An attacker that also chooses a severity publishes it alongside, in
    `last_severities`: {agent_handle: severity}. A handle missing from that dict means
    severity 1.0, the plausible value, so a two-factor attacker needs no change.

    perturb_vector() then only APPLIES an already-made decision to a different
    observation format (the maze policy's dense vector). It must be a pure
    function of (vec, ctype, severity): no RNG draw, no network query, no state update.
    Re-deciding there would mean the perturbation whose magnitude is recorded is
    not the perturbation the policy actually received, and -- for the learning
    attackers -- that the learned selection is discarded at the point of use.
    """

    def __init__(self, name="", defender=None):
        super().__init__(name)
        # The defender this attacker may query when scoring its own choices. None keeps
        # the black-box behaviour, which measurably does not work; see _attack_reward.
        self.defender = defender
        self.last_handle = None
        self.last_ctype = None
        # {handle: ctype} actually corrupted by the most recent perturb() call;
        # {} means "corrupt nothing", None means "no decision published".
        self.last_targets = None
        # {handle: severity} for the same call; a missing handle means severity 1.0.
        self.last_severities = {}
        self.last_severity = 1.0

    def _attack_reward(self, obs_dict, perturbed, ctype, severity):
        """
        Reward for the perturbation this agent just chose.

        Query-based when a defender is available: how much confidence the policy loses in
        the action it preferred, which is power grid's attacker objective. Otherwise the
        observation-only surrogate, with one warning, because that surrogate is why these
        attackers scored worse than random noise.

        A None from the query path means the defender could not answer, not that the
        corruption did nothing, so it falls back rather than reporting zero disruption.
        """
        from perturbation_agents.utils import (
            disruption_reward,
            query_disruption_reward,
            warn_black_box_fallback,
        )

        if self.defender is not None:
            reward = query_disruption_reward(self.defender, obs_dict, ctype, severity)
            if reward is not None:
                return reward
        warn_black_box_fallback(self.name)
        return disruption_reward(obs_dict, perturbed)

    def load(self, path):
        pass

    def _publish_targets(self, targets, severities=None):
        """
        Record which handles this perturb() call corrupted, how, and how hard.

        Args:
            targets (dict): {agent_handle: corruption_type}. Empty when the call
                corrupted nothing.
            severities (dict or None): {agent_handle: severity}. None, or a handle left
                out of it, means severity 1.0 -- the plausible value for that corruption
                type, which is what a two-factor attacker always played.
        """
        self.last_targets = dict(targets)
        self.last_severities = dict(severities) if severities else {}
        if targets:
            handle, ctype = next(iter(targets.items()))
            self.last_handle = handle
            self.last_ctype = ctype
            self.last_severity = float(self.last_severities.get(handle, 1.0))
        else:
            self.last_handle = None
            self.last_ctype = None
            self.last_severity = 1.0

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

    def perturb_vector(self, vec, ctype=None, severity=None, bound=None):
        """
        Apply an ALREADY-CHOSEN corruption to a generic 1-D observation vector.

        Called by MazeFlatlandDefender to inject the perturbation into the maze
        policy's own observations (dense numpy arrays, not TreeObs Nodes). The
        corruption type and severity come from the decision perturb() already made, so
        this is deterministic and carries no learning state.

        Args:
            vec (np.ndarray): Flat observation vector of any length.
            ctype (str or None): Corruption type to apply. Defaults to the type
                chosen by the last perturb() call, and to utils.DEFAULT_VECTOR_CTYPE
                when the attacker published no type at all.
            severity (float or None): How far past the plausible value to push the
                claim. Defaults to the severity the last perturb() call published, and
                to 1.0 when it published none.
            bound (tuple or None): (low, high) of the observation space, so the result
                stays a value the space can hold.

        Returns:
            np.ndarray: Corrupted copy of vec.
        """
        import numpy as np
        from perturbation_agents.utils import DEFAULT_VECTOR_CTYPE, corrupt_vector
        resolved = ctype or self.last_ctype or DEFAULT_VECTOR_CTYPE
        sev = self.last_severity if severity is None else severity
        return corrupt_vector(np.array(vec, dtype=np.float64), resolved,
                              severity=float(sev), bound=bound)
