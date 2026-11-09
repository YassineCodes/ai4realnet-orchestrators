"""
Maze-Flatland defender wrappers for the AI4REALNET robustness/resilience framework.

Integrates a trained maze-flatland behavioral cloning policy as a RailwayDefender.

IMPORTANT - perturbation semantics:
    The maze-flatland policy does not read the obs_dict the framework passes to act().
    Its input is the maze observation: the vector produced by
    GraphDirectionalObservationConversion plus train_move_mask, built from the
    environment's internal FlatlandMazeState. Corrupting a TreeObs node therefore
    cannot, by itself, change an action.

    How an attack reaches the policy:
        - attack_mode="obs_perturb"  -> act_and_step() applies the attacker's chosen
          corruption to the MAZE observation before the policy sees it. This does
          change actions, and it is the mode the KPIs are reported from.
        - attack_mode="state_inject" -> the RailEnv's internal state is corrupted, and
          the maze observation builder reads the corruption on the next flat step.
        - A query-based attacker that scores candidates through act() (GEPerturb)
          cannot reach the policy at all: act() answers from the maze observation and
          ignores the TreeObs it is handed, so every candidate scores identically.

    Robustness interpretation:
        An unchanged action is only evidence of robustness if the attack was actually
        applied. act_and_step() returns the perturbation it applied; when that vector
        is zero for a whole episode the attack never reached the policy, and the KPI
        should be reported as NOT APPLICABLE rather than as high robustness.

Usage:
    from defenders_maze import MazeFlatlandDefender, load_maze_policy
    from defenders import DoNothingDefender

    policy = load_maze_policy(
        checkpoint_path="maze-flatland/flatland_result/BC_offline-v2.2/state_dict-epoch_200.pt",
        spaces_config_path="maze-flatland/flatland_result/BC_offline-v2.2/spaces_config.pkl",
    )
    # Build maze env with same config used during training
    maze_env = build_maze_env(n_trains=3, width=40, height=40)
    defender = MazeFlatlandDefender(maze_env, policy)
    env = FlatlandEnvironment(raw_rail_env, defender)
"""

import logging
from typing import Dict, Optional

import numpy as np

try:  # torch is optional; only needed to perturb tensorised maze observations
    import torch as _TORCH
except ImportError:  # pragma: no cover
    _TORCH = None

from defenders import RailwayDefender

logger = logging.getLogger(__name__)


def load_maze_policy(checkpoint_path: str, spaces_config_path: str):
    """
    Load a trained maze-flatland TorchPolicy from a checkpoint.

    Args:
        checkpoint_path:    Path to state_dict-epoch_N.pt file.
        spaces_config_path: Path to spaces_config.pkl (saved alongside checkpoint).

    Returns:
        TorchPolicy ready for inference (eval mode, CPU).
    """
    import pickle
    import torch
    from maze.core.agent.torch_policy import TorchPolicy
    from maze.core.utils.factory import Factory
    from maze.perception.models.policies import ProbabilisticPolicyComposer
    from omegaconf import OmegaConf

    with open(spaces_config_path, "rb") as f:
        spaces_config = pickle.load(f)

    obs_spaces = spaces_config.observation_spaces_dict
    action_spaces = spaces_config.action_spaces_dict

    # Rebuild the same network as in bc_train.yaml: FlattenConcatMaskedPolicyNet [512, 256] Tanh
    from maze.perception.models.built_in.flatten_concat_masked import FlattenConcatMaskedPolicyNet
    from maze.distributions.distribution_mapper import DistributionMapper

    # Empty config — default_mapping already covers gymnasium.spaces.Discrete → Categorical
    distribution_mapper = DistributionMapper(
        action_space=action_spaces["train_move"],
        distribution_mapper_config=[],
    )

    network = FlattenConcatMaskedPolicyNet(
        obs_shapes={k: v.shape for k, v in obs_spaces["train_move"].spaces.items()},
        action_logits_shapes={k: (v.n,) for k, v in action_spaces["train_move"].spaces.items()},
        non_lin=torch.nn.Tanh,
        hidden_units=[512, 256],
        remove_mask_from_obs=True,
    )

    policy = TorchPolicy(
        networks={"train_move": network},
        distribution_mapper=distribution_mapper,
        device="cpu",
        substeps_with_separate_agent_nets=[],
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    logger.info(f"Loaded maze-flatland policy from {checkpoint_path}")
    return policy


def build_maze_env(n_trains: int = 3, width: int = 37, height: int = 37,
                   n_cities: int = 2, seed: int = 0):
    """
    Build a FlatlandMazeEnv matching the BC training config.

    Training used: 3 trains, 37x37, 2 cities, no malfunctions, speed=1.0 only.

    Args:
        n_trains:      Number of trains (default 3, matches training).
        width, height: Map size (default 37x37, matches training).
        n_cities:      Number of cities (default 2, matches training).
        seed:          Unused (FlatlandCoreEnvironment is seeded on reset).

    Returns:
        FlatlandEnvironment (maze-rl) instance.
    """
    from maze_flatland.env.maze_env import FlatlandEnvironment as MazeEnv
    from maze_flatland.env.core_env import FlatlandCoreEnvironment
    from maze_flatland.env.termination_condition import IncludeOutOfTimeTrainsInEarlyTermination
    from maze_flatland.reward.constant_reward import ConstantMinusOneReward
    from maze_flatland.env.renderer import FlatlandRendererBase
    from maze_flatland.space_interfaces.observation_conversion.graph_based_directional import (
        GraphDirectionalObservationConversion,
    )
    from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
    from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
    from flatland.envs.line_generators import SparseLineGen
    from flatland.envs.rail_generators import SparseRailGen

    core_env = FlatlandCoreEnvironment(
        n_trains=n_trains,
        map_width=width,
        map_height=height,
        termination_conditions=IncludeOutOfTimeTrainsInEarlyTermination(),
        reward_aggregator=ConstantMinusOneReward(),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(malfunction_rate=0, min_duration=1, max_duration=2)
        ),
        line_generator=SparseLineGen(speed_ratio_map={1.0: 1.0}),
        rail_generator=SparseRailGen(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=3,
        ),
        timetable_generator=None,
        renderer=FlatlandRendererBase(
            img_width=1500,
            agent_render_variant="flatland.utils.rendertools.AgentRenderVariant.AgentRenderVariant.ONE_STEP_BEHIND",
            highlight_current_train=False,
            render_out_of_map_trains=True,
        ),
    )

    obs_conv = GraphDirectionalObservationConversion(serialize_representation=True)
    act_conv = DirectionalAC()

    maze_env = MazeEnv(
        core_env=core_env,
        observation_conversion={"train_move": obs_conv},
        action_conversion={"train_move": act_conv},
    )

    # Wrap with masking to add train_move_mask (required by BC-trained network)
    from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper
    from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
    maze_env = FlatlandMaskingWrapper(
        env=maze_env,
        mask_builder=LogicMaskBuilder(mask_out_dead_ends=True, disable_stop_on_switches=False),
        explain_mask=False,
    )

    # SubStepSkippingWrapper is part of the stack the policy was PUBLISHED with (see the
    # run's .hydra/config.yaml). It skips sub-steps in which the agent has no real choice.
    # Omitting it asks the policy to act in states its published environment never
    # presents, and maze then rejects the resulting action (DirectionalAC asserts that
    # DO_NOTHING is illegal for an active train). Reproducing the published stack is what
    # makes the evaluation a measurement of the agent rather than of our harness.
    from maze_flatland.wrappers.skipping_wrapper import SubStepSkippingWrapper
    maze_env = SubStepSkippingWrapper(env=maze_env, do_skipping_in_reset=True)
    return maze_env


class MazeFlatlandDefender(RailwayDefender):
    """
    Wraps a trained maze-flatland behavioral cloning (BC) policy as a defender.

    The policy was trained to imitate ShortestPathPolicy via behavioral cloning.
    It drives a PARALLEL maze-rl env that mirrors the framework env step-for-step.
    The maze policy reads its own maze observation, not the perturbed TreeObs the
    framework passes in, so a TreeObs-level perturbation reaches it only because
    act_and_step() re-applies the attacker's chosen corruption to that maze
    observation. See the module docstring.

    Actor sequencing:
        The maze-rl env is sequential — each call to step() processes one actor.
        act() cycles through all n_trains actors: for each actor it queries the
        obs returned by the previous step (or reset), runs the policy, then steps
        the maze env.  The obs from the LAST step (actor n_trains-1 → actor 0 of
        the next flat step) is cached in self._next_obs for the next call to act().

    Args:
        maze_env: A FlatlandEnvironment (maze-rl) built by build_maze_env().
        policy:   A TorchPolicy loaded via load_maze_policy().
        n_trains: Number of trains (must match maze_env config).
    """

    def __init__(self, maze_env, policy, n_trains: int = 3,
                 attack_mode: str = "obs_perturb"):
        self._maze_env = maze_env
        self._policy = policy
        self._n_trains = n_trains
        # Tree observations for the framework, computed from the SAME RailEnv that the
        # maze env advances -- see _build_tree_builder().
        self._tree_builder = None
        self._maze_obs = None
        self._warned_query_attacker = False
        self._warned_blanket = False
        # Width of one train's perturbable maze observation, learned on first use so
        # the recorded perturbation vector keeps a constant length across steps.
        self._maze_delta_width = None
        # "obs_perturb": perturb maze obs dict (Phase 1, oracle/white-box)
        # "state_inject": corrupt rail_env internal state (Phase 2, black-box)
        self._attack_mode = attack_mode
        # obs returned by the last reset/step — ready for the next actor
        self._next_obs: Optional[dict] = None
        # perturbed actions from the most recent act() call
        self._cached_actions: Optional[dict] = None
        # clean (unperturbed) reference actions computed inside act() for comparison
        self._cached_clean_actions: Optional[dict] = None
        # active attacker registered by Environment.reset() via set_attacker()
        self._attacker = None

    def set_attacker(self, attacker) -> None:
        """Register the active attacker so act() injects perturbations into maze obs."""
        self._attacker = attacker

    @staticmethod
    def _as_numpy(val):
        """
        Return (array, restore) for a maze observation entry, or (None, None).

        TorchPolicy.compute_action() converts the observation dict IN PLACE from numpy
        to torch tensors, so an entry that was an ndarray on the first pass is a Tensor
        afterwards. Testing only for np.ndarray therefore skipped the observation and the
        perturbation silently became a no-op -- which scores the agent as perfectly
        robust against an attack that never happened.
        """
        if isinstance(val, np.ndarray):
            return val, lambda a: a.reshape(val.shape).astype(val.dtype)
        if _TORCH is not None and isinstance(val, _TORCH.Tensor):
            arr = val.detach().cpu().numpy()
            return arr, (lambda a: _TORCH.as_tensor(
                a.reshape(arr.shape).astype(arr.dtype), dtype=val.dtype, device=val.device))
        return None, None

    _MASK_KEY = "train_move_mask"

    def _perturbable_keys(self, obs: dict):
        """Keys of obs that carry perturbable numbers, in a deterministic order."""
        return [k for k in sorted(obs) if k != self._MASK_KEY]

    def _apply_vector_attack(self, obs: dict, ctype: str):
        """
        Apply the attacker's already-chosen corruption to the maze observation.

        The corruption TYPE was decided once, by the attacker's perturb() call on the
        tree observations; this only applies that decision in the maze observation's
        format. Deciding again here would mean the perturbation the policy receives is
        not the perturbation that was recorded, and that a learning attacker's selection
        is discarded at the point of use.

        The action mask is left exactly as the environment produced it: corrupting or
        widening it would change which actions the policy is ALLOWED to take, which is
        manipulating the agent rather than perturbing its perception.

        Args:
            obs: the clean maze observation for the current actor.
            ctype: corruption type published by the attacker for this handle.

        Returns:
            tuple: (perturbed_obs, delta) where delta is the flat
                (perturbed - clean) vector over _perturbable_keys(obs), i.e. the
                perturbation the policy ACTUALLY received.

        Raises:
            RuntimeError: if no entry could be perturbed. Silently returning the
                observation unchanged would report the agent as robust to an attack
                that was never applied.
        """
        from perturbation_agents.utils import corrupt_vector
        perturbed = {}
        deltas = {}
        for key, val in obs.items():
            arr, restore = (None, None) if key == self._MASK_KEY else self._as_numpy(val)
            if arr is None:
                perturbed[key] = val
                continue
            flat = arr.flatten().astype(np.float64)
            if self._attacker is not None and hasattr(self._attacker, "perturb_vector"):
                out = np.asarray(self._attacker.perturb_vector(flat, ctype), dtype=np.float64)
            else:
                out = corrupt_vector(flat, ctype)
            deltas[key] = out - flat
            perturbed[key] = restore(out)
        if not deltas:
            raise RuntimeError(
                "MazeFlatlandDefender: no maze observation entry could be perturbed "
                f"(keys={list(obs)}, types={[type(v).__name__ for v in obs.values()]}). "
                "Refusing to report robustness for an attack that was not applied."
            )
        delta = np.concatenate([deltas[k] for k in self._perturbable_keys(obs) if k in deltas])
        return perturbed, delta

    def _get_rail_env(self):
        """Traverse the maze wrapper chain to reach the underlying RailEnv."""
        maze = self._maze_env
        for candidate in (maze, getattr(maze, "env", None)):
            if candidate is not None and hasattr(candidate, "core_env"):
                return getattr(candidate.core_env, "_rail_env", None)
        return None

    def _inject_rail_state(self, handle: int, ctype: str) -> None:
        """
        Corrupt RailEnv internal state for the chosen agent.

        Phase 2 / black-box attack: the maze observation builder reads from this
        state on the next flat step, so the policy naturally sees corrupted inputs
        without any white-box access to the obs pipeline.

        Two injection strategies:
          malfunction — sets malfunction_down_counter so the train (and neighbours)
                        treat agent `handle` as broken for ~5 steps.
          all others  — adds large noise to the distance map slice for `handle`,
                        making all paths look unreachable or distorted.
        """
        rail_env = self._get_rail_env()
        if rail_env is None:
            return
        try:
            if ctype == "malfunction":
                agent = rail_env.agents[handle]
                agent.malfunction_handler.malfunction_down_counter = 5
            else:
                dm = rail_env.distance_map.distance_map
                if dm is not None and handle < dm.shape[0]:
                    scale = {"blank": 1e6, "target": 1e6,
                             "speed": 500.0, "direction": 200.0,
                             "conflict": 300.0}.get(ctype, 500.0)
                    dm[handle] = dm[handle] + scale
        except Exception as e:
            logger.debug(f"_inject_rail_state failed (handle={handle}, ctype={ctype}): {e}")

    # This defender OWNS the environment. The maze env internally advances a RailEnv, so
    # FlatlandEnvironment must drive the transition through act_and_step() instead of
    # stepping a RailEnv of its own -- otherwise the policy is evaluated against a
    # different map from the one it observes.
    drives_env = True

    @property
    def rail_env(self):
        """The RailEnv the maze env advances -- the single source of truth."""
        return self._get_rail_env()

    def _build_tree_builder(self):
        """
        Attach a TreeObs builder to the maze env's RailEnv.

        The maze env uses a DummyObservationBuilder, so the tree observation slot is
        free and tree observations can be computed from the very environment the policy
        acts on. This is what keeps observations, actions and rewards on one map.
        """
        from flatland.envs.observations import TreeObsForRailEnv
        from flatland.envs.predictions import ShortestPathPredictorForRailEnv

        rail = self.rail_env
        if rail is None:
            self._tree_builder = None
            return
        builder = TreeObsForRailEnv(
            max_depth=2, predictor=ShortestPathPredictorForRailEnv(20))
        builder.set_env(rail)
        builder.reset()
        self._tree_builder = builder

    def tree_obs(self) -> dict:
        """Tree observations for every agent, from the maze env's own RailEnv."""
        rail = self.rail_env
        if self._tree_builder is None or rail is None:
            return {}
        try:
            return self._tree_builder.get_many(list(range(rail.get_num_agents())))
        except Exception as e:
            logger.warning(f"tree_obs() failed: {e}")
            return {}

    def reset(self, seed: int = 0):
        """
        Reset the maze env and return tree observations taken from its RailEnv.

        The observation dict is returned so FlatlandEnvironment can adopt it directly;
        in this mode the framework keeps no environment of its own.
        """
        try:
            # Seed BEFORE reset. Without this every episode draws a fresh random map, so
            # two submissions are scored on different scenarios and their KPIs are not
            # comparable -- and no result is reproducible. result_getter passes the
            # episode index as the seed, mirroring the non-maze path.
            if seed is not None and hasattr(self._maze_env, "seed"):
                try:
                    self._maze_env.seed(int(seed))
                except Exception as e:
                    logger.warning(f"seeding the maze env failed: {e}")
            self._maze_obs = self._maze_env.reset()
        except Exception as e:
            logger.warning(f"MazeFlatlandDefender reset failed: {e}")
            self._maze_obs = None
        self._next_obs = self._maze_obs
        self._cached_actions = None
        self._cached_clean_actions = None
        self._build_tree_builder()
        return self.tree_obs()

    def act_and_step(self, perturbed_obs: dict, clean_obs: dict):
        """
        Advance the environment by exactly one flat step.

        The maze env is sequential: each step() serves one train, and the underlying
        RailEnv advances only once the last actor has been served. This drives all
        n_trains actors, so one call equals one RailEnv step.

        Perturbation semantics: the attacker decided once, in its own perturb() call on
        the tree observations (driven by FlatlandEnvironment.step()), which trains to
        corrupt and how. This method only APPLIES that decision in the maze
        observation's format, to exactly those trains. It does not call perturb()
        again: doing so drew a second, independent perturbation, so the magnitude that
        was recorded belonged to a different sample than the one the policy received,
        the attacker's exploration schedule decayed at twice its configured rate, and
        half of a learning attacker's bootstrapped updates saw s' = s.

        Args:
            perturbed_obs: tree observations after the attacker's perturbation.
            clean_obs: tree observations before it. Kept for interface compatibility;
                the attack decision is read from the attacker, not re-derived here.

        Returns:
            tuple: (actions, clean_actions, reward, done, applied_perturbation) where
                applied_perturbation is the flat (perturbed - clean) maze-observation
                delta the policy actually received, ordered by train handle.
        """
        from perturbation_agents.utils import DEFAULT_VECTOR_CTYPE as default_ctype

        rail = self.rail_env
        actions = {}
        clean_actions = {}

        attacker_active = self._attacker is not None
        published = getattr(self._attacker, "last_targets", None) if attacker_active else None
        if published is not None:
            # A published decision is authoritative, including the empty one: an
            # attacker whose roll corrupted nothing this step must corrupt nothing
            # here too, or a baseline that attacked one train gets applied to all of
            # them and stops being comparable with a targeted attacker.
            targets = dict(published)
            use_blanket = False
        else:
            handle = getattr(self._attacker, "last_handle", None) if attacker_active else None
            ctype = getattr(self._attacker, "last_ctype", None) if attacker_active else None
            targets = {handle: ctype or default_ctype} if handle is not None else {}
            # An attacker that publishes nothing at all gets the documented fallback of
            # corrupting every train. Skipping the step instead would report the agent
            # as robust against an attack that was never applied.
            use_blanket = attacker_active and not targets
        if use_blanket and not self._warned_blanket:
            self._warned_blanket = True
            logger.warning(
                "%s published no attack target; falling back to corrupting every "
                "train with '%s'. This gives it a breadth no targeted attacker has, "
                "so the comparison against targeted attackers is not breadth-matched.",
                getattr(self._attacker, "name", type(self._attacker).__name__),
                default_ctype,
            )

        if self._attack_mode == "state_inject" and targets:
            for handle, ctype in targets.items():
                self._inject_rail_state(handle, ctype or "malfunction")

        # One flat step means "the RailEnv advanced once", not "n_trains actor steps":
        # SubStepSkippingWrapper collapses sub-steps where the agent has no choice, so the
        # number of policy queries per flat step varies. Watch the RailEnv's own clock.
        def _elapsed():
            return getattr(rail, "_elapsed_steps", None) if rail is not None else None

        start_elapsed = _elapsed()
        max_substeps = max(1, self._n_trains) * 4  # generous cap; guards against a stall
        maze_done = False
        applied_deltas = {}
        for _ in range(max_substeps):
            if start_elapsed is not None and _elapsed() is not None and _elapsed() > start_elapsed:
                break
            actor_id = self._maze_env.actor_id()
            handle = getattr(actor_id, "agent_id", None)
            obs = self._maze_obs
            if obs is None:
                break

            if self._attack_mode == "obs_perturb" and attacker_active:
                # The attacker's published decision says WHICH trains are attacked and
                # with WHICH corruption type. Both come from the single perturb() call
                # on the tree observations; nothing is re-chosen here.
                #
                # ctype ("conflict", "target", ...) names a field of the TreeObs, and the
                # maze observation is a different representation in which those names
                # have no layout meaning, so corrupt_vector() maps each type onto a
                # format-agnostic operation (zero / amplify / negate). The mapping is
                # coarse, which is a documented limitation of comparing the two formats
                # -- but it is the attacker's own choice that is mapped.
                if handle in targets:
                    ctype = targets[handle] or default_ctype
                elif use_blanket:
                    ctype = default_ctype
                else:
                    ctype = None
                if ctype is not None:
                    policy_obs, delta = self._apply_vector_attack(obs, ctype)
                    if handle is not None:
                        applied_deltas[handle] = delta
                else:
                    policy_obs = obs
            else:
                policy_obs = obs

            if "train_move_mask" not in policy_obs:
                # Never invent a mask. An all-ones mask marks every action legal,
                # widening the agent's action space and letting it emit actions the
                # environment rejects (maze asserts DO_NOTHING is illegal for an active
                # train). That is manipulating the agent, not perturbing it.
                raise RuntimeError(
                    "maze observation is missing 'train_move_mask'; refusing to "
                    "substitute an all-permissive mask"
                )

            action = int(self._policy.compute_action(
                observation=policy_obs, maze_state=None, env=self._maze_env,
                actor_id=actor_id, deterministic=True).get("train_move", 0))
            if handle is not None:
                actions[handle] = action

            if attacker_active:
                ref = obs
                if "train_move_mask" not in ref:
                    raise RuntimeError(
                        "maze observation is missing 'train_move_mask'; refusing to "
                        "substitute an all-permissive mask"
                    )
                clean_action = int(self._policy.compute_action(
                    observation=ref, maze_state=None, env=self._maze_env,
                    actor_id=actor_id, deterministic=True).get("train_move", 0))
                if handle is not None:
                    clean_actions[handle] = clean_action

            step_out = self._maze_env.step({"train_move": action})
            self._maze_obs = step_out[0]
            maze_done = bool(step_out[2])
            if maze_done:
                break

        # Reward and termination are read from the RailEnv the policy just acted on.
        reward = 0.0
        done = maze_done
        if rail is not None:
            try:
                reward = float(sum(rail.rewards_dict.values())) if rail.rewards_dict else 0.0
                done = done or bool(rail.dones.get("__all__", False))
            except Exception as e:
                logger.debug(f"reading reward/done from rail env failed: {e}")

        self._cached_actions = dict(actions)
        self._cached_clean_actions = dict(clean_actions) if clean_actions else dict(actions)
        applied = self._assemble_applied_perturbation(applied_deltas)
        return actions, self._cached_clean_actions, reward, done, applied

    def _assemble_applied_perturbation(self, applied_deltas: dict) -> np.ndarray:
        """
        Concatenate the per-train applied deltas into one fixed-length vector.

        The maze env serves one train per sub-step and SubStepSkippingWrapper skips
        trains that have no choice, so the number and identity of the trains served
        varies from step to step. The recorded vector is therefore laid out by handle
        (0 .. n_trains-1), with zeros for a train that was not served or not attacked,
        which keeps it the same length at every step -- what the vulnerability metric
        needs in order to attribute action changes to individual features.

        Returns an all-zero vector when nothing was applied, which is the signal that
        the attack never reached the policy (report the KPI as not applicable rather
        than as high robustness).
        """
        width = self._maze_delta_width
        if width is None:
            for delta in applied_deltas.values():
                width = int(len(delta))
                self._maze_delta_width = width
                break
        if width is None:
            # Nothing applied and no width learned yet: fall back to the clean maze
            # observation's own width so the vector length stays stable from step one.
            if self._maze_obs is not None:
                try:
                    width = sum(
                        int(self._as_numpy(self._maze_obs[k])[0].size)
                        for k in self._perturbable_keys(self._maze_obs)
                        if self._as_numpy(self._maze_obs[k])[0] is not None
                    )
                    self._maze_delta_width = width
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"could not measure maze observation width: {e}")
                    width = 0
            else:
                width = 0
        out = np.zeros(max(1, self._n_trains) * width, dtype=np.float64)
        for handle, delta in applied_deltas.items():
            if handle is None or width == 0 or len(delta) != width:
                continue
            start = int(handle) * width
            if start + width <= len(out):
                out[start:start + width] = delta
        return out

    def act(self, obs_dict: dict) -> dict:
        """
        Pure policy query: return the current train's action WITHOUT touching the env.

        This method must have NO side effects. GEPerturbAttacker scores candidate
        perturbations by calling defender.act() once per candidate (nine times per step
        at the default n_candidates=8). The previous implementation advanced the maze env
        on every call, so the environment raced ahead of the framework by a factor of
        nine, corrupting every GEPerturb episode and making the pass take ~30 minutes
        instead of ~1. Stepping now lives solely in act_and_step().

        LIMITATION: the maze policy reads the maze observation, not the TreeObs dict
        passed in here, so a perturbation applied to obs_dict cannot reach it. A
        query-based attacker such as GEPerturb scores its candidates through this
        method and therefore sees every candidate score identically: its selection
        carries no gradient information. The attack it finally publishes IS applied
        (act_and_step() applies it to the maze observation), but it was chosen blind.
        Report that as a limitation of the attacker, not as robustness of the agent.

        Args:
            obs_dict: TreeObs per handle. Used only to determine which handles are active.

        Returns:
            dict: {handle: action} for the current actor, from the live maze observation.
        """
        if self._maze_obs is None:
            return {handle: 0 for handle in obs_dict}

        if not self._warned_query_attacker:
            self._warned_query_attacker = True
            logger.info(
                "MazeFlatlandDefender.act() is a side-effect-free policy query. The maze "
                "policy reads its own observation, so perturbations applied to the TreeObs "
                "passed here do not reach it; query-based attackers will see no change."
            )

        try:
            actor_id = self._maze_env.actor_id()
            handle = getattr(actor_id, "agent_id", None)
            policy_obs = self._maze_obs
            if "train_move_mask" not in policy_obs:
                raise RuntimeError(
                    "maze observation is missing 'train_move_mask'; refusing to "
                    "substitute an all-permissive mask"
                )
            action = int(self._policy.compute_action(
                observation=policy_obs, maze_state=None, env=self._maze_env,
                actor_id=actor_id, deterministic=True).get("train_move", 0))
        except RecursionError:
            raise
        except Exception as e:
            logger.warning(f"MazeFlatlandDefender.act() query failed: {e}")
            return {h: 0 for h in obs_dict}

        actions = {h: 0 for h in obs_dict}
        if handle is not None and handle in actions:
            actions[handle] = action
        return actions


    def act_comparison(self, obs_dict: dict) -> dict:
        """
        Return the CLEAN (unperturbed) reference actions computed inside act().

        These differ from self._cached_actions when the corruption applied to the maze
        observation caused the policy to choose a different action - giving a real
        n_actions_changed signal equivalent to the power-grid framework.

        Falls back to self._cached_actions (no difference) if no attacker is active.
        """
        if self._cached_clean_actions is not None:
            return dict(self._cached_clean_actions)
        if self._cached_actions is not None:
            return dict(self._cached_actions)
        return self.act(obs_dict)
