"""
Multi-Attacker Robustness & Resilience TestRunner for Railway Domain

This module implements a SINGLE TestRunner that evaluates defender agents against
multiple adversarial attack types and returns different KPI values based on test_id.

KPIs Implemented:
    - KPI-DF-069: Drop-off in reward
    - KPI-FF-070: Frequency changed output AI agent
    - KPI-SF-071: Severity of changed output AI agent
    - KPI-SF-072: Steps survived with perturbations
    - KPI-VF-073: Vulnerability to perturbation
    - KPI-AF-074: Area between reward curves
    - KPI-DF-075: Degradation time
    - KPI-RF-076: Restorative time
    - KPI-SF-077: Similarity state to unperturbed situation

Framework Path: Relative to this file at ./framework/

Design: Single evaluation runs ALL attackers once, computes ALL metrics,
then returns the appropriate metric based on which KPI is being evaluated.
Mirrors ai4realnet_orchestrators/power_grid/test_runner_robustness_resilience_kpi_069_077.py.

Author: INESC TEC
"""

import logging
import os
import sys
import pickle
import tempfile
import zipfile
import requests
from typing import Dict, List

import numpy as np

# Add framework to path BEFORE other imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(SCRIPT_DIR, "framework")
if FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, FRAMEWORK_PATH)

# Parent directory (ONE level up) - where test_runner.py lives
_parent_dir = os.path.dirname(SCRIPT_DIR)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from test_runner import TestRunner

# KPI-RF-078 target; defined next to the metric it belongs to.
from evaluation_framework.metrics import REWARD_PER_ACTION_TARGET_RATIO

logger = logging.getLogger(__name__)

# ============================================================================
# KPI ID Mapping
# ============================================================================

KPI_MAPPING = {
    # Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
    "0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc": {
        "name": "KPI-VF-073: Vulnerability to perturbation",
        "metric_key": "perturb_vulnerability",
        "description": "Proportion of features vulnerable to attack [0-1]"
    },
    "e5206c56-75a0-41fa-9db3-bec66359337e": {
        "name": "KPI-SF-072: Steps survived with perturbations",
        "metric_key": "n_steps_survived",
        "description": "Number of timesteps before failure"
    },
    "dce32e78-e827-4994-a0a2-06feee2528cc": {
        "name": "KPI-SF-071: Severity of changed output",
        "metric_key": "severity_of_change",
        "description": "Severity of action changes [0-1, higher=worse]"
    },
    "a94c858e-4bc3-4d67-bd78-5c81506e39f7": {
        "name": "KPI-DF-069: Drop-off in reward",
        "metric_key": "reward_drop_percent",
        "description": "Percentage decrease in reward [0-100]"
    },
    "5abadf6b-991c-4d37-810f-f77bb71d490d": {
        "name": "KPI-FF-070: Frequency changed output",
        "metric_key": "action_change_freq",
        "description": "Proportion of timesteps with changed actions [0-1]"
    },

    # Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)
    "707a1a4e-7073-432b-94fc-af4a5ee9f07d": {
        "name": "KPI-AF-074: Area between reward curves",
        "metric_key": "area_between_curves",
        "description": "Integrated performance degradation"
    },
    "2c4be118-6108-43b3-b09f-a4bee842167a": {
        "name": "KPI-DF-075: Degradation time",
        "metric_key": "degradation_time",
        "description": "Time until performance degrades"
    },
    "2cac54e0-aaf3-4f22-8307-f23878c432f0": {
        "name": "KPI-RF-076: Restorative time",
        "metric_key": "restoration_time",
        "description": "Time to restore performance"
    },
    "d432299f-dbee-46ba-9e15-77954086440a": {
        "name": "KPI-SF-077: Similarity to unperturbed state",
        "metric_key": "state_similarity",
        "description": "Cosine similarity to unperturbed states [-1 to 1]"
    },

    # Robustness KPI (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
    "8ebc88f0-896c-4910-8997-a44d107e7eb7": {
        "name": "KPI-RF-078: Reward per action",
        "metric_key": "reward_per_action_ratio",
        "description": "Perturbed reward-per-action as a fraction of the unperturbed baseline [0-1+]"
    },
}


class MultiAttackerRobustnessTestRunner(TestRunner):
    """
    Single TestRunner that handles ALL 9 robustness/resilience KPIs.

    This TestRunner:
    1. Runs evaluation ONCE against all attackers
    2. Computes ALL metrics
    3. Returns the appropriate metric based on test_id

    Advantages:
    - Efficient: Only one evaluation per submission
    - Maintainable: Single codebase for all KPIs
    - Cacheable: Results cached across KPI requests
    """

    # Evaluation configuration
    ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC", "RLPerturb"]
    NUM_EPISODES = 50

    # Flatland environment configuration (matches the maze-flatland BC training setup)
    N_TRAINS = 3
    MAP_WIDTH = 37
    MAP_HEIGHT = 37
    N_CITIES = 2
    TREE_DEPTH = 2
    PREDICTION_DEPTH = 20
    ENV_SEED = 0
    ENV_NAME = "flatland-sparse-3trains-37x37"

    # Defender used when no submission is supplied. There is deliberately NO default:
    # the built-in defenders are development fixtures, not reference agents, and
    # silently benchmarking against one would publish KPI values that describe a toy
    # rather than a submission. An evaluation without an agent fails loudly instead.
    #
    # Opt in explicitly for local smoke tests only:
    #   "shortest-path"   reads the RailEnv directly, so obs attacks cannot affect it
    #   "do-nothing"      takes no action at all
    BASELINE_DEFENDER = None

    # Framework location (overridable for local testing, see test_local_*.py)
    FRAMEWORK_PATH = FRAMEWORK_PATH

    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str):
        """
        Initialize the TestRunner.

        Args:
            test_id: UUID for the specific KPI being evaluated
            scenario_ids: List of scenario UUIDs to evaluate
            benchmark_id: UUID for the benchmark suite
        """
        super().__init__(test_id=test_id, scenario_ids=scenario_ids, benchmark_id=benchmark_id)

        # Validate test_id is one of our 9 KPIs
        if test_id not in KPI_MAPPING:
            raise ValueError(
                f"Unknown test_id: {test_id}. "
                f"Expected one of: {list(KPI_MAPPING.keys())}"
            )

        self.kpi_info = KPI_MAPPING[test_id]

        # Metrics cache (key: f"{scenario_id}_{submission_id}")
        self._metrics_cache = {}

        # Defender agent (loaded in init())
        self._defender_agent = None

        # Set when no submission is given: the built-in ShortestPathDefender baseline
        # is constructed in _initialize_environment(), because it needs the very env
        # instance it will be stepped on.
        self._use_baseline_defender = False

        # Framework initialization flag
        self._framework_initialized = False

        logger.info(
            f"Initialized MultiAttackerRobustnessTestRunner\n"
            f"  Test ID: {test_id}\n"
            f"  KPI: {self.kpi_info['name']}\n"
            f"  Metric: {self.kpi_info['metric_key']}\n"
            f"  Scenarios: {scenario_ids}"
        )

    def _initialize_framework(self):
        """Add framework to Python path and validate it exists."""
        if self._framework_initialized:
            return

        if not os.path.exists(self.FRAMEWORK_PATH):
            raise FileNotFoundError(
                f"Framework not found at {self.FRAMEWORK_PATH}\n"
                f"Please ensure the framework folder exists in the railway directory."
            )

        # Add framework to path if not already there
        if self.FRAMEWORK_PATH not in sys.path:
            sys.path.insert(0, self.FRAMEWORK_PATH)

        logger.info(f"Framework initialized from: {self.FRAMEWORK_PATH}")

        self._framework_initialized = True

    def init(self, submission_data_url: str, submission_id: str = None):
        """
        Initialize and load defender agent from submission.

        Args:
            submission_data_url: URL to download the submitted defender agent
            submission_id: UUID of the submission (used for logging / temp file names)
        """
        super().init(submission_data_url=submission_data_url, submission_id=submission_id)

        logger.info(f"Loading defender agent from: {submission_data_url}")

        # Initialize framework
        self._initialize_framework()

        # Determine submission format and load agent
        try:
            if submission_data_url is None:
                # The baseline may need the very RailEnv it will be stepped on,
                # so it is built in _initialize_environment() instead.
                logger.warning(
                    f"No submission URL given - falling back to the "
                    f"'{self.BASELINE_DEFENDER}' development fixture. This is NOT a "
                    f"reference agent; do not report these KPI values as an evaluation."
                )
                self._use_baseline_defender = True
                self._defender_agent = None
            elif submission_data_url.endswith('.pkl'):
                self._defender_agent = self._load_agent_from_pickle(submission_data_url)
            elif submission_data_url.endswith('.zip'):
                self._defender_agent = self._load_agent_from_zip(submission_data_url)
            elif submission_data_url.startswith(('ghcr.io/', 'docker://')) or 'docker' in submission_data_url.lower():
                self._defender_agent = self._load_agent_from_docker(submission_data_url)
            else:
                # Default: try pickle
                self._defender_agent = self._load_agent_from_pickle(submission_data_url)

            logger.info("Defender agent loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load defender agent: {e}")
            raise

    def run_scenario(self, scenario_id: str, submission_id: str) -> Dict:
        """
        Run evaluation for a specific scenario and return the KPI value.

        Args:
            scenario_id: UUID of the scenario to evaluate
            submission_id: UUID of the submission

        Returns:
            Dictionary with "primary" key containing the KPI value
        """
        logger.info(
            f"Running scenario evaluation\n"
            f"  Scenario: {scenario_id}\n"
            f"  Submission: {submission_id}\n"
            f"  KPI: {self.kpi_info['name']}"
        )

        # Check cache
        cache_key = f"{scenario_id}_{submission_id}"

        if cache_key in self._metrics_cache:
            logger.info(f"Using cached metrics for {cache_key}")
            all_metrics = self._metrics_cache[cache_key]
        else:
            # Run complete evaluation
            logger.info(f"No cache found - running complete evaluation")
            all_metrics = self._run_complete_evaluation(scenario_id, submission_id)

            # Cache results
            self._metrics_cache[cache_key] = all_metrics
            logger.info(f"Cached metrics for {cache_key}")

        # Extract KPI-specific value
        metric_key = self.kpi_info['metric_key']
        kpi_value = all_metrics[metric_key]

        logger.info(
            f"KPI Result: {self.kpi_info['name']} = {kpi_value}\n"
            f"  Description: {self.kpi_info['description']}"
        )

        return {"primary": float(kpi_value)}

    def _run_complete_evaluation(self, scenario_id: str, submission_id: str) -> Dict:
        """
        Run complete multi-attacker evaluation and compute ALL metrics.

        Args:
            scenario_id: UUID of the scenario to evaluate
            submission_id: UUID of the submission

        Returns:
            Dictionary containing ALL computed metrics for all 9 KPIs
        """
        logger.info(
            f"Starting complete evaluation\n"
            f"  Attackers: {self.ATTACKER_TYPES}\n"
            f"  Episodes: {self.NUM_EPISODES}\n"
            f"  Environment: {self.ENV_NAME}"
        )

        # Import framework modules (now from framework/ folder)
        from evaluation_framework.result_getter import result_getter
        from evaluation_framework.metrics import metrics

        # Initialize environment
        env = self._initialize_environment(scenario_id)

        # Load attackers
        attackers = self._load_attackers(env)

        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Running episodes in temp directory: {temp_dir}")

            # Run evaluation using result_getter
            rg = result_getter(
                env=env,
                defender=self._defender_agent,
                n_episodes=self.NUM_EPISODES,
                save_folder=temp_dir,
                attackers=attackers
            )

            # This runs all episodes and computes metrics
            rg.calculate_metrics()

            # Load computed metrics
            logger.info("Loading computed metrics from pickle files")

            # Load unperturbed data
            with open(f"{temp_dir}/unperturbed.pkl", "rb") as f:
                unperturbed_data = pickle.load(f)

            # Load metrics for each attacker
            metrics_dicts = []
            for attacker in attackers:
                with open(f"{temp_dir}/{attacker.pickle_file}", "rb") as f:
                    data_dict = pickle.load(f)

                m = metrics(
                    data_dict,
                    unperturbed_data,
                    env.do_nothing_action(),
                    env.get_similarity_score,
                    model_name=attacker.model_name
                )
                metrics_dicts.append(m)

            # Aggregate metrics across all attackers
            all_metrics = self._aggregate_metrics(metrics_dicts, unperturbed_data)

        logger.info("Complete evaluation finished")

        return all_metrics

    def _aggregate_metrics(self, metrics_dicts: List, unperturbed_data: Dict) -> Dict:
        """
        Aggregate metrics from all attackers into a single result.

        Args:
            metrics_dicts: List of metrics objects from each attacker
            unperturbed_data: Dictionary with unperturbed episode data

        Returns:
            Dictionary with aggregated metrics for all 9 KPIs
        """
        logger.info(f"Aggregating metrics from {len(metrics_dicts)} attackers")

        # Extract metrics from each attacker
        vulnerability_scores = []
        steps_survived = []
        similarity_scores = []
        reward_drops = []
        action_change_freqs = []
        areas_between_curves = []
        degradation_times = []
        restoration_times = []
        state_similarities = []
        reward_per_action_ratios = []

        # Unperturbed reward baseline.
        #
        # DIVERGENCE FROM POWER GRID (see README_robustness_resilience_kpi_069_077.md):
        # power grid normalises the reward drop by the unperturbed total, which assumes
        # a POSITIVE baseline. Flatland rewards are <= 0 (a penalty for every step a
        # train has not yet arrived) and are exactly 0 for an agent that solves every
        # episode — so that formula is either sign-flipped or divides by zero here.
        # Instead the gap is normalised by the worst reward reachable over the same
        # number of steps, -(n_agents * n_steps), which keeps KPI-DF-069 in [0, 100]
        # and well defined for a perfect baseline.
        total_reward_unperturbed = sum(
            sum(r for r in ep if not np.isnan(r)) for ep in unperturbed_data['rewards']
        )
        total_steps_unperturbed = sum(
            sum(1 for r in ep if not np.isnan(r)) for ep in unperturbed_data['rewards']
        )
        worst_case_reward_magnitude = float(self.N_TRAINS * total_steps_unperturbed)
        reward_scale = max(abs(total_reward_unperturbed), worst_case_reward_magnitude)

        for m in metrics_dicts:
            logger.info(f"\n{'='*60}")
            logger.info(f"ATTACKER: {m.model_name}")
            logger.info(f"{'='*60}")

            # Robustness metrics for this attacker
            # perturb_vulnerability is computed with divide-by-zero suppressed, so it
            # can carry NaN for features that were never perturbed.
            vuln = float(np.nan_to_num(m.perturb_vulnerability.mean(), nan=0.0))
            steps = m.metrics_robustness['n_steps'].mean()

            # DIVERGENCE FROM POWER GRID (see README_robustness_resilience_kpi_069_077.md):
            # metrics.similarity_score is a SUM of per-step similarities accumulated over
            # the steps whose action changed, not a mean. Power grid feeds that sum
            # straight into 1 - sim, so KPI-SF-071 leaves its declared [0, 1] range as
            # soon as an attacker changes more than one action (railway agents routinely
            # see hundreds). Dividing by the number of changed actions recovers the mean
            # similarity among changed actions, so severity = 1 - sim stays in [0, 1].
            # An attacker that changed nothing scores severity 0, not 1.
            n_changed_total = m.metrics_robustness['n_actions_changed'].sum()
            sim_total = m.metrics_robustness['similarity_score'].sum()
            sim = sim_total / n_changed_total if n_changed_total > 0 else 1.0

            # Reward drop, normalised by the worst reward reachable over the same steps
            total_reward_perturbed = m.metrics_robustness['total_reward'].sum()
            if reward_scale > 0:
                reward_drop = 100 * (total_reward_unperturbed - total_reward_perturbed) / reward_scale
                # KPI-DF-069 is declared on a [0, 100] scale
                reward_drop = float(np.clip(reward_drop, 0.0, 100.0))
            else:
                reward_drop = 0.0

            # Action change frequency
            n_total = m.metrics_robustness['n_steps_with_act'].sum()
            action_freq = n_changed_total / n_total if n_total > 0 else 0

            # Resilience metrics
            if 'area_per_1000_steps' in m.metrics_resilience.columns:
                area = m.metrics_resilience['area_per_1000_steps'].values[0]
            elif 'area' in m.metrics_resilience.columns:
                area = m.metrics_resilience['area'].values[0]
            else:
                area = 0.0

            degr = m.metrics_resilience['degradation_time'].values[0]
            rest = m.metrics_resilience['restoration_time'].values[0]

            # Peak/valley detection yields no degraded states when the reward curve is
            # flat, which produces NaN in the weighted aggregation. Treat as zero.
            area = float(np.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0))
            degr = float(np.nan_to_num(degr, nan=0.0, posinf=0.0, neginf=0.0))
            rest = float(np.nan_to_num(rest, nan=0.0, posinf=0.0, neginf=0.0))

            # State similarity
            state_sim = np.mean([np.mean(ep) for ep in m.cos_similarity_all])
            state_sim = float(np.nan_to_num(state_sim, nan=1.0, posinf=1.0, neginf=-1.0))

            # KPI-RF-078: pooled within this attacker as sum(reward)/sum(actions), where an
            # action is one TRAIN acting (see metrics.get_reward_per_action_single_ep).
            #
            # The raw ratio can exceed 1: an attack that suppresses the agent's
            # interventions (more DO_NOTHING, which is not counted as an action) without
            # reducing trains delivered scores as "more value per action". Measured 3.00
            # for GEPerturb against the maze policy. That is efficiency, not robustness,
            # and on a leaderboard it would read as the attack having improved the agent.
            #
            # RF-078 is a robustness KPI, so it is capped at 1.0 for scoring: "at least as
            # good as the unperturbed baseline" is no degradation. The uncapped value stays
            # on metrics.reward_per_action["reward_per_action_ratio"] so the
            # suppressed-intervention signal is not lost, and is logged below.
            rpa_ratio_raw = m.reward_per_action["reward_per_action_ratio"]
            rpa_ratio = (min(float(rpa_ratio_raw), 1.0)
                         if rpa_ratio_raw is not None and np.isfinite(rpa_ratio_raw)
                         else rpa_ratio_raw)

            # Print all metrics for this attacker
            logger.info(f"  Robustness Metrics:")
            logger.info(f"    - Vulnerability:        {vuln:.4f}")
            logger.info(f"    - Steps Survived:       {steps:.1f}")
            logger.info(f"    - Severity of Change:   {1.0 - sim:.4f}")
            logger.info(f"    - Reward Drop (%):      {reward_drop:.2f}")
            logger.info(f"    - Action Change Freq:   {action_freq:.4f}")
            logger.info(f"  Resilience Metrics:")
            logger.info(f"    - Area Between Curves:  {area:.4f}")
            logger.info(f"    - Degradation Time:     {degr:.1f}")
            logger.info(f"    - Restoration Time:     {rest:.1f}")
            logger.info(f"    - State Similarity:     {state_sim:.4f}")
            logger.info(f"    - Reward/Action Ratio:  {rpa_ratio:.4f} "
                        f"(raw {rpa_ratio_raw:.4f}, capped at 1.0; "
                        f"target >= {REWARD_PER_ACTION_TARGET_RATIO})")

            # Append to lists
            vulnerability_scores.append(vuln)
            steps_survived.append(steps)
            similarity_scores.append(sim)
            reward_drops.append(reward_drop)
            action_change_freqs.append(action_freq)
            areas_between_curves.append(area)
            degradation_times.append(degr)
            restoration_times.append(rest)
            state_similarities.append(state_sim)
            reward_per_action_ratios.append(rpa_ratio)

        # Compute means across attackers
        aggregated = {
            # Robustness KPIs
            'perturb_vulnerability': np.mean(vulnerability_scores),
            'n_steps_survived': np.mean(steps_survived),
            'severity_of_change': 1.0 - np.mean(similarity_scores),  # Inverted (higher=worse)
            'reward_drop_percent': np.mean(reward_drops),
            'action_change_freq': np.mean(action_change_freqs),

            # Resilience KPIs
            'area_between_curves': np.mean(areas_between_curves),
            'degradation_time': np.mean(degradation_times),
            'restoration_time': np.mean(restoration_times),
            'state_similarity': np.mean(state_similarities),

            # KPI-RF-078. Mean across ATTACKERS, matching the other KPIs here; the pooling
            # that matters (sum(reward)/sum(actions)) already happened across episodes
            # inside each attacker's metrics object.
            'reward_per_action_ratio': self._mean_defined(reward_per_action_ratios),
        }

        # A NaN would be serialised into the KPI result as-is, so make the contract
        # explicit: every KPI is a finite float.
        for key, value in aggregated.items():
            if not np.isfinite(value):
                logger.warning(f"Aggregated metric {key} is not finite ({value}); reporting 0.0")
                aggregated[key] = 0.0

        # These are computed from the arrivals curve (see metrics.
        # get_resilience_metrics_performance_single_ep). All-zero now means the perturbed
        # runs never fell behind the baseline -- a real result, not a degenerate metric.
        # It is still worth flagging, because it also happens when an agent takes no
        # actions at all and is therefore trivially unaffected by observation attacks.
        if all(v == 0.0 for v in (aggregated['area_between_curves'],
                                  aggregated['degradation_time'],
                                  aggregated['restoration_time'])):
            logger.warning(
                "Resilience KPIs (AF-074, DF-075, RF-076) are all zero: no perturbed run "
                "ever fell behind the unperturbed baseline in trains arrived. Check "
                "KPI-FF-070 -- if the agent's actions never changed either, it is likely "
                "ignoring the observation (see the notes on baseline defenders) rather "
                "than being perfectly resilient."
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"AGGREGATED METRICS (Average across {len(metrics_dicts)} attackers)")
        logger.info(f"{'='*60}")
        logger.info(
            "\n".join([f"  {k}: {v:.4f}" for k, v in aggregated.items()])
        )

        return aggregated

    def _initialize_environment(self, scenario_id: str):
        """
        Initialize the Flatland environment for the given scenario.

        Args:
            scenario_id: UUID of the scenario

        Returns:
            FlatlandEnvironment wrapper with attacker support
        """
        logger.info(f"Initializing environment for scenario: {scenario_id}")

        from flatland.envs.rail_env import RailEnv
        from flatland.envs.rail_generators import sparse_rail_generator
        from flatland.envs.line_generators import sparse_line_generator
        from flatland.envs.observations import TreeObsForRailEnv
        from flatland.envs.predictions import ShortestPathPredictorForRailEnv

        from Environment import FlatlandEnvironment

        obs_builder = TreeObsForRailEnv(
            max_depth=self.TREE_DEPTH,
            predictor=ShortestPathPredictorForRailEnv(self.PREDICTION_DEPTH),
        )

        # Create base Flatland environment
        base_env = RailEnv(
            width=self.MAP_WIDTH,
            height=self.MAP_HEIGHT,
            rail_generator=sparse_rail_generator(
                max_num_cities=self.N_CITIES,
                grid_mode=False,
                max_rails_between_cities=3,
                max_rail_pairs_in_city=3,
                seed=self.ENV_SEED,
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=self.N_TRAINS,
            obs_builder_object=obs_builder,
        )

        # ShortestPathDefender reads the env directly rather than the observation
        # dict, so it can only be built once the env instance exists.
        if self._use_baseline_defender and self._defender_agent is None:
            if self.BASELINE_DEFENDER is None:
                raise ValueError(
                    "No agent supplied and no baseline defender selected. The built-in "
                    "defenders are development fixtures, not reference agents; scoring "
                    "against one would report KPI values for a toy rather than for a "
                    "submission. Pass a submission, or set BASELINE_DEFENDER explicitly "
                    "('shortest-path' or 'do-nothing') to smoke-test."
                )
            if self.BASELINE_DEFENDER == "shortest-path":
                from defenders import ShortestPathDefender
                self._defender_agent = ShortestPathDefender(env=base_env)
                logger.warning(
                    "ShortestPathDefender derives its actions from the RailEnv, not "
                    "from the observation it is given, so observation-space attacks "
                    "cannot change its actions. Expect near-zero robustness KPIs. "
                )
            elif self.BASELINE_DEFENDER == "do-nothing":
                from defenders import DoNothingDefender
                self._defender_agent = DoNothingDefender()
            else:
                raise ValueError(
                    f"Unknown BASELINE_DEFENDER: {self.BASELINE_DEFENDER!r}. "
                    f"Expected 'shortest-path' or 'do-nothing'."
                )

        if self._defender_agent is None:
            raise RuntimeError(
                "No defender agent loaded. Call init(submission_data_url=...) "
                "before run_scenario(), or set ._defender_agent directly."
            )

        # Wrap with FlatlandEnvironment for attacker support
        env = FlatlandEnvironment(base_env, self._defender_agent)

        logger.info(f"Environment initialized: {self.ENV_NAME}")

        return env

    def _load_attackers(self, env) -> List:
        """
        Load all attacker agents from the framework.

        Args:
            env: FlatlandEnvironment wrapper instance

        Returns:
            List of attacker agent objects
        """
        logger.info(f"Loading {len(self.ATTACKER_TYPES)} attacker types")

        # Import attacker classes from framework
        from attack_models.SACAttacker import SACAttacker
        from attack_models.PPOAttacker import PPOAttacker
        from attack_models.RLPerturbAttacker import RLPerturbAttacker
        from attack_models.GEPerturbAttacker import GEPerturbAttacker
        from attack_models.RandomPerturbAttacker import RandomPerturbAttacker
        from attack_models.LambdaPIRAttacker import LambdaPIRAttacker

        attackers = []
        n_agents = self.N_TRAINS

        for attacker_type in self.ATTACKER_TYPES:
            logger.info(f"Loading attacker: {attacker_type}")

            try:
                if attacker_type == "SAC":
                    attacker = SACAttacker(
                        n_agents=n_agents,
                        gamma=0.99,
                        alpha=0.2,
                        seed=6,
                    )
                elif attacker_type == "PPO":
                    attacker = PPOAttacker(
                        n_agents=n_agents,
                        gamma=0.99,
                        entropy_coef=0.05,
                        seed=5,
                    )
                elif attacker_type == "RLPerturb":
                    attacker = RLPerturbAttacker(
                        n_agents=n_agents,
                        epsilon=1.0,
                        gamma=0.99,
                        seed=4,
                    )
                elif attacker_type == "GEPerturb":
                    attacker = GEPerturbAttacker(
                        defender=self._defender_agent,
                        n_agents=n_agents,
                        n_candidates=8,
                        seed=3,
                    )
                elif attacker_type == "Random":
                    attacker = RandomPerturbAttacker(
                        perturbation_rate=0.3,
                        seed=1,
                    )
                elif attacker_type == "LambdaPIR":
                    attacker = LambdaPIRAttacker(
                        n_agents=n_agents,
                        lambda_param=0.7,
                        initial_prob_policy=0.2,
                        epsilon=1.0,
                        gradient_step_size=0.1,
                        refinement_iterations=20,
                        decay_schedule="exponential",
                        gamma=0.99,
                        name="LambdaPIRAttacker",
                        seed=2,
                    )
                else:
                    logger.warning(f"Unknown attacker type: {attacker_type}, skipping")
                    continue

                attackers.append(attacker)
                logger.info(f"Loaded attacker: {attacker_type}")

            except Exception as e:
                logger.error(f"Failed to load attacker {attacker_type}: {e}")
                continue

        logger.info(f"Successfully loaded {len(attackers)} attackers")

        return attackers

    def _load_agent_from_pickle(self, url: str):
        """
        Load agent from pickle file.

        The pickle must contain either a RailwayDefender (used as-is) or a policy
        object exposing act_many(handles, obs_list), which is wrapped in an
        ObsPolicyDefender.

        Args:
            url: URL to pickle file

        Returns:
            RailwayDefender instance
        """
        logger.info(f"Loading agent from pickle: {url}")

        if os.path.exists(url):
            # Local path (used by test_local_*.py)
            with open(url, 'rb') as f:
                obj = pickle.load(f)
        else:
            # Download file
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                f.write(response.content)
                temp_path = f.name

            with open(temp_path, 'rb') as f:
                obj = pickle.load(f)

            os.remove(temp_path)

        logger.info("Agent loaded from pickle successfully")

        return self._wrap_as_defender(obj)

    def _load_agent_from_zip(self, url: str):
        """
        Load agent from zip file.

        Two layouts are supported:
          1. A maze-flatland checkpoint: state_dict-*.pt + spaces_config.pkl
             -> wrapped in MazeFlatlandDefender.
          2. A pickled policy at agent.pkl / policy.pkl
             -> wrapped in ObsPolicyDefender.

        Args:
            url: URL to zip file containing the agent

        Returns:
            RailwayDefender instance
        """
        logger.info(f"Loading agent from zip: {url}")

        if os.path.exists(url):
            temp_zip = url
            cleanup_zip = False
        else:
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as f:
                f.write(response.content)
                temp_zip = f.name
            cleanup_zip = True

        temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        if cleanup_zip:
            os.remove(temp_zip)

        # Layout 1: maze-flatland checkpoint
        checkpoint = None
        spaces_config = None
        for root, _dirs, files in os.walk(temp_dir):
            for name in files:
                if name.startswith("state_dict") and name.endswith(".pt"):
                    checkpoint = os.path.join(root, name)
                elif name == "spaces_config.pkl":
                    spaces_config = os.path.join(root, name)

        if checkpoint and spaces_config:
            logger.info("Detected maze-flatland checkpoint layout")
            from defenders_maze import load_maze_policy, build_maze_env, MazeFlatlandDefender

            policy = load_maze_policy(checkpoint, spaces_config)
            maze_env = build_maze_env(
                n_trains=self.N_TRAINS,
                width=self.MAP_WIDTH,
                height=self.MAP_HEIGHT,
                n_cities=self.N_CITIES,
            )
            logger.info("Agent loaded from zip successfully (maze-flatland)")
            return MazeFlatlandDefender(maze_env, policy, n_trains=self.N_TRAINS)

        # Layout 2: pickled policy
        for candidate in ("agent.pkl", "policy.pkl"):
            for root, _dirs, files in os.walk(temp_dir):
                if candidate in files:
                    with open(os.path.join(root, candidate), 'rb') as f:
                        obj = pickle.load(f)
                    logger.info("Agent loaded from zip successfully (pickled policy)")
                    return self._wrap_as_defender(obj)

        raise ValueError(
            f"Unrecognised submission layout in {url}. Expected either a "
            f"maze-flatland checkpoint (state_dict-*.pt + spaces_config.pkl) "
            f"or a pickled policy at agent.pkl / policy.pkl."
        )

    @staticmethod
    def _mean_defined(values):
        """
        Mean over the entries that are defined, ignoring NaN.

        KPI-RF-078 is NaN whenever the baseline reward-per-action is zero or undefined.
        On Flatland that is not a corner case: the reward is a per-step penalty that sums
        to exactly 0 for an agent that gets every train home, so a PERFECT baseline makes
        the ratio undefined. Reporting 0.0 in that situation would read as catastrophic
        rather than as "not measurable", so the warning below is important context.

        Returns:
            float: mean of the defined values, or 0.0 when none are defined.
        """
        defined = [v for v in values if v is not None and np.isfinite(v)]
        if not defined:
            logger.warning(
                "KPI-RF-078 is undefined for every attacker. On Flatland this usually "
                "means the unperturbed baseline scored exactly 0 (every train arrived), "
                "which leaves reward-per-action undefined rather than bad. Reporting 0.0 "
                "-- do not read it as a failing agent without checking the baseline."
            )
            return 0.0
        return float(np.mean(defined))

    def _load_agent_from_docker(self, url: str):
        """
        Load agent from Docker image.

        Args:
            url: Docker image URL

        Returns:
            RailwayDefender instance
        """
        logger.info(f"Loading agent from Docker: {url}")

        # The other railway KPI runners (see abstract_test_runner_railway.py) drive
        # containerised policies via flatland-trajectory-generate-from-policy, which
        # replays a trajectory rather than exposing a live act() the attacker can
        # perturb step by step. Perturbation requires in-process policy inference.
        raise NotImplementedError(
            "Docker agent loading is not supported for the robustness/resilience KPIs. "
            "These KPIs perturb the agent's observation at every step, which requires "
            "in-process inference rather than trajectory replay. "
            "Please submit agents as a pickled policy (.pkl) or a zip archive."
        )

    @staticmethod
    def _wrap_as_defender(obj):
        """
        Wrap a loaded submission object into a RailwayDefender.

        Args:
            obj: Either a RailwayDefender or a policy with act_many(handles, obs_list)

        Returns:
            RailwayDefender instance
        """
        from defenders import RailwayDefender, ObsPolicyDefender

        if isinstance(obj, RailwayDefender):
            return obj
        if hasattr(obj, "act_many") or hasattr(obj, "act"):
            return ObsPolicyDefender(obj)

        raise TypeError(
            f"Submitted object of type {type(obj).__name__} is neither a "
            f"RailwayDefender nor a policy exposing act_many(handles, obs_list)."
        )


# ============================================================================
# Convenience aliases for each KPI (all use the same TestRunner)
# These allow FAB orchestrator to reference specific KPIs
# ============================================================================

# Robustness KPIs
class TestRunner_KPI_DF_069_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-DF-069: Drop-off in reward"""
    pass

class TestRunner_KPI_FF_070_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-FF-070: Frequency changed output AI agent"""
    pass

class TestRunner_KPI_SF_071_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-SF-071: Severity of changed output AI agent"""
    pass

class TestRunner_KPI_SF_072_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-SF-072: Steps survived with perturbations"""
    pass

class TestRunner_KPI_VF_073_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-VF-073: Vulnerability to perturbation"""
    pass

# Resilience KPIs
class TestRunner_KPI_AF_074_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-AF-074: Area between reward curves"""
    pass

class TestRunner_KPI_DF_075_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-DF-075: Degradation time"""
    pass

class TestRunner_KPI_RF_076_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-RF-076: Restorative time"""
    pass

class TestRunner_KPI_SF_077_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-SF-077: Similarity state to unperturbed situation"""
    pass

class TestRunner_KPI_RF_078_Railway(MultiAttackerRobustnessTestRunner):
    """KPI-RF-078: Reward per action"""
    pass
