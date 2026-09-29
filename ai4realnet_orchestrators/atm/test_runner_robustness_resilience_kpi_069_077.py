"""
Multi-Attacker Robustness & Resilience TestRunner for the ATM domain (BlueSky-Gym).

One TestRunner evaluates a submitted ATM control agent against several bounded
observation-space attacks and returns a different KPI value depending on test_id.

KPIs implemented (ATM test ids, as registered in atm/orchestrator.py):
    - KPI-DF-069: Drop-off in reward
    - KPI-FF-070: Frequency changed output AI agent
    - KPI-SF-071: Severity of changed output AI agent
    - KPI-SF-072: Steps survived with perturbations
    - KPI-VF-073: Vulnerability to perturbation
    - KPI-AF-074: Area between reward curves
    - KPI-DF-075: Degradation time
    - KPI-RF-076: Restorative time
    - KPI-SF-077: Similarity state to unperturbed situation
    - KPI-RF-078: Reward per action

Design mirrors railway/test_runner_robustness_resilience_kpi_069_077.py: a single
evaluation runs every attacker once, computes all metrics, and serves the metric the
requested KPI asks for.

Environment: BlueSky-Gym "StaticObstacleEnv-v0" - the single-agent, single-objective ATM
control task (one scalar reward: waypoint reward, drift penalty, restricted-area
intrusion penalty). The submitted agent is a Stable-Baselines3 policy.

Author: INESC TEC
"""

import logging
import os
import pickle
import sys
import tempfile
import zipfile
from typing import Dict, List

import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(SCRIPT_DIR, "framework")
if FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, FRAMEWORK_PATH)

_parent_dir = os.path.dirname(SCRIPT_DIR)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from test_runner import TestRunner

from evaluation_framework.metrics import REWARD_PER_ACTION_TARGET_RATIO

logger = logging.getLogger(__name__)

ENV_ID = "StaticObstacleEnv-v0"
MAX_EPISODE_STEPS = 200

# ============================================================================
# KPI ID mapping (ATM test ids from atm/orchestrator.py)
# ============================================================================

KPI_MAPPING = {
    "4819e8f6-a2d4-497f-9b61-fc90883a0dfb": {
        "name": "KPI-DF-069: Drop-off in reward",
        "metric_key": "reward_drop_percent",
        "description": "Percentage decrease in reward [0-100]",
    },
    "f0f94fb1-2aef-44f6-ba80-5b8320725fb0": {
        "name": "KPI-FF-070: Frequency changed output AI agent",
        "metric_key": "action_change_freq",
        "description": "Fraction of steps whose action changed [0-1]",
    },
    "02bfbe09-6e9b-4243-a376-1a51b1beef19": {
        "name": "KPI-SF-071: Severity of changed output AI agent",
        "metric_key": "severity_of_change",
        "description": "Severity of action changes [0-1, higher=worse]",
    },
    "c466661d-12dc-4d1e-81a4-1db1623e3cc1": {
        "name": "KPI-SF-072: Steps survived with perturbations",
        "metric_key": "n_steps_survived",
        "description": "Number of timesteps before the episode ends",
    },
    "5cfc7e4d-024b-4dd1-82a5-c3d9bf25ba50": {
        "name": "KPI-VF-073: Vulnerability to perturbation",
        "metric_key": "perturb_vulnerability",
        "description": "Proportion of features vulnerable to attack [0-1]",
    },
    "5372decd-6a2a-4c50-bf7a-cd57cfebe3de": {
        "name": "KPI-AF-074: Area between reward curves",
        "metric_key": "area_between_curves",
        "description": "Area between perturbed and unperturbed performance curves",
    },
    "2baef867-c1f2-4b6e-b13c-0eac9463c2fa": {
        "name": "KPI-DF-075: Degradation time",
        "metric_key": "degradation_time",
        "description": "Steps spent degrading after a perturbation",
    },
    "7b15a7b3-2413-4953-b91a-24f5c0c5b6da": {
        "name": "KPI-RF-076: Restorative time",
        "metric_key": "restoration_time",
        "description": "Steps needed to recover after a perturbation",
    },
    "e3fb76a2-2121-4889-adf2-b60ca29c5c71": {
        "name": "KPI-SF-077: Similarity state to unperturbed situation",
        "metric_key": "state_similarity",
        "description": "Cosine similarity to unperturbed states [-1 to 1]",
    },
    "885cab0d-d4fc-4d93-95db-243870506405": {
        "name": "KPI-RF-078: Reward per action",
        "metric_key": "reward_per_action_ratio",
        "description": "Perturbed reward-per-action as a fraction of the unperturbed baseline [0-1]",
    },
}


class ATMRobustnessTestRunner(TestRunner):
    """Runs every attacker once, computes all metrics, serves the requested KPI."""

    # Same suite as power_grid: one lambda attacker (lambda-PIR), the two learned
    # attackers at two budgets (SAC_5 / SAC_10), PPO, RLPerturb, GEPerturb and Random.
    ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC_10", "SAC_5", "RLPerturb"]
    N_EPISODES = 10
    XI = 0.1                      # L-infinity budget per observation feature
    N_AIRCRAFT = 1                # StaticObstacleEnv-v0 controls a single aircraft

    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str = None, **kwargs):
        super().__init__(test_id=test_id, scenario_ids=scenario_ids, benchmark_id=benchmark_id, **kwargs)
        self._metrics_cache = {}
        self._defender_agent = None

        if test_id not in KPI_MAPPING:
            raise ValueError(
                f"Unknown ATM robustness/resilience test_id: {test_id}. "
                f"Expected one of: {list(KPI_MAPPING.keys())}"
            )
        self.kpi_info = KPI_MAPPING[test_id]
        logger.info(f"Initialized ATMRobustnessTestRunner for {self.kpi_info['name']}")

    # ------------------------------------------------------------------
    # Submission loading
    # ------------------------------------------------------------------

    def init(self, submission_data_url: str, submission_id: str = None):
        super().init(submission_data_url=submission_data_url, submission_id=submission_id)
        logger.info(f"Loading ATM defender from: {submission_data_url}")

        if submission_data_url is None:
            # The straight-line reference is built later, in _initialize_environment.
            logger.warning(
                "No submission URL given - falling back to the StraightLineDefender "
                "reference agent; do not report these KPI values as an evaluation."
            )
            self._defender_agent = None
            return

        local_path = submission_data_url
        if submission_data_url.startswith(("http://", "https://")):
            local_path = self._download(submission_data_url)

        if local_path.endswith(".zip") and not self._is_sb3_zip(local_path):
            local_path = self._extract_model_from_archive(local_path)

        from defenders import SB3Defender
        self._defender_agent = SB3Defender(local_path)
        logger.info(f"Defender loaded: {self._defender_agent.name}")

    @staticmethod
    def _is_sb3_zip(path: str) -> bool:
        """An SB3 checkpoint is a zip containing 'data' and 'policy.pth'."""
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
            return "data" in names and "policy.pth" in names
        except Exception:
            return False

    def _download(self, url: str) -> str:
        suffix = ".zip" if url.endswith(".zip") else ""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return path

    def _extract_model_from_archive(self, path: str) -> str:
        """A submission archive that wraps the SB3 zip: take the first SB3 checkpoint inside."""
        out_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
        for root, _, files in os.walk(out_dir):
            for name in sorted(files):
                candidate = os.path.join(root, name)
                if name.endswith(".zip") and self._is_sb3_zip(candidate):
                    return candidate
        raise RuntimeError(f"No Stable-Baselines3 checkpoint found inside {path}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_scenario(self, scenario_id: str, submission_id: str) -> Dict:
        cache_key = f"{scenario_id}_{submission_id}"
        if cache_key not in self._metrics_cache:
            self._metrics_cache[cache_key] = self._run_complete_evaluation(scenario_id, submission_id)
        all_metrics = self._metrics_cache[cache_key]

        value = all_metrics[self.kpi_info["metric_key"]]
        logger.info(f"{self.kpi_info['name']} = {value}")
        return {"primary": float(value)}

    def _initialize_environment(self, seed_env=True):
        import gymnasium as gym
        import bluesky_gym
        import bluesky_gym.envs  # noqa: F401  (registers the envs)

        bluesky_gym.register_envs()
        env = gym.make(ENV_ID, render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

        if self._defender_agent is None:
            from defenders import StraightLineDefender
            self._defender_agent = StraightLineDefender()

        from Environment import BlueSkyGymEnvironment
        return BlueSkyGymEnvironment(env, self._defender_agent)

    def _load_attackers(self) -> List:
        from attack_models.GEPerturbAttacker import GEPerturbAttacker
        from attack_models.LambdaPIRAttacker import LambdaPIRAttacker
        from attack_models.PPOAttacker import PPOAttacker
        from attack_models.RLPerturbAttacker import RLPerturbAttacker
        from attack_models.RPerturbAttacker import RPerturbAttacker
        from attack_models.SACAttacker import SACAttacker

        d = self._defender_agent
        attackers = []
        for name in self.ATTACKER_TYPES:
            try:
                if name == "Random":
                    attackers.append(RPerturbAttacker(xi=self.XI, seed=1))
                elif name == "GEPerturb":
                    attackers.append(GEPerturbAttacker(agent=d, n_candidates=8, xi=self.XI, seed=3))
                elif name == "LambdaPIR":
                    attackers.append(LambdaPIRAttacker(agent=d, xi=self.XI, seed=2))
                elif name == "RLPerturb":
                    attackers.append(RLPerturbAttacker(agent=d, xi=self.XI, seed=4))
                elif name == "PPO":
                    attackers.append(PPOAttacker(agent=d, xi=self.XI, seed=5))
                elif name == "SAC_5":
                    # SAC_5 / SAC_10 differ only in budget (5% / 10%), as power_grid's `factor` does
                    attackers.append(SACAttacker(agent=d, factor=5, seed=6))
                elif name == "SAC_10":
                    attackers.append(SACAttacker(agent=d, factor=10, seed=7))
                else:
                    logger.warning(f"Unknown attacker type: {name}, skipping")
            except Exception as e:
                logger.error(f"Failed to load attacker {name}: {e}")
        logger.info(f"Loaded {len(attackers)} attackers: {[a.model_name for a in attackers]}")
        return attackers

    def _run_complete_evaluation(self, scenario_id: str, submission_id: str) -> Dict:
        from evaluation_framework.metrics import metrics
        from evaluation_framework.result_getter import result_getter

        logger.info(
            f"Running ATM robustness evaluation\n"
            f"  Env: {ENV_ID}\n  Episodes: {self.N_EPISODES}\n"
            f"  Attackers: {self.ATTACKER_TYPES}\n  xi: {self.XI}"
        )

        env = self._initialize_environment()
        attackers = self._load_attackers()

        with tempfile.TemporaryDirectory() as temp_dir:
            rg = result_getter(env, self._defender_agent, self.N_EPISODES, temp_dir, attackers)
            rg.calculate_metrics()

            with open(f"{temp_dir}/unperturbed.pkl", "rb") as f:
                unperturbed_data = pickle.load(f)

            metrics_dicts = []
            for attacker in attackers:
                # result_getter writes each attacker's pickle under its own pickle_file name
                with open(f"{temp_dir}/{attacker.pickle_file}", "rb") as f:
                    data = pickle.load(f)
                metrics_dicts.append(
                    metrics(data, unperturbed_data, env.do_nothing_action(),
                            env.get_similarity_score, model_name=attacker.model_name)
                )

            # Realised budget per attacker: a KPI computed from an attack that exceeded its
            # declared budget would not describe the threat model it claims to.
            for attacker in attackers:
                stats = attacker.get_stats() if hasattr(attacker, "get_stats") else {}
                realised = stats.get("max_perturb_per_key", {})
                worst = max(realised.values()) if realised else 0.0
                budget = getattr(attacker, "xi", self.XI)   # SAC_5/SAC_10 carry their own budget
                logger.info(f"  {attacker.model_name}: max |perturbation| = {worst:.6g} (budget {budget})")
                if worst > budget * (1 + 1e-6):
                    logger.warning(f"  {attacker.model_name} EXCEEDED its budget: {worst} > {budget}")

            return self._aggregate_metrics(metrics_dicts, unperturbed_data)

    # ------------------------------------------------------------------
    # Aggregation (mirrors the railway runner)
    # ------------------------------------------------------------------

    def _aggregate_metrics(self, metrics_dicts: List, unperturbed_data: Dict) -> Dict:
        vulnerability_scores, steps_survived, similarity_scores = [], [], []
        reward_drops, action_change_freqs = [], []
        areas, degradation_times, restoration_times = [], [], []
        state_similarities, reward_per_action_ratios = [], []

        total_reward_unperturbed = sum(
            sum(r for r in ep if not np.isnan(r)) for ep in unperturbed_data["rewards"]
        )
        total_steps_unperturbed = sum(
            sum(1 for r in ep if not np.isnan(r)) for ep in unperturbed_data["rewards"]
        )

        # DIVERGENCE FROM POWER GRID, for the same reason as railway: normalising the
        # reward gap by the unperturbed total assumes a positive baseline. The ATM reward
        # mixes a +1 waypoint reward with -5 intrusion penalties and a small per-step drift
        # penalty, so the baseline total can be near zero or negative. Normalising instead
        # by the worst reward reachable over the same number of steps keeps KPI-DF-069 in
        # [0, 100] and defined for any baseline.
        worst_case_reward_magnitude = float(
            abs(RESTRICTED_AREA_INTRUSION_PENALTY) * self.N_AIRCRAFT * total_steps_unperturbed
        )
        reward_scale = max(abs(total_reward_unperturbed), worst_case_reward_magnitude)

        for m in metrics_dicts:
            vuln = float(np.nan_to_num(m.perturb_vulnerability.mean(), nan=0.0))
            steps = m.metrics_robustness["n_steps"].mean()

            # As in railway: similarity_score is a SUM over changed actions, so divide by
            # the number of changed actions to keep severity = 1 - sim inside [0, 1].
            n_changed_total = m.metrics_robustness["n_actions_changed"].sum()
            sim_total = m.metrics_robustness["similarity_score"].sum()
            sim = sim_total / n_changed_total if n_changed_total > 0 else 1.0

            total_reward_perturbed = m.metrics_robustness["total_reward"].sum()
            reward_drop = (float(np.clip(100 * (total_reward_unperturbed - total_reward_perturbed)
                                         / reward_scale, 0.0, 100.0))
                           if reward_scale > 0 else 0.0)

            n_total = m.metrics_robustness["n_steps_with_act"].sum()
            action_freq = n_changed_total / n_total if n_total > 0 else 0

            if "area_per_1000_steps" in m.metrics_resilience.columns:
                area = m.metrics_resilience["area_per_1000_steps"].values[0]
            elif "area" in m.metrics_resilience.columns:
                area = m.metrics_resilience["area"].values[0]
            else:
                area = 0.0
            degr = m.metrics_resilience["degradation_time"].values[0]
            rest = m.metrics_resilience["restoration_time"].values[0]

            area = float(np.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0))
            degr = float(np.nan_to_num(degr, nan=0.0, posinf=0.0, neginf=0.0))
            rest = float(np.nan_to_num(rest, nan=0.0, posinf=0.0, neginf=0.0))

            state_sim = float(np.nan_to_num(
                np.mean([np.mean(ep) for ep in m.cos_similarity_all]), nan=1.0, posinf=1.0, neginf=-1.0))

            # Capped at 1.0 for scoring, as in railway: "at least as good as the
            # unperturbed baseline" is no degradation. The uncapped value stays on the
            # metrics object.
            rpa_raw = m.reward_per_action["reward_per_action_ratio"]
            rpa = (min(float(rpa_raw), 1.0)
                   if rpa_raw is not None and np.isfinite(rpa_raw) else rpa_raw)

            logger.info(
                f"  {m.model_name}: vuln={vuln:.4f} steps={steps:.1f} severity={1.0 - sim:.4f} "
                f"reward_drop={reward_drop:.2f}% act_freq={action_freq:.4f} area={area:.4f} "
                f"degr={degr:.1f} rest={rest:.1f} state_sim={state_sim:.4f} "
                f"rpa={rpa if rpa is None else float(rpa):.4f} (target >= {REWARD_PER_ACTION_TARGET_RATIO})"
            )

            vulnerability_scores.append(vuln)
            steps_survived.append(steps)
            similarity_scores.append(sim)
            reward_drops.append(reward_drop)
            action_change_freqs.append(action_freq)
            areas.append(area)
            degradation_times.append(degr)
            restoration_times.append(rest)
            state_similarities.append(state_sim)
            reward_per_action_ratios.append(rpa)

        aggregated = {
            "perturb_vulnerability": np.mean(vulnerability_scores),
            "n_steps_survived": np.mean(steps_survived),
            "severity_of_change": 1.0 - np.mean(similarity_scores),
            "reward_drop_percent": np.mean(reward_drops),
            "action_change_freq": np.mean(action_change_freqs),
            "area_between_curves": np.mean(areas),
            "degradation_time": np.mean(degradation_times),
            "restoration_time": np.mean(restoration_times),
            "state_similarity": np.mean(state_similarities),
            "reward_per_action_ratio": self._mean_defined(reward_per_action_ratios),
        }

        for key, value in aggregated.items():
            if not np.isfinite(value):
                logger.warning(f"Aggregated metric {key} is not finite ({value}); reporting 0.0")
                aggregated[key] = 0.0

        return aggregated

    @staticmethod
    def _mean_defined(values):
        defined = [float(v) for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(defined)) if defined else 0.0


# The intrusion penalty that sets the worst-case reward scale above; taken from
# bluesky_gym/envs/static_obstacle_env.py so the two cannot drift apart silently.
try:
    from bluesky_gym.envs.static_obstacle_env import RESTRICTED_AREA_INTRUSION_PENALTY
except Exception:  # bluesky_gym not importable at import time (e.g. orchestrator registration)
    RESTRICTED_AREA_INTRUSION_PENALTY = -5
