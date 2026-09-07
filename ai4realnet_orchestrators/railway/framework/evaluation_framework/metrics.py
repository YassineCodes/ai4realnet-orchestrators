import numpy as np
import pandas as pd
from pandas import isna

from scipy.integrate import trapezoid
from scipy.signal import find_peaks


# KPI-RF-078: Reward per action. Same threshold and same headline number as power grid
# (see power_grid/framework/evaluation_framework/metrics.py) so the two domains stay
# comparable. What counts as an "action" necessarily differs -- see
# get_reward_per_action_single_ep() below.
REWARD_PER_ACTION_TARGET_RATIO = 0.90


class metrics:
    """
        Class for computing robustness and resilience metrics for reinforcement learning agents under perturbations.
        This class evaluates the performance of an agent by comparing its behavior and rewards in perturbed and unperturbed environments.         
        Attributes:
            similarity_score_fn (Callable): Function to compute similarity scores between two actions passed as np.ndarray.
            model_name (str): Name of the model being evaluated.
            
            rewards_unperturbed (list): Rewards from the environment without a perturbation agent.
            rewards_perturbed (list): Rewards from the environment with a perturbation agent.
            cos_similarity_all (list): Cosine similarity between the observation in the environment with and without perturbation agent in each step.
            euclidean_dist_all (list): Euclidean distances between the observation in the environment with and without perturbation agent in each step.

            metrics_robustness (pd.DataFrame): DataFrame containing raw robustness metrics for each episode.
            metrics_resilience (pd.DataFrame): DataFrame containing raw resilience metrics.
            metrics_resilience_obs_sim (pd.DataFrame): DataFrame containing resilience metrics calculated using observation similarity measure.
            perturb_vulnerability (np.ndarray): Proportion of significant perturbations resulting in changed actions.
        Methods:
            compute_metrics(...): Computes robustness and resilience metrics for all episodes and gets the average.
            get_robustness_metrics_single_ep(...): Computes robustness metrics for a single episode.
            get_resilience_metrics_single_ep(...): Computes resilience metrics for a single episode.
            aggregate_metrics_resilience(...): Aggregates resilience metrics across episodes.
            get_perturb_vulnerability(...): Computes the vulnerability of each data point in the observation to perturbations.
            compute_perturb_prop_single_ep(...): Computes the proportion of significant perturbations that result in changed actions for a single episode.
    """

    def __init__(self, data_dict_perturbed, data_dict_unperturbed, do_nothing_action, similarity_score_fn, model_name="",
                 recovery_action_fn=None):
        """
            Initializes the metrics instance and computes robustness and resilience metrics for the provided episodes.
            Args:
                data_dict_perturbed (dict): Dictionary containing data from the environment with the perturbation agent to evaluate, including observations, actions, perturbations, and rewards.
                data_dict_unperturbed (dict): Dictionary containing data from the environment without a perturbation agent, including observations and rewards.
                do_nothing_action (np.ndarray): The action representing a 'do nothing' or baseline action in the environment.
                similarity_score_fn (Callable): Function to compute similarity scores between two actions passed as np.ndarrays.
                model_name (str, optional): Name of the model being evaluated. Defaults to an empty string.
        """

        obs_unperturb = data_dict_unperturbed["observations"]
        obs_perturb = data_dict_perturbed["observations"]
        perturbations = data_dict_perturbed["perturbations"]
        actions_unperturbed = data_dict_perturbed["actions_unperturbed"]
        actions_perturbed = data_dict_perturbed["actions"]
        rewards_unperturbed = data_dict_unperturbed["rewards"]
        rewards_perturbed = data_dict_perturbed["rewards"]

        # KPI-RF-078 needs the action stream of the BASELINE rollout, which is a different
        # rollout from actions_unperturbed (the counterfactual actions recorded *inside* the
        # perturbed rollout). Older pickles predate this and simply yield NaN.
        actions_baseline = data_dict_unperturbed.get("actions", None)

        # Per-step performance (fraction of trains arrived), recorded by result_getter.
        # Absent in pickles written before this existed -> the resilience KPIs then fall
        # back to the old reward-curve behaviour.
        performance_perturbed = data_dict_perturbed.get("performance", None)
        performance_unperturbed = data_dict_unperturbed.get("performance", None)

        self.similarity_score_fn = similarity_score_fn
        self.recovery_action_fn = recovery_action_fn
        self.model_name = model_name

        self.rewards_unperturbed = rewards_unperturbed
        self.rewards_perturbed = rewards_perturbed

        self.metrics_robustness, self.metrics_resilience, self.metrics_resilience_obs_sim = None, None, None

        self.compute_metrics(obs_unperturb, obs_perturb, perturbations, actions_unperturbed, actions_perturbed, 
                            rewards_unperturbed, rewards_perturbed, do_nothing_action,
                            actions_baseline=actions_baseline,
                            performance_perturbed=performance_perturbed,
                            performance_unperturbed=performance_unperturbed)


    def compute_metrics(self, obs_unperturb, obs_perturb, perturbations, actions_unperturbed, actions_perturbed, rewards_unperturbed, 
            rewards_perturbed, do_nothing_action, actions_baseline=None,
            performance_perturbed=None, performance_unperturbed=None):
        # initialize lists to store metrics
        metrics_robustness, metrics_resilience, metrics_resilience_obs_sim = [[] for _ in range(3)]

        # initialize list of columns for metrics 
        # KPI-RF-078 columns are APPENDED, never inserted, and carry the same names as the
        # power grid implementation so downstream tooling can read either domain.
        cols_robustness = ["episode", "n_steps_with_act", "n_actions_changed", "similarity_score", "total_reward", "n_steps", "ave_reward_per_step",
                           "n_actions", "n_actions_excl_recovery", "ave_reward_per_action",
                           "n_actions_unperturbed", "ave_reward_per_action_unperturbed",
                           "reward_per_action_ratio"]
        cols_resilience = ["episode", "degradation_time", "restoration_time", "min_reward", "max_reward", "n_steps", "area", "n_degr_states"]
            
        self.cos_similarity_all = []
        self.euclidean_dist_all = []
        
        for ep in range(len(obs_perturb)):
            ep_rewards_unpert = rewards_unperturbed[ep] if ep < len(rewards_unperturbed) else []
            ep_rewards_pert = rewards_perturbed[ep] if ep < len(rewards_perturbed) else []
            
            if len(ep_rewards_unpert) == 0 or len(ep_rewards_pert) == 0:
                print(f"Warning: Empty rewards for episode {ep}, using default metrics")
                results_resilience = {'degradation_time': 0, 'restoration_time': 0, 'area': 0, 'area_per_1000_steps': 0}
            else:
                results_resilience = self.get_resilience_metrics_single_ep(
                    ep_rewards_unpert, ep_rewards_pert,
                    distance_peaks=500, distance_valleys=500
                )
            # compute robustness metrics and similarity in observation
            cos_similarity, euclidean_dist, actions_changed, similarity_score, n_actions = self.get_robustness_metrics_single_ep(obs_unperturb[ep], obs_perturb[ep], actions_unperturbed[ep], actions_perturbed[ep], do_nothing_action)
            r = [x for x in rewards_perturbed[ep] if not isna(x)]
            # ---- KPI-RF-078: reward per action (perturbed rollout) ----
            perf_p = (performance_perturbed[ep]
                      if performance_perturbed is not None and ep < len(performance_perturbed) else None)
            rpa = self.get_reward_per_action_single_ep(
                actions_perturbed[ep], rewards_perturbed[ep], do_nothing_action,
                performance=perf_p
            )
            # ---- KPI-RF-078: same measurement on the unperturbed baseline rollout ----
            if actions_baseline is not None and ep < len(actions_baseline):
                perf_u = (performance_unperturbed[ep]
                          if performance_unperturbed is not None and ep < len(performance_unperturbed) else None)
                rpa_base = self.get_reward_per_action_single_ep(
                    actions_baseline[ep], rewards_unperturbed[ep], do_nothing_action,
                    performance=perf_u
                )
            else:
                rpa_base = {"n_actions": np.nan, "n_actions_excl_recovery": np.nan,
                            "total_reward": np.nan, "ave_reward_per_action": np.nan}

            base_rpa = rpa_base["ave_reward_per_action"]
            if isna(base_rpa) or base_rpa == 0:
                # Undefined rather than 0. NOTE this fires routinely on Flatland: the
                # reward is a per-step penalty that sums to exactly 0 for an agent that
                # gets every train home, so a perfect baseline leaves the ratio undefined.
                reward_per_action_ratio = np.nan
            else:
                reward_per_action_ratio = rpa["ave_reward_per_action"] / base_rpa

            metrics_robustness_ep = [ep, n_actions, actions_changed, similarity_score, sum(r), len(r), sum(r) / len(r),
                                     rpa["n_actions"], rpa["n_actions_excl_recovery"], rpa["ave_reward_per_action"],
                                     rpa_base["n_actions"], base_rpa, reward_per_action_ratio]
            metrics_robustness.append(metrics_robustness_ep)

            # Scale peak-detection distance to episode length (~10% of steps, min 1).
            # Hardcoded 500 was tuned for power-grid (1000+ steps); railway episodes
            # are ~65 steps, making distance=500 guarantee zero peaks are found.
            ep_len = max(len(r), 1)
            dist = max(1, ep_len // 10)

            # ---- resilience metrics (KPI-AF-074 / DF-075 / RF-076) ----
            # Flatland's reward is sparse and terminal: exactly 0.0 every step with a
            # single penalty at the end. A reward curve therefore has no shape to detect
            # degradation or restoration in, and an agent that solves the scenario has an
            # all-zero baseline that the relative-difference formula divides by. Both make
            # these three KPIs identically zero for every agent.
            # The fraction of trains arrived is used instead when available: it varies
            # through the episode and falls behind the baseline exactly when an attack
            # delays trains. Compared as an ABSOLUTE difference in percentage points,
            # because the curve legitimately starts at zero.
            if (performance_perturbed is not None and performance_unperturbed is not None
                    and ep < len(performance_perturbed) and ep < len(performance_unperturbed)
                    and len(performance_perturbed[ep]) > 0):
                results_resilience = self.get_resilience_metrics_performance_single_ep(
                    performance_unperturbed[ep], performance_perturbed[ep])
            else:
                results_resilience = self.get_resilience_metrics_single_ep(rewards_unperturbed[ep], rewards_perturbed[ep], distance_peaks=dist, distance_valleys=dist)
            results_resilience = [ep] + [np.mean(r) if len(r) > 0 else 0 for r in results_resilience] + [len(results_resilience[0])]
            metrics_resilience.append(results_resilience)

            # compute resilience metrics for cosine similarity
            obs_dist = max(1, len(cos_similarity) // 10) if cos_similarity else 1
            results_resilience_similarity = self.get_resilience_metrics_single_ep(np.ones_like(cos_similarity), cos_similarity, distance_peaks=obs_dist, distance_valleys=obs_dist)
            results_resilience_similarity = [ep] + [np.mean(r) if len(r) > 0 else 0 for r in results_resilience_similarity] + [len(results_resilience_similarity[0])]
            metrics_resilience_obs_sim.append(results_resilience_similarity)

            self.cos_similarity_all.append(cos_similarity)
            self.euclidean_dist_all.append(euclidean_dist)

        # combine robustness metrics into one dataframe
        metrics_robustness = pd.DataFrame(metrics_robustness, columns=cols_robustness)
        metrics_robustness[metrics_robustness.columns[2:]] = metrics_robustness[metrics_robustness.columns[2:]].astype(float)
        self.metrics_robustness = metrics_robustness

        # KPI-RF-078 headline figure, pooled over episodes as sum(reward)/sum(actions).
        # NOT the mean of the per-episode ratios: episodes differ in length and in how many
        # train-actions they contain, so a plain mean would misweight short episodes.
        self.reward_per_action = self.aggregate_reward_per_action(metrics_robustness)

        # combine resilience metrics and get the mean for each perturbation agent
        metrics_resilience = pd.DataFrame(metrics_resilience, columns=cols_resilience)
        metrics_resilience = pd.DataFrame(self.aggregate_metrics_resilience(metrics_resilience)).T
        self.metrics_resilience = metrics_resilience

        # combine resilience metrics and get the mean for each perturbation agent
        metrics_resilience_obs_sim = pd.DataFrame(metrics_resilience_obs_sim, columns=cols_resilience)
        metrics_resilience_obs_sim = pd.DataFrame(self.aggregate_metrics_resilience(metrics_resilience_obs_sim)).T
        self.metrics_resilience_obs_sim = metrics_resilience_obs_sim

        np.seterr(divide = 'ignore', invalid='ignore') 
        perturb_vulnerability = self.get_perturb_vulnerability(perturbations, actions_perturbed, actions_unperturbed)
        self.perturb_vulnerability = perturb_vulnerability
        np.seterr(divide = 'warn', invalid='warn')

        return metrics_robustness, metrics_resilience, metrics_resilience_obs_sim, perturb_vulnerability

    def get_reward_per_action_single_ep(self, actions, rewards, do_nothing_action, performance=None):
        """
        Computes KPI-RF-078 (reward per action) for a single episode.

        WHAT COUNTS AS AN ACTION (differs from power grid -- read this)
        --------------------------------------------------------------
        Railway is multi-agent. An action vector here is one integer PER TRAIN
        (see Environment._flatten_actions), not a single flat action vector, so the
        power grid rule `(act != do_nothing).any()` would collapse a whole timestep into
        one count no matter how many trains moved.

        This implementation counts **per-train actions, summed across trains**:

            n_actions += (act != do_nothing_action).sum()

        So a step in which all 3 trains act contributes 3, not 1. Against the
        "any train acted" reading this differs by up to a factor of n_trains, which is why
        the choice is spelled out here rather than inherited silently.

        RailEnvActions.DO_NOTHING (0) is the only value that does not count.
        In particular **STOP_MOVING (4) DOES count as an action**: holding a train at a
        signal is a deliberate dispatching decision with a real cost, not an absence of
        one. Only a train the agent never addressed contributes nothing.

        TOPOLOGY-RECOVERY ACTIONS
        -------------------------
        There is no railway analogue of the power grid's `revert_topo` recovery action, so
        `n_actions_excl_recovery` is NaN unless a `recovery_action_fn` predicate is supplied.
        The column exists to keep the two domains schema-compatible.

        WHAT COUNTS AS VALUE DELIVERED (differs from power grid -- read this)
        ---------------------------------------------------------------------
        Power grid divides the episode's total REWARD by the action count. That cannot
        work here: Flatland's reward is a pure penalty whose optimum is exactly 0, so a
        agent that solves the scenario has a baseline of 0 and the KPI-RF-078 ratio
        divides by zero for every attacker.

        When the per-step `performance` curve is supplied, the numerator is instead the
        FRACTION OF TRAINS DELIVERED at the end of the episode -- the faithful railway
        reading of "value per intervention", and strictly positive whenever any train
        arrives, so the ratio is well defined. Without it the method falls back to the
        reward sum, preserving the original behaviour for older pickles.

        NaN HANDLING
        ------------
        Steps whose reward is NaN are dropped from BOTH the reward sum and the action
        count, so numerator and denominator always cover the same steps. Zero actions
        yields NaN, never 0 and never a division error.

        Args:
            actions (list): Per-step action vectors (one entry per train).
            rewards (list): Per-step summed rewards for the same episode.
            do_nothing_action (np.ndarray): Zero vector, one entry per train.
            performance (list, optional): Per-step fraction of trains arrived. When given,
                its final value is used as the delivered-value numerator instead of the
                reward sum.

        Returns:
            dict: n_actions, n_actions_excl_recovery, total_reward, ave_reward_per_action
        """
        if actions is None or rewards is None:
            return {"n_actions": np.nan, "n_actions_excl_recovery": np.nan,
                    "total_reward": np.nan, "ave_reward_per_action": np.nan}

        n_steps = min(len(actions), len(rewards))
        total_reward = 0.0
        n_actions = 0
        n_actions_excl_recovery = 0
        for step in range(n_steps):
            reward = rewards[step]
            if isna(reward):
                continue
            total_reward += reward
            act = np.asarray(actions[step])
            acting = np.asarray(act != do_nothing_action)
            n_actions += int(acting.sum())
            if self.recovery_action_fn is not None:
                n_actions_excl_recovery += int(
                    sum(1 for train in np.flatnonzero(acting.ravel())
                        if not self.recovery_action_fn(act.ravel()[train]))
                )

        # Delivered value: trains arrived when available, reward sum otherwise.
        if performance is not None and len(performance) > 0:
            delivered = float(performance[-1])
        else:
            delivered = total_reward

        if n_actions == 0:
            ave_reward_per_action = np.nan
        else:
            ave_reward_per_action = delivered / n_actions

        return {
            "n_actions": n_actions,
            "n_actions_excl_recovery": (n_actions_excl_recovery
                                        if self.recovery_action_fn is not None else np.nan),
            "total_reward": delivered,
            "ave_reward_per_action": ave_reward_per_action,
        }

    def aggregate_reward_per_action(self, metrics_robustness):
        """
        Pools KPI-RF-078 across episodes as sum(reward) / sum(actions).

        Identical to the power grid implementation: deliberately not a mean of the
        per-episode ratios, so each episode is weighted by the evidence it contributes
        rather than counting equally regardless of length.

        Args:
            metrics_robustness (pd.DataFrame): Per-episode robustness metrics.

        Returns:
            pd.Series: pooled totals, both per-action averages, the ratio, the target
                threshold and whether the ratio meets it.
        """
        # Pool the DELIVERED VALUE, not the "total_reward" column: on railway the value
        # signal is trains delivered, not the reward sum (see
        # get_reward_per_action_single_ep). Reconstructing it as ave * n_actions keeps the
        # numerator and the baseline denominator on the same signal -- mixing them yields
        # a nonsensical negative ratio.
        delivered_per_ep = (metrics_robustness["ave_reward_per_action"] *
                            metrics_robustness["n_actions"])
        total_reward = delivered_per_ep.sum(min_count=1)
        n_actions = metrics_robustness["n_actions"].sum()
        n_actions_excl_recovery = metrics_robustness["n_actions_excl_recovery"].sum(min_count=1)

        # Reconstruct the baseline reward total from its per-episode average and action
        # count, so the pooled baseline is also sum(reward)/sum(actions).
        base_reward_per_ep = (metrics_robustness["ave_reward_per_action_unperturbed"] *
                              metrics_robustness["n_actions_unperturbed"])
        total_reward_base = base_reward_per_ep.sum(min_count=1)
        n_actions_base = metrics_robustness["n_actions_unperturbed"].sum(min_count=1)

        ave = (total_reward / n_actions
               if not isna(n_actions) and n_actions > 0 and not isna(total_reward) else np.nan)
        if isna(n_actions_base) or n_actions_base <= 0 or isna(total_reward_base):
            ave_base = np.nan
        else:
            ave_base = total_reward_base / n_actions_base

        if isna(ave) or isna(ave_base) or ave_base == 0:
            ratio = np.nan
        else:
            ratio = ave / ave_base

        return pd.Series({
            "total_reward": total_reward,
            "n_actions": n_actions,
            "n_actions_excl_recovery": n_actions_excl_recovery,
            "ave_reward_per_action": ave,
            "n_actions_unperturbed": n_actions_base,
            "ave_reward_per_action_unperturbed": ave_base,
            "reward_per_action_ratio": ratio,
            "target_ratio": REWARD_PER_ACTION_TARGET_RATIO,
            "meets_target": (not isna(ratio)) and ratio >= REWARD_PER_ACTION_TARGET_RATIO,
        })

    def get_robustness_metrics_single_ep(self, obs_unperturb, obs_perturb, actions_unperturbed, actions_perturbed, do_nothing_action):
        """
        Computes robustness metrics for a single episode.

        Args:
            obs_unperturb (np.ndarray): Observations from the unperturbed environment.
            obs_perturb (np.ndarray): Observations from the perturbed environment.
            actions_unperturbed (np.ndarray): Actions taken in the unperturbed environment.
            actions_perturbed (np.ndarray): Actions taken in the perturbed environment.
            do_nothing_action (np.ndarray): The baseline 'do nothing' action.

        Returns:
            tuple: (cos_similarity, euclidean_dist, actions_changed, similarity_score, n_actions)
                - cos_similarity (list): Cosine similarity between observations at each step.
                - euclidean_dist (list): Euclidean distance between observations at each step.
                - actions_changed (int): Number of steps where the action changed due to perturbation.
                - similarity_score (float): Sum of similarity scores for changed actions.
                - n_actions (int): Number of steps where a non-baseline action was performed.
        """
        cos_similarity = []
        euclidean_dist = []
        actions_changed = 0
        similarity_score = 0
        n_actions = 0
        for step, act in enumerate(actions_perturbed):
            # Check if a non-baseline action was performed
            act_performed = (act != do_nothing_action).any()
            if act_performed:
                n_actions += 1

            # Compute observation similarity metrics
            if step < len(obs_unperturb):
                obs_orig = obs_unperturb[step]
                obs_ = obs_perturb[step]
                cos_similarity.append(np.dot(obs_orig, obs_) / (np.linalg.norm(obs_orig) * np.linalg.norm(obs_)))
                euclidean_dist.append(np.linalg.norm(obs_orig - obs_))

            # Count changed actions and accumulate similarity score
            if (act != actions_unperturbed[step]).any():
                actions_changed += 1
                act2 = actions_unperturbed[step]
                if act_performed and (act2 != do_nothing_action).any():
                    similarity_score += self.similarity_score_fn(act, act2)

        return cos_similarity, euclidean_dist, actions_changed, similarity_score, n_actions

    def get_resilience_metrics_single_ep(self, data_unperturbed, data_perturbed, distance_peaks=None, distance_valleys=None,
                                        drop_zeros=True, relative=True):
        """
        Computes resilience metrics for a single episode by analyzing the difference between unperturbed and perturbed data.

        Args:
            data_unperturbed (list or np.ndarray): Data from the unperturbed environment (e.g., rewards).
            data_perturbed (list or np.ndarray): Data from the perturbed environment (e.g., rewards).
            distance_peaks (int, optional): Minimum distance between peaks for peak detection.
            distance_valleys (int, optional): Minimum distance between valleys for valley detection.

        Returns:
            tuple: (degradation_times, restoration_times, min_rewards, max_rewards, [n_steps_perturbed], [area_unpert_rpa])
        """
        # ========== SAFETY CHECKS ==========
        # Handle None inputs
        if data_unperturbed is None or data_perturbed is None:
            return ([0], [0], [0], [0], [0], [0])
        
        # Convert to lists if needed and filter NaN/zero.
        # drop_zeros=False is used for the performance curve, which legitimately starts at
        # zero (no trains arrived yet) -- dropping those would delete the early episode.
        if drop_zeros:
            data_unpert_clean = [x for x in data_unperturbed if not isna(x) and x != 0]
            data_pert_clean = [x for x in data_perturbed if not isna(x) and x != 0]
        else:
            data_unpert_clean = [x for x in data_unperturbed if not isna(x)]
            data_pert_clean = [x for x in data_perturbed if not isna(x)]
        
        # Handle empty arrays
        if len(data_unpert_clean) == 0 or len(data_pert_clean) == 0:
            return ([0], [0], [0], [0], [0], [0])
        
        n_steps_perturbed = len(data_pert_clean)
        min_len = min(len(data_unpert_clean), n_steps_perturbed)
        
        if min_len == 0:
            return ([0], [0], [0], [0], [0], [0])
        
        data_ = np.array([data_unpert_clean[:min_len], data_pert_clean[:min_len]])
        
        # Compute percentage difference between unperturbed and perturbed data
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            if relative:
                diff = (100 * (data_[0] - data_[1]) / data_[0])[:-1]
            else:
                # Absolute gap in percentage points. Required when the baseline can be
                # zero, which makes a relative difference undefined.
                diff = (100 * (data_[0] - data_[1]))[:-1]
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        
        # If no degradation or empty diff, return zeros
        if len(diff) == 0 or diff.max() <= 0:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])
        
        # ========== END OF SAFETY CHECKS ==========

        # Find peaks (degradation) and valleys (restoration)
        peaks = list(find_peaks(diff, distance=distance_peaks)[0])
        
        positive_indices = np.where(diff > 0)[0]
        if len(positive_indices) == 0:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])
        
        valleys = [positive_indices[0]] + list(find_peaks(-diff, distance=distance_valleys)[0])

        if len(peaks) == 0 or len(valleys) == 1:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])

        prev_peak = 0
        peak = peaks.pop(0)
        prev_valley = 0
        valley = valleys.pop(0)

        degradation_times = []
        restoration_times = []
        min_rewards = []
        max_rewards = []
        
        while len(peaks) + len(valleys) > 0:
            if peak < valley:
                diff_reward = diff[peak]
                if prev_peak > prev_valley:
                    if diff[prev_peak] >= diff_reward:
                        if len(peaks) > 0:
                            peak = peaks.pop(0)
                        else:
                            peak = len(diff)
                        continue
                    else:
                        degradation_times[-1] += peak - prev_peak
                        min_rewards[-1] = diff_reward
                else:
                    degradation_times.append(peak - prev_valley)
                    min_rewards.append(diff_reward)

                prev_peak = peak
                if len(peaks) > 0:
                    peak = peaks.pop(0)
                else:
                    peak = len(diff)
            else:
                diff_reward = diff[valley]
                if prev_valley > prev_peak:
                    if diff[prev_valley] <= diff_reward or diff_reward < 0:
                        if len(valleys) > 0:
                            valley = valleys.pop(0)
                        else:
                            valley = len(diff)
                        continue
                    elif prev_peak > 0:
                        restoration_times[-1] += valley - prev_valley
                        max_rewards[-1] = diff_reward
                elif prev_peak > 0:
                    restoration_times.append(valley - prev_peak)
                    max_rewards.append(diff_reward)

                prev_valley = valley
                if len(valleys) > 0:
                    valley = valleys.pop(0)
                else:
                    valley = len(diff)

        # Handle unmatched degradation/restoration
        if len(degradation_times) > len(restoration_times):
            restoration_times.append(len(diff) - prev_peak)
            max_rewards.append(min(diff[prev_peak:]))
        
        # Handle empty lists
        if len(degradation_times) == 0:
            degradation_times = [0]
        if len(restoration_times) == 0:
            restoration_times = [0]
        if len(min_rewards) == 0:
            min_rewards = [0]
        if len(max_rewards) == 0:
            max_rewards = [0]

        # Compute area between unperturbed and perturbed curves
        area_unpert_rpa = trapezoid(data_[0]) - trapezoid(data_[1])

        return (degradation_times, restoration_times, min_rewards, max_rewards, [n_steps_perturbed], [area_unpert_rpa])
    def get_resilience_metrics_performance_single_ep(self, performance_unperturbed, performance_perturbed):
        """
        Resilience metrics computed on the per-step performance curve (fraction of trains
        arrived) rather than on the reward curve.

        WHY A SEPARATE DETECTOR
        -----------------------
        get_resilience_metrics_single_ep() locates degradation and restoration with
        scipy.find_peaks, which needs the gap between the two curves to rise AND fall --
        a local maximum. That fits an oscillating power-grid reward curve. A railway
        arrivals gap typically opens when the attack delays a train and then never closes,
        so it has no interior peak and find_peaks reports nothing, leaving KPI-DF-075 and
        KPI-RF-076 identically zero.

        Here the gap itself is measured directly:
            gap(t)           = 100 * (arrived_baseline(t) - arrived_perturbed(t))
            degradation_time = steps from the gap first opening to its widest point
            restoration_time = steps from the widest point until the gap closes again,
                               censored at the end of the episode when it never does
            area             = integral of the gap over the episode

        A run that never falls behind the baseline scores zero on all of them, which is
        the correct reading: nothing degraded, so there was nothing to restore.

        Args:
            performance_unperturbed (list): Baseline per-step fraction of trains arrived.
            performance_perturbed (list): Perturbed per-step fraction of trains arrived.

        Returns:
            tuple: (degradation_times, restoration_times, min_rewards, max_rewards,
                    [n_steps], [area]) -- same shape as get_resilience_metrics_single_ep.
        """
        if performance_unperturbed is None or performance_perturbed is None:
            return ([0], [0], [0], [0], [0], [0])

        base = np.array([x for x in performance_unperturbed if not isna(x)], dtype=float)
        pert = np.array([x for x in performance_perturbed if not isna(x)], dtype=float)
        n_steps_perturbed = len(pert)

        if len(base) == 0 or n_steps_perturbed == 0:
            return ([0], [0], [0], [0], [0], [0])

        # Compare over the overlap; a perturbed episode that runs longer has already been
        # penalised through the gap accumulated up to the baseline's end.
        min_len = min(len(base), n_steps_perturbed)
        gap = 100.0 * (base[:min_len] - pert[:min_len])

        # No point at which the agent fell behind the baseline.
        if min_len == 0 or gap.max() <= 0:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])

        positive = np.flatnonzero(gap > 0)
        onset = int(positive[0])
        widest = int(np.argmax(gap))

        degradation_time = widest - onset + 1

        closed = np.flatnonzero(gap[widest:] <= 0)
        if len(closed) > 0:
            restoration_time = int(closed[0])
        else:
            # Never recovered within the episode; censor at its end.
            restoration_time = min_len - widest

        area = float(trapezoid(gap))

        return (
            [degradation_time],
            [restoration_time],
            [float(gap.max())],      # worst gap, in percentage points
            [float(gap[-1])],        # residual gap when the episode ended
            [n_steps_perturbed],
            [area],
        )

    def aggregate_metrics_resilience(self, metrics_resilience_raw):
        """
        Aggregates resilience metrics across episodes, weighted by the number of degraded states.

        Args:
            metrics_resilience_raw (pd.DataFrame): DataFrame containing raw resilience metrics for each episode.

        Returns:
            pd.DataFrame: Aggregated resilience metrics as a DataFrame.
        """
        metrics_resilience_temp = metrics_resilience_raw.drop(columns=["episode"])
        if metrics_resilience_temp["n_degr_states"].sum() == 0:
            return pd.Series([0] * metrics_resilience_temp.shape[1], index=metrics_resilience_temp.columns)
        # Weighted average for each metric, weighted by number of degraded states
        metrics_resilience = np.array([
            np.average(
                metrics_resilience_temp[metrics_resilience_temp["n_degr_states"] > 0][col].values,
                weights=metrics_resilience_temp[metrics_resilience_temp["n_degr_states"] > 0]["n_degr_states"]
            )
            for col in metrics_resilience_temp.columns
        ])
        # Compute mean steps, area per 1000 steps, and degraded states per 1000 steps
        metrics_resilience[-3] = metrics_resilience_temp["n_steps"].mean()
        metrics_resilience[-2] = metrics_resilience_temp["area"].mean() / metrics_resilience[-3] * 1000
        metrics_resilience[-1] = metrics_resilience_temp["n_degr_states"].mean() / metrics_resilience[-3] * 1000
        metrics_resilience = pd.DataFrame(
            metrics_resilience,
            index=list(metrics_resilience_temp.columns[:-2]) + ["area_per_1000_steps", "n_degr_states_per_1000_steps"]
        )
        return metrics_resilience

    def get_perturb_vulnerability(self, perturbations, actions, actions_unperturbed):
        """
        Computes the vulnerability of each observation feature to perturbations.

        Args:
            perturbations (list of np.ndarray): Perturbations applied in each episode.
            actions (list of np.ndarray): Actions taken in each episode (perturbed).
            actions_unperturbed (list of np.ndarray): Actions taken in each episode (unperturbed).

        Returns:
            np.ndarray: Proportion of significant perturbations resulting in changed actions, averaged over episodes.
        """
        # Number of steps per episode
        n_steps_perturb = [len(perturbations[i]) for i in range(len(perturbations))]
        # Compute vulnerability for each episode
        succesful_perturb_prop = [
            self.compute_perturb_prop_single_ep(
                perturbations[i],
                np.not_equal(actions[i], actions_unperturbed[i]).any(axis=1)
            )
            for i in range(len(perturbations))
        ]
        # Weighted average across episodes
        succesful_perturb_prop = np.average(succesful_perturb_prop, weights=n_steps_perturb, axis=0)
        return succesful_perturb_prop

    def compute_perturb_prop_single_ep(self, perturbation, acts_changed):
        """
        Computes the proportion of times a significant perturbation of each feature resulted in a changed action for a single episode.

        Args:
            perturbation (np.ndarray): Perturbation values for each step and feature.
            acts_changed (np.ndarray): Boolean array indicating if the action changed at each step.

        Returns:
            np.ndarray: Proportion of significant perturbations resulting in changed actions for each feature.
        """
        # Mask out zero perturbations
        masked_data = np.ma.masked_equal(perturbation, 0)
        mean_perturb = masked_data.mean(axis=0).data
        std_perturb = masked_data.std(axis=0).data

        # Define significant perturbation thresholds
        lo = mean_perturb - std_perturb
        hi = mean_perturb + std_perturb
        # Identify significant perturbations
        perturbed_signif = (perturbation < lo) + (perturbation > hi)
        # Compute proportion for each feature
        perturb_prop = perturbed_signif[acts_changed].sum(axis=0) / perturbed_signif.sum(axis=0)
        perturb_prop[np.isnan(perturb_prop)] = 0

        return perturb_prop
