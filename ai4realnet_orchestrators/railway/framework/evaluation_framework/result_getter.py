"""
Episode runner for the Flatland railway robustness/resilience framework.

Mirrors the power-grid result_getter interface so callers are identical:

    rg = result_getter(env, defender, n_episodes, save_folder, attackers)
    rg.calculate_metrics()

The env argument must be a FlatlandEnvironment instance (see Environment.py).
"""

import os
import pickle
import time

import numpy as np
import pandas as pd

from evaluation_framework.metrics import metrics
import evaluation_framework.plots_and_tables as plots
from utility.UtilityHelper import FilenameHelper, UtilityHelper


class result_getter:
    """
    Runs Flatland episodes for each attacker, saves raw data to pickles,
    and computes robustness / resilience metrics — identical interface to
    the power-grid result_getter.

    Args:
        env:          FlatlandEnvironment instance.
        defender:     RailwayDefender instance (already embedded in env).
        n_episodes:   Number of episodes to run per attacker.
        save_folder:  Directory where pickle files, tables and plots are written.
        attackers:    List of attacker instances (BaseAttackerClass subclasses).
    """

    def __init__(self, env, defender, n_episodes, save_folder, attackers):
        self.env = env
        self.defender = defender
        self.n_episodes = n_episodes
        self.save_folder = save_folder
        self.attackers = attackers

    # ------------------------------------------------------------------
    # Core episode runner
    # ------------------------------------------------------------------

    def run_episodes(self, filename, attacker):
        """
        Run all episodes with a given attacker and persist results.

        Args:
            filename (str): Path to save the pickle file.
            attacker:       Attacker instance, or None for the unperturbed case.

        Returns:
            dict: Episode data dictionary.
        """
        observations = []
        perturbations = []
        actions = []
        actions_unperturbed = []
        rewards = []
        # Per-step performance (fraction of trains arrived). Flatland's reward is sparse
        # and terminal, so the resilience KPIs are computed on this curve instead --
        # see FlatlandEnvironment._compute_performance().
        performance = []
        # Environment clock per step. One step is one decision point and may span
        # several timesteps, by a different amount in the baseline than under attack,
        # so the resilience KPIs are aligned on this before the curves are compared.
        env_time = []
        # Per-step mask over agent handles: whose policy was actually queried. A train
        # skipped because it had one legal option, or already finished, has no decision
        # of its own and is excluded from the action comparison.
        queried = []

        self.env.attacker = attacker

        attacker_name = attacker.model_name if attacker else "Unperturbed"
        print(f"\n[TIMER] Starting episodes for: {attacker_name}")
        episode_start_time = time.time()
        episode_times = []

        for ep in range(self.n_episodes):
            obs = self.env.reset(seed=ep)
            done = False

            observations.append([obs])
            perturbations.append([])
            actions.append([])
            actions_unperturbed.append([])
            rewards.append([])
            performance.append([])
            env_time.append([])
            queried.append([])

            ep_start = time.time()
            step_times = []
            steps = 0

            while not done:
                step_start = time.time()
                obs, perturbation, act, act_unperturbed, reward, done = self.env.step()

                observations[ep].append(obs)
                perturbations[ep].append(perturbation)
                actions[ep].append(act)
                actions_unperturbed[ep].append(act_unperturbed)
                rewards[ep].append(reward)
                performance[ep].append(getattr(self.env, "last_performance", 0.0))
                env_time[ep].append(getattr(self.env, "last_env_time", steps + 1))
                mask = getattr(self.env, "last_queried_mask", None)
                queried[ep].append(None if mask is None else np.asarray(mask, dtype=bool).copy())

                step_times.append(time.time() - step_start)
                steps += 1

            ep_time = time.time() - ep_start
            episode_times.append(ep_time)
            avg_step = ep_time / steps if steps > 0 else 0
            print(
                f"[TIMER] Episode {ep + 1}/{self.n_episodes} completed: "
                f"{steps} steps in {ep_time:.1f}s ({avg_step:.3f}s/step)"
            )

        total_time = time.time() - episode_start_time
        total_steps = sum(len(r) for r in rewards)
        print(
            f"\n{'=' * 80}\n"
            f"SUMMARY: {attacker_name}\n"
            f"{'=' * 80}\n"
            f"Episodes:          {self.n_episodes}\n"
            f"Total steps:       {total_steps}\n"
            f"Total time:        {total_time:.1f}s ({total_time / 60:.1f} min)\n"
            f"Avg time/step:     {total_time / total_steps:.3f}s\n"
            f"Min episode time:  {min(episode_times):.1f}s\n"
            f"Max episode time:  {max(episode_times):.1f}s\n"
            f"Avg episode time:  {sum(episode_times)/len(episode_times):.1f}s\n"
            f"{'=' * 80}\n",
            # Flushed so a phase boundary survives the process being killed.
            # Without it the per-episode TIMER lines (which do flush) are the last
            # thing in the log and a phase that finished looks like one that stopped.
            flush=True,
        )

        # Did the attack ever reach the policy? perturbations holds what was APPLIED to
        # the observation the policy read, so an all-zero record means no perturbation
        # ever arrived -- against which an unchanged action is not evidence of
        # robustness. Recorded here as data rather than decided per attacker name, so a
        # new attacker is classified by what it did, not by a list someone maintains.
        attack_reached_policy = any(
            bool(np.any(np.asarray(p) != 0.0))
            for ep_perturbations in perturbations
            for p in ep_perturbations
        )
        if attacker is not None and not attack_reached_policy:
            print(
                f"[WARNING] {attacker_name}: every applied perturbation was zero -- the "
                f"attack never reached the policy. Robustness KPIs are NOT APPLICABLE "
                f"for this attacker; do not report an unchanged action as robustness."
            )

        data_dict = {
            "observations": observations,
            "perturbations": perturbations,
            "actions": actions,
            "actions_unperturbed": actions_unperturbed,
            "rewards": rewards,
            "performance": performance,
            "env_time": env_time,
            "queried": queried,
            "attack_reached_policy": attack_reached_policy,
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return data_dict

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------


    def _load_or_run(self, pickle_path, attacker, label):
        """
        Reuse a phase's archive only when it holds the episode count this run asks for.

        The resume path exists so that a campaign killed partway through does not repeat
        the phases it already finished, which is what --save-dir makes possible. It
        reused whatever pickle it found, with no check on what was in it. That is wrong
        whenever the save directory holds a run of a different length: the KPIs would be
        computed with a baseline of one size against perturbed phases of another, and
        nothing would say so. A mismatch re-runs the phase instead.

        A pickle truncated by a kill mid-write fails loudly in pickle.load, so the case
        that needs catching is the quiet one: a complete archive of the wrong length.
        """
        import os
        import pickle

        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            have = len(data.get("rewards", []))
            if have == self.n_episodes:
                print(f"[SKIPPED] {pickle_path} already holds {have} episodes, loading...",
                      flush=True)
                return data
            print(f"[REDO] {pickle_path} holds {have} episodes but this run asks for "
                  f"{self.n_episodes}; re-running {label} rather than mixing lengths.",
                  flush=True)

        print(f"[RUNNING] Running episodes for {label}...", flush=True)
        return self.run_episodes(pickle_path, attacker)

    def calculate_metrics(self):
        """
        Compute and save robustness / resilience metrics for all attackers.

        Saves (identical structure to power-grid result_getter):
          - <save_folder>/unperturbed.pkl
          - <save_folder>/<attacker.pickle_file>
          - <save_folder>/Episode<N>/reward_curve_<name>.svg
          - <save_folder>/Episode<N>/cos_sim_curve_<name>.svg
          - <save_folder>/robustness.svg
          - <save_folder>/robustness_table.tex
          - <save_folder>/reward_table.tex
          - <save_folder>/observation_table.tex
          - <save_folder>/report.html

        Returns:
            list of metrics: One metrics object per attacker.
        """
        os.makedirs(self.save_folder, exist_ok=True)

        # ---- Unperturbed baseline ----
        unperturbed_pickle = os.path.join(self.save_folder, "unperturbed.pkl")
        unperturbed_data = self._load_or_run(unperturbed_pickle, None, "the unperturbed baseline")

        unperturbed_metrics = metrics(
            unperturbed_data,
            unperturbed_data,
            self.env.do_nothing_action(),
            self.env.get_similarity_score,
            model_name="Unperturbed",
        )

        # ---- Per-attacker evaluation ----
        metrics_list = []

        for attacker in self.attackers:
            save_pickle = os.path.join(self.save_folder, attacker.pickle_file)
            data_dict = self._load_or_run(save_pickle, attacker, attacker.model_name)

            m = metrics(
                data_dict,
                unperturbed_data,
                self.env.do_nothing_action(),
                self.env.get_similarity_score,
                model_name=attacker.model_name,
            )
            metrics_list.append(m)

            # Per-episode reward and cosine-similarity curves
            sanitized_name = FilenameHelper.sanitize_filename(attacker.model_name)
            for episode in range(self.n_episodes):
                episode_folder = os.path.join(self.save_folder, f"Episode{episode}")
                os.makedirs(episode_folder, exist_ok=True)

                plots.plot_reward_curve_comparison(
                    m, episode, model_name=attacker.model_name,
                    filename=os.path.join(episode_folder, f"reward_curve_{sanitized_name}.svg"),
                )
                plots.plot_cos_similarity_curve_comparison(
                    m, episode, model_name=attacker.model_name,
                    filename=os.path.join(episode_folder, f"cos_sim_curve_{sanitized_name}.svg"),
                )

        # ---- Summary tables and plots ----
        robustness_table = plots.get_metrics_robustness_not_compared_to_unperturbed(
            metrics_list, print_as_latex=False
        )
        reward_table, observation_table = plots.get_metrics_resilience(
            metrics_list, print_as_latex=False
        )
        plots.plot_metrics_robustness_compared_to_unperturbed(
            metrics_list, unperturbed_metrics,
            filename=os.path.join(self.save_folder, "robustness.svg"),
        )

        robustness_table.to_latex(os.path.join(self.save_folder, "robustness_table.tex"))
        reward_table.to_latex(os.path.join(self.save_folder, "reward_table.tex"))
        observation_table.to_latex(os.path.join(self.save_folder, "observation_table.tex"))

        # ---- HTML report (same as power grid) ----
        UtilityHelper.create_html_report(
            save_folder=self.save_folder,
            output_file="report.html",
        )

        print(f"\n[DONE] Results saved to {self.save_folder}")
        return metrics_list
