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
            f"{'=' * 80}\n"
        )

        data_dict = {
            "observations": observations,
            "perturbations": perturbations,
            "actions": actions,
            "actions_unperturbed": actions_unperturbed,
            "rewards": rewards,
            "performance": performance,
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return data_dict

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------

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
        if os.path.exists(unperturbed_pickle):
            print(f"[SKIPPED] {unperturbed_pickle} already exists, loading...")
            with open(unperturbed_pickle, "rb") as f:
                unperturbed_data = pickle.load(f)
        else:
            print("[RUNNING] Running unperturbed episodes...")
            unperturbed_data = self.run_episodes(unperturbed_pickle, attacker=None)

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
            if os.path.exists(save_pickle):
                print(f"[SKIPPED] {save_pickle} already exists, loading...")
                with open(save_pickle, "rb") as f:
                    data_dict = pickle.load(f)
            else:
                print(f"[RUNNING] Running episodes for {attacker.model_name}...")
                data_dict = self.run_episodes(save_pickle, attacker)

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
