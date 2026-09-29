"""
Run the ATM robustness/resilience KPIs locally, without the orchestrator or the hub.

    python test_local_robustness_resilience_kpi_069_077.py --model /path/to/model.zip
    python test_local_robustness_resilience_kpi_069_077.py --model ... --episodes 3 --kpi DF-069

With no --model, the StraightLineDefender reference agent is used and the values describe
that baseline, not a submitted agent.

Requires bluesky-gym and stable-baselines3 importable (PYTHONPATH=/path/to/bluesky-gym).
"""

import argparse
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))          # ai4realnet_orchestrators/
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from ai4realnet_orchestrators.atm.test_runner_robustness_resilience_kpi_069_077 import (  # noqa: E402
    KPI_MAPPING, ATMRobustnessTestRunner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="SB3 .zip of the ATM agent to evaluate")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--xi", type=float, default=0.1)
    ap.add_argument("--attackers", default="GEPerturb,LambdaPIR,Random,PPO,SAC_10,SAC_5,RLPerturb")
    ap.add_argument("--kpi", default=None, help="e.g. DF-069; default: report all ten")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    wanted = [(tid, info) for tid, info in KPI_MAPPING.items()
              if args.kpi is None or args.kpi in info["name"]]
    if not wanted:
        raise SystemExit(f"No KPI matches {args.kpi!r}. Known: "
                         + ", ".join(i["name"].split(":")[0] for i in KPI_MAPPING.values()))

    # One runner computes every metric once; the rest read that runner's cache, so the
    # rollouts are shared instead of repeated per KPI.
    first_id, _ = wanted[0]
    runner = ATMRobustnessTestRunner(test_id=first_id, scenario_ids=["local"], benchmark_id="local")
    runner.N_EPISODES = args.episodes
    runner.XI = args.xi
    runner.ATTACKER_TYPES = [a.strip() for a in args.attackers.split(",") if a.strip()]
    runner.init(submission_data_url=args.model, submission_id="local")

    all_metrics = runner._run_complete_evaluation("local", "local")

    print(f"\nATM robustness/resilience KPIs  (env={runner.ATTACKER_TYPES}, "
          f"episodes={args.episodes}, xi={args.xi})")
    print(f"agent: {args.model or 'StraightLineDefender (reference, not a submission)'}\n")
    for tid, info in wanted:
        print(f"  {info['name']:<58} {all_metrics[info['metric_key']]:.4f}")
    print()


if __name__ == "__main__":
    main()
