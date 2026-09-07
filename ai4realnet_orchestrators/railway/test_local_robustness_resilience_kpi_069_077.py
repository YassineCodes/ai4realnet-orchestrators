"""
Local Test Script for Robustness & Resilience Framework (Railway)
==================================================================

Runs the multi-attacker evaluation framework LOCALLY without the FAB
orchestrator. Use for development, debugging, and testing agents before
submission.

Usage:
    python test_local_robustness_resilience_kpi_069_077.py

    # or pick a KPI and a number of episodes:
    python test_local_robustness_resilience_kpi_069_077.py --kpi KPI-FF-070 --episodes 10

    # or evaluate your own submission instead of the built-in baseline:
    python test_local_robustness_resilience_kpi_069_077.py --agent path/to/agent.pkl

Agent formats accepted by --agent:
    *.pkl  a pickled RailwayDefender, or a policy exposing act_many(handles, obs_list)
    *.zip  a maze-flatland checkpoint (state_dict-*.pt + spaces_config.pkl),
           or an archive containing agent.pkl / policy.pkl
    (omitted)  a built-in baseline defender, selected with --baseline

Note:
    This bypasses Celery/RabbitMQ orchestration. For production, use FAB.

Author: INESC TEC
"""

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "0"

# ============================================================
# PATH SETUP - Must be done BEFORE any framework imports
# ============================================================

# Get the directory where this script lives (railway folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAILWAY_DIR = SCRIPT_DIR
FRAMEWORK_DIR = os.path.join(SCRIPT_DIR, "framework")
ORCHESTRATORS_DIR = os.path.dirname(SCRIPT_DIR)   # ai4realnet_orchestrators
PARENT_DIR = os.path.dirname(ORCHESTRATORS_DIR)   # ai4realnet-orchestrators

# Add paths for imports
sys.path.insert(0, PARENT_DIR)          # For ai4realnet_orchestrators package
sys.path.insert(0, ORCHESTRATORS_DIR)   # For base test_runner
sys.path.insert(0, RAILWAY_DIR)         # For the test runner module
sys.path.insert(0, FRAMEWORK_DIR)       # For attack_models, defenders, etc.

# Remember where the user invoked us from: --agent is resolved against this, not
# against FRAMEWORK_DIR, so a relative path on the command line means what it looks like.
INVOCATION_DIR = os.getcwd()

# Change working directory to framework so relative paths work
os.chdir(FRAMEWORK_DIR)

# ============================================================
# NOW we can import (paths are set up)
# ============================================================

from test_runner_robustness_resilience_kpi_069_077 import (  # noqa: E402
    KPI_MAPPING,
    MultiAttackerRobustnessTestRunner,
)

# KPI label -> (test_id, scenario_id, benchmark_id)
ROBUSTNESS_BENCHMARK = "3810191b-8cfd-4b03-86b2-f7e530aab30d"
RESILIENCE_BENCHMARK = "31ea606b-681a-437a-85b9-7c81d4ccc287"

KPI_SCENARIOS = {
    "KPI-DF-069": ("a94c858e-4bc3-4d67-bd78-5c81506e39f7", "74dd5830-6e59-423f-89f4-b050319db14e", ROBUSTNESS_BENCHMARK),
    "KPI-FF-070": ("5abadf6b-991c-4d37-810f-f77bb71d490d", "ffcadd8d-207a-49af-8b09-54e922642f01", ROBUSTNESS_BENCHMARK),
    "KPI-SF-071": ("dce32e78-e827-4994-a0a2-06feee2528cc", "588ae37c-f583-47df-9154-ca12c9ac134a", ROBUSTNESS_BENCHMARK),
    "KPI-SF-072": ("e5206c56-75a0-41fa-9db3-bec66359337e", "8011c7bd-6082-4653-8d9d-887d23f1ec5c", ROBUSTNESS_BENCHMARK),
    "KPI-VF-073": ("0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc", "47a11418-fad5-4d55-a637-6b90a8351500", ROBUSTNESS_BENCHMARK),
    "KPI-AF-074": ("707a1a4e-7073-432b-94fc-af4a5ee9f07d", "a9e3fbf7-b5d5-477d-a0c8-d23880237d2d", RESILIENCE_BENCHMARK),
    "KPI-DF-075": ("2c4be118-6108-43b3-b09f-a4bee842167a", "dc03b9f1-bfb2-44b8-b124-f7eede10e0a7", RESILIENCE_BENCHMARK),
    "KPI-RF-076": ("2cac54e0-aaf3-4f22-8307-f23878c432f0", "56160b90-a287-4dec-acc7-f40967d60fa0", RESILIENCE_BENCHMARK),
    "KPI-SF-077": ("d432299f-dbee-46ba-9e15-77954086440a", "367a8b12-1b87-42f1-9400-4ecf96d6b617", RESILIENCE_BENCHMARK),
    "KPI-RF-078": ("8ebc88f0-896c-4910-8997-a44d107e7eb7", "dc8195e4-266d-4afb-ba60-2659f59acfa4", ROBUSTNESS_BENCHMARK),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kpi", default="KPI-VF-073", choices=sorted(KPI_SCENARIOS),
                        help="Which KPI to report (default: KPI-VF-073)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per attacker (default: 5; production uses 50)")
    parser.add_argument("--trains", type=int, default=None,
                        help="Number of trains (default: runner's N_TRAINS)")
    parser.add_argument("--agent", default=None,
                        help="Path to a .pkl or .zip submission "
                             "(default: use the built-in baseline, see --baseline)")
    parser.add_argument("--baseline", default=None,
                        choices=["shortest-path", "do-nothing"],
                        help="Built-in defender to evaluate when --agent is omitted. "
                             "'shortest-path' and 'do-nothing' derive actions without "
                             "reading the observation, so they are immune to these "
                             "attacks by construction. These are development fixtures, "
                             "not reference agents - no default, pass --agent instead)")
    parser.add_argument("--all", action="store_true",
                        help="Print all 9 KPI values from the single evaluation run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    test_id, scenario_id, benchmark_id = KPI_SCENARIOS[args.kpi]

    print("=" * 60)
    print("LOCAL TEST - FAB TestRunner (Railway)")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Framework directory: {FRAMEWORK_DIR}")

    runner = MultiAttackerRobustnessTestRunner(
        test_id=test_id,
        scenario_ids=[scenario_id],
        benchmark_id=benchmark_id,
    )

    # OVERRIDE framework path and episode count for local testing
    runner.FRAMEWORK_PATH = FRAMEWORK_DIR
    runner.NUM_EPISODES = args.episodes
    runner.BASELINE_DEFENDER = args.baseline
    if args.trains is not None:
        runner.N_TRAINS = args.trains

    print(f"[OK] TestRunner initialized")
    print(f"   KPI: {runner.kpi_info['name']}")
    print(f"   Framework: {runner.FRAMEWORK_PATH}")
    print(f"   Episodes: {runner.NUM_EPISODES}")
    print(f"   Attackers: {runner.ATTACKER_TYPES}")

    agent_path = args.agent
    if agent_path is not None:
        # Resolve against the directory the user ran from, not the framework dir we
        # chdir'd into at import time.
        agent_path = (agent_path if os.path.isabs(agent_path)
                      else os.path.join(INVOCATION_DIR, agent_path))
        agent_path = os.path.normpath(agent_path)
        if not os.path.exists(agent_path):
            fallback = os.path.normpath(os.path.join(FRAMEWORK_DIR, args.agent))
            if os.path.exists(fallback):
                agent_path = fallback
        print(f"\n[AGENT] {agent_path}")
        print(f"   Exists: {os.path.exists(agent_path)}")
        if not os.path.exists(agent_path):
            print("   -> file not found, aborting")
            return 1
    else:
        print(f"\n[AGENT] built-in baseline defender: {args.baseline}")

    try:
        print(f"\n[..] Initializing framework...")
        runner._initialize_framework()
        print("[OK] Framework initialized!")

        print(f"\n[..] Loading defender agent...")
        runner.init(submission_data_url=agent_path, submission_id="local-test-001")
        print("[OK] Agent loaded!")

        print("\n[..] Running evaluation (this may take a while)...")
        result = runner.run_scenario(
            scenario_id=scenario_id,
            submission_id="local-test-001",
        )

        print("\n" + "=" * 60)
        print("[OK] EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"{runner.kpi_info['name']}")
        print(f"Result: {result}")

        if args.all:
            all_metrics = runner._metrics_cache[f"{scenario_id}_local-test-001"]
            print("\nAll 9 KPI values from this run:")
            print("-" * 60)
            for uuid, info in KPI_MAPPING.items():
                value = all_metrics[info["metric_key"]]
                print(f"  {info['name']:<52} {value:>10.4f}")
            print("-" * 60)

        return 0

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
