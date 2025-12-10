import os
import pickle
from pathlib import Path

from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.integrations.interactiveai.interactiveai import FlatlandInteractiveAICallbacks
from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.test_runner import TestRunner


class TestRunnerPlaygroundInteractive(TestRunner):
    def run_scenario(self, scenario_id: str, submission_id: str):
        scenario = TestRunnerPlaygroundInteractive.load_scenario_data(scenario_id)

        # scenario Olten has step every 3 seconds for an hour
        STEPS_ONE_HOUR = 1300  # 1h + additional time for agents to leave the map
        # how many ms per step if replaying in real-time
        REALTIME_STEP_TO_MILLIS = 3600 / STEPS_ONE_HOUR * 1000
        # run faster... limiting factor becomes environment stepping time and blocking requests InteractiveAI platform
        SPEEDUP = 1000

        # https://github.com/flatland-association/flatland-scenarios/raw/refs/heads/scenario-olten-fix/scenario_olten/data/OLTEN_PARTIALLY_CLOSED_v1.zip
        _dir = os.getenv("SCENARIOS_FOLDER", "../scenarios")
        data_dir = Path(f"{_dir}/scenario_olten/data/{scenario}")

        with (data_dir / "position_to_latlon.pkl").resolve().open("rb") as file_in:
            position_to_latlon_olten = pickle.loads(file_in.read())

        trajectory = Trajectory.load_existing(data_dir=data_dir, ep_id=scenario)

        # see above for configuration options, use collect_only=False for live POSTing
        cb = FlatlandInteractiveAICallbacks(position_to_latlon_olten, collect_only=False,
                                            step_to_millis=REALTIME_STEP_TO_MILLIS / SPEEDUP,
                                            token_url="http://interactiveai.flatland.cloud/auth/token",
                                            event_api_host="http://interactiveai.flatland.cloud/cab_event",
                                            context_api_host="http://interactiveai.flatland.cloud/cabcontext",
                                            history_api_host="http://interactiveai.flatland.cloud/cabhistoric",
                                            )

        TrajectoryEvaluator(trajectory, cb).evaluate(end_step=150, skip_rewards_dones_infos=True)
        print(cb.events)
        print(cb.contexts)
        return {}

    @staticmethod
    def load_scenario_data(scenario_id: str) -> str:
        return {'cf8e0a9b-14af-43e1-b1fc-e43bf3aaddd7': "olten_partially_closed"}[scenario_id]
