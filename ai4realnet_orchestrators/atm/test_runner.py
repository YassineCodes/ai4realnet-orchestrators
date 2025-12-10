import os
import pathlib

import pandas as pd

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.test_runner import TestRunner

WORKDIR = os.environ.get("WORKDIR", "/data")


class BlueSkyRunner(TestRunner):
    def run_scenario(self, scenario_id: str, submission_id: str):
        # here you would implement the logic to run the test for the scenario:
        args = ["bluesky", "--detached", "--workdir", WORKDIR, "--scenfile", scenario_id]
        marker_path = pathlib.Path(WORKDIR) / "current_scenfile.txt"
        marker_path.write_text(scenario_id)

        exec_with_logging(args)

        # Read the generated data file
        output_dir = pathlib.Path(WORKDIR) / "output"

        # Gets the specific file for the scenario
        scenario_csv = output_dir / f"{scenario_id}_log.csv"
        if not scenario_csv.exists():
            raise RuntimeError(f"Expected output file not found: {scenario_csv}")
        data = pd.read_csv(scenario_csv, comment="#")

        # if scenario_is is ...
        # compute primary score accordingly
        # elif scenrio_id is ...
        # etc

        # test implementation
        primary_score = float(data['hdg'].mean())
        return {
          "primary": primary_score,
        }