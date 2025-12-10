from ai4realnet_orchestrators.test_runner import TestRunner


def load_submission_data(submission_data_url: str):
  raise NotImplementedError()


def load_model(submission_data):
  raise NotImplementedError()


def load_scenario_data(scenario_id: str):
  raise NotImplementedError()


class YourTestRunner(TestRunner):
  def init(self, submission_data_url: str):
    super().init(submission_data_url=submission_data_url)
    submission_data = load_submission_data(submission_data_url)
    self.model = load_model(submission_data)

  def run_scenario(self, scenario_id: str, submission_id: str):
    # here you would implement the logic to run the test for the scenario:
    scenario_data = load_scenario_data(scenario_id)
    model = self.model

    # data and other stuff initialized in the init method can be used here
    # for demonstration, we return a dummy result
    return {
      "primary": -999,
    }
