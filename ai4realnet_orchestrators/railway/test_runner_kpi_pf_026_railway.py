import logging
import os
from pathlib import Path

import numpy as np
from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")

logger = logging.getLogger(__name__)


# KPI-PF-026: Punctuality (Railway)
class TestRunner_KPI_PF_026_Railway(AbtractTestRunnerRailway):

  def run_scenario(self, scenario_id: str, submission_id: str):
    env_path = TestRunner_KPI_PF_026_Railway.load_scenario_data(scenario_id)
    # here you would implement the logic to run the test for the scenario
    # data and other stuff initialized in the init method can be used here
    # for demonstration, we return a dummy result

    data_dir = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}"
    generate_policy_args = [
      "--data-dir", data_dir,
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "PunctualityRewards",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
      "--snapshot-interval", "10",
    ]
    self.exec(generate_policy_args, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}")

    trajectory = Trajectory.load_existing(data_dir=Path(data_dir), ep_id=scenario_id)

    df_trains_arrived = trajectory.trains_arrived
    logger.info(f"trains arrived: {df_trains_arrived}")
    assert len(df_trains_arrived) == 1
    logger.info(f"trains arrived: {df_trains_arrived.iloc[0]}")
    success_rate = df_trains_arrived.iloc[0]["success_rate"]
    logger.info(f"success rate: {success_rate}")

    df_trains_rewards_dones_infos = trajectory.trains_rewards_dones_infos
    logger.info(f"trains dones infos: {trajectory.trains_rewards_dones_infos}")
    num_agents = df_trains_rewards_dones_infos["agent_id"].max() + 1
    logger.info(f"num_agents: {num_agents}")

    agent_scores = df_trains_rewards_dones_infos["reward"].to_list()
    logger.debug(f"agent_scores: {agent_scores[:10]}...")
    punctuality = mean_punctuality_aggregator(agent_scores)
    logger.info(f"punctuality: {punctuality}")

    self.upload_and_empty_local(submission_id=submission_id, scenario_id=scenario_id)

    return {
      'punctuality': punctuality,
      'success_rate': success_rate
    }

  @staticmethod
  def load_scenario_data(scenario_id: str) -> str:
    return {
      "5a60713d-01f2-4d32-9867-21904629e254": "Test_00/Level_0.pkl",
      "0db72a40-43e8-477b-89b3-a7bd1224660a": "Test_00/Level_1.pkl",
      "7def3118-2e9c-4de7-8d61-f0e76fbeee5d": "Test_00/Level_2.pkl",
      "3ae60635-6995-4fb1-8309-61fded3d6fd8": "Test_00/Level_3.pkl",
      "eeef8445-723d-4740-b89f-4dbaf75f9ae6": "Test_00/Level_4.pkl",
      "94af1ed1-3686-4a9e-99f5-3a7ad908f125": "Test_00/Level_5.pkl",
      "8250d0e2-700e-4051-85c3-a8d0d95a5f0f": "Test_00/Level_6.pkl",
      "c58759a7-a64a-4cbf-970b-948bae0c2254": "Test_00/Level_7.pkl",
      "f94f517f-c0a4-4415-b726-186cdc75f9c6": "Test_00/Level_8.pkl",
      "c0e2c3e0-c171-48dd-a312-5de070e3f937": "Test_00/Level_9.pkl",
      "26fb51b9-2466-48ab-8c0a-89d9536d4c34": "Test_01/Level_0.pkl",
      "66c23554-47bb-4268-b08e-518f6f163e9d": "Test_01/Level_1.pkl",
      "08641d25-9b18-41bb-9cbc-4039b4ad24f0": "Test_01/Level_2.pkl",
      "4d3dd85a-2b16-45de-b524-f83b4a58a2f4": "Test_01/Level_3.pkl",
      "bf9209c7-125c-42f9-b78e-4e5b7aacefcc": "Test_01/Level_4.pkl",
      "f511044d-2378-4c7f-af92-45c78146bdef": "Test_01/Level_5.pkl",
      "f89cee49-e1ae-427d-ae42-5cc411661a1c": "Test_01/Level_6.pkl",
      "262e43bd-bf35-4171-b38b-c77969db0b16": "Test_01/Level_7.pkl",
      "61bd29ec-0b09-4067-ada0-b43e48a8ac9a": "Test_01/Level_8.pkl",
      "92d22472-4696-40ba-924f-861a2f4343b6": "Test_01/Level_9.pkl",
      "6fc5f67a-40fa-45ce-819e-35a85e08e560": "Test_02/Level_0.pkl",
      "66bce513-502c-43b4-a155-8a16c410a7c6": "Test_02/Level_1.pkl",
      "eff645bf-7ea8-490d-ae8a-ebb0d16a774c": "Test_02/Level_2.pkl",
      "8397e6d6-babc-469b-a239-7eabcbd510da": "Test_02/Level_3.pkl",
      "c359f13c-d222-4b04-ad0a-2bb30fb9da5f": "Test_02/Level_4.pkl",
      "97203764-6717-4ca6-bae9-c35c4eb38206": "Test_02/Level_5.pkl",
      "adc4bf52-096c-4369-a85f-c9bf4b86bc64": "Test_02/Level_6.pkl",
      "72f93d48-ecef-4bf7-9d97-cb008b47e566": "Test_02/Level_7.pkl",
      "b470667b-d9c9-4af4-b64e-c32102c34387": "Test_02/Level_8.pkl",
      "4aa9e1b8-8669-466e-b4b9-c7db2a098bec": "Test_02/Level_9.pkl",
      "44354e67-e0b7-4faa-9385-2d6247c7a50c": "Test_03/Level_0.pkl",
      "4bca2964-ef35-4b03-a47e-829bc9078374": "Test_03/Level_1.pkl",
      "6190b6ac-a06b-4b07-8b82-0dfb1088663f": "Test_03/Level_2.pkl",
      "a51d4e4a-c841-416f-9292-fc64dead758b": "Test_03/Level_3.pkl",
      "d08f6539-5c3e-4a98-93ae-3e344611e3a8": "Test_03/Level_4.pkl",
      "8a2ec760-d2c2-4329-b373-3acd95395076": "Test_03/Level_5.pkl",
      "9b136147-f560-45d4-abd8-9f7de7cd7570": "Test_03/Level_6.pkl",
      "9723c2c5-ee55-441d-9dba-1a1dc23fdc5e": "Test_03/Level_7.pkl",
      "3406bbe9-cb19-44b7-af26-460b0a117a6c": "Test_03/Level_8.pkl",
      "ae0a88ad-3bfd-4201-b28a-e2c75d081cd5": "Test_03/Level_9.pkl",
      "8b308495-7ea6-4ddc-acb4-56eb5b3aec12": "Test_04/Level_0.pkl",
      "a8f69dc4-04a1-434a-ad97-27c745561b6a": "Test_04/Level_1.pkl",
      "8b244f56-50e1-411a-a7d8-a2b89dfab26e": "Test_04/Level_2.pkl",
      "8e6419c1-6470-4272-9c4b-43d9fe19dd3d": "Test_04/Level_3.pkl",
      "ec503b6e-3682-4dcd-9dc7-b194b67283d9": "Test_04/Level_4.pkl",
      "74fd9eab-d2e5-4222-8656-81fc2dde7c21": "Test_04/Level_5.pkl",
      "c16e54c1-33b2-45b8-95b0-33cc4f5400d5": "Test_04/Level_6.pkl",
      "c80effec-27b8-4103-b726-344a85f35407": "Test_04/Level_7.pkl",
      "9bec9335-3dd5-4d88-b2ac-c5d711bcab36": "Test_04/Level_8.pkl",
      "4a067d3c-75e6-4e91-a42d-cdf291016674": "Test_04/Level_9.pkl",
      "1f6c674b-fdb5-40c8-bbe3-d924c2b7146e": "Test_05/Level_0.pkl",
      "8bd15e2f-6089-4022-8383-26a36093dc80": "Test_05/Level_1.pkl",
      "5b026a8a-e0dc-47d2-b2ce-d0bfc083e6c5": "Test_05/Level_2.pkl",
      "ab2e2bba-baa0-45de-b639-c7ebf29bf947": "Test_05/Level_3.pkl",
      "8f8b6f67-d3c3-41ec-b60d-2924577e68f4": "Test_05/Level_4.pkl",
      "3e59642a-7b20-4989-b9e6-01a35a82b9da": "Test_05/Level_5.pkl",
      "0e4618c7-7ab7-4323-855c-7f95cbaef2d0": "Test_05/Level_6.pkl",
      "4a59929b-b391-464f-bfa1-628f7a45ac36": "Test_05/Level_7.pkl",
      "27f8ac34-3d09-4b68-9fb4-c51cfbfd09df": "Test_05/Level_8.pkl",
      "bacb4c11-11b4-4c76-910d-f92abc5b7a39": "Test_05/Level_9.pkl",
      "a5dff3c8-902e-4cb9-8466-d277d0ed4d67": "Test_06/Level_0.pkl",
      "aca25feb-6254-40b3-8d40-3c805797c69b": "Test_06/Level_1.pkl",
      "deb21442-0f94-4ff3-b78d-8d418415d646": "Test_06/Level_2.pkl",
      "6cf2cc89-d30e-4063-bced-051f3cdae92f": "Test_06/Level_3.pkl",
      "84bcbff5-346f-452c-87ab-08ceff6364f2": "Test_06/Level_4.pkl",
      "9acbe68e-2a45-420b-a142-34996dbcfb83": "Test_06/Level_5.pkl",
      "42786e4c-c80e-40f5-8237-bafc5f39979d": "Test_06/Level_6.pkl",
      "242a6240-b62c-48b4-a264-b6737e893fa5": "Test_06/Level_7.pkl",
      "3c38a1d3-2340-43ed-ac0b-4b76c6588b92": "Test_06/Level_8.pkl",
      "b89daede-405b-411a-a02b-ee32d7c9d020": "Test_06/Level_9.pkl",
      "736301d9-0cff-4e25-af78-4d6f78b48cd5": "Test_07/Level_0.pkl",
      "3b74e3a8-e740-4bb2-802f-2b2bedabad65": "Test_07/Level_1.pkl",
      "61ee09b0-96fb-4562-bf7d-a01606de424c": "Test_07/Level_2.pkl",
      "8d75b03c-2a34-4b2a-8408-d8db01a7ae1a": "Test_07/Level_3.pkl",
      "81c01671-1a30-41cd-9f6c-25b2f9253da9": "Test_07/Level_4.pkl",
      "01723b62-3f4d-4921-8904-752f092588f5": "Test_07/Level_5.pkl",
      "5d44cb89-3c63-4716-8551-bd25de881f89": "Test_07/Level_6.pkl",
      "cf7c97fc-1c61-4c0d-ad1e-952bf6d6f23a": "Test_07/Level_7.pkl",
      "60ddab54-dc12-4fbd-bbec-53b23d896c9d": "Test_07/Level_8.pkl",
      "7eab41ad-8fd5-4a04-8c06-4d6c2016a594": "Test_07/Level_9.pkl",
      "9e0aac9e-ddf9-4575-bf1c-d08a923e15fa": "Test_08/Level_0.pkl",
      "befa97fb-2a74-4f2e-91c8-ea2879d08dcf": "Test_08/Level_1.pkl",
      "0cc18965-c967-4b58-ac7f-38a443b4cd16": "Test_08/Level_2.pkl",
      "d0f62f51-5a51-443b-bf7b-18e3d5b191dc": "Test_08/Level_3.pkl",
      "c2ebb179-0a2d-4e84-95be-2837be406716": "Test_08/Level_4.pkl",
      "3ac76f3c-f560-4666-af61-c693e4cd3ad4": "Test_08/Level_5.pkl",
      "484bbf93-bc67-4726-8b81-6c4ab608c861": "Test_08/Level_6.pkl",
      "11b19a5f-4d61-4b5d-980c-98cf0c16906a": "Test_08/Level_7.pkl",
      "edecaeb7-53d3-411a-a00c-2ce6226fde50": "Test_08/Level_8.pkl",
      "a43cb746-fa63-4d39-87cd-43a81fbf3a8e": "Test_08/Level_9.pkl",
      "5a3cc8a5-584c-4171-ae2c-97bdbc5047a1": "Test_09/Level_0.pkl",
      "30778350-508b-4cbe-bff2-8882d0743aed": "Test_09/Level_1.pkl",
      "dc652dd4-c0b5-4036-bb70-f71cd9fd488a": "Test_09/Level_2.pkl",
      "6e4dd7f4-2155-407f-922b-25aeb04a47b7": "Test_09/Level_3.pkl",
      "ba75d8cb-bfb9-4d2b-ac2f-2b5b8697188c": "Test_09/Level_4.pkl",
      "10c621d6-1f99-4045-82a0-47d3ea107ddc": "Test_09/Level_5.pkl",
      "7a6b6aeb-af0e-441f-97b5-d5db846bb045": "Test_09/Level_6.pkl",
      "8e2cfb59-d31b-4346-b67a-b96a12ca04f5": "Test_09/Level_7.pkl",
      "2661fcc0-9b3c-45bd-b60e-3c6351acabd1": "Test_09/Level_8.pkl",
      "efcce7b2-e33a-4510-af56-09db1bfb5bd0": "Test_09/Level_9.pkl",
      "c643a6ba-a8a3-42de-afcc-fa92328397b7": "Test_10/Level_0.pkl",
      "86c8d140-2b1d-41df-ba97-a959c54d2c19": "Test_10/Level_1.pkl",
      "9826d43a-6be2-49ac-bfd7-fa2475f62985": "Test_10/Level_2.pkl",
      "151f38af-a59a-42f8-9b2e-2df3fef3f658": "Test_10/Level_3.pkl",
      "5b7b42ed-e41a-4e97-806a-6287ac918537": "Test_10/Level_4.pkl",
      "d897ffe6-43a8-4ebc-9881-6097be7711e7": "Test_10/Level_5.pkl",
      "95aa9a6b-b80a-4dcc-a0a4-228f53bc7959": "Test_10/Level_6.pkl",
      "0330f00b-412e-44d5-b7cc-23bebd26fa88": "Test_10/Level_7.pkl",
      "9742751d-d670-4310-97ab-a14973112470": "Test_10/Level_8.pkl",
      "72df8c4b-f0ef-438d-9858-88053cb188c1": "Test_10/Level_9.pkl",
      "05b4cb03-5576-4d79-9afa-1c6318d632ec": "Test_11/Level_0.pkl",
      "6db73b00-b6f1-4f63-9fd0-49f518361ee2": "Test_11/Level_1.pkl",
      "69df632e-d2aa-4005-a9e2-1c5e07eeebd9": "Test_11/Level_2.pkl",
      "09aff9df-7c67-4810-8e13-90f8c9bd05a1": "Test_11/Level_3.pkl",
      "3de1e810-7abe-4dd4-9663-e19270c37c52": "Test_11/Level_4.pkl",
      "afd8d475-9bd3-4740-a5be-293cd211b34d": "Test_11/Level_5.pkl",
      "de0a8389-d573-483c-811b-e7829bd58a54": "Test_11/Level_6.pkl",
      "65b60f43-6a71-4c7b-805f-6c3f564c87bb": "Test_11/Level_7.pkl",
      "8d1746ff-83c6-4675-acd7-01a2a654ec0a": "Test_11/Level_8.pkl",
      "011866d3-76a6-4b5a-9c42-447e2d567892": "Test_11/Level_9.pkl",
      "23d5ddd1-1fb0-4149-bf59-a2e7cd34213a": "Test_12/Level_0.pkl",
      "1990750e-de0f-4789-9dcc-dae5b9b99173": "Test_12/Level_1.pkl",
      "b127b87c-600d-4f28-b74a-e6c33d27e42f": "Test_12/Level_2.pkl",
      "bf0f6ceb-62fd-4a92-a7b0-29cf898b05e1": "Test_12/Level_3.pkl",
      "103881e7-8415-4d6b-90c5-cef06f36b5b3": "Test_12/Level_4.pkl",
      "8c558c8b-1a04-4c38-9f98-20cd5c8195a7": "Test_12/Level_5.pkl",
      "5c025c8a-a032-494f-8204-dd92b1067448": "Test_12/Level_6.pkl",
      "441fe9aa-79d7-4e27-8fb5-213c77c4f295": "Test_12/Level_7.pkl",
      "b630c1ad-f3a1-41c3-8e34-735a78dec9d1": "Test_12/Level_8.pkl",
      "3200dbee-2685-48c7-a7dc-2e780853efda": "Test_12/Level_9.pkl",
      "46583ddd-855b-4e6d-8711-d7b5a4fd26c1": "Test_13/Level_0.pkl",
      "a497e35d-fd84-4be2-a45d-f847962cd5f8": "Test_13/Level_1.pkl",
      "295d5dc4-f4d4-4016-8fc0-4badd1b9c94e": "Test_13/Level_2.pkl",
      "f8934f7d-e1f6-462e-8a12-dc82c440bc90": "Test_13/Level_3.pkl",
      "30fd755b-9f29-4330-b4d3-8eccc44ffade": "Test_13/Level_4.pkl",
      "a9cf3c28-8b08-451a-830f-b737936a9579": "Test_13/Level_5.pkl",
      "273b434f-74b7-4581-9d69-13f030b67313": "Test_13/Level_6.pkl",
      "fcf115d7-4246-4790-a89d-666f368b3356": "Test_13/Level_7.pkl",
      "c1306680-d5e0-4629-939d-ee9e3f4c439b": "Test_13/Level_8.pkl",
      "a5cafc37-5ab6-40b2-8c1b-19089e724b1d": "Test_13/Level_9.pkl",
      "34606fe8-3ba5-4778-a7f0-0275c1def3b8": "Test_14/Level_0.pkl",
      "2d6ffd36-f33d-4a68-9868-53c7aa3f4011": "Test_14/Level_1.pkl",
      "38e4e4d4-f801-42a0-8eac-a1a9a41a8a3e": "Test_14/Level_2.pkl",
      "bec0103f-bdf4-42b5-b04e-a44528b8c8d1": "Test_14/Level_3.pkl",
      "1e01bc5e-dcdb-4ef1-94a6-f4e3a77613b8": "Test_14/Level_4.pkl",
      "5debda2f-e5c1-447e-b025-d71252591074": "Test_14/Level_5.pkl",
      "2d8872a4-f002-4294-9396-91d9cefabdb7": "Test_14/Level_6.pkl",
      "ac4adfe9-a213-45bd-843b-f346c9891b2c": "Test_14/Level_7.pkl",
      "b59c4643-2b45-45e4-89f7-007ef1955c9f": "Test_14/Level_8.pkl",
      "f56f6f85-9aff-4f4e-bd84-8d763708e76f": "Test_14/Level_9.pkl"
    }[scenario_id]


def mean_punctuality_aggregator(scores):
  data = np.array(scores).transpose()
  # step rewards are (0,0), only take episode rewards with >0 agents in the second column:
  n_stops_on_time = data[0][data[1] > 0]
  n_stops = data[1][data[1] > 0]
  scenario_punctuality = np.divide(n_stops_on_time, n_stops)
  return np.mean(scenario_punctuality)
