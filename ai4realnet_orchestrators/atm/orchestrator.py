# based on https://github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py
import logging
import os
import ssl

from celery import Celery

from ai4realnet_orchestrators.atm.test_runner import BlueSkyRunner
from ai4realnet_orchestrators.atm.test_runner_robustness_resilience_kpi_069_077 import (
    ATMRobustnessTestRunner,
)
from ai4realnet_orchestrators.orchestrator import Orchestrator


logger = logging.getLogger(__name__)

app = Celery(
    broker=os.environ.get('BROKER_URL'),
    backend=os.environ.get('BACKEND_URL'),
    queue=os.environ.get("BENCHMARK_ID"),
    broker_use_ssl={
        'keyfile': os.environ.get("RABBITMQ_KEYFILE"),
        'certfile': os.environ.get("RABBITMQ_CERTFILE"),
        'ca_certs': os.environ.get("RABBITMQ_CA_CERTS"),
        'cert_reqs': ssl.CERT_REQUIRED
    }
)

bluesky_orchestrator = Orchestrator(
    test_runners={

        # KPI-AS-001: Ability to anticipate (ATM)
        "9a9b85fb-6b8b-4b42-af5d-1d81d515e6b1": BlueSkyRunner(
            test_id="9a9b85fb-6b8b-4b42-af5d-1d81d515e6b1", scenario_ids=['4cc07a90-fca8-4e96-b670-563c1e8f42fa'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-AS-009: Assistant disturbance (ATM)
        "c2a22379-524a-4294-80b2-c751be726b70": BlueSkyRunner(
            test_id="c2a22379-524a-4294-80b2-c751be726b70", scenario_ids=['76a0cad5-40fd-4b92-b904-5841aadf8a7d'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-DS-015: Decision support satisfaction (ATM)
        "290debb4-eeff-408c-a680-28fce5b376e7": BlueSkyRunner(
            test_id="290debb4-eeff-408c-a680-28fce5b376e7", scenario_ids=['22846c4e-e703-46c7-8069-38a6b6027a7d'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-HS-022: Human motivation (ATM)
        "ebd2a832-1c69-4d87-9de2-4a283e7c7f37": BlueSkyRunner(
            test_id="ebd2a832-1c69-4d87-9de2-4a283e7c7f37", scenario_ids=['4f81efc4-4c7d-4973-9fb6-1fccbe11fcd4'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-HS-023: Human response time (ATM)
        "e52b585d-0da8-4a88-97ff-0953a62a548c": BlueSkyRunner(
            test_id="e52b585d-0da8-4a88-97ff-0953a62a548c", scenario_ids=['f1278028-1c96-485c-bd6b-046d4356c335'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-SS-031: Situation awareness (ATM)
        "26bf1860-4b1c-4fed-8f81-53dac6a8f8f7": BlueSkyRunner(
            test_id="26bf1860-4b1c-4fed-8f81-53dac6a8f8f7", scenario_ids=['f8fc8712-e092-458e-a1e6-3733f5bc65ea'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-WS-040: Workload (ATM)
        "96396992-ab9a-45b8-9833-124b46925da6": BlueSkyRunner(
            test_id="96396992-ab9a-45b8-9833-124b46925da6", scenario_ids=['88a26ad8-bd41-4c4b-9f10-ee7876550752'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-CS-049: Cognitive Performance & Stress (ATM)
        "ebd44326-8c9b-4144-8e31-8c360b148dd3": BlueSkyRunner(
            test_id="ebd44326-8c9b-4144-8e31-8c360b148dd3", scenario_ids=['4b04f1b7-a4b0-4b71-8f75-6a222bb05284'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        ),
        
        # KPI-AS-002: Acceptance (ATM)
        "125cc932-43c2-40db-8fb3-e523557b98df": BlueSkyRunner(
            test_id="125cc932-43c2-40db-8fb3-e523557b98df", scenario_ids=['91da4c1c-3011-4f70-809f-8204cd3824ba'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        ),
        
        # KPI-AS-005: Agreement score (ATM)
        "7a1e72fb-942c-4f54-bc72-adba7d941a0e": BlueSkyRunner(
            test_id="7a1e72fb-942c-4f54-bc72-adba7d941a0e", scenario_ids=['5ec8e7f8-a26d-4263-95ea-6c7a0832f61d'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        ),
        
        # KPI-CS-013: Comprehensibility (ATM)
        "4795ec75-5e16-432a-9a5c-580d649471e2": BlueSkyRunner(
            test_id="4795ec75-5e16-432a-9a5c-580d649471e2", scenario_ids=['2057f6f2-015b-4370-a4db-40bb8cd9b244'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        ),
        
        # KPI-TS-038: Trust in AI solutions score (ATM)
        "626c4984-069a-4c6d-8bc6-b63d8eb91d4f": BlueSkyRunner(
            test_id="626c4984-069a-4c6d-8bc6-b63d8eb91d4f", scenario_ids=['e5063e27-d81b-499b-8827-b2a8ab0ff8d8'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        ),
        
        # KPI-TS-039: Trust towards the AI tool (ATM)
        "d3313fc9-8865-49c8-b3a2-ed0742bbcb8d": BlueSkyRunner(
            test_id="d3313fc9-8865-49c8-b3a2-ed0742bbcb8d", scenario_ids=['a5785b63-ccee-4108-ad34-32f861beeadf'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        ),
        
        # KPI-HS-003: Human intervention frequency (ATM)
        "8936e99a-3667-404d-97b7-eab2791c0cdc": BlueSkyRunner(
            test_id="8936e99a-3667-404d-97b7-eab2791c0cdc", scenario_ids=['0db9f3ea-ac49-4914-b491-ecf3f1f35263'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        ),
        
        # KPI-SS-030: Significance of human revisions (ATM)
        "a5cebe71-030b-4d3c-be27-d8c01a862952": BlueSkyRunner(
            test_id="a5cebe71-030b-4d3c-be27-d8c01a862952", scenario_ids=['a365a4df-98d6-4c7b-83bb-63b5ccb68581'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        ),
        
        # KPI-PS-089: Perceived decision novelty (ATM)
        "c489fc51-158f-4224-baba-8d18c34c19d3": BlueSkyRunner(
            test_id="c489fc51-158f-4224-baba-8d18c34c19d3", scenario_ids=['ab1d9b2b-b91f-440f-ad16-54ccabb25230'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        ),
        
        # KPI-AS-006: AI co-learning capability (ATM)
        "452bff66-df55-47fd-9ad8-725f327bedf3": BlueSkyRunner(
            test_id="452bff66-df55-47fd-9ad8-725f327bedf3", scenario_ids=['af91490d-d1cf-4644-aaec-887e320e4a36'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
        ),
        
        # KPI-HS-021: Human learning (ATM)
        "15de95cc-0ad4-4b78-9984-9bdeac8503dd": BlueSkyRunner(
            test_id="15de95cc-0ad4-4b78-9984-9bdeac8503dd", scenario_ids=['dd069d03-4ff6-4c9a-b3a6-bf1f1471e640'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
        ),
        
        # KPI-RF-027: Reduction in delay (ATM)
        "9b6bc151-9f25-4d85-bee1-919753934521": BlueSkyRunner(
            test_id="9b6bc151-9f25-4d85-bee1-919753934521", scenario_ids=['437971ac-6616-429b-ad27-f8796772c570'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        ),
        
        # KPI-SS-032: System efficiency (ATM)
        "a6cb2703-be7f-44da-a3d8-652fa8797627": BlueSkyRunner(
            test_id="a6cb2703-be7f-44da-a3d8-652fa8797627", scenario_ids=['22ef21e7-d00d-4c3b-8484-7110e024a4f5'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        ),
        
        # KPI-HS-018: Human control/autonomy over the process (ATM)
        "5be9d1d9-535a-4940-a29d-d32d087cb197": BlueSkyRunner(
            test_id="5be9d1d9-535a-4940-a29d-d32d087cb197", scenario_ids=['3324347e-e4cb-42a2-9f53-430421090075'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
        ),
        
        # KPI-IS-041: Impact on workload (ATM)
        "18ae6ae9-d0b4-4af8-bf37-4b7cbdb31a85": BlueSkyRunner(
            test_id="18ae6ae9-d0b4-4af8-bf37-4b7cbdb31a85", scenario_ids=['fddeb18e-789e-40e0-9680-ca58ae10851f'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
        ),
        
        # KPI-AF-050: AI-Agent Scalability Training (ATM)
        "4cba45eb-2512-4e8e-87da-d39a45e529f8": BlueSkyRunner(
            test_id="4cba45eb-2512-4e8e-87da-d39a45e529f8", scenario_ids=['89bbf582-bb03-4039-9318-1178da706760'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
        ),
        
        # KPI-AF-051: AI-Agent Scalability Testing (ATM)
        "bf76b70f-77ee-4cc0-9c4c-4165eca347d2": BlueSkyRunner(
            test_id="bf76b70f-77ee-4cc0-9c4c-4165eca347d2", scenario_ids=['b63f9a44-68e0-4850-b622-bac55d980b30'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
        ),
        
        # KPI-DF-052: Domain shift adaptation time (ATM)
        "68412020-58d7-4180-8cae-99f2ffa4dbf5": BlueSkyRunner(
            test_id="68412020-58d7-4180-8cae-99f2ffa4dbf5", scenario_ids=['c60afe79-dbc6-4c44-893e-387adf7bc02c'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-DF-053: Domain shift generalization gap (ATM)
        "aecbfe66-c49f-42cb-b624-d98c3d74fcb1": BlueSkyRunner(
            test_id="aecbfe66-c49f-42cb-b624-d98c3d74fcb1", scenario_ids=['1f8238f7-f6b6-482c-96ad-f4e0880b801d'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-DF-054: Domain shift out of domain detection accuracy (ATM)
        "a34741a1-d279-4b5b-a327-2a0f94bea125": BlueSkyRunner(
            test_id="a34741a1-d279-4b5b-a327-2a0f94bea125", scenario_ids=['5b92361c-927e-4aaa-83c5-4413989282a2'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-DF-055: Domain shift policy robustness (ATM)
        "a3adb18d-3b44-4498-8c2c-728d5ff40248": BlueSkyRunner(
            test_id="a3adb18d-3b44-4498-8c2c-728d5ff40248", scenario_ids=['33e32b92-51a9-4531-8b5d-8655b758a958'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-DF-056: Domain shift robustness to domain parameters (ATM)
        "defac2f8-5b3b-4de9-ae1b-8e2f74380257": BlueSkyRunner(
            test_id="defac2f8-5b3b-4de9-ae1b-8e2f74380257", scenario_ids=['3f521f00-d5f7-4883-923a-1ec2a3022dcb'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-DF-057: Domain shift success rate drop (ATM)
        "e75c588d-276f-4c6f-a329-90d570da9b67": BlueSkyRunner(
            test_id="e75c588d-276f-4c6f-a329-90d570da9b67", scenario_ids=['3ba0619c-8073-4e94-9c34-c3d5e17bcde8'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-DF-090: Domain shift forgetting rate (ATM)
        "9c6f7e45-b966-4b55-8315-790577683344": BlueSkyRunner(
            test_id="9c6f7e45-b966-4b55-8315-790577683344", scenario_ids=['3c03cc5e-f94e-4573-90a3-924b217442e6'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        
        # KPI-RS-058: Robustness to operator input (ATM)
        "07881948-f5e0-4772-83c9-3ca517ac245f": BlueSkyRunner(
            test_id="07881948-f5e0-4772-83c9-3ca517ac245f", scenario_ids=['2d805705-e6b1-45e2-8511-39d3d23a9994'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-DF-069: Drop-off in reward (ATM)
        "4819e8f6-a2d4-497f-9b61-fc90883a0dfb": ATMRobustnessTestRunner(
            test_id="4819e8f6-a2d4-497f-9b61-fc90883a0dfb", scenario_ids=['532a18a1-5e58-45a4-830e-b5ca016499f9'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-FF-070: Frequency changed output AI agent (ATM)
        "f0f94fb1-2aef-44f6-ba80-5b8320725fb0": ATMRobustnessTestRunner(
            test_id="f0f94fb1-2aef-44f6-ba80-5b8320725fb0", scenario_ids=['12c63b69-e6b1-4316-999b-7112e0b0c1d2'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-SF-071: Severity of changed output AI agent (ATM)
        "02bfbe09-6e9b-4243-a376-1a51b1beef19": ATMRobustnessTestRunner(
            test_id="02bfbe09-6e9b-4243-a376-1a51b1beef19", scenario_ids=['af261375-7fd2-4a89-92b6-3477b018a09d'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-SF-072: Steps survived with perturbations (ATM)
        "c466661d-12dc-4d1e-81a4-1db1623e3cc1": ATMRobustnessTestRunner(
            test_id="c466661d-12dc-4d1e-81a4-1db1623e3cc1", scenario_ids=['f903201d-a631-46c9-997f-f32bf7e3ff5d'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-VF-073: Vulnerability to perturbation (ATM)
        "5cfc7e4d-024b-4dd1-82a5-c3d9bf25ba50": ATMRobustnessTestRunner(
            test_id="5cfc7e4d-024b-4dd1-82a5-c3d9bf25ba50", scenario_ids=['3ced691e-d23c-47de-9967-5cf5d7be3e9e'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-RF-078: Reward per action (ATM)
        "885cab0d-d4fc-4d93-95db-243870506405": ATMRobustnessTestRunner(
            test_id="885cab0d-d4fc-4d93-95db-243870506405", scenario_ids=['eeb9b483-616d-4508-85b8-812a09f93d23'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # KPI-AF-074: Area between reward curves (ATM)
        "5372decd-6a2a-4c50-bf7a-cd57cfebe3de": ATMRobustnessTestRunner(
            test_id="5372decd-6a2a-4c50-bf7a-cd57cfebe3de", scenario_ids=['e80a5e55-dc60-459f-bf22-26f196a4711a'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        
        # KPI-DF-075: Degradation time (ATM)
        "2baef867-c1f2-4b6e-b13c-0eac9463c2fa": ATMRobustnessTestRunner(
            test_id="2baef867-c1f2-4b6e-b13c-0eac9463c2fa", scenario_ids=['2aaed2a4-7dd9-4ea6-a2df-e3ef2207680a'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        
        # KPI-RF-076: Restorative time (ATM)
        "7b15a7b3-2413-4953-b91a-24f5c0c5b6da": ATMRobustnessTestRunner(
            test_id="7b15a7b3-2413-4953-b91a-24f5c0c5b6da", scenario_ids=['00e749d7-baa3-4b24-8092-d3dd69cdea58'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        
        # KPI-SF-077: Similarity state to unperturbed situation (ATM)
        "e3fb76a2-2121-4889-adf2-b60ca29c5c71": ATMRobustnessTestRunner(
            test_id="e3fb76a2-2121-4889-adf2-b60ca29c5c71", scenario_ids=['afc812fc-e4f3-4380-a856-9987bc557d5c'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        
        # KPI-RS-091: Reflection on operator trust  (ATM)
        "4847c6f6-123d-4f5c-ab7c-6ff63b5407e5": BlueSkyRunner(
            test_id="4847c6f6-123d-4f5c-ab7c-6ff63b5407e5", scenario_ids=['12e1531b-8525-408d-b800-1d0df54f84bb'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),
        
        # KPI-RS-092: Reflection on operator agency  (ATM)
        "1f368c52-2c18-4658-a5b7-c2524e90b9dc": BlueSkyRunner(
            test_id="1f368c52-2c18-4658-a5b7-c2524e90b9dc", scenario_ids=['5afb6acf-df83-47e9-9e7e-cdfb85a3c5de'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),
        
        # KPI-RS-093: Reflection on operator de-skilling  (ATM)
        "8de9a511-8d9e-4289-a78b-0004737443c9": BlueSkyRunner(
            test_id="8de9a511-8d9e-4289-a78b-0004737443c9", scenario_ids=['e62d0769-c481-4641-b9d4-ad5586008e38'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),
        
        # KPI-RS-094: Reflection on over-reliance  (ATM)
        "7ce3a25e-219c-4146-b2a7-c4f46bc7b137": BlueSkyRunner(
            test_id="7ce3a25e-219c-4146-b2a7-c4f46bc7b137", scenario_ids=['a65a72e0-5ac4-4182-a919-da70d1241a59'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),
        
        # KPI-RS-095: Reflection on additional training  (ATM)
        "81b57209-24e3-4213-a93f-8129b3ca1c1c": BlueSkyRunner(
            test_id="81b57209-24e3-4213-a93f-8129b3ca1c1c", scenario_ids=['e82bc94e-ed5e-403a-8dce-a4ed9c7cb63d'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),
        
        # KPI-RS-096: Reflection on biases  (ATM)
        "733c3705-a1a3-45e1-9fd6-b19e9627bf50": BlueSkyRunner(
            test_id="733c3705-a1a3-45e1-9fd6-b19e9627bf50", scenario_ids=['3c3dd29c-67f1-49fa-8aef-384c957acaa4'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),
        
        # KPI-PS-097: Predicted long-term adoption  (ATM)
        "de69aa29-4aa0-4473-918e-14004116a9dc": BlueSkyRunner(
            test_id="de69aa29-4aa0-4473-918e-14004116a9dc", scenario_ids=['664cd46e-7eac-480f-92da-79574b470da4'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        ),

    }
)


# https://docs.celeryq.dev/en/stable/userguide/tasks.html#bound-tasks: A task being bound means the first argument to the task will always be the task instance (self).
# https://docs.celeryq.dev/en/stable/userguide/tasks.html#names: Every task must have a unique name.
@app.task(name='ATM', bind=True)
def orchestrator(self, submission_data_url: str, tests: list[str] = None, **kwargs):
    submission_id = self.request.id
    benchmark_id = orchestrator.name
    logger.info(
        f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}"
    )
    return bluesky_orchestrator.run(
        submission_id=submission_id,
        submission_data_url=submission_data_url,
        tests=tests,
    )
