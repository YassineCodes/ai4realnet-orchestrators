# Celery orchestrator for AI4REALNET Power Grid domain
# Based on: https://github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py

import logging
import os
import ssl
from typing import List
import sys
from celery import Celery

# Add parent directory to import the base Orchestrator class
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Now import base class from parent
from orchestrator import Orchestrator

from power_grid_test_runner import (
    TestRunner_KPI_VF_073_Power_Grid,
    TestRunner_KPI_SF_072_Power_Grid,
    TestRunner_KPI_SF_071_Power_Grid,
    TestRunner_KPI_DF_069_Power_Grid,
    TestRunner_KPI_FF_070_Power_Grid,
    TestRunner_KPI_AF_074_Power_Grid,
    TestRunner_KPI_DF_075_Power_Grid,
    TestRunner_KPI_RF_076_Power_Grid,
    TestRunner_KPI_SF_077_Power_Grid,
)

logger = logging.getLogger(__name__)

# Celery application configuration
app = Celery(
    broker=os.environ.get('BROKER_URL'),
    backend=os.environ.get('BACKEND_URL'),
    broker_use_ssl={
        'keyfile': os.environ.get("RABBITMQ_KEYFILE"),
        'certfile': os.environ.get("RABBITMQ_CERTFILE"),
        'ca_certs': os.environ.get("RABBITMQ_CA_CERTS"),
        'cert_reqs': ssl.CERT_REQUIRED
    }
)

# Configure queue routing - CORRECT QUEUE NAME
QUEUE_NAME = "Power Grid"  # This is the actual queue name from FAB
app.conf.task_default_queue = QUEUE_NAME
app.conf.task_routes = {
    'PowerGrid': {'queue': QUEUE_NAME}
}

# Power Grid orchestrator with registered test runners
# KPI definitions from: https://github.com/flatland-association/flatland-benchmarks
power_grid_orchestrator = Orchestrator(
    test_runners={
        # Multi-Attacker Robustness & Resilience KPIs (INESC TEC)
        # All KPIs share the same evaluation implementation
        
        # Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
        "b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920": TestRunner_KPI_VF_073_Power_Grid(
            test_id="b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920", 
            scenario_ids=['61063867-df62-4024-be42-c57507a15d7c'], 
            benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        "a121d8bd-1943-41ba-b3a7-472a0154f8f9": TestRunner_KPI_SF_072_Power_Grid(
            test_id="a121d8bd-1943-41ba-b3a7-472a0154f8f9", 
            scenario_ids=['9cd1a5e0-8445-4b9d-859b-76b096d33049'], 
            benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        "3d033ec6-942a-4b03-b26e-f8152ba48022": TestRunner_KPI_SF_071_Power_Grid(
            test_id="3d033ec6-942a-4b03-b26e-f8152ba48022", 
            scenario_ids=['70d937d5-742b-4838-a456-4a95ff994788'], 
            benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        "1cbb7783-47b4-4289-9abf-27939da69a2f": TestRunner_KPI_DF_069_Power_Grid(
            test_id="1cbb7783-47b4-4289-9abf-27939da69a2f", 
            scenario_ids=['900d5489-2539-4a49-b3fb-3ae2039be92f'], 
            benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        "acaf712a-c06c-4a04-a00f-0e7feeefb60c": TestRunner_KPI_FF_070_Power_Grid(
            test_id="acaf712a-c06c-4a04-a00f-0e7feeefb60c", 
            scenario_ids=['fdaac433-3ef0-4667-afb8-8014d0c1afa3'], 
            benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
        
        # Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)
        "534f5a1f-7115-48a5-b58c-4deb044d425d": TestRunner_KPI_AF_074_Power_Grid(
            test_id="534f5a1f-7115-48a5-b58c-4deb044d425d", 
            scenario_ids=['bbcf8224-c768-4469-8ff5-939d977383b4'], 
            benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        "04a23bfc-fc44-4ec4-a732-c29214130a83": TestRunner_KPI_DF_075_Power_Grid(
            test_id="04a23bfc-fc44-4ec4-a732-c29214130a83", 
            scenario_ids=['b355482b-30a2-431e-9536-8e3dd29d06d1'], 
            benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        "225aaee8-7c7f-4faf-810b-407b551e9f2a": TestRunner_KPI_RF_076_Power_Grid(
            test_id="225aaee8-7c7f-4faf-810b-407b551e9f2a", 
            scenario_ids=['2eaf04e3-090a-4c13-b923-ac86de1b6db1'], 
            benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
        "7fe4210f-1253-411c-ba03-49d8b37c71fa": TestRunner_KPI_SF_077_Power_Grid(
            test_id="7fe4210f-1253-411c-ba03-49d8b37c71fa", 
            scenario_ids=['4523d73e-427a-42a1-b841-c9668373fafb'], 
            benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        ),
    }
)


@app.task(name="PowerGrid", bind=True)
def orchestrator(self, submission_data_url: str, tests: List[str] = None, **kwargs):
    """
    Celery task for Power Grid domain evaluation.
    
    Args:
        submission_data_url: URL to download submitted agent
        tests: List of test IDs to run (None = run all)
        **kwargs: Additional task parameters
    
    Returns:
        Evaluation results
    """
    submission_id = self.request.id
    benchmark_id = orchestrator.name
    logger.info(
        f"Queue/task {benchmark_id} received submission {submission_id} "
        f"with submission_data_url={submission_data_url} for tests={tests}"
    )
    return power_grid_orchestrator.run(
        submission_id=submission_id,
        submission_data_url=submission_data_url,
        tests=tests,
    )