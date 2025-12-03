# test_local.py

import os
import sys
import zipfile
import tempfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# ============================================================
# PATH SETUP - Must be done BEFORE any imports
# ============================================================

# Get the directory where this script lives (power_grid folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POWER_GRID_DIR = SCRIPT_DIR
FRAMEWORK_DIR = os.path.join(SCRIPT_DIR, "framework")
ORCHESTRATORS_DIR = os.path.dirname(SCRIPT_DIR)  # ai4realnet_orchestrators
PARENT_DIR = os.path.dirname(ORCHESTRATORS_DIR)   # ai4realnet-orchestrators

# Add paths for imports
sys.path.insert(0, PARENT_DIR)          # For ai4realnet_orchestrators package
sys.path.insert(0, ORCHESTRATORS_DIR)   # For base test_runner
sys.path.insert(0, FRAMEWORK_DIR)       # For attack_models, etc.

# Change working directory to framework so relative paths work
os.chdir(FRAMEWORK_DIR)

print("=" * 60)
print("LOCAL TEST - FAB TestRunner")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print(f"Framework directory: {FRAMEWORK_DIR}")

# ============================================================
# NOW we can import (paths are set up)
# ============================================================

from power_grid_test_runner import MultiAttackerRobustnessTestRunner

runner = MultiAttackerRobustnessTestRunner(
    test_id="b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920",
    scenario_ids=['61063867-df62-4024-be42-c57507a15d7c'],
    benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
)

# OVERRIDE framework path for local testing
runner.FRAMEWORK_PATH = FRAMEWORK_DIR

print(f"✅ TestRunner initialized")
print(f"   KPI: {runner.kpi_info['name']}")
print(f"   Framework: {runner.FRAMEWORK_PATH}")

agent_path = os.path.join(FRAMEWORK_DIR, "agent.zip")

print(f"\n📦 Agent path: {agent_path}")
print(f"   Exists: {os.path.exists(agent_path)}")

try:
    print(f"\n🔄 Initializing framework...")
    runner._initialize_framework()
    print("✅ Framework initialized!")
    
    print(f"\n🔄 Loading agent from zip (local)...")
    
    # Extract zip locally
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(agent_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print(f"   Extracted to: {temp_dir}")
    print(f"   Contents: {os.listdir(temp_dir)}")
    
    # Load the agent using the framework's method
    from modified_curriculum_classes.baseline import CurriculumAgent
    import grid2op
    
    # Create environment
    env = grid2op.make(runner.ENV_NAME)
    
    # Create and load agent
    agent = CurriculumAgent(env.action_space, env.observation_space, "test_agent")
    
    # Find model and actions paths in extracted zip
    model_path = os.path.join(temp_dir, "model")
    actions_path = os.path.join(temp_dir, "actions")
    
    agent.load(model_path, actions_path)
    runner._defender_agent = agent
    
    print("✅ Agent loaded!")
    
    print("\n🔄 Running evaluation (this may take a while)...")
    result = runner.run_scenario(
        scenario_id="61063867-df62-4024-be42-c57507a15d7c",
        submission_id="local-test-001"
    )
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Result: {result}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()