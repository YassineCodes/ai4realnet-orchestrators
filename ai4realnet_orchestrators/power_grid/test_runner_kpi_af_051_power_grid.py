import logging
import time
from typing import Dict

from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner

logger = logging.getLogger(__name__)

# KPI ID to metric mapping
SCALABILITY_KPI_MAPPING = {
    # Scalability KPIs (Benchmark: 16706c82-75df-4969-932d-a7f5c941eca2)
    "1409dbf6-0f66-4570-97df-fda84c46c71d": {
        "name": "KPI-AF-051: AI-Agent scalability testing",
        "metric_key": "avg_time_per_timestep",
        "description": "Wall-clock seconds the agent needs per environment timestep"
    },
}


class ScalabilityTestRunner(PowerGridTestRunner):
    """
    The KPI is the average wall-clock time per timestep; comparing it across scenarios
    of increasing size (n_sub / n_line).
    """

    # Specific KPI mapping for this category
    KPI_MAPPING = SCALABILITY_KPI_MAPPING

    def getResult(self, env, env_shift, agent) -> Dict:
        """Timing depends on the scenario, so every scenario is measured instead of reusing the per-submission cache."""
        metrics = self._compute_all_metrics(env, env_shift, agent)
        return {"primary": float(metrics[self.kpi_info["metric_key"]])}

    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """Run a single episode and return the timing metrics."""
        obs = env.reset()
        reward, done, num_timesteps = 0.0, False, 0

        # only agent.act() is timed, not env.step()
        act_times = []
        start = time.perf_counter()
        while not done:
            act_start = time.perf_counter()
            action = agent.act(obs, reward, done)
            act_times.append(time.perf_counter() - act_start)
            obs, reward, done, _ = env.step(action)
            num_timesteps += 1
        total_time_seconds = time.perf_counter() - start
        # the first call is left out as one-off warm-up, unless it is the only call
        steady_act_times = act_times[1:] or act_times

        # time spent in the backend's powerflow computation, not attributable to the agent
        powerflow_time = getattr(env, "_time_powerflow", float("nan"))

        metrics = {
            "avg_time_per_timestep": sum(steady_act_times) / len(steady_act_times),
            "total_time_seconds": total_time_seconds,
            "num_timesteps": float(num_timesteps),
            "powerflow_time": powerflow_time,
            "agent_time": sum(act_times),
            "n_sub": float(env.n_sub),
            "n_line": float(env.n_line),
        }
        logger.info(f"Scalability metrics: {metrics}")
        return metrics


class TestRunner_KPI_AF_051_Power_Grid(ScalabilityTestRunner):
    """KPI-AF-051: AI-Agent Scalability Testing"""
    pass
