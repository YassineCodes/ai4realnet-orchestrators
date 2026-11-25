"""
A reported KPI has to say which environment produced it, not just which id was asked for.

`StaticObstacleCREnv-v1` exists only in the AI4REALNET bluesky-gym fork, which is
installed from a working directory rather than a versioned release. Its distribution
metadata carries a placeholder version, so two runs of the same id on two machines or two
dates can be two different tasks and the id alone cannot tell them apart. Every KPI here
depends on that code: the reward constants set the scale KPI-DF-069 normalises by, and the
observation layout and dynamics set the rest. The policy was already pinned by its
SHA-256; this pins the other half.

These tests pin env_provenance(): it names the module the registry actually resolves to,
fingerprints that module's file, and reports rather than raises when a field is
unavailable, because it runs inside the evaluation and must never be the thing that fails
a campaign.
"""
import hashlib
import importlib.util
import os
import re
import sys

import pytest

ATM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ATM_DIR)

RUNNER_PATH = os.path.join(
    ATM_DIR, "test_runner_robustness_resilience_kpi_069_077.py")


def _runner():
    """Load the runner module by path; importing it by name would need the orchestrator."""
    spec = importlib.util.spec_from_file_location("_atm_runner", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _runner()


def test_provenance_reports_every_field(runner):
    """The keys are the contract: a caller may store them without checking for presence."""
    prov = runner.env_provenance()
    assert set(prov) == {"env_id", "distribution", "module",
                         "module_file", "module_sha256"}
    assert all(isinstance(v, str) and v for v in prov.values())


def test_provenance_defaults_to_the_configured_env(runner):
    assert runner.env_provenance()["env_id"] == runner.ENV_ID


def test_an_unknown_id_degrades_instead_of_raising(runner):
    """
    This runs inside the evaluation. A missing env, a renamed module or a registry that
    has not been populated must cost the provenance record, never the campaign.
    """
    prov = runner.env_provenance("NoSuchEnv-v99")
    assert prov["env_id"] == "NoSuchEnv-v99"
    assert prov["module"] == "unknown"
    assert prov["module_sha256"] == "unknown"


@pytest.mark.skipif(
    importlib.util.find_spec("bluesky_gym") is None,
    reason="the AI4REALNET bluesky-gym fork is not installed")
def test_the_fingerprint_is_of_the_module_that_would_run(runner):
    """
    The point of the record: the hash must be of the file the registry resolves to, so a
    reviewer can compare it against their own checkout and see whether the task differs.
    """
    from _bluesky_compat import ensure_warnings_deprecated
    ensure_warnings_deprecated()
    import bluesky_gym
    import bluesky_gym.envs  # noqa: F401  (registers the envs)
    bluesky_gym.register_envs()

    prov = runner.env_provenance()
    assert prov["module"] != "unknown", "the env id did not resolve to a module"
    assert re.fullmatch(r"[0-9a-f]{64}", prov["module_sha256"])
    assert os.path.isfile(prov["module_file"])

    h = hashlib.sha256()
    with open(prov["module_file"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert prov["module_sha256"] == h.hexdigest()

    # The reward constants KPI-DF-069 normalises by come from this same module, so the
    # fingerprint covers them. If it changes, that normaliser may have changed too.
    assert set(runner.env_reward_constants()) == {"restricted_area", "aircraft"}
