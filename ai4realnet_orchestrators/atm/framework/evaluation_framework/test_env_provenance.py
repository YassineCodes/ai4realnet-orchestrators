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


# The script below runs in a fresh interpreter, because registration is process-global:
# any test above would have populated the registry and hidden what is checked here.
_UNREGISTERED_PROBE = r"""
import importlib, importlib.util, json, os, sys

atm_dir, runner_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(atm_dir, "framework"))
sys.path.insert(0, atm_dir)

spec = importlib.util.spec_from_file_location("_atm_runner", runner_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out = {"fork_imported_at_module_import": "bluesky_gym" in sys.modules}
out["prov"] = mod.env_provenance()
out["penalties"] = mod.env_reward_constants()

# What the record itself points at, so the expected values are read from the task that
# ran rather than written down here.
out["truth"] = None
if out["prov"]["module"] != "unknown":
    env_mod = importlib.import_module(out["prov"]["module"])
    out["truth"] = {
        "restricted_area": float(getattr(env_mod, "RESTRICTED_AREA_INTRUSION_PENALTY")),
        "aircraft": float(getattr(env_mod, "AC_INTRUSION_PENALTY", 0.0)),
    }
print(json.dumps(out))
"""


@pytest.mark.skipif(
    importlib.util.find_spec("bluesky_gym") is None,
    reason="the AI4REALNET bluesky-gym fork is not installed")
def test_both_resolvers_work_without_the_caller_registering_first(tmp_path):
    """
    The runner takes the provenance record at the top of _run_complete_evaluation(),
    before _initialize_environment() imports the fork and calls register_envs(). The
    fingerprint test above passes only because it registers the envs itself, so it pinned
    a convenient order instead of the one a campaign uses -- and a campaign logged no
    module, no file and no hash while that test was green.

    The reward constants resolve through the same registry and are the reason this is not
    only about a log line. They are read after initialisation, so they are correct today,
    and their fallback is the v1 set, which for the configured task is also what the
    module holds -- no measured KPI was ever normalised by the wrong scale. But ENV_ID is
    overridable, and on another task that fallback would apply v1's constants to an
    environment that may not award them, moving the scale KPI-DF-069 is normalised by
    with nothing but a debug line to show it. Only call order prevents that, so this
    checks both resolvers against the module the record names rather than against
    numbers written down here.
    """
    import json
    import subprocess

    probe = tmp_path / "probe_unregistered.py"
    probe.write_text(_UNREGISTERED_PROBE)
    proc = subprocess.run([sys.executable, str(probe), ATM_DIR, RUNNER_PATH],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])

    assert not got["fork_imported_at_module_import"], (
        "the runner imported the fork on import, so this does not test the order "
        "_run_complete_evaluation() actually asks in")

    prov = got["prov"]
    assert prov["module"] != "unknown", (
        "env_provenance() resolved nothing when called before the environment was "
        "initialised, which is when the runner calls it")
    assert re.fullmatch(r"[0-9a-f]{64}", prov["module_sha256"])
    assert os.path.isfile(prov["module_file"])

    assert got["penalties"] == got["truth"], (
        "env_reward_constants() fell back instead of reading the module the provenance "
        "record names, so KPI-DF-069 would be normalised by the wrong scale")
