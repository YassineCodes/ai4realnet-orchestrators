"""
Pin campaign_report.py, which regenerates the README's measured-results block from a log.

The script was written against a real campaign log and verified by reproducing the
published table from it byte for byte. That log is a build artefact on one machine, so it
cannot be the standing check. These tests pin the contract instead, on fixture logs built
here: the row order, the placement of the random baseline, and above all the emphasis
rule, which is the part a reader trusts without being able to verify it.

Emphasis is the reason this script exists. Bold marks the attacker that hurts the
defender most in a column, so it is a claim about the measurement, and a table carried by
hand into a new run keeps the previous run's winner while every number around it changes.
"""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location(
    "_campaign_report", os.path.join(HERE, "campaign_report.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

PREFIX = "INFO ai4realnet_orchestrators.atm.test_runner_robustness_resilience_kpi_069_077:"

# The field order the runner writes, which the parser's regex depends on.
FIELDS = ["vuln", "steps", "severity", "drop", "freq", "area", "degr", "rest",
          "state_sim", "rpa"]


def attacker_line(name, **values):
    """One result line in the runner's format. Unnamed fields default to a middling value."""
    v = {f: 0.5 for f in FIELDS}
    v.update(values)
    return (f"{PREFIX}   {name}: vuln={v['vuln']} steps={v['steps']} "
            f"severity={v['severity']} reward_drop={v['drop']}% act_freq={v['freq']} "
            f"area={v['area']} degr={v['degr']} rest={v['rest']} "
            f"state_sim={v['state_sim']} rpa={v['rpa']} (target >= 0.9)")


def build_log(tmp_path, lines, baseline_steps=1000, episodes=20, xi=0.1):
    text = [
        f"SUMMARY: Unperturbed",
        f"Episodes:          {episodes}",
        f"Total steps:       {baseline_steps}",
        f"Total time:        10.0s (0.2 min)",
        f"Avg time/step:     0.010s",
        *lines,
        "",
        f"ATM robustness/resilience KPIs  (env=StaticObstacleCREnv-v1, "
        f"episodes={episodes}, xi={xi})",
        "agent: /somewhere/model.zip",
        "",
        "  KPI-DF-069: Drop-off in reward                             0.1000",
        "  KPI-FF-070: Frequency changed output AI agent              0.2000",
        "  KPI-SF-071: Severity of changed output AI agent            0.3000",
        "  KPI-SF-072: Steps survived with perturbations             40.0000",
        "  KPI-VF-073: Vulnerability to perturbation                  0.5000",
        "  KPI-AF-074: Area between reward curves                   600.0000",
        "  KPI-DF-075: Degradation time                               7.0000",
        "  KPI-RF-076: Restorative time                               8.0000",
        "  KPI-SF-077: Similarity state to unperturbed situation      0.9000",
        "  KPI-RF-078: Reward per action                              1.0000",
    ]
    path = tmp_path / "campaign.log"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return str(path)


def cells(table_text, attacker):
    """The cells of one row, by the attacker's label rather than by position."""
    for line in table_text.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if parts and attacker in parts[0]:
            return parts[1:]
    raise AssertionError(f"no row for {attacker} in\n{table_text}")


def test_rows_run_from_most_to_least_severe(tmp_path):
    """
    KPI-SF-071 is the KPI the README ranks attackers by in this domain, so the row order
    is a statement and not a convenience.
    """
    log = build_log(tmp_path, [
        attacker_line("Mild", severity=0.10),
        attacker_line("Worst", severity=0.90),
        attacker_line("Middling", severity=0.50),
    ])
    report = cr.parse(log)
    assert cr._rows(report["attackers"]) == ["Worst", "Middling", "Mild"]


def test_the_random_baseline_is_last_and_italic_however_it_ranks(tmp_path):
    """
    Random noise is the reference every other row is read against, not a competitor, so
    it stays at the foot of the table even on a run where it out-perturbs a learned
    attacker -- which is exactly the run where a reader needs to see it as the baseline.
    """
    log = build_log(tmp_path, [
        attacker_line("Learned", severity=0.20),
        attacker_line("Random", severity=0.80),
    ])
    report = cr.parse(log)
    assert cr._rows(report["attackers"])[-1] == "Random"

    text = cr.table(report["attackers"])
    assert "*Random (the baseline)*" in text
    assert all(c.startswith("*") and c.endswith("*") for c in cells(text, "Random")), (
        "the baseline's numbers are italicised so they do not read as another result")


def test_bold_marks_the_strict_extreme_in_the_direction_that_is_worse(tmp_path):
    """
    Worse for the defender, which is not the same as larger: a high vulnerability is a
    worse result, a low reward-per-action is a worse result, and a table that bolded the
    maximum of every column would mark the healthiest agent on half of them.
    """
    log = build_log(tmp_path, [
        attacker_line("Harsh", vuln=0.99, rpa=0.10),
        attacker_line("Gentle", vuln=0.10, rpa=0.99),
    ])
    text = cr.table(cr.parse(log)["attackers"])
    harsh, gentle = cells(text, "Harsh"), cells(text, "Gentle")
    headers = [h for h, *_ in cr.COLUMNS]
    vuln, rpa = headers.index("VF-073 vuln"), headers.index("RF-078 rpa")

    assert harsh[vuln].startswith("**") and not gentle[vuln].startswith("**")
    assert harsh[rpa].startswith("**"), (
        "the lowest reward per action is the worst result, so it is the marked one")
    assert not gentle[rpa].startswith("**")


def test_a_tie_is_not_an_extreme(tmp_path):
    """
    Two attackers at the same value do not establish a worst one. This is the common case
    for the saturating KPIs, where bolding one of several identical cells would invent a
    ranking the numbers do not support.
    """
    log = build_log(tmp_path, [
        attacker_line("A", vuln=0.70),
        attacker_line("B", vuln=0.70),
    ])
    text = cr.table(cr.parse(log)["attackers"])
    assert "**" not in text, f"a tied column was still marked:\n{text}"


def test_the_undirected_kpis_are_never_marked(tmp_path):
    """
    KPI-FF-070 saturates in a continuous action space, and the direction of KPI-DF-075 and
    KPI-RF-076 is not settled -- a longer span reads as both deeper and slower. Marking an
    extreme in any of the three would assert a reading the README does not make.
    """
    log = build_log(tmp_path, [
        attacker_line("Extreme", freq=1.0, degr=99.0, rest=99.0),
        attacker_line("Other", freq=0.5, degr=1.0, rest=1.0),
    ])
    text = cr.table(cr.parse(log)["attackers"])
    headers = [h for h, *_ in cr.COLUMNS]
    row = cells(text, "Extreme")
    for name in ("FF-070 freq", "DF-075 degr", "RF-076 rest"):
        assert not row[headers.index(name)].startswith("**"), f"{name} was marked"


def test_the_baseline_line_is_computed_from_the_unperturbed_phase(tmp_path):
    log = build_log(tmp_path, [attacker_line("A")], baseline_steps=1478, episodes=20)
    assert cr.baseline_sentence(cr.parse(log)) == (
        "Unperturbed baseline: 1478 steps over 20 episodes, 73.9 steps per episode.")


def test_a_budget_that_differs_from_the_suite_is_reported(tmp_path, capsys):
    """
    SAC_5 runs at half the suite's xi by design, and its name is the only record of that.
    The exception is printed so that a run where some *other* attacker differs cannot look
    identical to a healthy one.
    """
    log = build_log(tmp_path, [
        f"{PREFIX}   SAC_5: max |perturbation| = 0.05 (budget 0.05)",
        f"{PREFIX}   PPO: max |perturbation| = 0.1 (budget 0.1)",
        attacker_line("SAC_5"), attacker_line("PPO"),
    ])
    cr.main(log)
    out = capsys.readouterr().out
    assert "budget differs from the suite xi: SAC_5 0.05" in out
    assert "PPO" not in out.split("budget differs")[1].splitlines()[0]


def test_the_env_fingerprint_is_read_from_the_runner_header(tmp_path):
    """
    The runner labels it "Env sha256:" in the run header, while training_metadata.json
    labels the same value "module_sha256". Reading only one label would report the
    fingerprint as unlogged on a real campaign log, which is the opposite of what it
    says: a reader would conclude the provenance record had not been written at all.
    """
    sha = "ce3f7b2e30822908d632d188dee0c675b787afb09d271c1c6557ff6db17ba544"
    for label in (f"  Env sha256: {sha}", f'  "module_sha256": "{sha}"'):
        log = build_log(tmp_path, [label, attacker_line("A")])
        assert cr.parse(log)["sha"] == sha, f"not parsed from {label!r}"


def test_a_log_without_a_fingerprint_says_so_rather_than_inventing_one(tmp_path):
    """Runs from before env_provenance() existed have no such line, and must read as such."""
    log = build_log(tmp_path, [attacker_line("A")])
    assert cr.parse(log)["sha"] is None


def test_an_unfinished_campaign_fails_instead_of_printing_an_empty_table(tmp_path):
    """
    The per-attacker lines are written only once the suite completes. A log read too early
    must not yield a well-formed table with nothing in it.
    """
    path = tmp_path / "partial.log"
    path.write_text("[TIMER] Episode 3/200 completed: 40 steps in 1.2s\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cr.parse(str(path))


def test_a_rerun_appended_to_a_log_reports_the_later_run(tmp_path):
    """Logs are appended to. The numbers printed must be the most recent ones present."""
    log = build_log(tmp_path, [
        attacker_line("A", severity=0.10),
        attacker_line("A", severity=0.80),
    ])
    assert cr.parse(log)["attackers"]["A"]["severity"] == 0.80
