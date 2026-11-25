"""
Regenerate the README's measured-results block from a campaign log.

The table of ten KPIs over seven attackers, the baseline line above it and the suite
aggregate sentence below it were transcribed by hand. That is seventy numbers to copy,
and every re-measurement copies them again; a stale cell is indistinguishable from a
measured one. The emphasis is the worse half of the problem, because bold marks the
attacker that hurts the defender most in each column, and a hand-carried table keeps the
previous run's winner even when the new numbers name a different one.

Everything printed here already exists in the log the runner writes, so this parses that
log instead. Emphasis is derived rather than carried: a cell is bold when it is the
strict extreme in the direction that is worse for the defender.

    python campaign_report.py <campaign.log>

KPI-FF-070, KPI-DF-075 and KPI-RF-076 are never emphasised. FF-070 saturates at 1.0000 in
a continuous action space, so an extreme there is a near-tie and ranking on it would be an
artefact of the action space (see action_change_report.py). The direction of DF-075 and
RF-076 is not settled -- a longer degradation span and a longer restoration span each
describe both a deeper disruption and a slower one, and the README makes no claim either
way -- so marking an extreme would assert a reading the document does not make.
"""
import re
import sys

# One INFO line per attacker, written by the runner once the suite completes. The table
# columns are drawn from these fields; nothing is recomputed here.
ATTACKER_LINE = re.compile(
    r"\b(?P<name>\w+): vuln=(?P<vuln>[-\d.]+) steps=(?P<steps>[-\d.]+) "
    r"severity=(?P<severity>[-\d.]+) reward_drop=(?P<drop>[-\d.]+)% "
    r"act_freq=(?P<freq>[-\d.]+) area=(?P<area>[-\d.]+) degr=(?P<degr>[-\d.]+) "
    r"rest=(?P<rest>[-\d.]+) state_sim=(?P<state_sim>[-\d.]+) rpa=(?P<rpa>[-\d.]+)")

AGGREGATE_LINE = re.compile(r"^\s+(KPI-[A-Z]{2}-\d{3}): .+?\s{2,}([-\d.]+)\s*$")
HEADER_LINE = re.compile(
    r"env=(?P<env>[\w-]+), episodes=(?P<episodes>\d+), xi=(?P<xi>[\d.]+)")
AGENT_LINE = re.compile(r"^agent: (?P<agent>.+?)\s*$")
BUDGET_LINE = re.compile(r"\b(?P<name>\w+): max \|perturbation\| = (?P<xi>[\d.]+) \(budget")
# The runner's header prints "Env sha256:"; training_metadata.json and the smoke
# scripts print "module_sha256". Both name the same fingerprint, and a reader who
# saw it reported as unlogged would conclude the provenance fix had not taken.
SHA_LINE = re.compile(
    r"(?:module_sha256|Env sha256)\"?\s*[=:]\s*\"?(?P<sha>[0-9a-f]{64})")

SUMMARY_NAME = re.compile(r"^SUMMARY: (?P<name>\w+)\s*$")
SUMMARY_FIELD = re.compile(r"^(Total steps|Total time|Avg time/step):\s+(?P<value>\S+)")

BASELINE = "Unperturbed"
RANDOM = "Random"
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten"}

# (header, field, decimals, which extreme is worse for the defender; None = never marked)
COLUMNS = [
    ("VF-073 vuln", "vuln", 4, max),
    ("SF-071 severity", "severity", 4, max),
    ("FF-070 freq", "freq", 4, None),
    ("AF-074 area", "area", 2, max),
    ("SF-072 steps", "steps", 1, max),
    ("DF-069 drop %", "drop", 2, max),
    ("DF-075 degr", "degr", 1, None),
    ("RF-076 rest", "rest", 1, None),
    ("SF-077 state sim", "state_sim", 4, min),
    ("RF-078 rpa", "rpa", 4, min),
]


def parse(path):
    """Read the log once. Later occurrences win, so a log with a re-run appended reports it."""
    attackers, aggregate, timings, budgets = {}, [], {}, {}
    header, agent, sha, current = {}, None, None, None

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ATTACKER_LINE.search(line)
            if m:
                attackers[m.group("name")] = {k: float(v)
                                              for k, v in m.groupdict().items()
                                              if k != "name"}
                continue
            m = AGGREGATE_LINE.match(line)
            if m:
                # The log's formatting is the published formatting; reusing its string
                # avoids rounding a second time a value that was rounded once already.
                aggregate.append((m.group(1), m.group(2)))
                continue
            m = BUDGET_LINE.search(line)
            if m:
                budgets[m.group("name")] = m.group("xi")
                continue
            m = HEADER_LINE.search(line)
            if m:
                header = m.groupdict()
                continue
            m = AGENT_LINE.match(line)
            if m:
                agent = m.group("agent")
                continue
            m = SHA_LINE.search(line)
            if m:
                sha = m.group("sha")
                continue
            m = SUMMARY_NAME.match(line)
            if m:
                current = m.group("name")
                timings[current] = {}
                continue
            m = SUMMARY_FIELD.match(line)
            if m and current:
                timings[current][m.group(1)] = m.group("value")

    if not attackers:
        raise SystemExit(
            f"no per-attacker result lines in {path}; has the campaign finished?")
    # The aggregate block holds one line per KPI and is printed once per suite.
    return {"attackers": attackers, "aggregate": aggregate[-len(COLUMNS):],
            "timings": timings, "budgets": budgets, "header": header,
            "agent": agent, "sha": sha}


def _rows(attackers):
    """Severity descending; the random baseline last, because it is a reference not a rank."""
    ranked = sorted((n for n in attackers if n != RANDOM),
                    key=lambda n: -attackers[n]["severity"])
    return ranked + ([RANDOM] if RANDOM in attackers else [])


def _marked(attackers):
    """The (field, attacker) cells that are the strict extreme in the worse direction."""
    out = set()
    for _, field, _, worse in COLUMNS:
        if worse is None:
            continue
        values = [a[field] for a in attackers.values()]
        target = worse(values)
        if values.count(target) == 1:
            out.add((field, next(n for n, a in attackers.items()
                                 if a[field] == target)))
    return out


def table(attackers):
    marked = _marked(attackers)
    head = [h for h, *_ in COLUMNS]
    out = ["| Attacker | " + " | ".join(head) + " |",
           "|----------|" + "|".join("-" * (len(h) + 1) + ":" for h in head) + "|"]
    for name in _rows(attackers):
        label = f"*{RANDOM} (the baseline)*" if name == RANDOM else name
        cells = []
        for _, field, places, _ in COLUMNS:
            text = f"{attackers[name][field]:.{places}f}"
            stars = "*" * (2 * ((field, name) in marked) + (name == RANDOM))
            cells.append(f"{stars}{text}{stars}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def aggregate_sentence(aggregate, n_attackers):
    count = NUMBER_WORDS.get(n_attackers, str(n_attackers))
    pairs = ", ".join(f"{kpi} {value}" for kpi, value in aggregate)
    return ("Suite aggregate, the arithmetic mean over the "
            f"{count} attackers: {pairs}.")


def baseline_sentence(report):
    t = report["timings"].get(BASELINE)
    episodes = report["header"].get("episodes")
    if not t or "Total steps" not in t or not episodes:
        return None
    steps, episodes = int(t["Total steps"]), int(episodes)
    return (f"Unperturbed baseline: {steps} steps over {episodes} episodes, "
            f"{steps / episodes:.1f} steps per episode.")


def main(path):
    report = parse(path)
    h = report["header"]
    print(f"log: {path}")
    if h:
        print(f"env {h['env']}, {h['episodes']} episodes per attacker, xi {h['xi']}")
    if report["agent"]:
        print(f"agent: {report['agent']}")
    print(f"env module sha256: {report['sha'] or 'not logged by this run'}")

    # SAC_5 carries a smaller budget by design, so the exceptions are reported rather
    # than assumed: a run where some other attacker differs should not look the same.
    odd = {n: x for n, x in report["budgets"].items() if x != h.get("xi")}
    if odd:
        print("budget differs from the suite xi: "
              + ", ".join(f"{n} {x}" for n, x in sorted(odd.items())))

    print("\n--- baseline line ---")
    print(baseline_sentence(report) or "no unperturbed summary block in this log")
    print("\n--- table ---")
    print(table(report["attackers"]))
    print("\n--- aggregate sentence ---")
    print(aggregate_sentence(report["aggregate"], len(report["attackers"])))
    print("\n--- phase timings ---")
    for name, t in report["timings"].items():
        print(f"{name:<12} {t.get('Total steps', '?'):>7} steps  "
              f"{t.get('Total time', '?'):>9}  {t.get('Avg time/step', '?'):>7}/step")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
