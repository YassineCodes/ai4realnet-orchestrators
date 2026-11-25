"""
How far each attacker actually moves the commanded action, from a campaign save-dir.

KPI-FF-070 counts the steps on which the action changed at all. In a continuous action
space that is almost every step, so the KPI saturates near 1.0 and cannot rank attackers.
The README argues that this is a property of the action space rather than a numerical
artefact, and supports it with three figures: the median largest per-component change,
the number of steps on which the action did not move at all, and the share of steps on
which the random baseline perturbs nothing by its own dropout parameters.

Those figures were produced by hand, so the document quoted numbers that nobody could
regenerate. This computes them from the archives the campaign already writes, which hold
the perturbed and clean action at every step, and reconstructs KPI-FF-070 from the same
counts as a check on the metric itself.

    python action_change_report.py <save-dir>

Attackers are discovered from the archives present, not listed here, so a campaign run
with a different `--attackers` set reports exactly what it ran.
"""
import glob
import os
import pickle
import sys

import numpy as np

# Written by result_getter alongside the per-attacker archives; it holds the clean rollout
# and so has no perturbed action to compare against.
UNPERTURBED = "unperturbed.pkl"


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def report(save_dir):
    paths = sorted(p for p in glob.glob(os.path.join(save_dir, "*.pkl"))
                   if os.path.basename(p) != UNPERTURBED)
    if not paths:
        raise SystemExit(f"no attacker archives in {save_dir}")

    rows = []
    for path in paths:
        d = load(path)
        acted, clean = d["actions"], d["actions_unperturbed"]
        perturbations = d.get("perturbations")

        deltas, unchanged, steps, null_perturbations = [], 0, 0, 0
        for ep in range(len(acted)):
            for t in range(len(acted[ep])):
                a = np.asarray(acted[ep][t], dtype=np.float64).ravel()
                c = np.asarray(clean[ep][t], dtype=np.float64).ravel()
                dmax = float(np.max(np.abs(a - c))) if a.size else 0.0
                deltas.append(dmax)
                steps += 1
                if dmax == 0.0:
                    unchanged += 1
                if perturbations is not None and t < len(perturbations[ep]):
                    p = np.asarray(perturbations[ep][t], dtype=np.float64)
                    if not np.any(p):
                        null_perturbations += 1

        rows.append({
            "attacker": os.path.splitext(os.path.basename(path))[0],
            "steps": steps,
            "median_max_change": float(np.median(deltas)) if deltas else 0.0,
            "max_max_change": float(np.max(deltas)) if deltas else 0.0,
            "unchanged": unchanged,
            "null_perturbations": null_perturbations,
            # KPI-FF-070 is the fraction of steps whose action changed. Recomputing it
            # from these counts is independent of the metrics code that reports it, so
            # agreement means the KPI measures what its name says.
            "ff070_reconstructed": (1.0 - unchanged / steps) if steps else 0.0,
        })

    rows.sort(key=lambda r: -r["median_max_change"])
    print(f"save-dir: {save_dir}\n")
    print(f"{'attacker':<12} {'steps':>7} {'median max':>11} {'largest':>9} "
          f"{'unchanged':>10} {'no perturbation':>16} {'FF-070 recomputed':>18}")
    for r in rows:
        unchanged = (f"1 in {r['steps'] // r['unchanged']}" if r["unchanged"]
                     else "none")
        share = (f"{r['null_perturbations']}/{r['steps']} "
                 f"({100.0 * r['null_perturbations'] / r['steps']:.2f}%)"
                 if r["null_perturbations"] else "-")
        print(f"{r['attacker']:<12} {r['steps']:>7} {r['median_max_change']:>11.4f} "
              f"{r['max_max_change']:>9.4f} {unchanged:>10} {share:>16} "
              f"{r['ff070_reconstructed']:>18.4f}")

    print("\nThe action space spans 2.0 per component, so a median largest change of x is")
    print("x/2 of the full command range. Compare 'FF-070 recomputed' against the reported")
    print("KPI-FF-070: they are computed from the same rollouts by different code.")
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report(sys.argv[1])
