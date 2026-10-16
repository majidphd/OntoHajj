"""E4 sensitivity (also: crowd exposure as a near-hard constraint) to the normalization of the criteria (reviewer concern: the most heavily weighted criterion,
crowd exposure, increased on average). Compares r* under per-question min-max normalization (Eq. 4, used in the
framework) with r* under fixed-scale normalization, in which every criterion is normalized with bounds taken over
all feasible routes of all route questions, so that a weight expresses an absolute priority.
Writes results/decision_normalization.json."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
src = open(os.path.join(ROOT, "eval", "decision_eval.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(ROOT, "eval", "decision_eval.py")}
exec(src[:src.index("stats = defaultdict(float)")], ns)
from ontohajj.decide import decide, CRITERIA, COST, WEIGHTS

decs = []
for c in ns["cases"]:
    d = decide(ns["kg"], c["o"], c["d"], c["t"], c["p"], c["end"])
    if d["best"] is not None:
        decs.append(d)
allraw = [f["raw"] for d in decs for f in d["feasible"]]
B = {k: (min(r[k] for r in allraw), max(r[k] for r in allraw)) for k in CRITERIA}


def u_fixed(raw):
    u = 0.0
    for k in CRITERIA:
        lo, hi = B[k]; x = 1.0 if hi == lo else (raw[k] - lo) / (hi - lo)
        u += WEIGHTS[k] * ((1 - x) if COST[k] else x)
    return u


def crowd_cap(feas, eps=0.10):
    """crowd exposure as a near-hard constraint: keep routes within 10% of the least crowded feasible route"""
    m = min(f["raw"]["u2_crowd_exposure"] for f in feas)
    keep = [f for f in feas if f["raw"]["u2_crowd_exposure"] <= m * (1 + eps) + 1e-9]
    return max(keep, key=lambda f: f["utility"])


out = {"n_answered": len(decs), "bounds": B}
for name in ("minmax", "fixed", "crowd_cap"):
    rows = []; more = less = same = differs = 0
    for d in decs:
        feas = d["feasible"]
        best = feas[0] if name == "minmax" else (max(feas, key=lambda f: u_fixed(f["raw"])) if name == "fixed" else crowd_cap(feas))
        fast = min(feas, key=lambda f: f["raw"]["u1_travel_time"])
        rows.append((best["raw"], fast["raw"]))
        if best["segments"] != fast["segments"]:
            differs += 1; dc = best["raw"]["u2_crowd_exposure"] - fast["raw"]["u2_crowd_exposure"]
            more += dc > 1e-9; less += dc < -1e-9; same += abs(dc) <= 1e-9
    rel = {}
    for k in CRITERIA:
        b = np.array([x[0][k] for x in rows]); f = np.array([x[1][k] for x in rows])
        rel[k] = round(100 * (b.mean() - f.mean()) / f.mean(), 1) if f.mean() else 0.0
    out[name] = {"differs_pct": round(100 * differs / len(decs), 1), "more_crowded": more, "less_crowded": less, "equal": same,
                 "relative_change_vs_fastest_pct": rel}
json.dump(out, open(os.path.join(ROOT, "results", "decision_normalization.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "bounds"}, indent=1))
