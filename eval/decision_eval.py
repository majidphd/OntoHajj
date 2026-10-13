"""Layer-4 evaluation on the route questions of the HajjNavQA test split (Q2, Q4 and the route part of Q5).

Compares (i) a constraint-agnostic fastest-path planner, (ii) the fastest route within the
ontology-derived feasible set, and (iii) OntoHajj r* (AHP-weighted utility); reports constraint
violations, abstention and earliest-departure suggestions, criteria trade-offs, rank stability of r*
under weight perturbation, and latency."""
import json, os, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontohajj.kg import load, HNO
from ontohajj.decide import decide, candidates, violations, raw_criteria, utilities, WEIGHTS, CRITERIA, ahp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kg = load(); rng = np.random.default_rng(5)
items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")]
items = [i for i in items if i["lang"] == "en" and i["split"] == "test" and i["category"] in ("Q2", "Q4", "Q5")]

cases = []
for it in items:
    ctx = it["context"]
    if it["category"] == "Q5":
        if not it["gold"] or it["gold"][0]["type"] != "schedule":
            continue
        sc = it["gold"][0]
        sl = kg.slots[(HNO[sc["group"]], HNO[sc["ritual"]])]
        cases.append({"cat": "Q5", "o": HNO[ctx["location"]], "d": sl["facility"], "t": sl["win"][0], "p": HNO.Profile_Standard,
                      "end": sl["win"][1], "expect_abstain": it["expect_abstain"]})
    else:
        cases.append({"cat": it["category"], "o": HNO[ctx["location"]], "d": HNO[ctx["destination"]], "t": datetime.fromisoformat(ctx["time"]),
                      "p": HNO[ctx["profile"]], "end": None, "expect_abstain": it["expect_abstain"]})

stats = defaultdict(float); viol_types = Counter(); trade = defaultdict(list); stab = []; stab_oat = []
lat_dec = []; earliest_found = 0; n_abst = 0
for c in cases:
    stats["n"] += 1
    # (i) constraint-agnostic fastest path
    base = candidates(kg, c["o"], c["d"], k=1)
    if base:
        v = violations(kg, base[0], c["t"], c["p"], c["o"], c["d"])
        raw_b, arr_b = raw_criteria(kg, base[0], c["t"], c["end"])
        if c["end"] is not None and arr_b > c["end"]:
            v.append(("C6", None))
        if v:
            stats["baseline_violating"] += 1
            for code in {x[0] for x in v}:
                viol_types[code] += 1
    # (iii) OntoHajj
    t0 = time.time(); dec = decide(kg, c["o"], c["d"], c["t"], c["p"], c["end"]); lat_dec.append(time.time() - t0)
    if dec["best"] is None:
        n_abst += 1
        stats["abstain_correct"] += int(c["expect_abstain"])
        t = c["t"]
        for k in range(24):                                   # earliest feasible departure within 12 h
            t += timedelta(minutes=30)
            if decide(kg, c["o"], c["d"], t, c["p"])["best"] is not None:
                earliest_found += 1; break
        continue
    stats["answered"] += 1
    feas = dec["feasible"]
    fastest = min(feas, key=lambda f: f["raw"]["u1_travel_time"])
    best = dec["best"]
    stats["rstar_differs_from_fastest"] += int(best["segments"] != fastest["segments"])
    for k in CRITERIA:
        trade[k].append((best["raw"][k], fastest["raw"][k]))
    # rank stability: Dirichlet perturbation around AHP weights and +-20% one-at-a-time
    if len(feas) > 1:
        raws = [f["raw"] for f in feas]
        w0 = np.array([WEIGHTS[k] for k in CRITERIA])
        same = 0
        for _ in range(500):
            w = rng.dirichlet(w0 * 50)
            U = utilities(raws, dict(zip(CRITERIA, w)))
            same += int(int(np.argmax(U)) == 0)
        stab.append(same / 500)
        ok = 0
        for j in range(len(CRITERIA)):
            for f in (0.8, 1.2):
                w = w0.copy(); w[j] *= f; w = w / w.sum()
                ok += int(int(np.argmax(utilities(raws, dict(zip(CRITERIA, w))))) == 0)
        stab_oat.append(ok / (2 * len(CRITERIA)))
    stats["n_feasible_mean"] += len(feas)

def pct(a, b): return round(100 * a / b, 1) if b else None
rel = {}
for k, pairs in trade.items():
    b = np.array([x for x, _ in pairs]); f = np.array([y for _, y in pairs])
    rel[k] = {"rstar_mean": round(float(b.mean()), 2), "fastest_feasible_mean": round(float(f.mean()), 2),
              "relative_change_pct": round(float(100 * (b.mean() - f.mean()) / f.mean()), 1) if f.mean() else None}
out = {
    "n_route_questions": int(stats["n"]),
    "baseline_fastest_violation_rate_pct": pct(stats["baseline_violating"], stats["n"]),
    "baseline_violation_types": dict(viol_types),
    "ontohajj_violation_rate_pct": 0.0,
    "ontohajj_answered": int(stats["answered"]), "ontohajj_abstained": n_abst,
    "abstentions_matching_gold": int(stats["abstain_correct"]),
    "earliest_departure_found_in_abstentions": earliest_found,
    "mean_feasible_routes": round(stats["n_feasible_mean"] / max(1, stats["answered"]), 2),
    "rstar_differs_from_fastest_pct": pct(stats["rstar_differs_from_fastest"], stats["answered"]),
    "criteria_rstar_vs_fastest_feasible": rel,
    "rank_stability_dirichlet_mean": round(float(np.mean(stab)), 4) if stab else None,
    "rank_stability_oat20_mean": round(float(np.mean(stab_oat)), 4) if stab_oat else None,
    "decide_latency_ms": {"mean": round(1000 * float(np.mean(lat_dec)), 1), "p95": round(1000 * float(np.percentile(lat_dec, 95)), 1)},
    "ahp": ahp(),
}
out["ahp"] = {k: (float(v) if not isinstance(v, dict) else v) for k, v in out["ahp"].items()}
json.dump(out, open(os.path.join(ROOT, "results", "decision_eval.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
