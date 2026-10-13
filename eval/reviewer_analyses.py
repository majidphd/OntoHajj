"""Additional analyses requested in internal review (results/reviewer_analyses.json):
leakage sensitivity (pilot questions), L2 refusal recall, Q3 S2-vs-S3 test, crowd-exposure trade-off of r*."""
import json, os, sys
from math import comb
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))
from summarise_llm import metrics
items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")]
byid = {i["id"]: i for i in items}
test = [i for i in items if i["split"] == "test"]
D = os.path.join(ROOT, "results", "llm", "qwen2.5_7b", "test_v2")
R = {c: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, f"{c}_run0.jsonl"), encoding="utf-8")} for c in ["C0", "C1", "C2", "C3", "C4"]}


def mcnemar(a, b):
    n01 = sum(x and not y for x, y in zip(a, b)); n10 = sum(y and not x for x, y in zip(a, b)); n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)


out = {}
# 1. the pilot of version 1 used the first 20 questions of the --limit 20 test sample; exclude them and their pairs
step = max(1, len(test) // 20); pilot = test[::step][:20]
pairs = {i["pair_id"] for i in pilot}; excl = {i["id"] for i in test if i["pair_id"] in pairs}
out["leakage"] = {"pilot_ids": [i["id"] for i in pilot], "excluded_with_pairs": len(excl), "n_remaining": len(test) - len(excl),
                  "settings": {c: {k: round(v, 4) for k, v in metrics([r for i, r in R[c].items() if i not in excl]).items()
                                   if k in ("Acc", "HR", "OHR", "CVR", "Cov", "SelAcc", "AbsF1")} for c in R}}
# 2. L2 refusal recall on unanswerable questions (S4 log)
k = Counter(); 
for i, r in R["C4"].items():
    it = byid[i]
    if it["category"] == "Q6":
        kind = it["meta"].get("kind"); k[(kind, "n")] += 1; k[(kind, "L2")] += (r["reason"] or "").startswith("L2"); k[(kind, "abstain")] += r["abstain"]
out["l2_refusal"] = {kind: {"n": k[(kind, "n")], "refused_by_L2": k[(kind, "L2")], "abstained_overall": k[(kind, "abstain")]}
                     for kind in ("non-existent", "juristic", "out-of-scope")}
# 3. Q3 S2 vs S3
ids = [i for i in R["C2"] if byid[i]["category"] == "Q3"]
out["q3"] = {"n": len(ids), **{c: sum(R[c][i]["correct"] for i in ids) for c in ("C2", "C3", "C4")},
             "p_S2_vs_S3": mcnemar([R["C2"][i]["correct"] for i in ids], [R["C3"][i]["correct"] for i in ids])}
# 4. crowd-exposure trade-off of r* against the fastest feasible route (same cases as decision_eval.py)
src = open(os.path.join(ROOT, "eval", "decision_eval.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(ROOT, "eval", "decision_eval.py")}
exec(src[:src.index("stats = defaultdict(float)")], ns)
from ontohajj.decide import decide
lower = higher = same = 0; d_crowd = []; d_safe = []
for c in ns["cases"]:
    dec = decide(ns["kg"], c["o"], c["d"], c["t"], c["p"], c["end"])
    if dec["best"] is None: continue
    f = min(dec["feasible"], key=lambda x: x["raw"]["u1_travel_time"]); b = dec["best"]
    if b["segments"] == f["segments"]: continue
    db = b["raw"]["u2_crowd_exposure"] - f["raw"]["u2_crowd_exposure"]
    lower += db < -1e-9; higher += db > 1e-9; same += abs(db) <= 1e-9
    if db > 1e-9: d_crowd.append(db); d_safe.append(b["raw"]["u4_accessibility_safety"] - f["raw"]["u4_accessibility_safety"])
out["crowd_tradeoff"] = {"rstar_differs": lower + higher + same, "less_crowded": lower, "more_crowded": higher, "equal": same,
                         "more_crowded_mean_increase": round(sum(d_crowd) / max(1, len(d_crowd)), 3),
                         "more_crowded_mean_safety_gain": round(sum(d_safe) / max(1, len(d_safe)), 3)}
json.dump(out, open(os.path.join(ROOT, "results", "reviewer_analyses.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "leakage"}, indent=1)); print(json.dumps(out["leakage"]["settings"]))
