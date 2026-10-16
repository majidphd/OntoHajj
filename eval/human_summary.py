"""Summary of the human-written question set (HajjNavQA-H): results/human_summary.json"""
import json, os, sys
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "eval"))
from summarise_llm import metrics, bootstrap
P = {"S2": "qwen2.5_7b/human_v2/C2_run0.jsonl", "S4": "qwen2.5_7b/human_v2/C4_run0.jsonl", "S4-Template": "template/human_v2/C4_run0.jsonl"}
D = {k: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "results", "llm", p), encoding="utf-8")} for k, p in P.items()}
ids = sorted(D["S4"])
def mc(a, b):
    n01 = sum(x and not y for x, y in zip(a, b)); n10 = sum(y and not x for x, y in zip(a, b)); n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
out = {}
for k, d in D.items():
    rows = [d[i] for i in ids]; m = metrics(rows); m["Acc_CI95"] = bootstrap(rows, "Acc")
    m["cat"] = {c: f"{sum(r['correct'] for r in rows if r['category'] == c)}/{sum(1 for r in rows if r['category'] == c)}" for c in sorted({r['category'] for r in rows})}
    m["p_vs_S4"] = mc([D["S4"][i]["correct"] for i in ids], [d[i]["correct"] for i in ids])
    out[k] = m
out["L2"] = json.load(open(os.path.join(ROOT, "results", "human_l2.json")))
json.dump(out, open(os.path.join(ROOT, "results", "human_summary.json"), "w"), indent=1)
for k in P: print(k, round(out[k]["Acc"], 3), out[k]["Acc_CI95"], round(out[k]["CVR"], 3), out[k]["cat"], round(out[k]["p_vs_S4"], 4))
