"""Summary of the human-written question set (HajjNavQA-H): results/human_summary.json"""
import json, os, sys
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "eval"))
from summarise_llm import metrics, bootstrap
P = {"S2": "qwen2.5_7b/human_v2/C2_run0.jsonl", "S4": "qwen2.5_7b/human_v2/C4_run0.jsonl", "S4-Template": "template/human_v2/C4_run0.jsonl",
     "S4-LLM-L2": "qwen2.5_7b/human_v2/C4llm_run0.jsonl"}
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
# LLM-based L2: intent accuracy on the human questions and on the template test subset (C4llm vs rule-based C4)
GI = {"Q1": "Location", "Q2": "Route", "Q3": "Temporal", "Q4": "Accessibility", "Q5": "MultiHop"}
def llm_intent(rows):
    a = [r for r in rows if r["category"] in GI]
    return sum((r.get("l2_llm") or {}).get("intent") == GI[r["category"]] for r in a), len(a)
out["llm_intent_human"] = llm_intent(list(D["S4-LLM-L2"].values()))
T = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "results", "llm", "qwen2.5_7b", "test_llml2", "C4llm_run0.jsonl"), encoding="utf-8")}
Q = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "results", "llm", "qwen2.5_7b", "test_v2", "C4_run0.jsonl"), encoding="utf-8")}
tid = sorted(T)
out["template_subset"] = {"n": len(tid), "llm_l2_correct": sum(T[i]["correct"] for i in tid), "rule_l2_correct": sum(Q[i]["correct"] for i in tid),
                          "p": mc([T[i]["correct"] for i in tid], [Q[i]["correct"] for i in tid]), "llm_intent": llm_intent(list(T.values())),
                          "llm_unsafe": sum(T[i]["unsafe"] for i in tid)}
json.dump(out, open(os.path.join(ROOT, "results", "human_summary.json"), "w"), indent=1)
print(out['llm_intent_human'], out['template_subset'])
for k in P: print(k, round(out[k]["Acc"], 3), out[k]["Acc_CI95"], round(out[k]["CVR"], 3), out[k]["cat"], round(out[k]["p_vs_S4"], 4))
