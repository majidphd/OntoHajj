"""L2 on the human-written questions (HajjNavQA-H): intent accuracy, unresolved-entity detection, refusals.
Writes results/human_l2.json."""
import json, os, sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ontohajj.understand import Linker, intent
from ontohajj.kg import HNO
L = Linker()
GI = {"Q1": "Location", "Q2": "Route", "Q3": "Temporal", "Q4": "Accessibility", "Q5": "MultiHop"}
items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa_h.jsonl"), encoding="utf-8")]
ok = Counter(); n = Counter(); confusion = Counter(); ne = [0, 0]; false_unres = 0
for it in items:
    found, unres = L.link(it["question"]); z = intent(it["question"])
    if it["category"] in GI:
        n[it["category"]] += 1; ok[it["category"]] += z == GI[it["category"]]
        if z != GI[it["category"]]: confusion[(GI[it["category"]], z)] += 1
        false_unres += bool(unres)
    if it["meta"].get("kind") == "non-existent":
        ne[1] += 1; ne[0] += bool(unres)
out = {"intent_accuracy": round(sum(ok.values()) / sum(n.values()), 4), "intent_n": sum(n.values()),
       "intent_by_category": {k: f"{ok[k]}/{n[k]}" for k in n}, "confusions": {f"{a}->{b}": v for (a, b), v in confusion.items()},
       "nonexistent_detected": f"{ne[0]}/{ne[1]}", "false_unresolved_on_answerable": false_unres}
json.dump(out, open(os.path.join(ROOT, "results", "human_l2.json"), "w"), indent=1); print(json.dumps(out, indent=1))
