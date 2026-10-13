"""Re-runs specific questions of one configuration (e.g. after an infrastructure failure such as a
time-out) and replaces their records in the result file.
Example: python eval/rerun_ids.py --model qwen2.5:7b --split test --config C4 --ids HNQ-0242-AR"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from ontohajj.pipeline import OntoHajj
from ontohajj.generate import LLM
from run_llm_eval import score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--split", default="test"); ap.add_argument("--version", default="v2")
ap.add_argument("--config", required=True); ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--run", type=int, default=0)
a = ap.parse_args()
path = os.path.join(ROOT, "results", "llm", a.model.replace("/", "_").replace(":", "_"), f"{a.split}_{a.version}", f"{a.config}_run{a.run}.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")}
sys_ = OntoHajj(LLM(model=a.model))
new = {}
for qid in a.ids:
    it = items[qid]
    rec = sys_.answer(it["question"], it["context"], a.config, it["lang"])
    s = score(it, rec, sys_)
    new[qid] = {"id": qid, "category": it["category"], "lang": it["lang"], "config": a.config, "run": a.run,
                "expect_abstain": it["expect_abstain"], **s, "latency": rec.get("latency", {}).get("total"),
                "tokens": rec.get("tokens", 0), "repairs": rec.get("repairs", 0), "answer": rec.get("answer"),
                "claims": rec.get("claims"), "attempts": rec.get("attempts"), "reason": rec.get("reason"), "rerun": True}
    print(qid, "correct =", s["correct"], "| abstain =", s["abstain"], "| seconds =", round(new[qid]["latency"] or 0))
with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(new.get(r["id"], r), ensure_ascii=False) + "\n")
print("updated", path)
