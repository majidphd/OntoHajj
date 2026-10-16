"""End-to-end evaluation of configurations C0-C4 on the HajjNavQA test split.

Usage (example with a local model served by Ollama):
  export ONTOHAJJ_PROVIDER=openai ONTOHAJJ_BASE_URL=http://localhost:11434/v1 ONTOHAJJ_API_KEY=ollama
  python eval/run_llm_eval.py --model qwen2.5:7b-instruct --configs C0 C1 C2 C3 C4 --runs 3
Results are written to results/llm/<model>/ and summarised by eval/summarise_llm.py (Tables 8-9).

Automatic scoring (Section 4.3):
  correct  - category-specific comparison with the gold claims (route answers must be feasible);
  HR       - share of claims not supported by the KG (symbolic oracle, C7 excluded);
  OHR      - share of mentioned entities that do not exist in HNO;
  CVR      - share of answered questions with >=1 safety-constraint violation (C2-C6, C8, C9);
  Cov, SelAcc, AbsF1 - abstention behaviour; latency and tokens per question."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontohajj.kg import load, HNO, local
from ontohajj.pipeline import OntoHajj
from ontohajj.generate import LLM, QuotaExhausted
from ontohajj.verify import SAFETY
from ontohajj.retrieve import resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT_KEYS = ("facility", "site", "from", "group", "ritual", "origin", "destination")


def score(item, rec, sys_):
    kg, V, LK = sys_.kg, sys_.verifier, sys_.linker
    claims = rec.get("claims") or []
    abst = rec.get("abstain", False)
    res = V.verify(claims, HNO[item["context"].get("profile", "Profile_Standard")]) if claims else {"violations": {}}
    unsupported = sum(1 for i in range(len(claims)) if [c for c in res["violations"].get(i, []) if c != "C7"])
    ents = [c.get(k) for c in claims for k in ENT_KEYS if c.get(k)]
    fake = sum(1 for e in ents if not kg.exists(resolve(kg, LK, e)))
    unsafe = any(set(v) & SAFETY for v in res["violations"].values())
    correct = False
    if item["expect_abstain"]:
        correct = abst
    elif not abst and claims and not unsafe:
        gold = item["gold"]
        def match(g):
            for c in claims:
                if c.get("type") != g["type"]:
                    continue
                R = lambda x: resolve(kg, LK, x)
                if g["type"] == "location" and R(c.get("facility")) == HNO[g["facility"]] and R(c.get("site")) == HNO[g["site"]]:
                    return True
                if g["type"] == "nearest" and R(c.get("facility")) == HNO[g["facility"]]:
                    return True
                if g["type"] == "schedule" and str(c.get("start", ""))[:16] == g["start"][:16] and R(c.get("facility")) == HNO[g["facility"]]:
                    return True
                if g["type"] == "ritual_time" and str(c.get("start", ""))[:16] == g["start"][:16] and str(c.get("end", ""))[:16] == g["end"][:16]:
                    return True
                if g["type"] == "route" and R(c.get("destination")) == HNO[g["destination"]] and R(c.get("origin")) == HNO[g["origin"]]:
                    return True        # feasibility already enforced by 'not unsafe'
            return False
        correct = all(match(g) for g in gold)
    return {"correct": correct, "abstain": abst, "n_claims": len(claims), "unsupported": unsupported,
            "n_entities": len(ents), "fake_entities": fake, "unsafe": unsafe and not abst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--provider", default=None)
    ap.add_argument("--configs", nargs="+", default=["C0", "C1", "C2", "C3", "C4"],
                    help="settings S0-S4 (or C0-C4); ablations S4-NoRepair (C4k0), S4-Hint (C4hint); S4-LLM-L2 (C4llm)")
    ap.add_argument("--runs", type=int, default=1); ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default="hajjnavqa.jsonl", help="benchmark file in data/ (hajjnavqa_h.jsonl = human-written questions, split 'human')")
    ap.add_argument("--version", default="v2", help="tag of the system version (results go to <model>/<split>_<version>)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(ROOT, "data", a.data), encoding="utf-8")]
    items = [i for i in items if i["split"] == a.split]
    if a.limit:                       # spread the trial sample over all categories and both languages
        step = max(1, len(items) // a.limit)
        items = items[::step][: a.limit]
    sys_ = OntoHajj(LLM(model=a.model, provider=a.provider))
    outdir = os.path.join(ROOT, "results", "llm", a.model.replace("/", "_").replace(":", "_"), f"{a.split}_{a.version}")
    os.makedirs(outdir, exist_ok=True)
    print("results folder:", outdir, flush=True)
    for cfg in a.configs:
        for run in range(a.runs):
            path = os.path.join(outdir, f"{cfg}_run{run}.jsonl")
            done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
            with open(path, "a", encoding="utf-8") as f:
                for it in items:
                    if it["id"] in done:
                        continue
                    try:
                        rec = sys_.answer(it["question"], it["context"], cfg, it["lang"])
                    except QuotaExhausted as e:
                        print(f"\nSTOPPED: {e}\nNothing was recorded for {it['id']}. Run the same command again later "
                              f"(for a daily quota: tomorrow); it continues from this question.", flush=True)
                        sys.exit(2)
                    except Exception as e:
                        print(f"   ERROR on {it['id']} ({type(e).__name__}: {str(e)[:200]}); not recorded, will be retried on the next run", flush=True)
                        continue
                    s = score(it, rec, sys_)
                    f.write(json.dumps({"id": it["id"], "category": it["category"], "lang": it["lang"], "config": cfg, "run": run,
                                        "expect_abstain": it["expect_abstain"], **s, "latency": rec.get("latency", {}).get("total"),
                                        "tokens": rec.get("tokens", 0), "repairs": rec.get("repairs", 0),
                                        "answer": rec.get("answer"), "claims": rec.get("claims"),
                                        "attempts": rec.get("attempts"), "reason": rec.get("reason"), "l2_llm": rec.get("l2_llm"), "intent": rec.get("intent")}, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{cfg} run{run} {it['id']} correct={s['correct']} abstain={s['abstain']}", flush=True)
            print("finished", cfg, "run", run, "->", path, flush=True)


if __name__ == "__main__":
    main()
