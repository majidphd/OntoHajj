"""Experiment B2: a strong LLM-as-judge baseline for the hallucination-injection study (E3).

Uses exactly the perturbed and gold claims of eval/verifier_eval.py (same seed). For a stratified sample of test
cases (default 40 per kind: gold, entity, relational, temporal, constraint), an LLM receives the claim as English
sentences together with reference evidence from the pilgrim-guide corpus (the verbalised knowledge graph, which
includes segment closures and step-free access) and decides whether the claim is SUPPORTED, CONTRADICTED or
NOT_ENOUGH_INFO (a RefChecker-style check). A claim is flagged when the verdict is not SUPPORTED.
For route claims the evidence contains the passages of every claimed segment (oracle evidence), so the judge is
not limited by retrieval. Results are appended to results/judge/e3_<judge>.jsonl (restartable).

Usage:  python eval/llm_judge_e3.py --judge openai/gpt-oss-120b         (add --summary to print the table only)
"""
import argparse, json, os, random, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)

SYSTEM = """You check statements made by a Hajj navigation assistant against REFERENCE passages from the official
pilgrim guide of this season. Use ONLY the reference passages, not outside knowledge.
Answer SUPPORTED if every part of the statement is backed by the passages (including times, places, levels,
order of the route legs, closures at the stated time, and step-free access for wheelchair users);
CONTRADICTED if any part conflicts with the passages; NOT_ENOUGH_INFO otherwise.
Return ONLY a JSON object: {"verdict": "SUPPORTED"|"CONTRADICTED"|"NOT_ENOUGH_INFO", "reason": "short"}"""


def load_cases():
    """re-create the cases of verifier_eval.py without running its evaluation loop"""
    src = open(os.path.join(ROOT, "eval", "verifier_eval.py"), encoding="utf-8").read()
    ns = {"__file__": os.path.join(ROOT, "eval", "verifier_eval.py"), "__name__": "e3cases"}
    exec(src[:src.index("# threshold for the similarity detector")], ns)
    return ns


def evidence(ns, c):
    from ontohajj.generate import TextRAG
    kg, HNO = ns["kg"], ns["HNO"]
    if "rag" not in ns:
        ns["rag"] = TextRAG(ns["corpus"])
    sents = ns["verbalise"](c); ev = []
    if c["type"] == "route":
        for s in c["segments"]:
            d = kg.seg.get(HNO[s])
            if d:
                key = f"From {kg.label(d['from'])} to {kg.label(d['to'])} by"
                ev += [p for p in ns["corpus"] if p.startswith(key)][:1]
    for s in sents:
        ev += ns["rag"].search(s, k=4)
    seen, out = set(), []
    for p in ev:
        if p not in seen:
            seen.add(p); out.append(p)
    return out[:16]


def statement(ns, c):
    s = " ".join(ns["verbalise"](c))
    if c["type"] == "route":
        prof = {"Profile_Wheelchair": "a wheelchair user", "Profile_Elderly": "an elderly pilgrim"}.get(c.get("profile"), "a pilgrim")
        s = f"Recommended route for {prof}, departing at {ns['fmt'](ns['T'](c['depart']), 'en')}: " + s
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="openai/gpt-oss-120b"); ap.add_argument("--per_kind", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results", "judge"), exist_ok=True)
    out = os.path.join(ROOT, "results", "judge", f"e3_{a.judge.replace('/', '_').replace(':', '_')}.jsonl")
    if not a.summary:
        from ontohajj.generate import LLM, QuotaExhausted, parse_json
        ns = load_cases()
        test = [(i, k, c, code) for i, (sp, k, c, code) in enumerate(ns["cases"]) if sp == "test"]
        rng = random.Random(a.seed); sample = []
        for kind in ("gold", "entity", "relational", "temporal", "constraint"):
            ks = [x for x in test if x[1] == kind]; rng.shuffle(ks); sample += ks[:a.per_kind]
        done = {json.loads(l)["case"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
        llm = LLM(model=a.judge)
        with open(out, "a", encoding="utf-8") as f:
            for idx, kind, c, code in sample:
                if idx in done:
                    continue
                ev = evidence(ns, c)
                user = "STATEMENT: " + statement(ns, c) + "\n\nREFERENCE PASSAGES:\n" + "\n".join(f"- {p}" for p in ev)
                try:
                    text, tok, lat = llm(SYSTEM, user)
                except QuotaExhausted as e:
                    print(f"\nSTOPPED: {e}\nRun the same command again later; it continues where it stopped.", flush=True); sys.exit(2)
                except Exception as e:
                    print(f"   ERROR on case {idx}: {str(e)[:150]} (will be retried next run)", flush=True); continue
                j = parse_json(text); v = str(j.get("verdict", "")).upper()
                row = {"case": idx, "kind": kind, "claim_type": c["type"], "expected": code, "verdict": v,
                       "flagged": v != "SUPPORTED", "reason": j.get("reason", ""), "judge": a.judge}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                print(f"case {idx} {kind:10s} {c['type']:11s} -> {v}", flush=True)
    st = defaultdict(lambda: [0, 0])
    for l in open(out, encoding="utf-8"):
        r = json.loads(l); st[r["kind"]][0] += r["flagged"]; st[r["kind"]][1] += 1
    print("\nKind         flagged / n   rate   (gold row = false-positive rate)")
    for k in ("gold", "entity", "relational", "temporal", "constraint"):
        if st[k][1]:
            print(f"{k:11s} {st[k][0]:4d} / {st[k][1]:<4d} {100*st[k][0]/st[k][1]:6.1f}%")
    json.dump({k: {"flagged": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in st.items() if v[1]},
              open(out.replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
