"""Experiment B1: LLM-as-judge of the free-text answer against the typed claims.

The verifier (L5) checks typed claims only. This script asks an independent LLM whether the free text of each
answered question (i) contradicts the claims, or (ii) states navigation-relevant content (places, times, routes,
closures, rulings) that is supported neither by the claims nor by the question.

Sample: for each setting S2, S3, S4 of Qwen2.5-7B, a stratified random sample of answered (non-abstaining)
test questions (default 60 per setting, 10 per category, seed 2026). Results are appended to
results/judge/freetext_<judge>.jsonl, so the script can be stopped and restarted at any time.

Usage (GroqCloud, same environment variables as run_llm_eval.py):
  python eval/llm_judge_freetext.py --judge openai/gpt-oss-120b
  python eval/llm_judge_freetext.py --judge openai/gpt-oss-120b --summary      # print the table only
  python eval/llm_judge_freetext.py --judge openai/gpt-oss-120b --v2           # v2: claims spelled out with their KG content
"""
import argparse, json, os, random, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ontohajj.generate import LLM, QuotaExhausted, parse_json

SYSTEM = """You are a strict evaluator of a Hajj navigation assistant. You receive a pilgrim's question, its context,
the list of typed CLAIMS that the assistant's answer was checked against, and the assistant's free-text ANSWER.
Judge ONLY the free text against the claims and the question. Do not use outside knowledge about Hajj.
Return ONLY a JSON object:
{"contradiction": true|false,   // the text states something that conflicts with a claim (other place, level, time, date, route order)
 "unsupported": true|false,     // the text states a navigation-relevant fact (place, time, date, closure, distance, route step, rule or ruling)
                                //   that is neither in the claims nor in the question/context
 "consistent": true|false,      // true only if contradiction and unsupported are both false
 "evidence": "short quote of the problematic text, or empty"}
Paraphrases, the same fact in other words, translations, identifiers in brackets, polite closing phrases and
advice to contact the service provider are NOT problems. An arrival time equal to the departure time plus the
stated travel time is supported."""


KG = None


def claim_facts(claims):
    """v2: spell out what each verified claim entails in the knowledge graph (names in both languages, route legs
    with mode, length, minutes and step-free access, distance to the nearest facility, end of a dispatch slot),
    so that the judge does not count attributes of a claimed entity as unsupported content."""
    global KG
    from datetime import datetime, timedelta
    from ontohajj.kg import load, HNO
    from ontohajj.retrieve import nearest
    KG = KG or load()
    nm = lambda x: f"{KG.label(HNO[x], 'en')} / {KG.label(HNO[x], 'ar')}" if isinstance(x, str) and KG.exists(HNO[x]) else str(x)
    out = []
    for c in claims or []:
        t = c.get("type")
        if t == "location":
            out.append(f"{nm(c.get('facility'))} is located in {nm(c.get('site'))}.")
        elif t == "nearest":
            try:
                f, d = nearest(KG, HNO[c["from"]], HNO[c["facility_type"]])
                out.append(f"The nearest {c['facility_type']} to {nm(c['from'])} is {nm(c.get('facility'))}; walking distance {int(d)} m.")
            except Exception:
                out.append(json.dumps(c, ensure_ascii=False))
        elif t == "schedule":
            sl = KG.slots.get((HNO[c.get("group", "")], HNO[c.get("ritual", "")])) if c.get("group") and c.get("ritual") else None
            end = sl["win"][1].isoformat(timespec="minutes") if sl else "?"
            out.append(f"Dispatch slot of {nm(c.get('group'))} for {nm(c.get('ritual'))}: {nm(c.get('facility'))}, from {c.get('start')} to {end} (Makkah time; 10 Dhu al-Hijjah = 27 May 2026).")
        elif t == "ritual_time":
            out.append(f"{nm(c.get('ritual'))} window: from {c.get('start')} to {c.get('end')}.")
        elif t == "route":
            segs = [HNO[s] for s in c.get("segments") or [] if isinstance(s, str) and HNO[s] in KG.seg]
            legs, tot = [], 0
            for s in segs:
                d = KG.seg[s]; tot += d["min"]
                legs.append(f"{nm(str(d['from']).split('#')[-1])} -> {nm(str(d['to']).split('#')[-1])} by {KG.label(d['mode'], 'en')}, "
                            f"{int(d['len'])} m, {d['min']} min, {'step-free' if HNO.Profile_Wheelchair in d['access'] else 'has stairs'}")
            dep = c.get("depart")
            arr = (datetime.fromisoformat(dep) + timedelta(minutes=tot)).isoformat(timespec="minutes") if dep and segs else "?"
            out.append(f"Verified route from {nm(c.get('origin'))} to {nm(c.get('destination'))}, departing {dep}, arriving about {arr} "
                       f"({round(tot)} min). Legs in order: " + "; ".join(legs))
        else:
            out.append(json.dumps(c, ensure_ascii=False))
    return "\n".join(f"- {x}" for x in out) or "- (none)"


def user_msg(it, rec, v2=False):
    m = (f"QUESTION: {it['question']}\nCONTEXT: {json.dumps(it['context'], ensure_ascii=False)}\n"
         f"CLAIMS: {json.dumps(rec.get('claims') or [], ensure_ascii=False)}\n")
    if v2:
        m += "WHAT THE CLAIMS STATE (from the verified knowledge graph; content stated here counts as supported):\n" + claim_facts(rec.get("claims")) + "\n"
    return m + f"ANSWER: {rec.get('answer') or ''}"


def sample(model_dir, cfg, n_per_cat, seed):
    path = os.path.join(ROOT, "results", "llm", model_dir, "test_v2", f"{cfg}_run0.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if not r["abstain"] and (r.get("answer") or "").strip()]
    rng = random.Random(seed); out = []
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]; rng.shuffle(rs); out += rs[:n_per_cat]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="openai/gpt-oss-120b")
    ap.add_argument("--model_dir", default="qwen2.5_7b")
    ap.add_argument("--configs", nargs="+", default=["C4", "C3", "C2"])
    ap.add_argument("--per_cat", type=int, default=10); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--v2", action="store_true", help="give the judge the knowledge-graph content of every claim (recommended)")
    a = ap.parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")}
    os.makedirs(os.path.join(ROOT, "results", "judge"), exist_ok=True)
    out = os.path.join(ROOT, "results", "judge", f"freetext{'_v2' if a.v2 else ''}_{a.judge.replace('/', '_').replace(':', '_')}.jsonl")
    done = {(json.loads(l)["model"], json.loads(l)["config"], json.loads(l)["id"]) for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    if not a.summary:
        llm = LLM(model=a.judge)
        with open(out, "a", encoding="utf-8") as f:
            for cfg in a.configs:
                for rec in sample(a.model_dir, cfg, a.per_cat, a.seed):
                    key = (a.model_dir, cfg, rec["id"])
                    if key in done:
                        continue
                    try:
                        text, tok, lat = llm(SYSTEM, user_msg(items[rec["id"]], rec, a.v2))
                    except QuotaExhausted as e:
                        print(f"\nSTOPPED: {e}\nRun the same command again later; it continues where it stopped.", flush=True); sys.exit(2)
                    except Exception as e:
                        print(f"   ERROR on {rec['id']}: {str(e)[:150]} (will be retried next run)", flush=True); continue
                    j = parse_json(text)
                    row = {"model": a.model_dir, "config": cfg, "id": rec["id"], "category": rec["category"], "lang": rec["lang"],
                           "scored_correct": rec["correct"], "contradiction": bool(j.get("contradiction")),
                           "unsupported": bool(j.get("unsupported")), "consistent": bool(j.get("consistent")) and not j.get("parse_error"),
                           "evidence": j.get("evidence", ""), "parse_error": bool(j.get("parse_error")), "judge": a.judge}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    print(f"{cfg} {rec['id']} consistent={row['consistent']} contradiction={row['contradiction']} unsupported={row['unsupported']}", flush=True)
    # summary
    st = defaultdict(lambda: defaultdict(int))
    for l in open(out, encoding="utf-8"):
        r = json.loads(l); s = st[(r["model"], r["config"])]
        s["n"] += 1; s["consistent"] += r["consistent"]; s["contradiction"] += r["contradiction"]; s["unsupported"] += r["unsupported"]
        s["parse_error"] += r["parse_error"]
    NAME = {"C2": "S2", "C3": "S3", "C4": "S4"}
    print("\nModel        Setting  n   consistent  contradiction  unsupported  parse errors")
    summ = {}
    for (m, c), s in sorted(st.items()):
        n = max(1, s["n"])
        print(f"{m:12s} {NAME.get(c, c):7s} {s['n']:3d}  {100*s['consistent']/n:9.1f}%  {100*s['contradiction']/n:12.1f}%  {100*s['unsupported']/n:10.1f}%  {s['parse_error']}")
        summ[f"{m}/{c}"] = dict(s)
    json.dump(summ, open(out.replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
