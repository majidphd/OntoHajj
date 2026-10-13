"""Lexical check of the free-text answer against the verified typed claims (reviewer concern: the verifier
checks typed claims only). For every answered (non-abstaining) question we flag
  - fabricated mention : a code-like mention in the text that resolves to no KG entity (e.g. 'Mina Station 5');
  - ungrounded entity  : a KG entity named in the text that is neither in the question, nor in a claim,
                         nor a stop of a claimed route, nor the holy site of such an entity;
  - ungrounded time    : a clock time (HH:MM) in the text that appears neither in the question context nor in a claim.
This is a conservative lexical proxy; it does not judge paraphrase or juristic content.
Usage: python eval/freetext_consistency.py   (writes results/freetext_consistency.json)"""
import json, os, re, sys, glob
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ontohajj.kg import load, HNO
from ontohajj.understand import Linker, normalize

kg = load(); L = Linker(kg)
items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")}
PLACES = set(kg.site_of) | set(kg.net.nodes) | set(getattr(kg, "group_camps", {}))
SITES = {HNO.Mina, HNO.Arafat, HNO.Muzdalifah, HNO.MasjidAlHaram}
DIST = re.compile(r"\(?\s*(?:about\s*|≈\s*|حوالي\s*)?\d+(?:[.,]\d+)?\s*(?:km|m|meters|metres|minutes|min|كم|م|متر|مترًا|دقيقة|دقائق)(?![\w])\s*\)?")
TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:٫.]([0-5]\d)(?!\d)")
ENT_KEYS = ("facility", "site", "from", "group", "origin", "destination")


def hm(s):
    m = TIME.search(s or ""); return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def grounded(rec, it):
    ents, times = set(SITES), set()
    q_ents, _ = L.link(it["question"]); ents |= set(q_ents)
    for k, v in it["context"].items():
        if isinstance(v, str) and k != "time" and v and not v.startswith("Profile"):
            ents.add(HNO[v])
    if it["context"].get("time"): times.add(hm(it["context"]["time"].replace("T", " ")))
    for c in rec.get("claims") or []:
        for k in ENT_KEYS:
            if isinstance(c.get(k), str) and c[k]:
                u = HNO[c[k]] if c[k] in {str(x).split("#")[-1] for x in kg.named} else None
                if u is not None: ents.add(u)
        for k in ("start", "end", "depart"):
            if isinstance(c.get(k), str): times.add(hm(c[k].replace("T", " ")))
        if c.get("type") == "schedule" and c.get("group") and c.get("ritual"):   # end of the claimed dispatch slot
            sl = kg.slots.get((HNO[c["group"]], HNO[c["ritual"]]))
            if sl: times.add(sl["win"][1].strftime("%H:%M"))
        segs = [HNO[s] for s in c.get("segments") or [] if isinstance(s, str) and HNO[s] in kg.seg]
        for s in segs:
            ents |= {kg.seg[s]["from"], kg.seg[s]["to"]}
        if segs and isinstance(c.get("depart"), str):      # arrival time derived from the claimed route
            from datetime import datetime, timedelta
            t0 = datetime.fromisoformat(c["depart"]); tot = sum(kg.seg[s]["min"] for s in segs)
            for dm in range(-2, 3):
                times.add((t0 + timedelta(minutes=tot + dm)).strftime("%H:%M"))
    ents |= {kg.site_of[e] for e in list(ents) if e in kg.site_of}
    return ents, times


def check(rec):
    it = items[rec["id"]]
    text = rec.get("answer") or ""
    ents, times = grounded(rec, it)
    plain = DIST.sub(" ", text)                         # distances such as "(700 m)" are not entity codes
    found, unres = L.link(plain)
    found = [u for u in found if u in PLACES]          # places and groups only (not modes, profiles, rituals)
    t_text = {f"{int(a):02d}:{b}" for a, b in TIME.findall(normalize(text).replace(" ", " ") and text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))}
    return {"fabricated": bool(unres), "ungrounded_entity": [str(u).split("#")[-1] for u in found if u not in ents],
            "ungrounded_time": sorted(t for t in t_text if t not in times)}


out = {}
for folder in sorted(glob.glob(os.path.join(ROOT, "results", "llm", "*", "test_v2"))):
    model = os.path.basename(os.path.dirname(folder))
    for p in sorted(glob.glob(os.path.join(folder, "C[0-4]_run0.jsonl"))):
        cfg = os.path.basename(p).split("_")[0]
        st = defaultdict(int); ex = []
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["abstain"]:
                continue
            res = check(r); st["answered"] += 1
            bad = res["fabricated"] or res["ungrounded_entity"] or res["ungrounded_time"]
            st["fabricated"] += res["fabricated"]; st["ungrounded_entity"] += bool(res["ungrounded_entity"])
            st["ungrounded_time"] += bool(res["ungrounded_time"]); st["any"] += bool(bad)
            st["any_but_claims_ok"] += bool(bad) and r["correct"]
            if bad and len(ex) < 8: ex.append({"id": r["id"], **res, "answer": r["answer"][:240]})
        n = max(1, st["answered"])
        out[f"{model}/{cfg}"] = {**st, "rate_any": round(st["any"] / n, 4), "examples": ex}
        print(f"{model:24s} {cfg}: answered {st['answered']:3d}  inconsistent {st['any']:3d} ({100*st['any']/n:5.1f}%)  "
              f"fabricated {st['fabricated']}  entity {st['ungrounded_entity']}  time {st['ungrounded_time']}  (of which scored correct: {st['any_but_claims_ok']})")
json.dump(out, open(os.path.join(ROOT, "results", "freetext_consistency.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
