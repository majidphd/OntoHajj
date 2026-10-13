"""Layer-5 evaluation by controlled hallucination injection.

Gold claims of answerable HajjNavQA items are perturbed with four hallucination types
(entity, relational, temporal, constraint). Three detectors are compared:
  (a) entity-only guardrail (checks that every mentioned entity exists: C1 only);
  (b) similarity-based grounding check (max fuzzy similarity of the verbalised claim to the
      text corpus; threshold set on dev gold claims for 5% false positives);
  (c) the OntoHajj neuro-symbolic verifier (SHACL C1-C8 + procedural C9).
"""
import copy, json, os, random, sys, time
from collections import defaultdict
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rapidfuzz import fuzz, process
from ontohajj.kg import load, HNO, local
from ontohajj.understand import Linker
from ontohajj.verify import Verifier
from ontohajj.retrieve import resolve, fmt, nearest
from ontohajj.generate import build_corpus
from ontohajj.decide import candidates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(11)
kg = load(); LK = Linker(kg); V = Verifier(kg, LK)
items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")]
items = [i for i in items if i["lang"] == "en" and i["answerable"] and not i["expect_abstain"] and i["gold"]]
corpus = build_corpus(kg, "en")
SRC = "Src_HNOPrototype"


def T(s): return datetime.fromisoformat(s)
def iso(t): return t.isoformat(timespec="minutes")


def lab(x):
    u = HNO[x] if x and kg.exists(HNO[x]) else None
    return kg.label(u, "en") if u is not None else str(x)


def verbalise(c):
    t = c["type"]
    if t == "location":
        return [f"{lab(c['facility'])} is located in {lab(c['site'])}."]
    if t == "nearest":
        tn = "medical centre" if "Medical" in c["facility_type"] else "metro station"
        return [f"The nearest {tn} to {lab(c['from'])} is {lab(c['facility'])}."]
    if t == "schedule":
        return [f"Dispatch slot of {lab(c['group'])} for {lab(c['ritual'])}: {lab(c['facility'])}, from {fmt(T(c['start']), 'en')}."]
    if t == "ritual_time":
        return [f"{lab(c['ritual'])} takes place between {fmt(T(c['start']), 'en')} and {fmt(T(c['end']), 'en')}."]
    if t == "route":
        out = []
        for s in c["segments"]:
            d = kg.seg.get(HNO[s])
            out.append(f"From {kg.label(d['from'])} to {kg.label(d['to'])} by {kg.label(d['mode'])}." if d else f"Use segment {s}.")
        return out


def sim_score(c):
    return min(process.extractOne(v, corpus, scorer=fuzz.token_set_ratio)[1] for v in verbalise(c))


def entity_only(c):
    keys = ("facility", "site", "from", "group", "ritual", "origin", "destination")
    for k in keys:
        if c.get(k) and not kg.exists(resolve(kg, LK, c[k])):
            return True
    for s in c.get("segments", []) or []:
        if HNO[s] not in kg.seg:
            return True
    return False


# ------------------------------------------------------------ perturbations
FAKE = {"facility": ["Mina Station 6", "Mina camp M77", "Mina Medical Centre F", "Arafat Station 5", "Jamarat level 7", "Gate 45 of Masjid al-Haram"],
        "ritual": ["Stoning of Jamarat al-Wusta (14 Dhu al-Hijjah)", "Night stay in Arafat"]}
SITES = ["Mina", "Muzdalifah", "Arafat", "MasjidAlHaram"]


def perturb(c, kind):
    c = copy.deepcopy(c); t = c["type"]
    if kind == "entity":
        if t in ("location", "nearest", "schedule"):
            c["facility"] = random.choice(FAKE["facility"]); return c, "C1"
        if t == "ritual_time":
            c["ritual"] = random.choice(FAKE["ritual"]); return c, "C1"
        if t == "route":
            i = random.randrange(len(c["segments"])); c["segments"][i] = f"Seg_{random.randint(500, 999)}"; return c, "C2"
    if kind == "relational":
        if t == "location":
            c["site"] = random.choice([s for s in SITES if s != c["site"]]); return c, "C8"
        if t == "nearest":
            cls = HNO[c["facility_type"]]
            others = [local(f) for f in kg.instances(cls) if local(f) != c["facility"]]
            c["facility"] = random.choice(others); return c, "C9"
        if t == "route" and len(c["segments"]) >= 3:
            del c["segments"][random.randrange(1, len(c["segments"]) - 1)]; return c, "C2"
    if kind == "temporal":
        if t == "schedule":
            c["start"] = iso(T(c["start"]) + timedelta(hours=random.choice([-2, 2, 4]))); return c, "C6"
        if t == "ritual_time":
            key = random.choice(["start", "end"]); c[key] = iso(T(c[key]) + timedelta(minutes=random.choice([-90, 60, 120]))); return c, "C4"
        if t == "route":
            base = T(c["depart"]).replace(hour=0, minute=0)
            segs = [HNO[s] for s in c["segments"]]
            slots = [base + timedelta(minutes=30 * k) for k in range(48)]
            random.shuffle(slots)
            for t0 in slots:
                tt, bad = t0, False
                for s in segs:
                    if kg.is_restricted(s, tt): bad = True; break
                    tt += timedelta(minutes=kg.seg[s]["min"])
                if bad:
                    c["depart"] = iso(t0); return c, "C3"
    if kind == "constraint":
        if t == "schedule":
            c["facility"] = random.choice([f"JamaratLevel_{i}" for i in range(4) if f"JamaratLevel_{i}" != c["facility"]]); return c, "C6"
        if t == "route":
            o, d = HNO[c["origin"]], HNO[c["destination"]]
            for cand in candidates(kg, o, d, k=12):
                if any(HNO.Profile_Wheelchair not in kg.seg[s]["access"] for s in cand):
                    c["segments"] = [local(s) for s in cand]; c["profile"] = "Profile_Wheelchair"; return c, "C5"
    return None, None


cases = []           # (split, kind, claim, expected_code)
for it in items:
    for c in it["gold"]:
        c = dict(c, source=SRC)
        cases.append((it["split"], "gold", c, None))
        for kind in ("entity", "relational", "temporal", "constraint"):
            p, code = perturb(c, kind)
            if p is not None:
                cases.append((it["split"], kind, p, code))

# threshold for the similarity detector: 5th percentile of dev gold scores
dev_gold = sorted(sim_score(c) for sp, k, c, _ in cases if sp == "dev" and k == "gold")
tau = dev_gold[int(0.05 * len(dev_gold))]

res = defaultdict(lambda: defaultdict(lambda: [0, 0]))     # kind -> detector -> [flagged, n]
by_type = defaultdict(lambda: [0, 0])
code_ok = defaultdict(lambda: [0, 0])
lat = []
for sp, kind, c, code in cases:
    if sp != "test":
        continue
    t0 = time.time(); r = V.verify([c]); lat.append(time.time() - t0)
    v = [x for x in r["violations"][0] if x != "C7"]
    flags = {"entity_only": entity_only(c), "similarity": sim_score(c) < tau, "ontohajj": bool(v)}
    for d, f in flags.items():
        res[kind][d][0] += int(f); res[kind][d][1] += 1
    by_type[(kind, c["type"])][0] += int(bool(v)); by_type[(kind, c["type"])][1] += 1
    if code:
        code_ok[kind][0] += int(code in v); code_ok[kind][1] += 1

out = {"similarity_threshold_tau": tau, "n_test_cases": sum(v["ontohajj"][1] for v in res.values()),
       "detection": {k: {d: {"rate": round(a / n, 4), "flagged": a, "n": n} for d, (a, n) in dd.items()} for k, dd in res.items()},
       "ontohajj_by_claim_type": {f"{k}|{t}": {"rate": round(a / n, 4), "n": n} for (k, t), (a, n) in sorted(by_type.items())},
       "correct_constraint_attribution": {k: round(a / n, 4) for k, (a, n) in code_ok.items()},
       "verify_latency_ms": {"mean": round(1000 * sum(lat) / len(lat), 1), "p95": round(1000 * sorted(lat)[int(0.95 * len(lat))], 1)}}
json.dump(out, open(os.path.join(ROOT, "results", "verifier_eval.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
