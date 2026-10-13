"""Evaluates Layer 2 on HajjNavQA: intent typing accuracy, entity-linking P/R/F1 and detection of
non-existent entities, per language."""
import json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontohajj.kg import load, HNO, local
from ontohajj.understand import Linker, intent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kg = load(); L = Linker(kg)
items = [json.loads(l) for l in open(os.path.join(ROOT, "data", "hajjnavqa.jsonl"), encoding="utf-8")]
GOLD_INTENT = {"Q1": "Location", "Q2": "Route", "Q3": "Temporal", "Q4": "Accessibility", "Q5": "MultiHop"}
ENT_KEYS = ("facility", "site", "from", "group", "ritual", "origin", "destination")

st = defaultdict(lambda: defaultdict(float))
times = []
for it in items:
    lang = it["lang"] + "_" + it["split"]; q = it["question"]
    t0 = time.time(); found, unres = L.link(q); pi = intent(q); times.append(time.time() - t0)
    s = st[lang]
    # intent
    if it["category"] in GOLD_INTENT:
        s["intent_n"] += 1; s["intent_ok"] += pi == GOLD_INTENT[it["category"]]
    elif it["meta"].get("kind") == "juristic":
        s["intent_n"] += 1; s["intent_ok"] += pi == "OutOfScope"
    # entity linking: gold entities = entities of gold claims that are named in the question text
    if it["answerable"] and it["gold"]:
        gold = set()
        QSIDE = {"location": ("facility",), "nearest": ("from",), "schedule": ("group", "ritual"), "ritual_time": ("ritual",),
                 "route": ("origin", "destination")}
        for c in it["gold"]:
            for k in QSIDE.get(c["type"], ()):
                if c.get(k):
                    u = HNO[c[k]]
                    names = list(kg.labels[u].values()) + kg.alt[u]
                    from ontohajj.understand import normalize
                    nq = " ".join(w[1:] if len(w) > 3 and w[0] in "لبوف" else w for w in normalize(q).split())  # Arabic proclitics
                    if any(f" {normalize(n)} " in f" {normalize(q)} " or f" {normalize(n)} " in f" {nq} " for n in names):
                        gold.add(u)
        pred = set(found)
        s["tp"] += len(gold & pred); s["fp"] += len(pred - gold - {HNO.Mina, HNO.Arafat, HNO.Muzdalifah, HNO.MasjidAlHaram} if False else set()) ; s["fn"] += len(gold - pred)
        s["fp_strict"] += len(pred - gold)
        s["unres_false_alarm"] += bool(unres); s["answerable_n"] += 1
    if it["meta"].get("kind") == "non-existent":
        s["ne_n"] += 1; s["ne_detected"] += bool(unres)

out = {}
for lang, s in st.items():
    p = s["tp"] / max(1, s["tp"] + s["fp_strict"]); r = s["tp"] / max(1, s["tp"] + s["fn"])
    out[lang] = {"intent_accuracy": round(s["intent_ok"] / s["intent_n"], 4), "intent_n": int(s["intent_n"]),
                 "linking_precision_strict": round(p, 4), "linking_recall": round(r, 4), "linking_f1": round(2 * p * r / max(1e-9, p + r), 4),
                 "nonexistent_detection_recall": round(s["ne_detected"] / max(1, s["ne_n"]), 4), "nonexistent_n": int(s["ne_n"]),
                 "false_alarm_rate_answerable": round(s["unres_false_alarm"] / max(1, s["answerable_n"]), 4)}
out["mean_ms_per_query"] = round(1000 * sum(times) / len(times), 2)
json.dump(out, open(os.path.join(ROOT, "results", "understanding_eval.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
