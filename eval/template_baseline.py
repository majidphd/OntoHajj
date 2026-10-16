"""No-LLM baseline (S4-Template): the full pipeline L1-L6, but the LLM is replaced by a deterministic renderer
that copies every claim skeleton supplied by L3/L4 and verbalises it with fixed bilingual templates.
It answers the reviewer question "what does the LLM add?": on template-generated questions the symbolic
layers alone determine the claims, and the renderer cannot produce free-text errors by construction.
Usage: python eval/template_baseline.py   (results/llm/template/test_v2/C4_run0.jsonl, then summarise_llm.py template/test_v2)"""
import json, os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))
from ontohajj.kg import load, HNO
from ontohajj.pipeline import OntoHajj
from run_llm_eval import score


class TemplateRenderer:
    """drop-in replacement for ontohajj.generate.LLM"""
    def __init__(self, kg):
        self.kg = kg

    def lab(self, x, lang):
        return self.kg.label(HNO[x], lang) if isinstance(x, str) and self.kg.exists(HNO[x]) else str(x)

    def render(self, c, lang):
        L = lambda x: self.lab(x, lang); ar = lang == "ar"; t = c.get("type")
        if t == "location":
            return f"يقع {L(c['facility'])} في {L(c['site'])}." if ar else f"{L(c['facility'])} is located in {L(c['site'])}."
        if t == "nearest":
            return f"أقرب مرفق من {L(c['from'])} هو {L(c['facility'])}." if ar else f"The nearest facility of this type to {L(c['from'])} is {L(c['facility'])}."
        if t == "schedule":
            return (f"يبدأ موعد {L(c['group'])} لـ{L(c['ritual'])} في {c['start'].replace('T', ' ')} في {L(c['facility'])}." if ar else
                    f"The slot of {L(c['group'])} for {L(c['ritual'])} starts at {c['start'].replace('T', ' ')} at {L(c['facility'])}.")
        if t == "ritual_time":
            return (f"نافذة {L(c['ritual'])}: من {c['start'].replace('T', ' ')} إلى {c['end'].replace('T', ' ')}." if ar else
                    f"{L(c['ritual'])}: from {c['start'].replace('T', ' ')} to {c['end'].replace('T', ' ')}.")
        if t == "route":
            segs = [HNO[s] for s in c["segments"] if HNO[s] in self.kg.seg]
            stops = [self.kg.seg[segs[0]]["from"]] + [self.kg.seg[s]["to"] for s in segs] if segs else []
            sep = " ← " if ar else " → "
            path = sep.join(self.kg.label(s, lang) for s in stops)
            return f"المسار الموصى به (انطلاق {c['depart'].replace('T', ' ')}): {path}." if ar else f"Recommended route (departure {c['depart'].replace('T', ' ')}): {path}."
        return ""

    def __call__(self, system, user):
        t0 = time.time()
        if "failed verification" in user:                       # never repair: abstain
            return json.dumps({"answer": "", "abstain": True, "claims": []}), 0, time.time() - t0
        q = re.search(r"Question: (.*)", user).group(1)
        lang = "ar" if re.search(r"[؀-ۿ]", q) else "en"
        claims = []
        m = re.search(r"Ready-made claims supported by the facts \(copy the relevant ones unchanged\):\n(.*?)(?:\n\n|$)", user, re.S)
        if m:
            for line in m.group(1).splitlines():
                try:
                    claims.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not claims:
            return json.dumps({"answer": "", "abstain": True, "claims": []}), 0, time.time() - t0
        text = " ".join(self.render(c, lang) for c in claims)
        return json.dumps({"answer": text, "abstain": False, "claims": claims}, ensure_ascii=False), 0, time.time() - t0


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="hajjnavqa.jsonl"); ap.add_argument("--split", default="test")
    a = ap.parse_args()
    kg = load()
    sys_ = OntoHajj(TemplateRenderer(kg))
    items = [json.loads(l) for l in open(os.path.join(ROOT, "data", a.data), encoding="utf-8")]
    items = [i for i in items if i["split"] == a.split]
    outdir = os.path.join(ROOT, "results", "llm", "template", f"{a.split}_v2"); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "C4_run0.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            rec = sys_.answer(it["question"], it["context"], "C4", it["lang"])
            s = score(it, rec, sys_)
            f.write(json.dumps({"id": it["id"], "category": it["category"], "lang": it["lang"], "config": "C4", "run": 0,
                                "expect_abstain": it["expect_abstain"], **s, "latency": rec.get("latency", {}).get("total"),
                                "tokens": 0, "repairs": rec.get("repairs", 0), "answer": rec.get("answer"), "claims": rec.get("claims"),
                                "attempts": rec.get("attempts"), "reason": rec.get("reason")}, ensure_ascii=False) + "\n")
    print("written", outdir)


if __name__ == "__main__":
    main()
