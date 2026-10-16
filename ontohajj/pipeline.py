import json
"""OntoHajj inference (Algorithm 1) and the five evaluation configurations C0-C4."""
import time
from datetime import datetime, timedelta
from .kg import HNO, local, load
from .understand import Linker, intent
from .retrieve import retrieve, fmt
from .decide import decide, candidates
from .verify import Verifier, verdict_feedback, verbose_feedback, SAFETY
from .generate import system_prompt, user_prompt, parse_json, TextRAG, build_corpus

REFERRAL = {"en": "I cannot confirm this information. Please contact your Hajj service provider or the official pilgrim help line.",
            "ar": "لا أستطيع تأكيد هذه المعلومة. يرجى التواصل مع مقدم الخدمة أو مركز الاتصال الرسمي لخدمة الحجاج."}


class OntoHajj:
    def __init__(self, llm, kg=None, k_repair=2):
        self.kg = kg or load()
        self.linker = Linker(self.kg)
        self.verifier = Verifier(self.kg, self.linker)
        self.llm, self.k = llm, k_repair
        self._rag = {}

    def rag(self, lang):
        if lang not in self._rag:
            self._rag[lang] = TextRAG(build_corpus(self.kg, lang))
        return self._rag[lang]

    def route_text(self, best, lang, origin, dest, profile, start):
        L = lambda u: self.kg.label(u, lang)
        legs = []
        for s in best["segments"]:
            d = self.kg.seg[s]
            legs.append(f"[{local(s)}] {L(d['from'])} → {L(d['to'])} ({local(d['mode'])}, {int(d['len'])} m)")
        return (f"origin={local(origin)}, destination={local(dest)}, profile={local(profile)}, depart={start.isoformat(timespec='minutes')}, "
                f"arrival={best['arrival'].isoformat(timespec='minutes')}, utility={best['utility']}\n" + "\n".join(legs))

    def earliest_departure(self, origin, dest, start, profile, horizon_h=12):
        t = start
        while t <= start + timedelta(hours=horizon_h):
            t += timedelta(minutes=30)
            d = decide(self.kg, origin, dest, t, profile)
            if d["best"] is not None:
                return t, d
        return None, None

    # ------------------------------------------------------------------
    def answer(self, question, context, config="C4", lang="en"):
        """config: C0-C4; ablations: C4k0 (no repair), C4hint (verbose hints instead of verdicts)"""
        ALIAS = {"S0": "C0", "S1": "C1", "S2": "C2", "S3": "C3", "S4": "C4", "S4-NoRepair": "C4k0", "S4-Hint": "C4hint", "S4-LLM-L2": "C4llm"}
        config = ALIAS.get(config, config)
        variant = config
        k_repair = 0 if config == "C4k0" else self.k
        feedback_fn = verbose_feedback if config == "C4hint" else verdict_feedback
        if config.startswith("C4"):
            config = "C4"
        kg, rec = self.kg, {"config": variant, "latency": {}, "tokens": 0, "repairs": 0}
        T = lambda: time.time()
        t0 = T()
        t_ctx = datetime.fromisoformat(context["time"]) if context.get("time") else None
        profile = HNO[context.get("profile", "Profile_Standard")]
        facts, passages, route_txt, best_info, route_claim = None, None, None, None, None
        # ---------------- L2
        ents, unresolved = self.linker.link(question)
        it = intent(question)
        if variant == "C4llm":                      # S4-LLM-L2: LLM intent typing + canonical mentions, linked to HNO
            from .understand_llm import understand
            z, mentions, tok = understand(self.llm, question)
            rec["tokens"] += tok
            unresolved = []
            for m in mentions:
                f, u = self.linker.link(m)
                ents += [e for e in f if e not in ents]; unresolved += u
            if z is not None:
                it = z
            rec["l2_llm"] = {"intent": z, "mentions": mentions}
        for key in ("location", "group"):
            if context.get(key) and HNO[context[key]] not in ents:
                ents.append(HNO[context[key]])
        rec.update(intent=it, entities=[local(e) for e in ents], unresolved=unresolved)
        rec["latency"]["L2"] = T() - t0
        if config == "C4" and (it == "OutOfScope" or unresolved or not ents):
            rec.update(answer=REFERRAL[lang], abstain=True, claims=[], verified=True, reason="L2: out of scope or unresolved entity")
            rec["latency"]["total"] = T() - t0
            return rec
        # ---------------- L3
        t1 = T()
        if config == "C1":
            passages = self.rag(lang).search(question, k=10)
        elif config in ("C2", "C3", "C4"):
            facts = retrieve(kg, ents, t_ctx, lang)
            if it in ("Route", "Accessibility", "MultiHop") and config == "C2":
                o, d = self._od(ents, context)
                if o is not None and d is not None:
                    for c in candidates(kg, o, d, k=3)[:3]:
                        for s in c:
                            sd = kg.seg[s]
                            closed = "; ".join(f"{fmt(b, lang)}–{fmt(e, lang)}" for b, e in sd["restr"]) or "none"
                            acc = "step-free" if HNO.Profile_Wheelchair in sd["access"] else "has stairs"
                            facts.append({"text": f"[{local(s)}] {kg.label(sd['from'], lang)} → {kg.label(sd['to'], lang)} by {local(sd['mode'])}, "
                                                  f"{int(sd['len'])} m, {sd['min']} min, {acc}; closed: {closed}", "source": "Src_HNOPrototype"})
        rec["latency"]["L3"] = T() - t1
        # ---------------- L4
        t2 = T()
        if config in ("C3", "C4") and it in ("Route", "Accessibility", "MultiHop"):
            o, d = self._od(ents, context)
            target_end = None
            start = t_ctx
            if it == "MultiHop" and context.get("group") and context.get("ritual"):
                sl = kg.slots.get((HNO[context["group"]], HNO[context["ritual"]]))
                if sl:
                    d = sl["facility"]; start = sl["win"][0]; target_end = sl["win"][1]
            if o is not None and d is not None and start is not None:
                dec = decide(kg, o, d, start, profile, target_end)
                if dec["best"] is not None:
                    route_txt = self.route_text(dec["best"], lang, o, d, profile, start)
                    best_info = {"segments": [local(s) for s in dec["best"]["segments"]], "utility": dec["best"]["utility"]}
                    route_claim = {"type": "route", "origin": local(o), "destination": local(d), "profile": local(profile),
                                   "depart": start.strftime("%Y-%m-%dT%H:%M"), "segments": best_info["segments"],
                                   "source": "Src_HNOPrototype"}
                elif config == "C4":
                    te, d2 = self.earliest_departure(o, d, start, profile)
                    extra = ""
                    if te is not None:
                        extra = (f" The earliest verified departure is {fmt(te, lang)}." if lang == "en" else f" أقرب موعد انطلاق موثّق هو {fmt(te, lang)}.")
                    rec.update(answer=REFERRAL[lang] + extra, abstain=True, claims=[], verified=True,
                               reason="L4: empty feasible set", earliest=te.isoformat() if te else None)
                    rec["latency"]["L4"] = T() - t2; rec["latency"]["total"] = T() - t0
                    return rec
        rec["latency"]["L4"] = T() - t2
        rec["best_route"] = best_info
        skeletons = None
        if config in ("C2", "C3", "C4") and facts:
            skeletons, seen = [], set()
            for f in facts:
                dd = f.get("data") or {}
                if dd.get("type"):
                    key = json.dumps(dd, sort_keys=True)
                    if key not in seen:
                        seen.add(key); skeletons.append(dd)
            if route_claim is not None:
                skeletons.append(route_claim)
        # ---------------- generation (Eq. 6)
        sysmsg = system_prompt(config)
        usr = user_prompt(question, context, facts, passages, route_txt, skeletons=skeletons)
        t3 = T()
        text, tok, _ = self.llm(sysmsg, usr)
        rec["tokens"] += tok
        out = parse_json(text)
        rec["latency"]["LLM"] = T() - t3
        # ---------------- L5 / L6
        if config == "C4":
            t4 = T()
            for j in range(k_repair + 1):
                if out.get("abstain"):
                    break
                if out["claims"]:
                    res = self.verifier.verify(out["claims"], profile)
                    rec.setdefault("attempts", []).append({"claims": out["claims"], "violations": res["violations"]})
                    if res["conforms"]:
                        break
                    fb = feedback_fn(out["claims"], res)
                else:
                    rec.setdefault("attempts", []).append({"claims": [], "violations": "no claims"})
                    fb = "Your answer contains no checkable claims; state your claims with sources or abstain."
                if j == k_repair:
                    out = {"answer": REFERRAL[lang], "abstain": True, "claims": [], "abstain_reason": "L6: unverifiable after repair"}
                    break
                text, tok, _ = self.llm(sysmsg, user_prompt(question, context, facts, passages, route_txt, feedback=fb, skeletons=skeletons))
                rec["tokens"] += tok; rec["repairs"] += 1
                out = parse_json(text)
            rec["latency"]["L5L6"] = T() - t4
        rec.update(answer=out.get("answer", ""), abstain=bool(out.get("abstain")), claims=out.get("claims", []),
                   parse_error=out.get("parse_error", False))
        rec["latency"]["total"] = T() - t0
        return rec

    def _od(self, ents, context):
        kg = self.kg
        locs = [e for e in ents if HNO.Location in kg.types[e]]
        o = HNO[context["location"]] if context.get("location") else (locs[0] if locs else None)
        rest = [e for e in locs if e != o and e in kg.net]          # holy sites are areas, not network nodes
        d = HNO[context["destination"]] if context.get("destination") else (rest[0] if rest else None)
        if o is not None and o not in kg.net:
            o = None
        return o, d
