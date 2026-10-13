"""Layer 3: ontology-guided retrieval (Eq. 2) and verbalisation (pi)."""
from datetime import datetime, timedelta
from .kg import HNO, local, load

TYPE_NAMES = {HNO.Camp: ("camp", "مخيم"), HNO.MetroStation: ("metro station", "محطة قطار"), HNO.JamaratLevel: ("Jamarat level", "دور الجمرات"),
              HNO.MedicalPoint: ("medical centre", "مركز طبي"), HNO.Mosque: ("mosque", "مسجد"), HNO.Gate: ("gate", "باب"),
              HNO.BusStation: ("bus station", "محطة حافلات"), HNO.HolySite: ("holy site", "مشعر"), HNO.Waypoint: ("waypoint", "نقطة عبور"),
              HNO.PilgrimGroup: ("pilgrim group", "مجموعة حجاج"), HNO.Ritual: ("ritual", "نسك")}


def fmt(t, lang):
    day = {25: 8, 26: 9, 27: 10, 28: 11, 29: 12, 30: 13}[t.day]
    iso = t.strftime("%Y-%m-%dT%H:%M")
    return (f"{day} Dhu al-Hijjah {t:%H:%M} ({iso})" if lang == "en" else f"{day} ذو الحجة الساعة {t:%H:%M} ({iso})")


def kind(kg, u):
    for c in TYPE_NAMES:
        if c in kg.types[u] and c != HNO.Ritual:
            return c
    return HNO.Ritual if HNO.Ritual in kg.types[u] else None


def nearest(kg, frm, cls, profile=None):
    best, bd = None, float("inf")
    for f in kg.instances(cls):
        d = kg.walking_distance(frm, f, profile)
        if d < bd:
            best, bd = f, d
    return best, bd


def retrieve(kg, entities, t, lang="en", h=1):
    """returns list of facts: (text, source_local_name, structured dict)"""
    facts = []
    L = lambda u: kg.label(u, lang)
    def add(txt, src, data):
        if data and data.get("type"):
            data = dict(data, source=src)          # claim skeleton in the HNO output schema
        facts.append({"text": txt, "source": src, "data": data})
    day_lo = t.replace(hour=0, minute=0) if t else None
    for e in entities:
        k = kind(kg, e)
        src = local(kg.g.value(e, HNO.derivedFrom) or HNO.Src_HNOPrototype)
        tn = TYPE_NAMES.get(k, ("entity", "كيان"))[0 if lang == "en" else 1]
        add(f"[{local(e)}] {L(e)} is a {tn}." if lang == "en" else f"[{local(e)}] {L(e)} هو {tn}.", src, {"id": local(e)})
        if e in kg.site_of:
            s = kg.site_of[e]
            add(f"{L(e)} is located in {L(s)} [{local(s)}]." if lang == "en" else f"يقع {L(e)} في {L(s)} [{local(s)}].", src,
                {"type": "location", "facility": local(e), "site": local(s)})
        if k in (HNO.Camp, HNO.Waypoint, HNO.Mosque, HNO.Gate, HNO.MetroStation):
            for cls in (HNO.MedicalPoint, HNO.MetroStation):
                if cls == k:
                    continue
                n, d = nearest(kg, e, cls)
                if n is not None and d < float("inf"):
                    add(f"The nearest {TYPE_NAMES[cls][0]} to {L(e)} is {L(n)} [{local(n)}], {int(d)} m on foot."
                        if lang == "en" else f"أقرب {TYPE_NAMES[cls][1]} إلى {L(e)} هو {L(n)} [{local(n)}] على بعد {int(d)} م سيرًا.",
                        "Src_HNOPrototype", {"type": "nearest", "from": local(e), "facility_type": local(cls), "facility": local(n)})
                    if n in kg.site_of:                       # one more hop from the derived node
                        sn = kg.site_of[n]
                        add(f"{L(n)} is located in {L(sn)} [{local(sn)}]." if lang == "en" else f"يقع {L(n)} في {L(sn)} [{local(sn)}].",
                            local(kg.g.value(n, HNO.derivedFrom) or HNO.Src_HNOPrototype),
                            {"type": "location", "facility": local(n), "site": local(sn)})
        if k == HNO.PilgrimGroup:
            for c in kg.group_camps[e]:
                add(f"{L(e)} is assigned to {L(c)} [{local(c)}]." if lang == "en" else f"{L(e)} مسكنها {L(c)} [{local(c)}].", src, {})
            for (gr, r), sl in kg.slots.items():
                if gr == e:
                    b, en = sl["win"]
                    add(f"Dispatch slot of {L(e)} for {L(r)} [{local(r)}]: {L(sl['facility'])} [{local(sl['facility'])}], from {fmt(b, 'en')} to {fmt(en, 'en')}."
                        if lang == "en" else f"موعد تفويج {L(e)} لـ{L(r)} [{local(r)}]: {L(sl['facility'])} [{local(sl['facility'])}] من {fmt(b, 'ar')} إلى {fmt(en, 'ar')}.",
                        "Src_HNOPrototype", {"type": "schedule", "group": local(e), "ritual": local(r), "facility": local(sl["facility"]),
                                             "start": b.isoformat(timespec="minutes")})
        if HNO.Ritual in kg.types[e]:
            b, en = kg.ritual_win[e]
            sites = [L(x) for x in kg.g.objects(e, HNO.performedAt)]
            add(f"{L(e)} [{local(e)}] takes place at {', '.join(sites)} between {fmt(b, 'en')} and {fmt(en, 'en')} (operational window)."
                if lang == "en" else f"{L(e)} [{local(e)}] يؤدى في {'، '.join(sites)} بين {fmt(b, 'ar')} و{fmt(en, 'ar')} (نافذة تشغيلية).",
                "Src_HNOPrototype", {"type": "ritual_time", "ritual": local(e), "start": b.isoformat(timespec="minutes"), "end": en.isoformat(timespec="minutes")})
        if k == HNO.MetroStation and t is not None:
            for s, d in kg.seg.items():
                if d["from"] == e and d["mode"] == HNO.MashaerMetro:
                    closed = [(b, en) for b, en in d["restr"] if en >= day_lo and b <= day_lo + timedelta(days=1)]
                    opens = "; ".join(f"closed {fmt(b, lang)}–{fmt(en, lang)}" for b, en in closed) or "open all day"
                    add(f"Metro {L(e)} → {L(d['to'])}: {opens}." if lang == "en" else f"القطار من {L(e)} إلى {L(d['to'])}: {opens}.",
                        "Src_HNOPrototype", {})
    return facts


def resolve(kg, linker, s):
    """map an entity string from an LLM claim (id or name) to a KG URI.
    Whole-string matching only: a partial match (e.g. 'Mina' inside 'Mina Station 5') is NOT accepted,
    so fabricated entities stay unresolved and fail constraint C1."""
    import re as _re
    from rapidfuzz import fuzz
    from .understand import normalize, codes
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("hno:"):
        s = s[4:]
    if _re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s) and kg.exists(HNO[s]):
        return HNO[s]
    q = normalize(s)
    best, bs = None, 0
    for nn, u in linker.index:
        if nn == q:
            return u
        if abs(len(nn) - len(q)) <= 3 and codes(nn) == codes(q):
            r = fuzz.ratio(nn, q)
            if r > bs:
                best, bs = u, r
    if bs >= linker.TAU and not linker._unresolved(q, [best]):
        return best
    return HNO["UNRESOLVED_" + "".join(ch if ch.isalnum() else "_" for ch in s)[:40]]
