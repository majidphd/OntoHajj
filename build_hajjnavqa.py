"""Generates HajjNavQA: bilingual (Arabic/English) navigation questions grounded in HNO.
Each item: id, pair_id, lang, category, question, context, answerable, expect_abstain, gold (claims), reference answer."""
import json, os, random, sys
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontohajj.kg import load, HNO, local
from ontohajj.retrieve import nearest, fmt
from ontohajj.decide import decide

random.seed(2026)
kg = load()
L = lambda u, l: kg.label(u, l)
DAYS = {8: "2026-05-25", 9: "2026-05-26", 10: "2026-05-27", 11: "2026-05-28", 12: "2026-05-29", 13: "2026-05-30"}
DAY_AR = {8: "الثامن", 9: "التاسع", 10: "العاشر", 11: "الحادي عشر", 12: "الثاني عشر", 13: "الثالث عشر"}
dt = lambda d, hm: datetime.fromisoformat(f"{DAYS[d]}T{hm}")
iso = lambda t: t.isoformat(timespec="minutes")
items, pid = [], [0]


def add(cat, q_en, q_ar, ctx, gold, ref_en, ref_ar, answerable=True, expect_abstain=False, meta=None):
    pid[0] += 1
    for lang, q, ref in (("en", q_en, ref_en), ("ar", q_ar, ref_ar)):
        items.append({"id": f"HNQ-{pid[0]:04d}-{lang.upper()}", "pair_id": pid[0], "lang": lang, "category": cat, "question": q,
                      "context": ctx, "answerable": answerable, "expect_abstain": expect_abstain, "gold": gold,
                      "reference": ref, "meta": meta or {}})


mina_camps = [HNO[f"Camp_M{i:02d}"] for i in range(1, 25)]
arafat_camps = [HNO[f"Camp_A{i:02d}"] for i in range(1, 13)]
JL = [HNO[f"JamaratLevel_{i}"] for i in range(4)]
groups = [HNO[f"Group_G{i:02d}"] for i in range(1, 25)]

# ---------------- Q1 Location (40 site questions + 40 nearest questions) ----------------
facs = [u for u in kg.named if HNO.Facility in kg.types[u] and u not in mina_camps]
random.shuffle(facs)
for f in (facs * 2)[:40]:
    s = kg.site_of[f]
    tpl = random.choice([("In which holy site is {x} located?", "في أي مشعر يقع {y}؟"), ("Which holy site is {x} in?", "{y} في أي مشعر؟")])
    add("Q1", tpl[0].format(x=L(f, "en")), tpl[1].format(y=L(f, "ar")), {"time": None},
        [{"type": "location", "facility": local(f), "site": local(s)}],
        f"{L(f, 'en')} is located in {L(s, 'en')}.", f"يقع {L(f, 'ar')} في {L(s, 'ar')}.")
src_points = mina_camps + arafat_camps
random.shuffle(src_points)
for i, c in enumerate(src_points[:40]):
    cls, en_t, ar_t = (HNO.MedicalPoint, "medical centre", "مركز طبي") if i % 2 == 0 else (HNO.MetroStation, "metro station", "محطة قطار")
    n, d = nearest(kg, c, cls)
    add("Q1", f"Where is the nearest {en_t} to {L(c, 'en')}?", f"أين أقرب {ar_t} إلى {L(c, 'ar')}؟", {"time": None, "location": local(c)},
        [{"type": "nearest", "from": local(c), "facility_type": local(cls), "facility": local(n)}],
        f"The nearest {en_t} is {L(n, 'en')} ({int(d)} m on foot).", f"أقرب {ar_t} هو {L(n, 'ar')} ({int(d)} م سيرًا).")

# ---------------- Q2 Route (80) ----------------
def route_item(cat, o, d, day, hm, profile, q_en, q_ar):
    t = dt(day, hm)
    dec = decide(kg, o, d, t, profile)
    ctx = {"time": iso(t), "location": local(o), "destination": local(d), "profile": local(profile)}
    if dec["best"] is None:
        add(cat, q_en, q_ar, ctx, [], "No feasible route at this time; refer to the service provider.", "لا يوجد مسار ممكن في هذا الوقت؛ يُحال إلى مقدم الخدمة.",
            answerable=True, expect_abstain=True, meta={"n_candidates": dec["n_candidates"]})
        return
    b = dec["best"]
    gold = [{"type": "route", "origin": local(o), "destination": local(d), "profile": local(profile), "depart": iso(t),
             "segments": [local(s) for s in b["segments"]]}]
    stops = [kg.seg[b["segments"][0]]["from"]] + [kg.seg[s]["to"] for s in b["segments"]]
    add(cat, q_en, q_ar, ctx, gold, "Route: " + " → ".join(L(x, "en") for x in stops),
        "المسار: " + " ← ".join(L(x, "ar") for x in stops),
        meta={"n_feasible": len(dec["feasible"]), "n_candidates": dec["n_candidates"], "best_utility": b["utility"]})

TPL_R = [("How do I get from {o} to {d} on {day} Dhu al-Hijjah at {t}?", "كيف أصل من {o} إلى {d} يوم {day} من ذي الحجة الساعة {t}؟"),
         ("What is the best way to go from {o} to {d} at {t} on {day} Dhu al-Hijjah?", "ما أفضل طريق للذهاب من {o} إلى {d} الساعة {t} يوم {day} من ذي الحجة؟")]
scen = []
for _ in range(30):
    scen.append((random.choice(mina_camps), random.choice(JL), random.choice([10, 11, 12]), random.choice(["07:00", "09:30", "13:00", "15:30", "19:00", "22:00"])))
for _ in range(20):
    scen.append((random.choice(arafat_camps), random.choice([HNO.Station_Muzdalifah1, HNO.MasjidAlMasharAlHaram, HNO.Med_Muzdalifah]), 9,
                 random.choice(["15:00", "17:30", "19:10", "20:30", "23:00"])))
for _ in range(15):
    scen.append((random.choice(mina_camps), random.choice([HNO.Gate_KingAbdulaziz, HNO.Gate_KingFahd, HNO.Gate_AlSalam]), random.choice([10, 11]),
                 random.choice(["06:30", "10:00", "16:00", "21:00"])))
for _ in range(15):
    scen.append((random.choice([HNO.Station_Muzdalifah2, HNO.MasjidAlMasharAlHaram]), random.choice(mina_camps), 10, random.choice(["00:30", "03:00", "05:30", "08:00"])))
for o, d, day, hm in scen:
    tp = random.choice(TPL_R)
    route_item("Q2", o, d, day, hm, HNO.Profile_Standard, tp[0].format(o=L(o, "en"), d=L(d, "en"), day=day, t=hm),
               tp[1].format(o=L(o, "ar"), d=L(d, "ar"), day=DAY_AR[day], t=hm))

# ---------------- Q3 Temporal (36 schedule + 16 ritual windows) ----------------
rit_names = {HNO.StoningAqabah: 10, HNO.StoningDay11: 11, HNO.StoningDay12: 12}
for gr in random.sample(groups, 12):
    for r, day in rit_names.items():
        sl = kg.slots[(gr, r)]
        b = sl["win"][0]
        add("Q3", f"When does the dispatch slot of {L(gr, 'en')} for the {L(r, 'en')} start, and at which Jamarat level?",
            f"متى يبدأ موعد تفويج {L(gr, 'ar')} لـ{L(r, 'ar')}، وفي أي دور من الجمرات؟", {"time": None, "group": local(gr)},
            [{"type": "schedule", "group": local(gr), "ritual": local(r), "start": iso(b), "facility": local(sl["facility"])}],
            f"It starts at {fmt(b, 'en')} at {L(sl['facility'], 'en')}.", f"يبدأ {fmt(b, 'ar')} في {L(sl['facility'], 'ar')}.")
for r, (b, e) in kg.ritual_win.items():
    for tpl in [("When does the operational window for the {x} start and end?", "متى تبدأ وتنتهي النافذة التشغيلية لـ{y}؟"),
                ("What are the start and end times of the {x} window?", "ما وقت بداية ونهاية نافذة {y}؟")]:
        add("Q3", tpl[0].format(x=L(r, "en")), tpl[1].format(y=L(r, "ar")), {"time": None},
            [{"type": "ritual_time", "ritual": local(r), "start": iso(b), "end": iso(e)}],
            f"From {fmt(b, 'en')} to {fmt(e, 'en')}.", f"من {fmt(b, 'ar')} إلى {fmt(e, 'ar')}.")

# ---------------- Q4 Accessibility (50) ----------------
hill = [HNO[f"Camp_M{i:02d}"] for i in (5, 10, 15, 20)]
TPL_A = [("I use a wheelchair. Which route should I take from {o} to {d} on {day} Dhu al-Hijjah at {t}?", "أستخدم كرسيًا متحركًا، ما الطريق المناسب من {o} إلى {d} يوم {day} من ذي الحجة الساعة {t}؟"),
         ("Which step-free route is suitable for a wheelchair user from {o} to {d} at {t} on {day} Dhu al-Hijjah?", "ما الطريق الخالي من الدرج المناسب لمستخدم كرسي متحرك من {o} إلى {d} الساعة {t} يوم {day} من ذي الحجة؟")]
TPL_E = [("I am an elderly pilgrim. How do I get from {o} to {d} on {day} Dhu al-Hijjah at {t}?", "أنا حاج كبير السن، كيف أصل من {o} إلى {d} يوم {day} من ذي الحجة الساعة {t}؟")]
for k in range(50):
    o = random.choice(hill + mina_camps) if k % 3 else random.choice(hill)
    d = random.choice(JL + [HNO.Med_MinaC, HNO.Med_Jamarat, HNO.Station_Mina3])
    day = random.choice([10, 11, 12]); hm = random.choice(["02:00", "07:30", "10:00", "14:00", "18:30", "21:00"])
    prof, tp = (HNO.Profile_Wheelchair, random.choice(TPL_A)) if k % 4 else (HNO.Profile_Elderly, TPL_E[0])
    route_item("Q4", o, d, day, hm, prof, tp[0].format(o=L(o, "en"), d=L(d, "en"), day=day, t=hm),
               tp[1].format(o=L(o, "ar"), d=L(d, "ar"), day=DAY_AR[day], t=hm))

# ---------------- Q5 Multi-hop (25 nearest+location, 25 schedule+route) ----------------
for c in random.sample(mina_camps + arafat_camps, 25):
    n, d = nearest(kg, c, HNO.MetroStation)
    s = kg.site_of[n]
    add("Q5", f"What is the nearest metro station to {L(c, 'en')}, and in which holy site is that station?",
        f"ما أقرب محطة قطار إلى {L(c, 'ar')}، وفي أي مشعر تقع هذه المحطة؟", {"time": None, "location": local(c)},
        [{"type": "nearest", "from": local(c), "facility_type": "MetroStation", "facility": local(n)},
         {"type": "location", "facility": local(n), "site": local(s)}],
        f"{L(n, 'en')}, located in {L(s, 'en')}.", f"{L(n, 'ar')}، وتقع في {L(s, 'ar')}.")
for gr in random.sample(groups, 24) + random.sample(groups, 1):
    r, day = random.choice(list(rit_names.items()))
    sl = kg.slots[(gr, r)]
    camp = [c for c in kg.group_camps[gr] if kg.site_of[c] == HNO.Mina][0]
    t = sl["win"][0]
    dec = decide(kg, camp, sl["facility"], t, HNO.Profile_Standard, sl["win"][1])
    gold = [{"type": "schedule", "group": local(gr), "ritual": local(r), "start": iso(t), "facility": local(sl["facility"])}]
    if dec["best"] is not None:
        gold.append({"type": "route", "origin": local(camp), "destination": local(sl["facility"]), "profile": "Profile_Standard",
                     "depart": iso(t), "segments": [local(x) for x in dec["best"]["segments"]]})
    add("Q5", f"We are {L(gr, 'en')}. When does our dispatch slot for the {L(r, 'en')} start, and how do we get from our Mina camp to the assigned Jamarat level at that time?",
        f"نحن {L(gr, 'ar')}. متى يبدأ موعد تفويجنا لـ{L(r, 'ar')}، وكيف نصل من مخيمنا في منى إلى دور الجمرات المخصص في ذلك الوقت؟",
        {"time": iso(t), "group": local(gr), "ritual": local(r), "location": local(camp)}, gold,
        f"Slot at {fmt(t, 'en')}, {L(sl['facility'], 'en')}.", f"الموعد {fmt(t, 'ar')}، {L(sl['facility'], 'ar')}.",
        expect_abstain=dec["best"] is None)

# ---------------- Q6 Unanswerable (30 non-existent + 15 juristic + 15 out-of-scope) ----------------
ne = []
for n in random.sample(range(25, 90), 8):
    ne.append((f"Where is the nearest medical centre to Mina camp M{n}?", f"أين أقرب مركز طبي إلى مخيم منى م{n}؟"))
for n in (4, 5, 6, 7, 8, 9):
    ne.append((f"How do I get from Mina Station {n} to Jamarat level 1?", f"كيف أصل من محطة منى {n} إلى الدور الأول للجمرات؟"))
for n in random.sample(range(10, 120), 8):
    ne.append((f"Which bus goes directly from Arafat to gate {n} of the Haram?", f"ما الحافلة التي تذهب مباشرة من عرفات إلى باب رقم {n} في الحرم؟"))
for n in random.sample(range(30, 99), 8):
    ne.append((f"When does the dispatch slot of group G{n} for the stoning of Jamarat al-Aqabah start?", f"متى يبدأ موعد تفويج المجموعة ج{n} لرمي جمرة العقبة؟"))
for en, ar in ne:
    add("Q6", en, ar, {"time": None}, [], "Unknown entity: abstain and refer.", "كيان غير موجود: يُمتنع ويُحال.", answerable=False, expect_abstain=True, meta={"kind": "non-existent"})
jur = [("Is my Hajj valid if I leave Muzdalifah before midnight?", "هل يصح حجي إذا خرجت من مزدلفة قبل منتصف الليل؟"),
       ("Is it permissible to delegate the stoning to someone else?", "هل يجوز توكيل غيري في رمي الجمرات؟"),
       ("What is the expiation if I miss the stoning on 11 Dhu al-Hijjah?", "ما الفدية إذا فاتني الرمي يوم 11 ذي الحجة؟"),
       ("Is it a sin to leave Arafat before sunset?", "هل علي إثم إذا خرجت من عرفة قبل الغروب؟"),
       ("Is staying overnight in Mina obligatory?", "هل المبيت بمنى واجب؟"),
       ("Can a woman perform Tawaf al-Ifadah during her period?", "هل يجوز للحائض طواف الإفاضة؟"),
       ("Is it sunnah to walk from Mina to Arafat?", "هل المشي من منى إلى عرفة سنة؟"),
       ("Does missing the night in Muzdalifah invalidate Hajj?", "هل يبطل الحج بترك المبيت بمزدلفة؟"),
       ("What is the ruling on stoning before noon on 11 Dhu al-Hijjah?", "ما حكم الرمي قبل الزوال يوم 11 ذي الحجة؟"),
       ("Is it allowed to combine the stoning of two days?", "هل يجوز جمع رمي يومين؟"),
       ("Must I offer a sacrifice if I leave Mina early?", "هل يلزمني دم إذا تعجلت من منى؟"),
       ("Is Tawaf al-Wada obligatory for me?", "هل طواف الوداع واجب علي؟"),
       ("Is my Hajj accepted if I could not reach Masjid Namirah?", "هل يصح حجي إذا لم أصل إلى مسجد نمرة؟"),
       ("Is it permissible to stone from the upper level of the Jamarat?", "هل يجوز الرمي من الدور العلوي للجمرات؟"),
       ("What is the ruling on shaving before stoning?", "ما حكم الحلق قبل الرمي؟")]
for en, ar in jur:
    add("Q6", en, ar, {"time": None}, [], "Juristic question: out of scope, refer to qualified scholars.", "سؤال فقهي خارج النطاق، يُحال إلى المختصين.",
        answerable=False, expect_abstain=True, meta={"kind": "juristic"})
oos = [("What will the temperature be in Arafat tomorrow?", "كم ستكون درجة الحرارة في عرفات غدًا؟"),
       ("Can you book me a hotel room near the Haram?", "هل يمكنك حجز غرفة فندقية لي قرب الحرم؟"),
       ("What is the exchange rate of the riyal today?", "ما سعر صرف الريال اليوم؟"),
       ("Which restaurant in Mina has the best food?", "أي مطعم في منى طعامه أفضل؟"),
       ("How many pilgrims are in Mina right now?", "كم عدد الحجاج في منى الآن؟"),
       ("What is the price of a sacrifice voucher this year?", "كم سعر صك الأضحية هذا العام؟"),
       ("Can I extend my Hajj visa?", "هل يمكنني تمديد تأشيرة الحج؟"),
       ("Where can I buy a SIM card in Makkah?", "أين أشتري شريحة اتصال في مكة؟"),
       ("Is there Wi-Fi in the Mina camps?", "هل يتوفر واي فاي في مخيمات منى؟"),
       ("When does my flight home leave?", "متى تقلع رحلة عودتي؟"),
       ("Which hospital in Jeddah is best for heart patients?", "أي مستشفى في جدة أفضل لمرضى القلب؟"),
       ("How do I apply for Hajj next year?", "كيف أسجل للحج العام القادم؟"),
       ("What is the history of Masjid al-Khayf?", "ما تاريخ مسجد الخيف؟"),
       ("Can you recommend a good supplication for Arafat day?", "هل تقترح دعاءً مناسبًا ليوم عرفة؟"),
       ("Who is the imam of Masjid Namirah this year?", "من خطيب مسجد نمرة هذا العام؟")]
for en, ar in oos:
    add("Q6", en, ar, {"time": None}, [], "Out of scope: abstain and refer.", "خارج النطاق: يُمتنع ويُحال.", answerable=False, expect_abstain=True, meta={"kind": "out-of-scope"})

# stratified dev/test split by pair (30% dev for rule and prompt development, 70% held-out test)
rs = random.Random(7)
bycat = {}
for it in items:
    bycat.setdefault(it["category"], set()).add(it["pair_id"])
dev = set()
for c, ps in bycat.items():
    ps = sorted(ps); rs.shuffle(ps); dev |= set(ps[: round(0.3 * len(ps))])
for it in items:
    it["split"] = "dev" if it["pair_id"] in dev else "test"
os.makedirs("../data", exist_ok=True)
with open("../data/hajjnavqa.jsonl", "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
from collections import Counter
cnt = Counter((i["category"], i["lang"]) for i in items)
summary = {"total": len(items), "pairs": pid[0], "by_category": {c: cnt[(c, "en")] for c in sorted({i["category"] for i in items})},
           "expect_abstain": sum(i["expect_abstain"] for i in items) // 2,
           "answerable_expect_abstain": sum(i["expect_abstain"] and i["answerable"] for i in items) // 2,
           "dev_pairs": len(dev), "test_pairs": pid[0] - len(dev),
           "test_by_category": {c: sum(1 for i in items if i["category"] == c and i["lang"] == "en" and i["split"] == "test") for c in sorted(bycat)}}
json.dump(summary, open("../data/hajjnavqa_summary.json", "w", encoding="utf-8"), indent=1)
print(summary)
