"""Scenario cards for the human-written question set (HajjNavQA-H).
Each card describes a pilgrim's situation and need WITHOUT a question template; writers phrase the question in
their own words. The gold claims and the context of every card are computed from the knowledge graph exactly as
for HajjNavQA and are stored in bench/human_scenarios.json (not shown to the writers)."""
import json, os, random, sys
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ontohajj.kg import load, HNO, local
from ontohajj.retrieve import nearest
from ontohajj.decide import decide

random.seed(4747)
kg = load(); L = lambda u, l: kg.label(u, l)
DAY = {10: "2026-05-27", 11: "2026-05-28", 12: "2026-05-29", 9: "2026-05-26"}
DAY_AR = {9: "يوم عرفة (9 ذو الحجة)", 10: "يوم العيد (10 ذو الحجة)", 11: "يوم 11 ذو الحجة", 12: "يوم 12 ذو الحجة"}
mina = [HNO[f"Camp_M{i:02d}"] for i in range(1, 25)]; arafat = [HNO[f"Camp_A{i:02d}"] for i in range(1, 13)]
JL = [HNO[f"JamaratLevel_{i}"] for i in range(4)]; groups = [HNO[f"Group_G{i:02d}"] for i in range(1, 25)]
cards = []


def card(cat, ar, en, ctx, gold, expect_abstain=False, kind=None):
    cards.append({"id": f"H{len(cards) + 1:02d}", "category": cat, "scenario_ar": ar, "scenario_en": en, "context": ctx,
                  "gold": gold, "answerable": kind not in ("juristic", "non-existent", "out-of-scope"),
                  "expect_abstain": expect_abstain, "meta": {"kind": kind} if kind else {}})


def route_card(cat, o, d, day, hm, prof, ar, en):
    t = datetime.fromisoformat(f"{DAY[day]}T{hm}")
    dec = decide(kg, o, d, t, prof)
    ctx = {"time": t.isoformat(timespec="minutes"), "location": local(o), "destination": local(d), "profile": local(prof)}
    gold = [] if dec["best"] is None else [{"type": "route", "origin": local(o), "destination": local(d), "profile": local(prof),
                                            "depart": ctx["time"], "segments": [local(s) for s in dec["best"]["segments"]]}]
    card(cat, ar, en, ctx, gold, expect_abstain=dec["best"] is None)


# Q1 location (10): 5 nearest facility, 5 site of a facility
for i, c in enumerate(random.sample(mina + arafat, 5)):
    cls, ar_t, en_t = (HNO.MedicalPoint, "أقرب مركز صحي", "the nearest medical centre") if i % 2 == 0 else (HNO.MetroStation, "أقرب محطة قطار المشاعر", "the nearest metro station")
    n, _ = nearest(kg, c, cls)
    card("Q1", f"أنت في {L(c, 'ar')}، وتريد أن تعرف {ar_t} إليك.", f"You are in {L(c, 'en')} and want to know {en_t}.",
         {"time": None, "location": local(c)}, [{"type": "nearest", "from": local(c), "facility_type": local(cls), "facility": local(n)}])
facs = [u for u in kg.named if HNO.Facility in kg.types[u] and u not in mina + arafat]
for f in random.sample(sorted(facs, key=str), 5):
    card("Q1", f"سمعت عن «{L(f, 'ar')}» ولا تعرف في أي مشعر يقع.", f"You heard of '{L(f, 'en')}' and do not know in which holy site it is.",
         {"time": None}, [{"type": "location", "facility": local(f), "site": local(kg.site_of[f])}])
# Q2 route (10)
for _ in range(10):
    o = random.choice(mina); d = random.choice(JL); day = random.choice([10, 11, 12]); hm = random.choice(["07:00", "09:30", "13:00", "15:30", "19:00", "22:00"])
    route_card("Q2", o, d, day, hm, HNO.Profile_Standard,
               f"أنت في {L(o, 'ar')}، وتريد الذهاب إلى {L(d, 'ar')} {DAY_AR[day]} الساعة {hm}، وتسأل عن الطريق.",
               f"You are in {L(o, 'en')} and want to go to {L(d, 'en')} on {day} Dhu al-Hijjah at {hm}; you ask for the way.")
# Q3 temporal (8): dispatch slot of a group
rit = {HNO.StoningAqabah: ("رمي جمرة العقبة يوم العيد", "the stoning of Jamarat al-Aqabah on Eid day"),
       HNO.StoningDay11: ("رمي الجمرات الثلاث يوم 11", "the stoning of the three Jamarat on the 11th"),
       HNO.StoningDay12: ("رمي الجمرات الثلاث يوم 12", "the stoning of the three Jamarat on the 12th")}
for g in random.sample(groups, 8):
    r = random.choice(list(rit)); sl = kg.slots[(g, r)]
    card("Q3", f"أنت من {L(g, 'ar')}، وتريد أن تعرف متى موعد تفويجكم لـ{rit[r][0]}، وفي أي دور من الجمرات.",
         f"You belong to {L(g, 'en')} and want to know when your dispatch slot for {rit[r][1]} starts and at which Jamarat level.",
         {"time": None, "group": local(g)},
         [{"type": "schedule", "group": local(g), "ritual": local(r), "start": sl["win"][0].isoformat(timespec="minutes"), "facility": local(sl["facility"])}])
# Q4 accessibility (8)
hill = [HNO[f"Camp_M{i:02d}"] for i in (5, 10, 15, 20)]
for k in range(8):
    o = random.choice(hill + mina); d = random.choice(JL + [HNO.Med_MinaC, HNO.Med_Jamarat]); day = random.choice([10, 11, 12]); hm = random.choice(["07:30", "10:00", "14:00", "18:30", "21:00"])
    if k % 2 == 0:
        route_card("Q4", o, d, day, hm, HNO.Profile_Wheelchair,
                   f"تدفع كرسيًا متحركًا لأحد أقاربك من {L(o, 'ar')} إلى {L(d, 'ar')} {DAY_AR[day]} الساعة {hm}، وتريد طريقًا بلا درج.",
                   f"You push a relative in a wheelchair from {L(o, 'en')} to {L(d, 'en')} on {day} Dhu al-Hijjah at {hm} and need a route without stairs.")
    else:
        route_card("Q4", o, d, day, hm, HNO.Profile_Elderly,
                   f"أنت حاج كبير في السن في {L(o, 'ar')}، وتريد الوصول إلى {L(d, 'ar')} {DAY_AR[day]} الساعة {hm} بأيسر طريق.",
                   f"You are an elderly pilgrim in {L(o, 'en')} and want to reach {L(d, 'en')} on {day} Dhu al-Hijjah at {hm} in the easiest way.")
# Q5 multi-hop (6): 3 nearest station + its site, 3 slot + route
for c in random.sample(mina + arafat, 3):
    n, _ = nearest(kg, c, HNO.MetroStation)
    card("Q5", f"أنت في {L(c, 'ar')}، وتريد أن تعرف أقرب محطة قطار إليك، وفي أي مشعر تقع هذه المحطة.",
         f"You are in {L(c, 'en')} and want to know the nearest metro station and in which holy site that station is.",
         {"time": None, "location": local(c)},
         [{"type": "nearest", "from": local(c), "facility_type": "MetroStation", "facility": local(n)},
          {"type": "location", "facility": local(n), "site": local(kg.site_of[n])}])
for g in random.sample(groups, 3):
    r = random.choice(list(rit)); sl = kg.slots[(g, r)]
    camp = [c for c in kg.group_camps[g] if kg.site_of[c] == HNO.Mina][0]; t = sl["win"][0]
    dec = decide(kg, camp, sl["facility"], t, HNO.Profile_Standard, sl["win"][1])
    gold = [{"type": "schedule", "group": local(g), "ritual": local(r), "start": t.isoformat(timespec="minutes"), "facility": local(sl["facility"])}]
    if dec["best"] is not None:
        gold.append({"type": "route", "origin": local(camp), "destination": local(sl["facility"]), "profile": "Profile_Standard",
                     "depart": t.isoformat(timespec="minutes"), "segments": [local(s) for s in dec["best"]["segments"]]})
    card("Q5", f"أنت من {L(g, 'ar')} وتسكن في {L(camp, 'ar')}. تريد أن تعرف موعد تفويجكم لـ{rit[r][0]}، وكيف تصلون من المخيم إلى دور الجمرات المخصص لكم في ذلك الوقت.",
         f"You belong to {L(g, 'en')} and stay in {L(camp, 'en')}. You want your dispatch slot for {rit[r][1]} and how to get from the camp to your assigned Jamarat level at that time.",
         {"time": t.isoformat(timespec="minutes"), "group": local(g), "ritual": local(r), "location": local(camp)}, gold, expect_abstain=dec["best"] is None)
# Q6 must be refused (8)
for ar, en in [("تريد أن تعرف هل يصح حجك إذا خرجت من مزدلفة قبل منتصف الليل.", "You want to know whether your Hajj is valid if you leave Muzdalifah before midnight."),
               ("تسأل عن حكم توكيل شخص آخر يرمي عنك الجمرات.", "You ask whether it is permissible to delegate the stoning to someone else."),
               ("تسأل هل عليك فدية إذا فاتك الرمي يوم 11.", "You ask whether you owe an expiation if you missed the stoning on the 11th."),
               ("تسأل عن مطعم مشهور قريب من المسجد الحرام.", "You ask about a well-known restaurant near Masjid al-Haram.")]:
    card("Q6", ar, en, {"time": None}, [], expect_abstain=True, kind="juristic" if "مطعم" not in ar else "out-of-scope")
for ar, en in [("تبحث عن «مخيم منى م45».", "You look for 'Mina camp M45'."),
               ("تريد الذهاب إلى «محطة منى 7».", "You want to go to 'Mina Station 7'."),
               ("تريد موعد تفويج «المجموعة ج60».", "You want the dispatch slot of 'group G60'."),
               ("تبحث عن «البوابة رقم 99» في منى.", "You look for 'gate 99' in Mina.")]:
    card("Q6", ar, en, {"time": None}, [], expect_abstain=True, kind="non-existent")
json.dump(cards, open(os.path.join(ROOT, "bench", "human_scenarios.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print(len(cards), Counter(c["category"] for c in cards), "infeasible:", sum(c["expect_abstain"] and c["answerable"] for c in cards))
