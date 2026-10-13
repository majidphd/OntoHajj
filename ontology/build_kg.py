"""Builds the prototype HNO knowledge graph (ABox) for the 1447H (2026) season.

Site names, the Mashaer metro line order and the coarse layout are public;
fine-grained operational data (camps, path segments, closures, dispatch slots,
crowd observations) are SYNTHETIC because the authorities do not release them.
All synthetic assertions are linked to hno:Src_HNOPrototype via hno:derivedFrom.
"""
import random, json
from datetime import datetime, timedelta
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, SKOS, OWL

HNO = Namespace("https://w3id.org/ontohajj/hno#")
TIME = Namespace("http://www.w3.org/2006/time#")
random.seed(1447)

g = Graph()
g.bind("hno", HNO); g.bind("time", TIME); g.bind("skos", SKOS)

DAY = {8: "2026-05-25", 9: "2026-05-26", 10: "2026-05-27", 11: "2026-05-28", 12: "2026-05-29", 13: "2026-05-30"}
def dt(day, hhmm):
    return datetime.fromisoformat(f"{DAY[day]}T{hhmm}:00")
SEASON_START, SEASON_END = dt(8, "00:00"), dt(13, "23:59")

SRC_PROTO, SRC_GAZ = HNO.Src_HNOPrototype, HNO.Src_SiteGazetteer
for s, en, ar in [(SRC_PROTO, "HNO prototype operational data, season 1447H (synthetic)", "بيانات تشغيلية افتراضية لنموذج HNO لموسم 1447هـ"),
                  (SRC_GAZ, "Gazetteer of holy-site names and public metro line order", "معجم أسماء المشاعر وترتيب محطات قطار المشاعر المعلن")]:
    g.add((s, RDF.type, HNO.Source)); g.add((s, RDFS.label, Literal(en, lang="en"))); g.add((s, RDFS.label, Literal(ar, lang="ar")))

def ent(local, cls, en, ar, alt=(), src=SRC_PROTO, located=None):
    u = HNO[local]
    g.add((u, RDF.type, cls)); g.add((u, RDF.type, OWL.NamedIndividual))
    g.add((u, RDFS.label, Literal(en, lang="en"))); g.add((u, RDFS.label, Literal(ar, lang="ar")))
    for a in alt:
        g.add((u, SKOS.altLabel, Literal(a)))
    g.add((u, HNO.derivedFrom, src))
    if located is not None:
        g.add((u, HNO.locatedIn, located))
    return u

wcount = [0]
def window(start, end, name=None):
    wcount[0] += 1
    w = HNO[name or f"W{wcount[0]:04d}"]
    b, e = HNO[f"{w.split('#')[1]}_b"], HNO[f"{w.split('#')[1]}_e"]
    g.add((w, RDF.type, HNO.TimeWindow)); g.add((b, RDF.type, TIME.Instant)); g.add((e, RDF.type, TIME.Instant))
    g.add((w, TIME.hasBeginning, b)); g.add((w, TIME.hasEnd, e))
    g.add((b, TIME.inXSDDateTime, Literal(start.isoformat(), datatype=XSD.dateTime)))
    g.add((e, TIME.inXSDDateTime, Literal(end.isoformat(), datatype=XSD.dateTime)))
    return w

# ---------------- holy sites, modes, profiles ----------------
HARAM = ent("MasjidAlHaram", HNO.HolySite, "Masjid al-Haram", "المسجد الحرام", ("Haram", "Al-Haram", "الحرم"), SRC_GAZ)
MINA = ent("Mina", HNO.HolySite, "Mina", "منى", ("Mena", "مني"), SRC_GAZ)
MUZ = ent("Muzdalifah", HNO.HolySite, "Muzdalifah", "مزدلفة", ("Muzdalifa", "Muzdalefah", "مزدلفه"), SRC_GAZ)
ARAFAT = ent("Arafat", HNO.HolySite, "Arafat", "عرفات", ("Arafah", "Arafa", "عرفة", "عرفه"), SRC_GAZ)
WALK = ent("Walking", HNO.TransportMode, "walking", "المشي", ("on foot", "مشيا", "سيرا"), SRC_GAZ)
METRO = ent("MashaerMetro", HNO.TransportMode, "Al Mashaaer Al Mugaddassah Metro", "قطار المشاعر المقدسة", ("metro", "train", "القطار", "المترو"), SRC_GAZ)
BUS = ent("Bus", HNO.TransportMode, "bus", "الحافلات", ("buses", "الباص", "حافلة"), SRC_GAZ)
P_STD = ent("Profile_Standard", HNO.PilgrimProfile, "standard pilgrim", "حاج دون احتياجات خاصة")
P_WC = ent("Profile_Wheelchair", HNO.PilgrimProfile, "wheelchair user", "مستخدم كرسي متحرك", ("wheelchair", "كرسي متحرك"))
P_ELD = ent("Profile_Elderly", HNO.PilgrimProfile, "elderly pilgrim", "حاج كبير السن", ("elderly", "كبار السن"))

# ---------------- facilities ----------------
ent("MasjidNamirah", HNO.Mosque, "Masjid Namirah", "مسجد نمرة", ("Namira", "Namirah Mosque", "نمره"), SRC_GAZ, ARAFAT)
ent("MasjidAlMasharAlHaram", HNO.Mosque, "Masjid al-Mashar al-Haram", "مسجد المشعر الحرام", ("Mashar al-Haram",), SRC_GAZ, MUZ)
ent("MasjidAlKhayf", HNO.Mosque, "Masjid al-Khayf", "مسجد الخيف", ("Khayf", "Kheef", "الخيف"), SRC_GAZ, MINA)
for g_local, en, ar in [("Gate_KingAbdulaziz", "King Abdulaziz Gate", "باب الملك عبدالعزيز"), ("Gate_KingFahd", "King Fahd Gate", "باب الملك فهد"), ("Gate_AlSalam", "Al-Salam Gate", "باب السلام")]:
    ent(g_local, HNO.Gate, en, ar, (), SRC_GAZ, HARAM)
stations = []
for site, site_en, site_ar in [(ARAFAT, "Arafat", "عرفات"), (MUZ, "Muzdalifah", "مزدلفة"), (MINA, "Mina", "منى")]:
    for i in (1, 2, 3):
        stations.append(ent(f"Station_{site_en}{i}", HNO.MetroStation, f"{site_en} Station {i}", f"محطة {site_ar} {i}",
                            (f"{site_en} {i}", f"{site_en}{i}", f"{site_ar} {i}", f"محطة {site_ar}{i}"), SRC_GAZ, site))
JL = []
for i, (en, ar) in enumerate([("Jamarat ground level", "الدور الأرضي للجمرات"), ("Jamarat level 1", "الدور الأول للجمرات"),
                              ("Jamarat level 2", "الدور الثاني للجمرات"), ("Jamarat level 3", "الدور الثالث للجمرات")]):
    JL.append(ent(f"JamaratLevel_{i}", HNO.JamaratLevel, en, ar, (f"Jamarat L{i}", f"Jamarat floor {i}"), SRC_PROTO, MINA))
BUS_HARAM = ent("BusStation_Haram", HNO.BusStation, "Haram bus station", "محطة حافلات الحرم", (), SRC_PROTO, HARAM)
BUS_MINA = ent("BusStation_MinaNorth", HNO.BusStation, "Mina North bus station", "محطة حافلات منى الشمالية", (), SRC_PROTO, MINA)
meds = {}
for local, en, ar, site in [("Med_MinaA", "Mina Medical Centre A", "مركز منى الطبي أ", MINA), ("Med_MinaB", "Mina Medical Centre B", "مركز منى الطبي ب", MINA),
                            ("Med_MinaC", "Mina Medical Centre C", "مركز منى الطبي ج", MINA), ("Med_Jamarat", "Jamarat Medical Centre", "مركز الجمرات الطبي", MINA),
                            ("Med_Muzdalifah", "Muzdalifah Medical Centre", "مركز مزدلفة الطبي", MUZ), ("Med_ArafatWest", "Arafat West Medical Centre", "مركز عرفات الغربي الطبي", ARAFAT),
                            ("Med_ArafatEast", "Arafat East Medical Centre", "مركز عرفات الشرقي الطبي", ARAFAT)]:
    meds[local] = ent(local, HNO.MedicalPoint, en, ar, (), SRC_PROTO, site)
wps = {}
for local, site in [("WP_MinaA", MINA), ("WP_MinaB", MINA), ("WP_MinaC", MINA), ("WP_JamaratPlaza", MINA), ("WP_MinaMuzBorder", MINA),
                    ("WP_MuzCentral", MUZ), ("WP_MuzArafatBorder", MUZ), ("WP_ArafatCentral", ARAFAT), ("WP_ArafatEast", ARAFAT), ("WP_MakkahTunnel", HARAM)]:
    nm = local.replace("WP_", "")
    wps[local] = ent(local, HNO.Waypoint, f"Waypoint {nm}", f"نقطة عبور {nm}", (), SRC_PROTO, site)
camps_mina = {}
for i in range(1, 25):
    sector = "ABC"[(i - 1) // 8]
    camps_mina[i] = ent(f"Camp_M{i:02d}", HNO.Camp, f"Mina camp M{i:02d}", f"مخيم منى م{i:02d}", (f"M{i:02d}", f"camp M{i:02d}", f"camp {i}", f"مخيم م{i:02d}", f"مخيم {i}", f"م{i:02d}"), SRC_PROTO, MINA)
camps_arafat = {}
for i in range(1, 13):
    camps_arafat[i] = ent(f"Camp_A{i:02d}", HNO.Camp, f"Arafat camp A{i:02d}", f"مخيم عرفات ع{i:02d}", (f"A{i:02d}", f"camp A{i:02d}", f"مخيم ع{i:02d}", f"ع{i:02d}"), SRC_PROTO, ARAFAT)

# ---------------- path segments ----------------
segs = []
def seg(a, b, length, mode=WALK, stepfree=True, shaded=False, minutes=None, both=True, cap=6000, tag=""):
    out = []
    for (x, y, d) in ([(a, b, "f"), (b, a, "r")] if both else [(a, b, "f")]):
        n = len(segs) + 1
        s = HNO[f"Seg_{n:03d}"]
        g.add((s, RDF.type, HNO.PathSegment)); g.add((s, HNO.fromLocation, x)); g.add((s, HNO.toLocation, y))
        g.add((s, HNO.usesMode, mode)); g.add((s, HNO.lengthMeters, Literal(length, datatype=XSD.decimal)))
        mins = minutes if minutes is not None else round(length / 50.0, 1)   # 50 m/min crowded walking speed
        g.add((s, HNO.travelMinutes, Literal(mins, datatype=XSD.decimal)))
        g.add((s, HNO.shaded, Literal(bool(shaded))))
        g.add((s, HNO.capacityPerHour, Literal(cap, datatype=XSD.integer)))
        g.add((s, HNO.derivedFrom, SRC_PROTO))
        g.add((s, HNO.accessibleFor, P_STD))
        if stepfree:
            g.add((s, HNO.accessibleFor, P_WC)); g.add((s, HNO.accessibleFor, P_ELD))
        segs.append({"uri": s, "from": x, "to": y, "mode": mode, "len": length, "min": mins, "stepfree": stepfree, "tag": tag, "dir": d})
        out.append(s)
    return out

W = wps; S = {str(s).split("#")[1]: s for s in stations}
# Mina camps -> sector waypoints (every 5th camp is on a hillside: short stepped path + longer step-free path)
for i, c in camps_mina.items():
    wp = W[f"WP_Mina{'ABC'[(i - 1) // 8]}"]
    if i % 5 == 0:
        seg(c, wp, 180 + 20 * (i % 8), stepfree=False, tag="hillside-stairs")
        seg(c, wp, 650 + 30 * (i % 8), stepfree=True, shaded=True, tag="hillside-ramp")
    else:
        seg(c, wp, 200 + 45 * ((i - 1) % 8), stepfree=True, shaded=(i % 2 == 0))
seg(W["WP_MinaA"], W["WP_MinaB"], 1200, shaded=True); seg(W["WP_MinaB"], W["WP_MinaC"], 1200, shaded=True)
seg(W["WP_MinaC"], W["WP_JamaratPlaza"], 800, shaded=True, cap=9000, tag="jamarat-approach")
seg(W["WP_MinaA"], S["Station_Mina1"], 300); seg(W["WP_MinaB"], S["Station_Mina2"], 300); seg(W["WP_MinaC"], S["Station_Mina3"], 300)
seg(S["Station_Mina3"], W["WP_JamaratPlaza"], 400, shaded=True, cap=9000, tag="jamarat-approach")
# Jamarat levels (one-way returns restricted at peak, see below)
j_in, j_out = {}, {}
for lvl, (src, length, sf) in enumerate([(W["WP_JamaratPlaza"], 150, True), (W["WP_JamaratPlaza"], 400, True),
                                          (W["WP_JamaratPlaza"], 200, False), (W["WP_JamaratPlaza"], 250, False)]):
    a, b = seg(src, JL[lvl], length, stepfree=sf, cap=12000, tag=f"jamarat-L{lvl}")
    j_in[lvl], j_out[lvl] = a, b
seg(W["WP_MinaB"], JL[2], 1500, stepfree=True, shaded=True, cap=10000, tag="jamarat-L2-east-ramp")
seg(W["WP_MinaA"], JL[3], 2600, stepfree=True, shaded=True, cap=10000, tag="jamarat-L3-walkway")
seg(W["WP_JamaratPlaza"], meds["Med_Jamarat"], 120)
seg(W["WP_MinaA"], meds["Med_MinaA"], 250); seg(W["WP_MinaB"], meds["Med_MinaB"], 350); seg(W["WP_MinaC"], meds["Med_MinaC"], 300, stepfree=False)
seg(W["WP_MinaC"], meds["Med_MinaB"], 900)
seg(W["WP_MinaB"], HNO.MasjidAlKhayf, 500, shaded=True)
# Mina <-> Muzdalifah <-> Arafat pedestrian corridor
seg(W["WP_MinaA"], W["WP_MinaMuzBorder"], 1000, shaded=True, cap=20000)
seg(W["WP_MinaMuzBorder"], W["WP_MuzCentral"], 1500, shaded=True, cap=20000, tag="mina-muz")
seg(W["WP_MuzCentral"], S["Station_Muzdalifah1"], 300); seg(W["WP_MuzCentral"], S["Station_Muzdalifah2"], 450); seg(W["WP_MuzCentral"], S["Station_Muzdalifah3"], 500)
seg(W["WP_MuzCentral"], HNO.MasjidAlMasharAlHaram, 300); seg(W["WP_MuzCentral"], meds["Med_Muzdalifah"], 200)
ar_muz_walk = seg(W["WP_ArafatCentral"], W["WP_MuzArafatBorder"], 2000, shaded=True, cap=25000, tag="arafat-muz")
seg(W["WP_MuzArafatBorder"], W["WP_MuzCentral"], 2500, shaded=True, cap=25000, tag="arafat-muz")
seg(W["WP_ArafatCentral"], W["WP_ArafatEast"], 1300, shaded=True)
seg(W["WP_ArafatCentral"], HNO.MasjidNamirah, 600, shaded=True)
seg(W["WP_ArafatCentral"], S["Station_Arafat1"], 400); seg(W["WP_ArafatCentral"], S["Station_Arafat2"], 500); seg(W["WP_ArafatEast"], S["Station_Arafat3"], 350)
seg(W["WP_ArafatCentral"], meds["Med_ArafatWest"], 300); seg(W["WP_ArafatEast"], meds["Med_ArafatEast"], 250)
for i, c in camps_arafat.items():
    seg(c, W["WP_ArafatCentral"] if i <= 6 else W["WP_ArafatEast"], 250 + 60 * ((i - 1) % 6), stepfree=(i != 9), shaded=True)
# Makkah links
seg(W["WP_MinaA"], W["WP_MakkahTunnel"], 2500, shaded=True, cap=15000, tag="pedestrian-tunnel")
seg(W["WP_MakkahTunnel"], HNO.Gate_KingAbdulaziz, 2300, shaded=True, cap=15000)
seg(BUS_HARAM, HNO.Gate_KingAbdulaziz, 400); seg(BUS_HARAM, HNO.Gate_KingFahd, 500); seg(BUS_HARAM, HNO.Gate_AlSalam, 700)
seg(BUS_MINA, W["WP_MinaA"], 300)
bus_segs = seg(BUS_MINA, BUS_HARAM, 7000, mode=BUS, minutes=40, cap=8000, tag="bus-mina-haram")
# Metro line: Arafat1-2-3, Muzdalifah1-2-3, Mina1-2-3 (public line order), forward = towards Arafat
order = ["Station_Arafat1", "Station_Arafat2", "Station_Arafat3", "Station_Muzdalifah1", "Station_Muzdalifah2", "Station_Muzdalifah3", "Station_Mina1", "Station_Mina2", "Station_Mina3"]
metro_to_arafat, metro_from_arafat = [], []
for a, b in zip(order[:-1], order[1:]):
    toward_mina, toward_arafat = seg(S[a], S[b], 2200, mode=METRO, minutes=4, cap=72000, tag="metro")
    metro_from_arafat.append((a, b, toward_mina)); metro_to_arafat.append((b, a, toward_arafat))

# ---------------- restrictions (closures, operating windows, one-way rules) ----------------
def restrict(s, start, end):
    g.add((s, HNO.restrictedDuring, window(start, end)))
def open_only(s, windows):
    """restrict a segment outside the given open windows within the season"""
    cur = SEASON_START
    for (a, b) in sorted(windows):
        if a > cur: restrict(s, cur, a - timedelta(minutes=1))
        cur = b + timedelta(minutes=1)
    if cur < SEASON_END: restrict(s, cur, SEASON_END)
SUNSET9 = dt(9, "18:55")
for a, b, s in metro_to_arafat:          # Mina -> Arafat direction
    open_only(s, [(dt(8, "12:00"), dt(9, "11:00"))])
for a, b, s in metro_from_arafat:        # Arafat -> Muzdalifah -> Mina direction
    if a.startswith("Station_Arafat") or b.startswith("Station_Muzdalifah1"):
        open_only(s, [(SUNSET9, dt(10, "02:00"))])
    elif a.startswith("Station_Muzdalifah"):
        open_only(s, [(SUNSET9, dt(10, "10:00"))])
    else:                                 # Mina1 -> Mina2 -> Mina3 internal shuttle days 10-13
        open_only(s, [(SUNSET9, dt(10, "10:00"))] + [(dt(d, "06:00"), dt(d, "23:59")) for d in (10, 11, 12, 13)])
for a, b, s in metro_to_arafat:
    if a.startswith("Station_Mina") and b.startswith("Station_Mina"):   # Mina internal, reverse direction
        g.remove((s, HNO.restrictedDuring, None))
        open_only(s, [(dt(8, "12:00"), dt(9, "11:00"))] + [(dt(d, "06:00"), dt(d, "23:59")) for d in (10, 11, 12, 13)])
# leaving Arafat on foot is not permitted before sunset on 9 Dhu al-Hijjah (operational rule)
restrict(ar_muz_walk[0], dt(9, "00:00"), dt(9, "18:54"))
for s in [x["uri"] for x in segs if x["tag"] == "arafat-muz" and x["dir"] == "f"]:
    restrict(s, dt(9, "00:00"), dt(9, "18:54"))
# one-way flow at the Jamarat: returns from levels to the plaza closed during peaks
for lvl in range(4):
    for d in (10, 11, 12):
        restrict(j_out[lvl], dt(d, "06:00") if d == 10 else dt(d, "12:00"), dt(d, "11:59") if d == 10 else dt(d, "17:59"))
# bus service Mina<->Haram suspended on 9 DH (pilgrims in Arafat) and overnight 9-10
restrict(bus_segs[0], dt(9, "00:00"), dt(10, "05:59")); restrict(bus_segs[1], dt(9, "00:00"), dt(10, "05:59"))
# hillside stairs closed during the night for safety
for x in segs:
    if x["tag"] == "hillside-stairs":
        for d in (10, 11, 12):
            restrict(x["uri"], dt(d, "00:00"), dt(d, "04:59"))

# ---------------- rituals ----------------
rit = {}
for local, cls, en, ar, site, (d1, t1), (d2, t2) in [
    ("TarwiyahStay", HNO.StayRitual, "Tarwiyah stay in Mina", "المبيت بمنى يوم التروية", MINA, (8, "06:00"), (9, "06:00")),
    ("ArafatStanding", HNO.StayRitual, "Standing at Arafat", "الوقوف بعرفة", ARAFAT, (9, "12:20"), (9, "18:55")),
    ("MuzdalifahOvernight", HNO.StayRitual, "Overnight stay in Muzdalifah", "المبيت بمزدلفة", MUZ, (9, "18:55"), (10, "06:00")),
    ("StoningAqabah", HNO.StoningRitual, "Stoning of Jamarat al-Aqabah", "رمي جمرة العقبة", None, (10, "06:00"), (10, "23:59")),
    ("TawafIfadah", HNO.Ritual, "Tawaf al-Ifadah", "طواف الإفاضة", HARAM, (10, "06:00"), (13, "23:59")),
    ("StoningDay11", HNO.StoningRitual, "Stoning of the three Jamarat (11 Dhu al-Hijjah)", "رمي الجمرات الثلاث يوم 11", None, (11, "12:30"), (11, "23:59")),
    ("StoningDay12", HNO.StoningRitual, "Stoning of the three Jamarat (12 Dhu al-Hijjah)", "رمي الجمرات الثلاث يوم 12", None, (12, "12:30"), (12, "23:59")),
    ("StoningDay13", HNO.StoningRitual, "Stoning of the three Jamarat (13 Dhu al-Hijjah)", "رمي الجمرات الثلاث يوم 13", None, (13, "12:30"), (13, "18:00"))]:
    r = ent(local, cls, en, ar, (), SRC_PROTO)
    if site is not None: g.add((r, HNO.performedAt, site))
    else:
        for l in JL: g.add((r, HNO.performedAt, l))
    g.add((r, HNO.hasTimeWindow, window(dt(d1, t1), dt(d2, t2), f"W_{local}")))
    rit[local] = r
chain = ["TarwiyahStay", "ArafatStanding", "MuzdalifahOvernight", "StoningAqabah", "StoningDay11", "StoningDay12", "StoningDay13"]
for a, b in zip(chain[:-1], chain[1:]):
    g.add((rit[a], HNO.precedes, rit[b]))
g.add((rit["StoningAqabah"], HNO.precedes, rit["TawafIfadah"]))

# ---------------- groups and dispatch (Tafweej) slots ----------------
groups = {}
slot_times = ["06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"]
for i in range(1, 25):
    gr = ent(f"Group_G{i:02d}", HNO.PilgrimGroup, f"group G{i:02d}", f"المجموعة ج{i:02d}", (f"G{i:02d}", f"group G{i:02d}", f"ج{i:02d}", f"المجموعة ج{i:02d}", f"group {i}", f"مجموعة {i}"))
    g.add((gr, HNO.assignedCamp, camps_mina[i])); g.add((gr, HNO.assignedCamp, camps_arafat[(i + 1) // 2]))
    groups[i] = gr
    for d, ritual in [(10, "StoningAqabah"), (11, "StoningDay11"), (12, "StoningDay12")]:
        if d == 10:
            t = slot_times[(i - 1) % 7]
        else:
            t = ["12:30", "14:30", "16:30", "18:30", "20:30", "22:00"][(i + d) % 6]
        start = dt(d, t); end = start + timedelta(hours=2) - timedelta(minutes=1)
        lvl = (i + d) % 4
        sl = HNO[f"Slot_G{i:02d}_D{d}"]
        g.add((sl, RDF.type, HNO.TafweejSlot)); g.add((sl, HNO.slotFacility, JL[lvl])); g.add((sl, HNO.slotRitual, rit[ritual]))
        g.add((sl, HNO.hasTimeWindow, window(start, end, f"W_Slot_G{i:02d}_D{d}"))); g.add((sl, HNO.derivedFrom, SRC_PROTO))
        g.add((sl, HNO.hijriDay, Literal(d, datatype=XSD.integer)))
        g.add((gr, HNO.hasSlot, sl))

# ---------------- crowd observations (3-hour slots, days 9-12) ----------------
LV = {}
for k, (en, ar, sc) in enumerate([("low", "منخفضة", 1), ("medium", "متوسطة", 2), ("high", "مرتفعة", 3)]):
    LV[sc] = ent(f"Crowd_{en.capitalize()}", HNO.CrowdLevel, f"{en} density", f"كثافة {ar}", (), SRC_PROTO)
    g.add((LV[sc], HNO.densityScore, Literal(sc, datatype=XSD.integer)))
nobs = 0
for x in segs:
    if x["mode"] != WALK: continue
    for d in (9, 10, 11, 12):
        for h in range(0, 24, 3):
            start = dt(d, f"{h:02d}:00"); end = start + timedelta(hours=3) - timedelta(minutes=1)
            base = 1
            if "jamarat" in x["tag"]:
                base = 3 if (d == 10 and 6 <= h < 15) or (d in (11, 12) and 12 <= h < 21) else 2
            elif x["tag"] in ("arafat-muz", "mina-muz"):
                base = 3 if (d == 9 and h >= 18) or (d == 10 and h < 6) else 1
            lvl = max(1, min(3, base + random.choice([-1, 0, 0, 0, 1])))
            nobs += 1
            o = HNO[f"Obs_{nobs:05d}"]
            g.add((o, RDF.type, HNO.CrowdObservation)); g.add((o, HNO.onSegment, x["uri"]))
            g.add((o, HNO.during, window(start, end))); g.add((o, HNO.hasLevel, LV[lvl])); g.add((o, HNO.derivedFrom, SRC_PROTO))

g.serialize("../data/hno-kg.ttl", format="turtle")
meta = {"segments": len(segs), "observations": nobs, "triples": len(g)}
json.dump(meta, open("../data/kg_meta.json", "w", encoding="utf-8"), indent=1)
print(meta)
