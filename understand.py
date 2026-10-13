"""Layer 2: bilingual normalisation, ontology-based entity linking (Eq. 1) and intent typing."""
import re
from rapidfuzz import fuzz
from .kg import HNO, load

AR_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize(text):
    t = text.translate(AR_DIGITS)
    t = AR_DIAC.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"(?<=[a-z\u0621-\u064a])(?=\d)", " ", t)   # "m07" -> "m 07", "منى3" -> "منى 3"
    return re.sub(r"\s+", " ", t).strip()


def codes(text):
    """identifier-like tokens (short tokens or tokens containing digits), which must match exactly"""
    return {t for t in normalize(text).split() if len(t) <= 2 or any(ch.isdigit() for ch in t)}


class Linker:
    """Lexical/fuzzy linker over rdfs:label and skos:altLabel (sim in Eq. 1; tau = 92)."""
    TAU = 92

    def __init__(self, kg=None):
        self.kg = kg or load()
        self.index = []          # (normalized label, uri)
        for u in self.kg.named:
            names = list(self.kg.labels[u].values()) + self.kg.alt[u]
            for n in names:
                nn = normalize(n)
                if len(nn) >= 2:
                    self.index.append((nn, u))
        self.index.sort(key=lambda x: -len(x[0]))
        # code-like mentions: camps, groups, stations, gates, levels (to detect non-existent entities)
        self.code_patterns = [
            (re.compile(r"\b(?:camp|مخيم)\s*(?:mina\s*|مني\s*|arafat\s*|عرفات\s*)?([mاعa]|م|ع)?\s*(\d{1,3})\b"), "camp"),
            (re.compile(r"\b(?:group|المجموعه|مجموعه)\s*([gج])?\s*(\d{1,3})\b"), "group"),
            (re.compile(r"\b(?:station|محطه)\s*(mina|muzdalifah|arafat|منى|مني|مزدلفه|عرفات)\s*(\d{1,2})\b"), "station"),
            (re.compile(r"\b(mina|muzdalifah|arafat|منى|مني|مزدلفه|عرفات)\s*station\s*(\d{1,2})\b"), "station"),
            (re.compile(r"\b(?:gate|باب)\s*(?:no\s*|رقم\s*)?(\d{1,3})\b"), "gate"),
            (re.compile(r"\b(?:level|floor|الدور)\s*(\d{1,2})\b"), "level"),
        ]

    def link(self, text):
        q = normalize(text)
        found, spans = [], []
        padded = f" {q} "
        for nn, u in self.index:                       # exact (longest first)
            i = padded.find(f" {nn} ")
            while i >= 0:
                span = (i, i + len(nn))
                if not any(a < span[1] and span[0] < b for a, b in spans):
                    spans.append(span); found.append(u)
                i = padded.find(f" {nn} ", i + 1)
        if not found:                                  # fuzzy fallback on word n-grams
            toks = q.split()
            for n in (4, 3, 2, 1):
                for k in range(len(toks) - n + 1):
                    cand = " ".join(toks[k:k + n])
                    if len(cand) < 4:
                        continue
                    for nn, u in self.index:
                        if (abs(len(nn) - len(cand)) <= 3 and fuzz.ratio(cand, nn) >= self.TAU and u not in found
                                and codes(cand) == codes(nn)):
                            found.append(u)
        unresolved = self._unresolved(q, found)
        return list(dict.fromkeys(found)), unresolved

    def _unresolved(self, q, found):
        """code-like mentions (e.g. 'camp M99', 'Mina station 5') that do not resolve to any KG entity"""
        bad = []
        for pat, kind in self.code_patterns:
            for m in pat.finditer(q):
                num = int(m.groups()[-1])
                ok = False
                if kind == "camp":
                    ok = any(n.startswith("Camp_") and int(n[-2:]) == num for n in map(lambda u: str(u).split("#")[1], found))
                elif kind == "group":
                    ok = any(str(u).endswith(f"Group_G{num:02d}") for u in found)
                elif kind == "station":
                    ok = num in (1, 2, 3)
                elif kind == "gate":
                    ok = False                         # numbered gates are not modelled in HNO
                elif kind == "level":
                    ok = 0 <= num <= 3
                if not ok:
                    bad.append(m.group(0))
        return bad


JURISTIC = ["valid", "permissible", "allowed to", "ruling", "fatwa", "sin", "expiation", "obligatory", "sunnah", "is it sunna",
            "يجوز", "حكم", "صحيح", "فتوى", "فديه", "كفاره", "واجب", "سنه", "يبطل", "دم", "اثم"]
ROUTE = ["how do i get", "how can i get", "how do i reach", "how do i go", "how do we get", "route", "way to", "which way", "get from", "go from", "best way",
         "كيف اصل", "كيف نصل", "كيف اذهب", "الطريق", "طريق", "كيف انتقل", "اسلك", "الوصول من", "اذهب من", "انتقل من"]
ACCESS = ["wheelchair", "كرسيا متحركا", "متحرك", "elderly", "step free", "step-free", "disab", "كرسي متحرك", "كبار السن", "كبير السن", "دون درج", "ذوي الاعاقه"]
TEMPORAL = ["when", "what time", "start", "starts", "end", "open", "until", "متى", "موعد", "وقت", "يبدا", "ينتهي", "ساعه", "يعمل"]
LOCATION = ["where", "which site", "which holy site", "holy site", "located", "nearest", "closest", "in which", "اين", "اقرب", "يقع", "في اي مشعر", "اي مشعر"]


def _has(q, kws):
    """whole-word (token sequence) matching on normalised text"""
    stripped = " ".join(w[1:] if w.startswith("و") and len(w) > 3 else w for w in q.split())   # conjunction clitic
    pads = (f" {q} ", f" {stripped} ")
    return any(f" {normalize(k)} " in p for k in kws for p in pads)


def intent(text):
    q = normalize(text)
    has = lambda kws: _has(q, kws)
    if has(JURISTIC):
        return "OutOfScope"
    route, tem, loc = has(ROUTE), has(TEMPORAL), has(LOCATION)
    two_loc = has(["nearest", "closest", "اقرب"]) and has(["which holy site", "in which", "اي مشعر"])
    if (route and tem) or (loc and tem) or two_loc:
        return "MultiHop"
    if has(ACCESS) and (route or has(["route", "طريق", "الطريق"])):
        return "Accessibility"
    if route:
        return "Route"
    if tem:
        return "Temporal"
    if loc:
        return "Location"
    return "Procedural"
