"""Layer 5: claim-level neuro-symbolic verification (Eq. 7).

Claims (JSON) are mapped to RDF (tau) and validated with pySHACL against the SHACL shapes
C1-C8 over the ontology and the operational part of the knowledge graph. The nearest-facility
relation (C9) is checked procedurally on the path network."""
import re
from datetime import datetime, timedelta
from rdflib import Graph, Literal, URIRef, BNode, Namespace
from rdflib.namespace import RDF, XSD, RDFS
from pyshacl import validate
from .kg import HNO, TIME, local, load
from .retrieve import resolve, nearest
from .understand import Linker

SH = Namespace("http://www.w3.org/ns/shacl#")
CN = Namespace("https://w3id.org/ontohajj/claim#")
SAFETY = {"C2", "C3", "C4", "C5", "C6", "C8", "C9"}


def _core(kg):
    """ontology + KG without the bulky crowd observations (not needed by any shape)"""
    if not hasattr(kg, "_core"):
        obs = set(kg.g.subjects(RDF.type, HNO.CrowdObservation))
        wins = set(kg.g.objects(None, HNO.during))
        drop = obs | wins | {i for w in wins for i in kg.g.objects(w, None) if isinstance(i, URIRef)}
        core = Graph()
        for t in kg.g:
            if t[0] not in drop:
                core.add(t)
        kg._core = core
    return kg._core


def _t(x):
    if x is None:
        return None
    x = str(x)
    try:
        return datetime.fromisoformat(x[:16])
    except ValueError:
        return None


class Verifier:
    def __init__(self, kg=None, linker=None):
        self.kg = kg or load()
        self.linker = linker or Linker(self.kg)
        self.core = _core(self.kg)

    # ---------------------------------------------------------------
    def to_rdf(self, claims, default_profile=HNO.Profile_Standard):
        kg, R = self.kg, (lambda s: resolve(self.kg, self.linker, s))
        cg, owner, nodes = Graph(), {}, []
        for i, c in enumerate(claims):
            n = CN[f"c{i}"]; nodes.append(n); owner[n] = i
            typ = c.get("type")
            src = c.get("source")
            if src:
                cg.add((n, HNO.derivedFrom, HNO[str(src).replace("hno:", "")]))
            mention = lambda u: cg.add((n, HNO.mentions, u)) if u is not None else None
            if typ == "location":
                f, s = R(c.get("facility")), R(c.get("site"))
                cg.add((n, RDF.type, HNO.LocationClaim)); cg.add((n, HNO.aboutFacility, f)); cg.add((n, HNO.claimedSite, s))
                mention(f); mention(s)
            elif typ == "nearest":
                fr, f = R(c.get("from")), R(c.get("facility"))
                cg.add((n, RDF.type, HNO.NearestClaim)); cg.add((n, HNO.fromPoint, fr)); cg.add((n, HNO.aboutFacility, f))
                cg.add((n, HNO.facilityType, HNO[str(c.get("facility_type", "MedicalPoint")).replace("hno:", "")]))
                mention(fr); mention(f)
            elif typ == "schedule":
                gr, r, f = R(c.get("group")), R(c.get("ritual")), R(c.get("facility")) if c.get("facility") else None
                cg.add((n, RDF.type, HNO.ScheduleClaim)); cg.add((n, HNO.forGroup, gr)); cg.add((n, HNO.aboutRitual, r))
                if _t(c.get("start")): cg.add((n, HNO.claimedStart, Literal(_t(c["start"]).isoformat(), datatype=XSD.dateTime)))
                if f is not None: cg.add((n, HNO.claimedFacility, f)); mention(f)
                mention(gr); mention(r)
            elif typ == "ritual_time":
                r = R(c.get("ritual"))
                cg.add((n, RDF.type, HNO.RitualTimeClaim)); cg.add((n, HNO.aboutRitual, r)); mention(r)
                if _t(c.get("start")): cg.add((n, HNO.claimedStart, Literal(_t(c["start"]).isoformat(), datatype=XSD.dateTime)))
                if _t(c.get("end")): cg.add((n, HNO.claimedEnd, Literal(_t(c["end"]).isoformat(), datatype=XSD.dateTime)))
            elif typ == "route":
                o, d = R(c.get("origin")), R(c.get("destination"))
                prof = R(c.get("profile")) if c.get("profile") else default_profile
                t0 = _t(c.get("depart")) or datetime(2026, 5, 27, 8, 0)
                cg.add((n, RDF.type, HNO.RouteRecommendation)); cg.add((n, HNO.origin, o)); cg.add((n, HNO.destination, d))
                cg.add((n, HNO.forProfile, prof)); mention(o); mention(d)
                segs = self._route_segments(c, R)
                t = t0
                for j, s in enumerate(segs, start=1):
                    st = CN[f"c{i}_s{j}"]; owner[st] = i
                    cg.add((n, HNO.hasStep, st)); cg.add((st, RDF.type, HNO.RouteStep))
                    cg.add((st, HNO.stepIndex, Literal(j, datatype=XSD.integer))); cg.add((st, HNO.segment, s))
                    cg.add((st, HNO.plannedAt, Literal(t.isoformat(), datatype=XSD.dateTime)))
                    if s in kg.seg:
                        t = t + timedelta(minutes=kg.seg[s]["min"])
                for stop in c.get("stops", []) or []:
                    mention(R(stop))
            else:
                cg.add((n, RDF.type, HNO.Claim))
                for k in ("entity", "facility", "site"):
                    if c.get(k): mention(R(c[k]))
        return cg, owner, nodes

    def _route_segments(self, c, R):
        kg = self.kg
        if c.get("segments"):
            return [HNO[str(s).replace("hno:", "")] for s in c["segments"]]
        stops = [R(s) for s in (c.get("stops") or [c.get("origin"), c.get("destination")])]
        modes = c.get("modes") or []
        out = []
        for j, (a, b) in enumerate(zip(stops[:-1], stops[1:])):
            want = R(modes[j]) if j < len(modes) and modes[j] else None
            cand = [k for k in (kg.net[a][b].keys() if kg.net.has_edge(a, b) else [])]
            if want is not None:
                cand = [k for k in cand if kg.seg[k]["mode"] == want]
            out.append(cand[0] if cand else HNO[f"NOSEG_{j}"])
        return out

    # ---------------------------------------------------------------
    def verify(self, claims, default_profile=HNO.Profile_Standard):
        """returns per-claim verdicts V(a_i) and the list of violated constraint codes"""
        kg = self.kg
        cg, owner, nodes = self.to_rdf(claims, default_profile)
        data = self.core + cg
        conforms, rg, _ = validate(data, shacl_graph=kg.shapes, inference="none", advanced=True, allow_warnings=True)
        viol = {i: set() for i in range(len(claims))}
        for r in rg.subjects(RDF.type, SH.ValidationResult):
            fn = rg.value(r, SH.focusNode)
            if fn in owner:
                msg = str(rg.value(r, SH.resultMessage) or "")
                code = re.match(r"(C\d)", msg)
                viol[owner[fn]].add(code.group(1) if code else "C?")
        for i, c in enumerate(claims):                      # C9: nearest-facility relation (procedural)
            if c.get("type") == "nearest":
                fr = resolve(kg, self.linker, c.get("from")); f = resolve(kg, self.linker, c.get("facility"))
                cls = HNO[str(c.get("facility_type", "MedicalPoint")).replace("hno:", "")]
                if kg.exists(fr) and kg.exists(f):
                    best, bd = nearest(kg, fr, cls)
                    if best is not None and f != best and kg.walking_distance(fr, f) > bd + 1e-6:
                        viol[i].add("C9")
                    if cls not in kg.types[f]:
                        viol[i].add("C9")
        verdicts = [0 if viol[i] else 1 for i in range(len(claims))]
        return {"verdicts": verdicts, "violations": {i: sorted(v) for i, v in viol.items()},
                "conforms": all(verdicts)}


def verdict_feedback(claims, result):
    """concise verdict-only feedback for repair (Section 3.8)"""
    lines = []
    for i, v in result["violations"].items():
        if v:
            lines.append(f"Claim {i + 1} ({claims[i].get('type')}) is INVALID: violates {', '.join(v)}.")
    return "\n".join(lines)


CONSTRAINT_TEXT = {
    "C1": "every entity you mention must be one of the identifiers listed in the facts; do not invent places, stations, gates, camps or groups",
    "C2": "a route must use consecutive, connected segments from the origin to the destination; take the segments in order from the recommended route",
    "C3": "a route must not enter a segment while it is closed; check the closure windows listed in the facts and the planned time of each leg",
    "C4": "ritual start and end times must be exactly the window given in the facts",
    "C5": "for a wheelchair user or an elderly pilgrim, every segment must be step-free; do not use segments marked as having stairs",
    "C6": "the dispatch time and Jamarat level must be exactly those of the group's slot given in the facts",
    "C7": "every claim must include a 'source' field with the source identifier given in the facts",
    "C8": "the site of a facility must be the site stated in the facts",
    "C9": "the nearest facility must be the one stated as nearest in the facts",
}


def verbose_feedback(claims, result):
    """verbose corrective hints (ablation 'verbose hints instead of verdicts')"""
    lines = []
    for i, v in result["violations"].items():
        if v:
            lines.append(f"Claim {i + 1} ({claims[i].get('type')}) is invalid. " + " ".join(
                f"Constraint {c}: {CONSTRAINT_TEXT.get(c, c)}." for c in v) +
                f" The claim you made was: {claims[i]}. Re-read the facts, explain to yourself which value is wrong, "
                f"and replace it with the correct value from the facts before answering again.")
    return "\n".join(lines)
