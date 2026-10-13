"""Layer 1 access: loads HNO (TBox), the knowledge graph (ABox) and the SHACL shapes,
and precomputes the indexes used by the other layers."""
import os
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import networkx as nx
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS

HNO = Namespace("https://w3id.org/ontohajj/hno#")
TIME = Namespace("http://www.w3.org/2006/time#")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local(u):
    return str(u).split("#")[-1]


class KG:
    def __init__(self, root=ROOT):
        self.tbox = Graph().parse(os.path.join(root, "ontology", "hno.ttl"))
        self.shapes = Graph().parse(os.path.join(root, "ontology", "hno-shapes.ttl"))
        self.abox = Graph().parse(os.path.join(root, "data", "hno-kg.ttl"))
        self.g = self.tbox + self.abox
        self._index()

    # ------------------------------------------------------------------
    def _dt(self, inst):
        return datetime.fromisoformat(str(self.g.value(inst, TIME.inXSDDateTime)))

    def window(self, w):
        return (self._dt(self.g.value(w, TIME.hasBeginning)), self._dt(self.g.value(w, TIME.hasEnd)))

    def _index(self):
        g = self.g
        # class closure
        self.sub = defaultdict(set)
        for c in set(g.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))):
            for sup in g.transitive_objects(c, RDFS.subClassOf):
                if isinstance(sup, URIRef):
                    self.sub[sup].add(c)
        self.types = defaultdict(set)
        for s, o in g.subject_objects(RDF.type):
            if isinstance(o, URIRef) and str(o).startswith(str(HNO)):
                self.types[s].add(o)
                for sup in g.transitive_objects(o, RDFS.subClassOf):
                    if isinstance(sup, URIRef):
                        self.types[s].add(sup)
        self.named = {s for s, t in self.types.items() if HNO.NamedEntity in t}
        # labels
        self.labels = defaultdict(dict)          # uri -> {lang: label}
        self.alt = defaultdict(list)
        for s in self.named | set(g.subjects(RDF.type, HNO.CrowdLevel)) | set(g.subjects(RDF.type, HNO.Source)):
            for l in g.objects(s, RDFS.label):
                self.labels[s][l.language or "en"] = str(l)
            for l in g.objects(s, SKOS.altLabel):
                self.alt[s].append(str(l))
        # spatial
        self.located = {s: set(g.transitive_objects(s, HNO.locatedIn)) - {s} for s in self.named if HNO.Location in self.types[s]}
        self.site_of = {}
        for s, anc in self.located.items():
            for a in anc:
                if HNO.HolySite in self.types[a]:
                    self.site_of[s] = a
        # network
        self.seg = {}
        self.net = nx.MultiDiGraph()
        for s in g.subjects(RDF.type, HNO.PathSegment):
            d = {
                "uri": s, "from": g.value(s, HNO.fromLocation), "to": g.value(s, HNO.toLocation),
                "mode": g.value(s, HNO.usesMode), "len": float(g.value(s, HNO.lengthMeters)),
                "min": float(g.value(s, HNO.travelMinutes)), "shaded": str(g.value(s, HNO.shaded)).lower() == "true",
                "access": set(g.objects(s, HNO.accessibleFor)),
                "restr": [self.window(w) for w in g.objects(s, HNO.restrictedDuring)],
            }
            self.seg[s] = d
            self.net.add_edge(d["from"], d["to"], key=s, weight=d["min"], length=d["len"])
        # crowd: seg -> list of (start, end, score)
        self.crowd = defaultdict(list)
        for o in g.subjects(RDF.type, HNO.CrowdObservation):
            s = g.value(o, HNO.onSegment)
            b, e = self.window(g.value(o, HNO.during))
            self.crowd[s].append((b, e, int(g.value(g.value(o, HNO.hasLevel), HNO.densityScore))))
        # rituals, groups, slots
        self.ritual_win = {r: self.window(g.value(r, HNO.hasTimeWindow)) for r in self.instances(HNO.Ritual)}
        self.group_camps = {gr: set(g.objects(gr, HNO.assignedCamp)) for gr in self.instances(HNO.PilgrimGroup)}
        self.slots = {}
        for gr in self.group_camps:
            for sl in g.objects(gr, HNO.hasSlot):
                r = g.value(sl, HNO.slotRitual)
                self.slots[(gr, r)] = {"slot": sl, "facility": g.value(sl, HNO.slotFacility), "win": self.window(g.value(sl, HNO.hasTimeWindow))}

    # ------------------------------------------------------------------
    def instances(self, cls):
        return sorted([s for s, t in self.types.items() if cls in t and s not in self.sub], key=str)

    def label(self, u, lang="en"):
        return self.labels.get(u, {}).get(lang) or self.labels.get(u, {}).get("en") or local(u)

    def exists(self, u):
        return u in self.named

    def is_restricted(self, s, t):
        return any(b <= t <= e for b, e in self.seg[s]["restr"])

    def density(self, s, t):
        for b, e, sc in self.crowd.get(s, []):
            if b <= t <= e:
                return sc
        return 1

    def walking_distance(self, a, b, profile=None):
        """shortest walking distance in metres (ignores time restrictions)"""
        def w(u, v, k):
            d = self.seg[k]
            if d["mode"] != HNO.Walking:
                return None
            if profile is not None and profile not in d["access"]:
                return None
            return d["len"]
        if not hasattr(self, "_wc"):
            self._wc = {}
        if profile not in self._wc:
            sub = nx.DiGraph()
            for u, v, k in self.net.edges(keys=True):
                x = w(u, v, k)
                if x is not None and (not sub.has_edge(u, v) or sub[u][v]["w"] > x):
                    sub.add_edge(u, v, w=x)
            self._wc[profile] = sub
        sub = self._wc[profile]
        try:
            return nx.shortest_path_length(sub, a, b, weight="w")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")


@lru_cache(maxsize=1)
def load():
    return KG()
