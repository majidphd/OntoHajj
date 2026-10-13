"""Ontology metrics, HermiT consistency/classification and competency-question (CQ) evaluation."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL
from ontohajj.kg import load, HNO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kg = load()
T = kg.tbox
res = {}
hno_cls = [c for c in T.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef) and str(c).startswith(str(HNO))]
res["classes"] = len(hno_cls)
res["object_properties"] = len([p for p in T.subjects(RDF.type, OWL.ObjectProperty) if str(p).startswith(str(HNO))])
res["data_properties"] = len([p for p in T.subjects(RDF.type, OWL.DatatypeProperty) if str(p).startswith(str(HNO))])
logical = 0
logical += len([1 for s, o in T.subject_objects(RDFS.subClassOf)])
logical += len(list(T.subject_objects(OWL.equivalentClass)))
logical += len(list(T.subjects(RDF.type, OWL.AllDisjointClasses)))
logical += len(list(T.subject_objects(RDFS.domain))) + len(list(T.subject_objects(RDFS.range)))
logical += len(list(T.subjects(RDF.type, OWL.TransitiveProperty))) + len(list(T.subjects(RDF.type, OWL.FunctionalProperty)))
res["logical_axioms_tbox"] = logical
res["individuals"] = len(set(kg.abox.subjects(RDF.type, OWL.NamedIndividual)))
res["kg_resources"] = len(set(kg.abox.subjects()))
res["triples_tbox"] = len(T); res["triples_abox"] = len(kg.abox); res["triples_shapes"] = len(kg.shapes)
res["path_segments"] = len(kg.seg); res["restriction_windows"] = sum(len(d["restr"]) for d in kg.seg.values())
res["crowd_observations"] = len(list(kg.abox.subjects(RDF.type, HNO.CrowdObservation)))
res["dispatch_slots"] = len(kg.slots); res["groups"] = len(kg.group_camps)
res["labels_ar"] = sum(1 for u in kg.labels if "ar" in kg.labels[u]); res["labels_en"] = sum(1 for u in kg.labels if "en" in kg.labels[u])
res["alt_labels"] = sum(len(v) for v in kg.alt.values())
res["shapes"] = len([s for s in kg.shapes.subjects(RDF.type, URIRef("http://www.w3.org/ns/shacl#NodeShape"))])
res["expressivity"] = "SOQ(D)"

# ---------------- HermiT consistency + classification (owlready2) ----------------
try:
    import owlready2
    from owlready2 import get_ontology, sync_reasoner, World
    nt = os.path.join(ROOT, "results", "_hno_full.nt")
    (kg.tbox + kg.abox).serialize(nt, format="nt")
    w = World()
    onto = w.get_ontology("file://" + nt).load()
    t0 = time.time()
    with onto:
        sync_reasoner(w, infer_property_values=False, debug=0)
    res["hermit_seconds"] = round(time.time() - t0, 1)
    res["hermit_consistent"] = True
    sf = onto.search_one(iri=str(HNO.StepFreeSegment))
    res["inferred_stepfree_segments"] = len(list(sf.instances())) if sf else None
    os.remove(nt)
except Exception as e:  # inconsistency raises OwlReadyInconsistentOntologyError
    res["hermit_consistent"] = "error: " + type(e).__name__ + ": " + str(e)[:200]

# ---------------- Competency questions ----------------
P = "PREFIX hno: <https://w3id.org/ontohajj/hno#> PREFIX time: <http://www.w3.org/2006/time#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
CQ = {
    "CQ1 Which path segments connect a given camp (M07) to the Jamarat plaza?":
        "SELECT ?s WHERE { ?s hno:fromLocation hno:Camp_M07 . } ",
    "CQ2 Which segments are restricted at a given time (10 DH 08:00)?":
        "SELECT ?s WHERE { ?s hno:restrictedDuring ?w . ?w time:hasBeginning/time:inXSDDateTime ?b ; time:hasEnd/time:inXSDDateTime ?e . "
        "FILTER (\"2026-05-27T08:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime> >= ?b && \"2026-05-27T08:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime> <= ?e) }",
    "CQ3 Which facilities are accessible to a wheelchair user (via step-free segments)?":
        "SELECT DISTINCT ?f WHERE { ?s hno:accessibleFor hno:Profile_Wheelchair ; hno:toLocation ?f . ?f a ?c . ?c rdfs:subClassOf* hno:Facility }",
    "CQ4 Which ritual must precede the stoning on 11 Dhu al-Hijjah, and in which window?":
        "SELECT ?r ?b ?e WHERE { ?r hno:precedes hno:StoningDay11 ; hno:hasTimeWindow ?w . ?w time:hasBeginning/time:inXSDDateTime ?b ; time:hasEnd/time:inXSDDateTime ?e }",
    "CQ5 In which holy site is a given facility (Masjid Namirah) located?":
        "SELECT ?site WHERE { hno:MasjidNamirah hno:locatedIn+ ?site . ?site a hno:HolySite }",
    "CQ6 When and at which Jamarat level is the dispatch slot of group G05 on 10 Dhu al-Hijjah?":
        "SELECT ?lvl ?b WHERE { hno:Group_G05 hno:hasSlot ?sl . ?sl hno:hijriDay 10 ; hno:slotFacility ?lvl ; hno:hasTimeWindow ?w . ?w time:hasBeginning/time:inXSDDateTime ?b }",
    "CQ7 When does the metro operate from Arafat Station 3 towards Muzdalifah?":
        "SELECT ?b ?e WHERE { ?s hno:fromLocation hno:Station_Arafat3 ; hno:toLocation hno:Station_Muzdalifah1 ; hno:restrictedDuring ?w . ?w time:hasBeginning/time:inXSDDateTime ?b ; time:hasEnd/time:inXSDDateTime ?e }",
    "CQ8 What is the crowd level on the Jamarat approach at a given time (10 DH 07:00)?":
        "SELECT ?seg ?lvl WHERE { ?o hno:onSegment ?seg ; hno:hasLevel ?lvl ; hno:during ?w . ?seg hno:toLocation hno:WP_JamaratPlaza . "
        "?w time:hasBeginning/time:inXSDDateTime ?b ; time:hasEnd/time:inXSDDateTime ?e . "
        "FILTER (\"2026-05-27T07:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime> >= ?b && \"2026-05-27T07:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime> <= ?e) }",
    "CQ9 Which source asserts a given fact (provenance of segment Seg_001)?":
        "SELECT ?src WHERE { hno:Seg_001 hno:derivedFrom ?src }",
    "CQ10 Which camps are assigned to a group (G05)?":
        "SELECT ?c WHERE { hno:Group_G05 hno:assignedCamp ?c }",
}
cq = {}
for name, q in CQ.items():
    t0 = time.time(); rows = list(kg.g.query(P + q)); ms = (time.time() - t0) * 1000
    cq[name] = {"answers": len(rows), "ms": round(ms, 1), "answered": len(rows) > 0}
res["competency_questions"] = cq
res["cq_answered"] = f"{sum(v['answered'] for v in cq.values())}/{len(cq)}"
json.dump(res, open(os.path.join(ROOT, "results", "ontology_eval.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
