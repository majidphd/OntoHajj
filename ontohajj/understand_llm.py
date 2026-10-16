"""LLM-based query understanding for L2 (variant S4-LLM-L2, config C4llm).

The LLM types the question into the intent types of the ontology and extracts the entity mentions in the canonical
naming scheme of the HNO labels; the mentions are then linked by the ontology linker (Eq. 1), so that only
ontology entities can enter the pipeline, and code-like mentions that do not resolve (e.g. 'Mina Station 7') are
reported as unresolved exactly as in the rule-based L2."""
import json
from .generate import parse_json

INTENTS = ["Location", "Route", "Temporal", "Accessibility", "MultiHop", "OutOfScope"]

SYSTEM = """You analyse questions that Hajj pilgrims send to a navigation assistant. Questions may be in English or in any
Arabic dialect, informal, and misspelt. Return ONLY a JSON object: {"intent": I, "mentions": [M, ...]}

I is exactly one of:
- "Location": where a place or facility is, in which holy site it is, or which facility is nearest to a place.
- "Route": how to get from one place to another (directions, the way, walking or metro), at a given time.
- "Temporal": when something starts or ends, e.g. the dispatch slot (tafweej) of a group and its Jamarat level, or a ritual window.
- "Accessibility": a route for a wheelchair user, an elderly pilgrim, or someone who must avoid stairs.
- "MultiHop": two linked needs in one question, e.g. the nearest station AND its holy site, or a dispatch slot AND the route to it.
- "OutOfScope": a juristic question (validity of Hajj, permissibility, sin, expiation, delegation of rituals) or a request
  unrelated to navigation (restaurants, shopping, history, people).

Each M is a place, facility, or group named in the question, rewritten in this canonical English form:
"Mina camp M07", "Arafat camp A03" (camp numbers always with two digits), "group G05", "Mina Station 2",
"Muzdalifah Station 1", "Arafat Station 3", "Jamarat ground level", "Jamarat level 1" (to 3), "Mina Medical Centre A",
"Jamarat Medical Centre", "Masjid al-Khayf", "Masjid al-Haram", "gate 45", or the holy sites "Mina", "Arafat", "Muzdalifah".
Rewrite only what the question names; never add or invent a place. Keep a numbered place even if it may not exist."""


def understand(llm, question):
    text, tok, _ = llm(SYSTEM, "Question: " + question)
    j = parse_json(text)
    z = j.get("intent") if j.get("intent") in INTENTS else None
    mentions = [m for m in (j.get("mentions") or []) if isinstance(m, str) and m.strip()]
    return z, mentions, tok
