"""Layer 4: candidate generation, ontology-derived feasible set (Eq. 3), AHP weights and
multi-attribute utility ranking (Eqs. 4-5)."""
from datetime import timedelta
import itertools
import numpy as np
import networkx as nx
from .kg import HNO, load

CRITERIA = ["u1_travel_time", "u2_crowd_exposure", "u3_walking_distance", "u4_accessibility_safety", "u5_schedule_margin"]
COST = {"u1_travel_time": True, "u2_crowd_exposure": True, "u3_walking_distance": True,
        "u4_accessibility_safety": False, "u5_schedule_margin": False}

# Preliminary pairwise comparison matrix (Saaty 1-9 scale), rows/cols in CRITERIA order.
AHP_MATRIX = np.array([
    [1,   1/3, 2,   1/2, 1/2],
    [3,   1,   5,   2,   2  ],
    [1/2, 1/5, 1,   1/3, 1/3],
    [2,   1/2, 3,   1,   1  ],
    [2,   1/2, 3,   1,   1  ],
])
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}


def ahp(matrix=AHP_MATRIX):
    vals, vecs = np.linalg.eig(matrix)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real); w = w / w.sum()
    n = matrix.shape[0]
    lam = vals[k].real
    ci = (lam - n) / (n - 1)
    return {"weights": dict(zip(CRITERIA, w.round(4).tolist())), "lambda_max": round(lam, 4), "CI": round(ci, 4), "CR": round(ci / RI[n], 4)}


WEIGHTS = ahp()["weights"]


def candidates(kg, origin, dest, k=12, profile=None):
    """k shortest simple paths (by travel minutes) over the sub-network accessible to the profile
    (all segments if profile is None), expanded to segment sequences"""
    simple = nx.DiGraph()
    for u, v, key, d in kg.net.edges(keys=True, data=True):
        if profile is not None and profile not in kg.seg[key]["access"]:
            continue
        if not simple.has_edge(u, v) or simple[u][v]["weight"] > d["weight"]:
            simple.add_edge(u, v, weight=d["weight"])
    out = []
    try:
        for path in itertools.islice(nx.shortest_simple_paths(simple, origin, dest, weight="weight"), k):
            choices = [[key for key in kg.net[u][v].keys() if profile is None or profile in kg.seg[key]["access"]]
                       for u, v in zip(path[:-1], path[1:])]
            for combo in itertools.product(*choices):
                out.append(list(combo))
                if len(out) >= 4 * k:
                    return out
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return out


def schedule(kg, segs, start):
    """time-dependent traversal: returns list of (segment, entry_time) and arrival time"""
    t, steps = start, []
    for s in segs:
        steps.append((s, t))
        t = t + timedelta(minutes=kg.seg[s]["min"])
    return steps, t


def violations(kg, segs, start, profile, origin=None, dest=None):
    """compiled form of shapes C2, C3, C5 (same semantics as the SHACL shapes)"""
    v = []
    steps, _ = schedule(kg, segs, start)
    for (a, _), (b, _) in zip(steps[:-1], steps[1:]):
        if kg.seg[a]["to"] != kg.seg[b]["from"]:
            v.append(("C2", a))
    if origin is not None and steps and kg.seg[steps[0][0]]["from"] != origin:
        v.append(("C2", steps[0][0]))
    if dest is not None and steps and kg.seg[steps[-1][0]]["to"] != dest:
        v.append(("C2", steps[-1][0]))
    for s, t in steps:
        if kg.is_restricted(s, t):
            v.append(("C3", s))
        if profile not in kg.seg[s]["access"]:
            v.append(("C5", s))
    return v


def raw_criteria(kg, segs, start, target_end=None):
    steps, arrival = schedule(kg, segs, start)
    tt = (arrival - start).total_seconds() / 60
    crowd = sum(kg.density(s, t) * kg.seg[s]["len"] for s, t in steps if kg.seg[s]["mode"] == HNO.Walking) / 1000
    walk = sum(kg.seg[s]["len"] for s, _ in steps if kg.seg[s]["mode"] == HNO.Walking)
    tot = sum(kg.seg[s]["len"] for s, _ in steps) or 1
    safe = sum(kg.seg[s]["len"] for s, _ in steps if kg.seg[s]["shaded"] or kg.seg[s]["mode"] != HNO.Walking) / tot
    margin = (target_end - arrival).total_seconds() / 60 if target_end else 0.0
    return {"u1_travel_time": tt, "u2_crowd_exposure": crowd, "u3_walking_distance": walk,
            "u4_accessibility_safety": safe, "u5_schedule_margin": margin}, arrival


def utilities(raws, weights=None):
    weights = weights or WEIGHTS
    U = []
    for i in range(len(raws)):
        u = 0.0
        for c in CRITERIA:
            vals = [r[c] for r in raws]
            lo, hi = min(vals), max(vals)
            x = 1.0 if hi == lo else (raws[i][c] - lo) / (hi - lo)
            if COST[c] and hi != lo:
                x = 1 - x
            u += weights[c] * x
        U.append(u)
    return U


def decide(kg, origin, dest, start, profile, target_end=None, weights=None, k=12):
    """returns dict with feasible routes ranked by utility and the best route r*"""
    cands = candidates(kg, origin, dest, k, profile)
    feas, infeas = [], []
    for c in cands:
        v = violations(kg, c, start, profile, origin, dest)
        raw, arrival = raw_criteria(kg, c, start, target_end)
        if target_end is not None and arrival > target_end:
            v.append(("C6", None))
        (infeas if v else feas).append({"segments": c, "raw": raw, "arrival": arrival, "violations": v})
    if feas:
        U = utilities([f["raw"] for f in feas], weights)
        for f, u in zip(feas, U):
            f["utility"] = round(u, 4)
        feas.sort(key=lambda f: -f["utility"])
    return {"feasible": feas, "infeasible": infeas, "best": feas[0] if feas else None, "n_candidates": len(cands)}


def shortest_baseline(kg, origin, dest):
    """constraint-agnostic baseline: fastest path ignoring closures, accessibility and schedules"""
    c = candidates(kg, origin, dest, k=1)
    return c[0] if c else None
