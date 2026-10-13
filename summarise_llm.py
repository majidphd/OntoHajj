"""Aggregates results/llm/<model>/*.jsonl into the metrics of Tables 8-9 with 95% bootstrap CIs
and exact McNemar tests (C4 vs. each other configuration)."""
import glob, json, os, sys, random
from collections import defaultdict
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(rows):
    n = len(rows); ans = [r for r in rows if not r["abstain"]]
    claims = sum(r["n_claims"] for r in rows); ents = sum(r["n_entities"] for r in rows)
    q6 = [r for r in rows if r["expect_abstain"]]; ab = [r for r in rows if r["abstain"]]
    p_abs = sum(r["expect_abstain"] for r in ab) / len(ab) if ab else 0.0
    r_abs = sum(r["abstain"] for r in q6) / len(q6) if q6 else 0.0
    lat = [r["latency"] for r in rows if r.get("latency")]
    return {"Acc": sum(r["correct"] for r in rows) / n,
            "HR": sum(r["unsupported"] for r in rows) / claims if claims else 0.0,
            "OHR": sum(r["fake_entities"] for r in rows) / ents if ents else 0.0,
            "CVR": sum(r["unsafe"] for r in ans) / len(ans) if ans else 0.0,
            "Cov": len(ans) / n,
            "SelAcc": sum(r["correct"] for r in ans if not r["expect_abstain"]) / len(ans) if ans else 0.0,
            "AbsF1": 2 * p_abs * r_abs / (p_abs + r_abs) if (p_abs + r_abs) else 0.0,
            "Latency_s": sum(lat) / len(lat) if lat else None, "n": n}


def bootstrap(rows, key, B=1000):
    vals = []
    for _ in range(B):
        s = [random.choice(rows) for _ in rows]
        vals.append(metrics(s)[key])
    vals.sort(); return vals[int(0.025 * B)], vals[int(0.975 * B)]


def mcnemar(a, b):
    """exact two-sided McNemar test on paired correctness"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(folder):
    """folder relative to results/llm, e.g. qwen2.5_7b/test_v2"""
    d = os.path.join(ROOT, "results", "llm", folder)
    if not glob.glob(os.path.join(d, "*.jsonl")):
        sys.exit(f"No result files in {d}. Example: python eval/summarise_llm.py qwen2.5_7b/dev_v2")
    data = defaultdict(list)
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); data[r["config"]].append(r)
    out = {}
    for cfg, rows in sorted(data.items()):
        m = metrics(rows)
        m["Acc_CI95"] = bootstrap(rows, "Acc"); m["HR_CI95"] = bootstrap(rows, "HR")
        for lang in ("en", "ar"):
            m[f"Acc_{lang}"] = metrics([r for r in rows if r["lang"] == lang])["Acc"]
        out[cfg] = m
    if "C4" in data:
        key = lambda r: (r["id"], r["run"])
        c4 = {key(r): r["correct"] for r in data["C4"]}
        for cfg, rows in data.items():
            if cfg != "C4":
                pairs = [(c4[key(r)], r["correct"]) for r in rows if key(r) in c4]
                out[cfg]["McNemar_p_vs_C4"] = mcnemar([x for x, _ in pairs], [y for _, y in pairs])
    json.dump(out, open(os.path.join(d, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    print("\nSummary table (Acc / HR / OHR / CVR / Cov / SelAcc / AbsF1 / latency s):")
    NAME = {"C0": "S0", "C1": "S1", "C2": "S2", "C3": "S3", "C4": "S4", "C4k0": "S4-NoRepair", "C4hint": "S4-Hint"}
    for cfg, v in out.items():
        print(f"{NAME.get(cfg, cfg):12s} {v['Acc']:.3f} {v['HR']:.3f} {v['OHR']:.3f} {v['CVR']:.3f} {v['Cov']:.3f} {v['SelAcc']:.3f} {v['AbsF1']:.3f} {v['Latency_s'] or 0:.1f}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main(sys.argv[1])
