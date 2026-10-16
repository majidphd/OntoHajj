"""Summary of the GPU timing run (Experiment B3): results/gpu_timing.json"""
import json, os, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for c in ("C4", "C2"):
    g = [json.loads(l) for l in open(os.path.join(ROOT, "results", "llm", "qwen2.5_7b", "test_gpu", f"{c}_run0.jsonl"), encoding="utf-8")]
    cpu = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "results", "llm", "qwen2.5_7b", "test_v2", f"{c}_run0.jsonl"), encoding="utf-8")}
    lat = sorted(r["latency"] for r in g); llm = [r["latency"] for r in g if not (r["reason"] or "").startswith(("L2", "L4"))]
    out[c] = {"n": len(g), "correct_gpu": sum(r["correct"] for r in g), "correct_cpu": sum(cpu[r["id"]]["correct"] for r in g),
              "agree": sum(r["correct"] == cpu[r["id"]]["correct"] for r in g), "unsafe_gpu": sum(r["unsafe"] for r in g),
              "lat_mean": round(statistics.mean(lat), 2), "lat_median": round(statistics.median(lat), 2),
              "lat_p95": round(lat[int(0.95 * len(lat)) - 1], 2), "lat_llm_mean": round(statistics.mean(llm), 2),
              "cpu_lat_mean": round(statistics.mean(cpu[r["id"]]["latency"] for r in g), 1)}
json.dump(out, open(os.path.join(ROOT, "results", "gpu_timing.json"), "w"), indent=1); print(json.dumps(out, indent=1))
