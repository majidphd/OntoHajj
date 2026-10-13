"""Step-by-step connection test: checks the LLM server, sends one question in C0 and C4,
and prints the raw model output, the parsed claims and the verification verdicts."""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from ontohajj.generate import LLM, parse_json

model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ONTOHAJJ_MODEL")
base = os.environ.get("ONTOHAJJ_BASE_URL", "http://localhost:11434/v1").rstrip("/")
print("1) Server:", base)
try:
    req = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + os.environ.get("ONTOHAJJ_API_KEY", ""),
                                                  "User-Agent": "OntoHajj/1.0 (research evaluation; python)", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        names = [m["id"].replace("models/", "") for m in json.loads(r.read()).get("data", [])]
    print("   models available:", names if len(names) <= 40 else names[:40] + ["..."])
    if model not in names:
        print(f"   WARNING: '{model}' is not in the list; check the spelling (the call below will show the provider's answer).")
except Exception as e:
    print("   note: could not list models:", e, "- continuing with the test call")

llm = LLM(model=model)
print("2) Raw call (a short JSON request)...")
text, tok, sec = llm("Return only JSON.", 'Return {"ok": true} and nothing else.')
print(f"   {sec:.1f} s, {tok} tokens, output: {text[:300]!r}")

print("3) Loading OntoHajj (knowledge graph, about 10 s)...")
from ontohajj.pipeline import OntoHajj
sysm = OntoHajj(llm)
q = {"question": "Where is the nearest medical centre to Mina camp M07?", "context": {"time": None, "location": "Camp_M07"}, "lang": "en"}
for cfg in ("C0", "C4"):
    print(f"4) Configuration {cfg} ...")
    rec = sysm.answer(q["question"], q["context"], cfg, q["lang"])
    print("   answer :", (rec.get("answer") or "")[:400])
    print("   abstain:", rec.get("abstain"), "| repairs:", rec.get("repairs"), "| seconds:", round(rec["latency"].get("total", 0), 1))
    print("   claims :", json.dumps(rec.get("claims"), ensure_ascii=False)[:600])
    if rec.get("claims"):
        print("   verify :", sysm.verifier.verify(rec["claims"])["violations"])
print("Done. Expected gold answer: Mina Medical Centre A [Med_MinaA].")
