"""LLM access (Eq. 6), prompts with the HNO-derived output schema, and the text-RAG baseline (C1).

Providers (set via environment variables):
  ONTOHAJJ_PROVIDER = openai | anthropic
  ONTOHAJJ_BASE_URL = e.g. https://api.openai.com/v1  or  http://localhost:11434/v1 (Ollama, vLLM)
  ONTOHAJJ_API_KEY, ONTOHAJJ_MODEL
"""
import json, math, os, re, time, urllib.request, urllib.error


class QuotaExhausted(RuntimeError):
    """raised when the provider keeps refusing requests (e.g. the free daily quota is used up)"""
from collections import Counter
from .kg import HNO, local
from .understand import normalize

DATES = "Dates: 8 Dhu al-Hijjah = 2026-05-25, 9 = 2026-05-26, 10 = 2026-05-27, 11 = 2026-05-28, 12 = 2026-05-29, 13 = 2026-05-30 (Makkah local time)."

SCHEMA = """Return ONLY a JSON object (no markdown) with the keys:
  "answer": the message to the pilgrim, in the language of the question;
  "abstain": true if you cannot give a safe and supported answer, otherwise false;
  "claims": a list of the checkable facts your answer asserts, each one object of these types:
    {"type":"location","facility":X,"site":Y,"source":S}
    {"type":"nearest","from":X,"facility_type":"MedicalPoint"|"MetroStation","facility":Y,"source":S}
    {"type":"schedule","group":G,"ritual":R,"start":"YYYY-MM-DDTHH:MM","facility":JamaratLevel,"source":S}
    {"type":"ritual_time","ritual":R,"start":"YYYY-MM-DDTHH:MM","end":"YYYY-MM-DDTHH:MM","source":S}
    {"type":"route","origin":X,"destination":Y,"profile":P,"depart":"YYYY-MM-DDTHH:MM","segments":[ordered segment identifiers],"source":S}
      (if you do not know the segment identifiers, give "stops":[every place in order] and "modes":[one per leg: "Walking"|"MashaerMetro"|"Bus"] instead of "segments")
Entities may be given by identifier (preferred, e.g. Camp_M07) or by name. "source" is the identifier of the source of the fact.
If a list of ready-made claims is provided, copy the claims that your answer states exactly as given, without changing any value."""

POLICY = {
    "C0": "Answer from your own knowledge.",
    "C1": "Use the retrieved passages below to answer.",
    "C2": "Use ONLY the facts below. Use the identifiers in square brackets. If the facts do not allow a safe answer, set abstain to true and refer the pilgrim to the service provider or the official help line.",
}
POLICY["C3"] = POLICY["C2"] + " When a recommended route is given, describe that route; do not invent another one."
POLICY["C4"] = POLICY["C3"]


def system_prompt(config):
    return ("You are OntoHajj, a navigation assistant for Hajj pilgrims. " + POLICY[config] + "\n" + DATES + "\n" + SCHEMA)


def user_prompt(question, context, facts=None, passages=None, route_text=None, feedback=None, skeletons=None):
    parts = [f"Pilgrim context: {json.dumps(context, ensure_ascii=False)}", f"Question: {question}"]
    if passages:
        parts.append("Retrieved passages:\n" + "\n".join(f"- {p}" for p in passages))
    if facts:
        parts.append("Facts (source in parentheses):\n" + "\n".join(f"- {f['text']} ({f['source']})" for f in facts))
    if route_text:
        parts.append("Recommended route (verified feasible, ranked by multi-criteria utility):\n" + route_text)
    if skeletons:
        parts.append("Ready-made claims supported by the facts (copy the relevant ones unchanged):\n"
                     + "\n".join(json.dumps(c, ensure_ascii=False) for c in skeletons))
    if feedback:
        parts.append("Your previous answer failed verification:\n" + feedback + "\nCorrect the invalid claims or abstain.")
    return "\n\n".join(parts)


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)   # reasoning models (e.g. Qwen3)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)                # unmatched closing tag
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"answer": text, "abstain": False, "claims": [], "parse_error": True}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"answer": text, "abstain": False, "claims": [], "parse_error": True}
    obj.setdefault("claims", []); obj.setdefault("abstain", False); obj.setdefault("answer", "")
    obj["claims"] = [c for c in obj["claims"] if isinstance(c, dict)]
    return obj


class LLM:
    def __init__(self, model=None, provider=None, base_url=None, api_key=None, temperature=0.0, max_tokens=None):
        self.provider = provider or os.environ.get("ONTOHAJJ_PROVIDER", "openai")
        self.model = model or os.environ.get("ONTOHAJJ_MODEL")
        self.base_url = (base_url or os.environ.get("ONTOHAJJ_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("ONTOHAJJ_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens or int(os.environ.get("ONTOHAJJ_MAX_TOKENS", "4096"))
        self.no_think = os.environ.get("ONTOHAJJ_NO_THINK", "0") == "1"
        self.timeout = int(os.environ.get("ONTOHAJJ_TIMEOUT", "900"))
        self.rpm = float(os.environ.get("ONTOHAJJ_RPM", "0"))        # requests per minute (0 = no throttle)
        self.max_retries = int(os.environ.get("ONTOHAJJ_RETRIES", "8"))
        self._last = 0.0

    def __call__(self, system, user):
        t0 = time.time()
        if self.no_think:
            user = user + "\n/no_think"
        if self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                    "system": system, "messages": [{"role": "user", "content": user}]}
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        else:
            url = self.base_url + "/chat/completions"
            body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if os.environ.get("ONTOHAJJ_REASONING_EFFORT"):          # reasoning models, e.g. openai/gpt-oss-120b
                body["reasoning_effort"] = os.environ["ONTOHAJJ_REASONING_EFFORT"]
            headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        headers["User-Agent"] = "OntoHajj/1.0 (research evaluation; python)"   # default urllib agent is blocked by some CDNs
        headers["Accept"] = "application/json"
        out = self._post(url, body, headers)
        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in out["content"])
            usage = out.get("usage", {}); tok = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        else:
            text = out["choices"][0]["message"].get("content") or ""
            tok = out.get("usage", {}).get("total_tokens", 0)
        return text, tok, time.time() - t0


    def _post(self, url, body, headers):
        """POST with throttling and exponential back-off on rate limits and transient server errors"""
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            if self.rpm > 0:
                wait = 60.0 / self.rpm - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    sleep = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    print(f"   [server {e.code}] waiting {sleep:.0f} s and retrying ({attempt + 1}/{self.max_retries})", flush=True)
                    time.sleep(sleep); delay = min(delay * 2, 300)
                    continue
                if e.code == 429:
                    raise QuotaExhausted(f"rate limit or daily quota reached: {detail}")
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < self.max_retries:
                    print(f"   [network error: {e}] waiting {delay:.0f} s and retrying", flush=True)
                    time.sleep(delay); delay = min(delay * 2, 300)
                    continue
                raise


# ------------------------------------------------------------------ text RAG (C1)
class TextRAG:
    """BM25 over passages that verbalise the same KG content (the 'pilgrim guide' corpus)."""
    def __init__(self, passages, k1=1.5, b=0.75):
        self.p = passages
        self.docs = [normalize(x).split() for x in passages]
        self.N = len(self.docs); self.avg = sum(map(len, self.docs)) / max(1, self.N)
        df = Counter(w for d in self.docs for w in set(d))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]; self.k1, self.b = k1, b

    def search(self, query, k=8):
        q = normalize(query).split()
        sc = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]; s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg))
            sc.append(s)
        idx = sorted(range(self.N), key=lambda i: -sc[i])[:k]
        return [self.p[i] for i in idx if sc[i] > 0]


def build_corpus(kg, lang="en"):
    """verbalise the KG (without crowd observations) into guide-style passages"""
    from .retrieve import retrieve, fmt
    L = lambda u: kg.label(u, lang)
    passages = []
    for e in sorted(kg.named, key=str):
        passages += [f["text"] for f in retrieve(kg, [e], None, lang)]
    for s, d in kg.seg.items():
        mode = L(d["mode"])
        acc = ("step-free" if HNO.Profile_Wheelchair in d["access"] else "has stairs") if lang == "en" else ("خالٍ من الدرج" if HNO.Profile_Wheelchair in d["access"] else "يحتوي على درج")
        closed = "; ".join(f"{fmt(b, lang)}–{fmt(e, lang)}" for b, e in d["restr"]) or ("none" if lang == "en" else "لا يوجد")
        passages.append(f"From {L(d['from'])} to {L(d['to'])} by {mode}: {int(d['len'])} m, about {d['min']} min, {acc}; closed: {closed}."
                        if lang == "en" else f"من {L(d['from'])} إلى {L(d['to'])} عبر {mode}: {int(d['len'])} م، نحو {d['min']} دقيقة، {acc}؛ الإغلاق: {closed}.")
    return passages
