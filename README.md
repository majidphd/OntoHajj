# OntoHajj — Ontology-Guided Neuro-Symbolic Decision Support for Hajj Navigation

Reference implementation of the OntoHajj framework, the Hajj Navigation Ontology (HNO) and the
HajjNavQA benchmark.

## Contents
| Path | Description |
|---|---|
| `ontology/hno.ttl` | HNO TBox (OWL 2, bilingual labels, axioms A1–A6) |
| `ontology/hno-shapes.ttl` | SHACL shapes for constraints C1–C8 |
| `ontology/build_kg.py` | Builds the prototype knowledge graph for season 1447H |
| `data/hno-kg.ttl` | Prototype knowledge graph (67,266 triples) |
| `data/hajjnavqa.jsonl` | HajjNavQA (736 questions, 368 Arabic–English pairs, dev/test split) |
| `ontohajj/` | Layers L2–L6: `understand.py`, `retrieve.py`, `decide.py`, `generate.py`, `verify.py`, `pipeline.py` |
| `eval/` | Ontology, understanding, verifier and decision evaluations; end-to-end LLM harness |
| `results/` | JSON outputs of all symbolic evaluations; `results/llm/` holds every answer of every setting (JSONL) |

**Data note.** Holy-site names and the public Mashaer metro line order are real. Camps, path
segments, closures, dispatch (Tafweej) slots and crowd observations are synthetic, because
operational data are not publicly released. Ritual windows are operational approximations for
navigation and are not juristic rulings.

## Reproduce
```bash
pip install -r requirements.txt
cd ontology && python build_kg.py && cd ..
cd bench && python build_hajjnavqa.py && cd ..
python eval/ontology_eval.py        # metrics, HermiT, competency questions
python eval/understanding_eval.py   # intent typing, entity linking
python eval/verifier_eval.py        # hallucination-injection evaluation of L5
python eval/decision_eval.py        # feasible set, AHP ranking, sensitivity
```

## End-to-end LLM evaluation (Tables 12–13)
```bash
# any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama) or Anthropic
export ONTOHAJJ_PROVIDER=openai ONTOHAJJ_BASE_URL=http://localhost:11434/v1 ONTOHAJJ_API_KEY=ollama
python eval/run_llm_eval.py --model qwen2.5:7b --split dev --configs C0 C4 --limit 20      # quick check
python eval/run_llm_eval.py --model qwen2.5:7b --split test --configs C0 C1 C2 C3 C4       # full run
python eval/summarise_llm.py qwen2.5_7b/test_v2
```

## Ablation variants (Table 14)
`--configs C4k0` runs OntoHajj without repair (k = 0); `--configs C4hint` uses verbose corrective
hints instead of verdict-only feedback. C3 and C2 give the other two ablation rows.

## Setting names used in the paper
| Code in this repository | Name in the paper | Meaning |
|---|---|---|
| `C0` | S0 | LLM only |
| `C1` | S1 | text RAG (BM25) |
| `C2` | S2 | ontology-guided retrieval (L1–L3) |
| `C3` | S3 | S2 + constrained decision making (L4) |
| `C4` | S4 | full OntoHajj (L1–L6) |
| `C4k0` | S4-NoRepair | S4 without repair (k = 0) |
| `C4hint` | S4-Hint | S4 with verbose corrective hints |

## Larger model (gpt-oss-120b via GroqCloud)
```bash
export ONTOHAJJ_PROVIDER=openai ONTOHAJJ_BASE_URL=https://api.groq.com/openai/v1
export ONTOHAJJ_API_KEY=<your key>  ONTOHAJJ_REASONING_EFFORT=low ONTOHAJJ_RPM=3
python eval/run_llm_eval.py --model openai/gpt-oss-120b --split test --configs C2 C1 C4 --limit 100
python eval/summarise_llm.py openai_gpt-oss-120b/test_v2
```
`--limit 100` draws the same stratified 100-question subset used for the ablations. Never commit an API key.

## Manual check of the benchmark
`bench/annotation_sample_ids.txt` lists the 100 test questions (random, seed 2026) checked by two
annotators; agreement is reported as Cohen's kappa.

## Additional analyses (v1.1)
| Script | What it does |
|---|---|
| `eval/reviewer_analyses.py` | results without the 20 pilot questions and their translations; L2 refusal recall; Q3 S2-vs-S3 test; crowd-exposure trade-off of r* |
| `eval/freetext_consistency.py` | lexical check of the free-text answer against its typed claims, for every answered question |
| `eval/llm_judge_freetext.py --v2` | Experiment B1: an independent LLM judges whether the free text contradicts or goes beyond the claims, given the knowledge-graph content of each claim (sample of S2–S4 answers). The first run without `--v2` (claim identifiers only) is kept for transparency but is not used |
| `eval/llm_judge_e3.py` | Experiment B2: LLM-as-judge baseline (claim vs. reference passages) for the hallucination-injection study |
| `eval/template_baseline.py` | S4-Template: the full pipeline with the LLM replaced by a deterministic renderer of the claim skeletons (no LLM) |
| `eval/decision_normalization.py` | E4 with fixed-scale normalization and with crowd exposure as a near-hard constraint |
| `notebooks/gpu_timing_colab.ipynb`, `eval/gpu_summary.py` | Experiment B3: S4 and S2 on a free Colab T4 GPU, same 100 questions; results in `results/llm/qwen2.5_7b/test_gpu` and `results/gpu_timing.json` |

The two judge scripts use the same environment variables as `run_llm_eval.py` (e.g. GroqCloud) and can be stopped
and restarted; results go to `results/judge/`.

**Benchmark note.** In v1.1 the reference *text* of the 50 schedule-and-route questions of Q5 also states the route
(found during the manual check); the gold claims, which are used for scoring, are unchanged. The released
`data/hajjnavqa.jsonl` is the canonical benchmark; re-running `bench/build_hajjnavqa.py` produces an equivalent
benchmark whose random choices can differ with the Python hash seed.

## Useful environment variables
| Variable | Default | Meaning |
|---|---|---|
| `ONTOHAJJ_MAX_TOKENS` | 4096 | maximum output tokens per call |
| `ONTOHAJJ_TIMEOUT` | 900 | seconds before a call is abandoned |
| `ONTOHAJJ_NO_THINK` | 0 | `1` appends `/no_think` (disables thinking in Qwen3-family models) |
| `ONTOHAJJ_RPM` | 0 | requests per minute (throttling for rate-limited APIs) |
| `ONTOHAJJ_RETRIES` | 8 | retries on HTTP 429/5xx with back-off |
| `ONTOHAJJ_REASONING_EFFORT` | – | passed to reasoning models (e.g. `low`) |

`python eval/test_connection.py <model>` checks the server and runs one question end to end.

## Version history
- v1: first pilot (20 questions). Route and schedule claims of a 7B model failed format checks, causing over-abstention.
- v2: facts carry ISO timestamps; L3/L4 attach ready-made claims in the HNO output schema (route claims list
  segment identifiers); identifier-like tokens must match exactly when resolving names; every verification
  attempt is logged. Results are written to `results/llm/<model>/<split>_v2/`.

## Cloud models on free tiers (OpenAI-compatible endpoints)
| Provider | ONTOHAJJ_BASE_URL | Example model |
|---|---|---|
| Google AI Studio (Gemini) | `https://generativelanguage.googleapis.com/v1beta/openai` | a Gemini Flash model listed in AI Studio |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile`, `openai/gpt-oss-120b` |

Set `ONTOHAJJ_RPM` below the provider's per-minute limit (e.g. 8 for Gemini free, 25 for Groq). Rate-limit
responses are retried with back-off; when a daily quota is exhausted the run stops without recording the
question, and the same command resumes later. Use the same `--limit` for every cloud model so that all of
them are evaluated on the identical stratified subset.

## Naming of settings
Settings are named S0-S4 in the paper (S0 LLM only, S1 text RAG, S2 ontology-guided RAG, S3 + constrained
decision, S4 full OntoHajj; ablations S4-NoRepair and S4-Hint). The code accepts both these names and the
earlier internal names C0-C4, C4k0 and C4hint; result files of the qwen2.5:7b run use the internal names.
Constraints remain C1-C9 and design requirements R1-R7.

## License
Code: MIT License (see `LICENSE`). Ontology, knowledge graph, benchmark and results: CC BY 4.0.

## Citation
If you use OntoHajj, HNO or HajjNavQA, please cite the article (see `CITATION.cff`).
