# ReguRAG_MM — Local Hebrew Regulation Assistant

A fully local, air-gapped Hebrew RAG chatbot for community regulation documents:
every answer is grounded in — and cited from — the regulation PDFs, never from
parametric memory. Built for a kibbutz/community deployment ( Hebrew RTL, single user,
zero cloud dependency).

Corpus: regulation PDFs you supply in `data/regulations/` (the shipped setup ran on
~146 PDFs / ~1,169 chunks), including annex/parent document links and cross-document
contradiction detection.

## Architecture

```
Browser (Streamlit RTL) → FastAPI (embedded ChromaDB + SQLite) → Ollama (127.0.0.1:11434)
```
- **ChromaDB embedded** (`chromadb.PersistentClient` in-process, not a separate container)
- **SQLite** `data/regulator.db` — documents / chunks / document_links / approved_answers / conflicts / ingestion_log
- **Chroma** collection `regurag_chunks` stores chunk-text embeddings with full metadata
  (`file_name`, `file_hash`, `page_number`, `section_number`, `kind='chunk'`);
  optional LLM question pairs behind `performance.qa_pairs_enabled` (default off)
- **Ollama** bound to `127.0.0.1` only, `user: 1000:1000` on every service
- **Document links** — `backend/services/document_links.py` builds annex→parent edges
  (Hebrew `נספח/נספחים/תוספת לתקנון X` name pattern) + sibling edges (same folder);
  retrieval widens the candidate pool with linked docs' chunks (no score boosts)
- **Okapi BM25 & Contextual Retrieval** — `backend/services/hybrid_retrieval.py` implements true
  Okapi BM25 with smoothed IDF, Hebrew grammatical prefix stripping (`והבלמשכ`), stopword filtering,
  and query coordination / document title congruence reranking. Dense vectors embed context-enriched
  headers (`get_searchable_text`) while preserving section continuity across page breaks.
- **Tolerant Citations & Grounding Guard** — `backend/services/answer_generator.py` normalizes
  formatting variations into canonical citations (`[doc | עמוד X | סעיף Y]`), verifies quoted snippets,
  and guards against factual/numerical hallucinations.
- **Contradiction scan** wired post-ingestion: cosine ≥ 0.75 prefilter → one LLM judgment
  per document pair → `conflicts` table (user decides precedence in the UI)
- **OCR for scanned PDFs** — `backend/services/ocr.py` is a Tesseract `heb+eng` fallback inside
  `pdf_parser.extract_pages(use_ocr)` (config `ocr.enabled`/`ocr.languages`), with spaced-letter
  repair for wide-tracked Hebrew output. Requires system packages (Fedora):
  `sudo dnf install -y tesseract tesseract-langpack-heb ocrmypdf`; without them the parser
  behaves exactly as before (binary-gated, no new pip deps)

## Quickstart

```bash
# 1. Prereqs: Docker + Python 3.11; drop your regulation PDFs into data/regulations/

# 2. Local dev (without Docker)
python3 -m pip install -r backend/requirements.txt
python3 -m pip install --user pdfplumber chromadb  # for parser tests
pytest -q                                          # 237 passed (offline)

# 3. Full stack
docker compose up -d --build
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/documents/refresh
curl http://127.0.0.1:8000/api/ingest/status       # processing → done
open http://127.0.0.1:8501                          # Streamlit RTL
```

## Manual Test (approved)

> Note: answers are generated strictly from retrieved chunks with citations; without a citation the system refuses (audit fix — no canned fallbacks).

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"מתי מותר להיכנס לבריכה?"}' | jq
```
Real output (dev_gpu, `ornith-1.5:9b`):
```json
{
  "answer": "כניסה לבריכה מותרת בשעות הפעילות בלבד...\n\n[דוגמה - נוהל בריכה.pdf | עמוד 1 | סעיף 13]",
  "citations": ["[דוגמה - נוהל בריכה.pdf | עמוד 1 | סעיף 13]"],
  "pdf_links": ["/data/%D7%93%D7%95%D7%92%D7%9E%D7%94%20-%20%D7%A0%D7%95%D7%94%D7%9C%20%D7%91%D7%A8%D7%99%D7%9B%D7%94.pdf#page=1"],
  "related_documents": [{"file_name": "נספח לנוהל בריכה.pdf", "link_type": "annex"}]
}
```
- `pdf_links` are URL-encoded (Hebrew filenames open in the browser)
- `related_documents` — linked docs (annex/parent/sibling) of the cited sources, rendered as
  "מסמכים קשורים" cards in the UI
- Out-of-scope → exact refusal `המידע אינו קיים בתקנונים שבמערכת.`
  (guardrail in `backend/prompts/qa_system.he.txt` + `backend/services/answer_generator.py`)

## Config

Single source of truth `config.yaml`:

```yaml
system: {topic_name: kibbutz_regulations, documents_path: ./data/regulations, language: he}
hardware_profile: dev_gpu  # dev_gpu | prod_cpu
models:
  prod_cpu: {llm_name: qwen2.5:3b-instruct-q4_K_M, num_thread: 8}
  dev_gpu: {llm_name: ornith-1.5:9b, num_ctx: 32768}   # GPU-resident on a 24GB card
  embedding_model: bge-m3
  ollama_base_url: http://127.0.0.1:11434
retrieval: {top_k: 5, hybrid_search_alpha: 0.5, stage3_relevance_threshold: 0.6,
            link_expansion: true, link_expansion_max_chunks: 20}
performance: {chat_timeout_seconds: 180, ingestion_batch_size: 5, qa_pairs_enabled: false}
ocr: {enabled: true, languages: heb+eng}   # Tesseract fallback for scanned pages
```
- 9B answers on a 24GB GPU stay VRAM-resident (~10-30 s); on CPU switch
  `hardware_profile: prod_cpu`
- Ollama requests send `keep_alive: -1` so the embedding + generation models stay co-resident
  in VRAM (no per-query reload)
- Override via `REGURAG_CONFIG` env or `OllamaClient(base_url=...)`.

## Scripts

```bash
bash scripts/package_for_offline.sh --dry-run   # dummy tars + SHA256SUMS (no Docker/Ollama)
bash scripts/package_for_offline.sh             # pull models, tar ~/.ollama/models, docker save
bash scripts/install_offline.sh --dry-run       # verify SHA256SUMS only
bash scripts/install_offline.sh                 # verify, docker load, restore, compose up
bash scripts/sqlite_backup.sh                   # → data/backups/regulator_*.db (600)
```

## Key Endpoints

- `GET /api/health` / `GET /health` — liveness (reports version)
- `POST /api/documents/refresh` + `GET /api/ingest/status` — background ingestion (non-blocking;
  also rebuilds document links and runs the contradiction scan)
- `POST /api/chat {query}` — `{answer, citations, pdf_links, source_hashes, related_documents,
  cached, refusal, timings}`; failures → HTTP 503
- `POST /api/answers/approve` — approved_answers cache with hash invalidation
- `GET /api/conflicts` / `POST /api/conflicts/{id}/resolve` — contradiction detection
  (the user decides which document takes precedence)
- `POST /api/backup` — consistent SQLite snapshot (online backup API, keep last 10)
- `GET /data/<file>` — documents served read-only (powers `pdf_links`)

## Evaluation & Security

- `eval/run_eval_5_questions.py` — 5-question evaluation suite verifying Top-1 accuracy, BM25 retrieval, tolerant citations, and groundedness checks
- `eval/benchmark_models.py` — validates prod vs dev model choice; outputs to `artifacts/` (gitignored)
- Hardened by default: Ollama/API bound to 127.0.0.1 only, all containers non-root (`user: 1000:1000`), base images pinned by digest, `chmod 700 data/`, SQLite online-backup API (`POST /api/backup`, keeps last 10)
- `docker compose config` must show `host_ip: 127.0.0.1` and `user: 1000:1000` for every service

## Project Layout

```
backend/core/{config,database}.py  backend/services/{ollama_client,pdf_parser,ocr,document_scanner,qa_generator,ingestion_pipeline,document_links,contradiction_detector,hybrid_retrieval,agentic_refinement,answer_generator,approved_answers}.py
backend/api/{chat,conflicts,approved,backup}_api.py  backend/prompts/qa_system.he.txt
frontend/app.py  eval/benchmark_models.py  config.yaml
```
See `docs/CONFIG_REFERENCE.md` for the full `config.yaml` key reference. Tests: `tests/test_*.py` (offline via MockTransport / tmp DB).

## Tech Stack

- **Backend:** Python 3.11, FastAPI, ChromaDB (embedded), SQLite, pdfplumber, Tesseract OCR (heb+eng)
- **Frontend:** Streamlit (RTL UI)
- **LLM stack:** Ollama — `bge-m3` embeddings + Qwen2.5 / 9B instruct generation, hybrid BM25 + dense retrieval
- **Ops:** Docker Compose, offline packaging scripts, GitHub Actions CI (pytest, ruff, mypy, compose gates)

## License

Runs fully local (air-gapped capable). Provided as-is for portfolio demonstration.
