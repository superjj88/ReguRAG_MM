# Configuration Reference — ReguRAG_MM

Single source of truth: `config.yaml` (validated by `backend/core/config.py`).
Path resolution order: `REGURAG_CONFIG` env → repo-root `config.yaml`.

| Key | Type | Constraint | Default in repo | Used by |
|---|---|---|---|---|
| `system.topic_name` | str | non-empty | `kibbutz_regulations` | DB path resolution, display |
| `system.documents_path` | str | non-empty | `./data/regulations` | scanner, PDF parsing, static `/data` mount |
| `system.language` | str | non-empty | `he` | reserved (product language) |
| `hardware_profile` | str | `prod_cpu` \| `dev_gpu` | `dev_gpu` | model selection |
| `models.<profile>.llm_name` | str | non-empty | `ornith-1.5:9b` (dev_gpu) | Ollama generate |
| `models.<profile>.num_thread` | int \| null | > 0 | 8 (prod) | Ollama options |
| `models.<profile>.num_ctx` | int \| null | > 0 | 32768 (dev_gpu) | Ollama context window (`options.num_ctx`); omitted → ollama default |
| `models.embedding_model` | str | non-empty | `bge-m3` | Ollama embed |
| `models.ollama_base_url` | str | http(s) URL | `http://127.0.0.1:11434` | Ollama client |
| `retrieval.top_k` | int | > 0 | 5 | chat, stage 3 |
| `retrieval.hybrid_search_alpha` | float | (0,1) | 0.5 | dense/lexical fusion |
| `retrieval.stage3_relevance_threshold` | float | (0,1) | 0.6 | refinement trigger |
| `retrieval.link_expansion` | bool | — | `true` | chat: widen candidate pool with linked docs' chunks |
| `retrieval.link_expansion_max_chunks` | int | > 0 | 20 | cap on link-expansion chunks added per query |
| `performance.chat_timeout_seconds` | int | > 0 | 180 | total chat budget (retrieval + bounded stage3 + up to 2 generation attempts) |
| `performance.ingestion_batch_size` | int | > 0 | 5 | QA batch flush |
| `performance.qa_pairs_enabled` | bool | — | `false` | ingestion: store LLM question pairs in Chroma (`kind='question'`) |
| `performance.cache_semantic_threshold` | float | in (0,1) | `0.85` | approved-answers cache: cosine between the approved query embedding and a new question above which the cached answer is served instantly (`cached=true`) |
| `ocr.enabled` | bool | — | `true` | ingestion: OCR fallback (tesseract) for scanned pages with no text layer |
| `ocr.languages` | str | non-empty | `heb+eng` | tesseract `-l` value for the OCR fallback |

## Environment overrides

| Variable | Effect |
|---|---|
| `REGURAG_CONFIG` | config file path (highest priority) |
| `OLLAMA_BASE_URL` | overrides `models.ollama_base_url` (Docker host wiring) |
| `BACKEND_URL` | frontend → backend base URL (default `http://127.0.0.1:8000`) |
| `REGURAG_LOG_LEVEL` | backend log level (default `INFO`) |

Validation errors raise `ConfigError` with the exact key path (see `tests/test_config.py`).
