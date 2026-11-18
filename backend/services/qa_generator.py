"""Vector store writer — chunk-text embeddings primary, optional QA pairs.

Design (link-expansion decision):
* The Chroma document IS the chunk text (primary retrieval path). Metadata
  carries the full citation set: chunk_id, file_hash, file_name, page_number,
  section_number, kind='chunk'.
* Optional LLM-generated question pairs (``performance.qa_pairs_enabled``)
  are stored as kind='question' items whose metadata maps back to the chunk
  via ``chunk_id``; the chunk's real text is resolved from the chunks table
  at query time. Question embeddings widen lexical recall only — answers
  never come from questions.
* Chroma writes never silently drop data — every failure is logged and
  recorded in the progress report. Items whose embedding failed are skipped
  and counted (adding without embeddings would trigger Chroma's default
  embedding function, which downloads a model — impossible offline).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from backend.core.config import get_config

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT_QA = "אתה מחולל שאלות בעברית על בסיס הטקסט הבא. צור 3 שאלות שונות ותשובות קצרות."

COLLECTION_NAME = "regurag_chunks"
CHROMA_FLUSH_BATCH = 64

_chroma_clients: dict[str, Any] = {}


def _get_chroma_client(persist_dir: str | Path = "./data/vectordb"):
    """One PersistentClient per persist_dir per process (Chroma anti-pattern
    otherwise: multiple clients to the same path break updates)."""
    key = str(Path(persist_dir).resolve())
    if key in _chroma_clients:
        return _chroma_clients[key]
    try:
        import chromadb
    except Exception:
        LOG.warning("chromadb not installed — vector store disabled")
        return None
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    _chroma_clients[key] = client
    return client


def get_or_create_collection(
    name: str = COLLECTION_NAME,
    persist_dir: str | Path = "./data/vectordb",
):
    client = _get_chroma_client(persist_dir)
    if client is None:
        return None
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def delete_doc_vectors(file_hash: str, persist_dir: str | Path = "./data/vectordb") -> int:
    """Purge every vector (chunk + question items) belonging to file_hash.

    Returns the number of deleted items. Failures are logged and raised so
    the ingestion pipeline can surface them (no orphaned vectors).
    """
    col = get_or_create_collection(persist_dir=persist_dir)
    if col is None:
        return 0
    try:
        got = col.get(where={"file_hash": file_hash}, include=[])
        ids = list(got.get("ids") or [])
        if ids:
            col.delete(ids=ids)
        return len(ids)
    except Exception:
        LOG.error("chroma delete failed for file_hash=%s", file_hash[:12], exc_info=True)
        raise


def generate_questions_for_chunk(chunk: dict, ollama_client=None) -> list[str]:
    """Call LLM to generate 3 questions. Falls back to heuristic if LLM fails."""
    text = chunk.get("text", "")[:2000]
    if ollama_client is None:
        try:
            from backend.services.ollama_client import get_ollama_client

            ollama_client = get_ollama_client()
        except Exception:
            LOG.warning("LLM unavailable for question generation — heuristic fallback", exc_info=True)
            ollama_client = None
    if ollama_client is not None:
        try:
            prompt = f"{SYSTEM_PROMPT_QA}\n\nטקסט:\n{text}\n\nהחזר 3 שאלות מופרדות בשורה חדשה."
            res = ollama_client.generate(prompt, temperature=0.7, timeout_s=30)
            lines = [line.strip("-• \t") for line in res.text.strip().splitlines() if line.strip()]
            # filter question-like lines
            qs = [line for line in lines if "?" in line or "מה" in line or "האם" in line][:3]
            if len(qs) >= 1:
                return qs[:3]
        except Exception:
            LOG.warning("LLM question generation failed — heuristic fallback", exc_info=True)
    # fallback heuristic — ensures 1-3 variants without LLM
    base = text[:80].strip() or "התקנון"
    return [f"מה אומר {base}?", f"האם {base} חל?", f"פרט את {base}"][:3]


def _chunk_meta(chunk: dict, kind: str) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "file_hash": chunk["file_hash"],
        "file_name": chunk["file_name"],
        "page_number": chunk["page_number"],
        "section_number": chunk.get("section_number", ""),
        "kind": kind,
    }


class QAGenerator:
    """Batched vector-store writer with progress tracking (non-blocking via caller)."""

    def __init__(self, ollama_client=None, batch_size: int | None = None, persist_dir: str | Path = "./data/vectordb"):
        cfg = get_config()
        self.batch_size = batch_size or cfg.performance.ingestion_batch_size
        self.qa_pairs_enabled = cfg.performance.qa_pairs_enabled
        self.collection = get_or_create_collection(persist_dir=persist_dir)
        self.ollama_client = ollama_client
        self._progress: dict = {
            "processed": 0,
            "total": 0,
            "state": "idle",
            "eta_seconds": None,
            "last_error": None,
            "start_ts": None,
            "skipped_no_embedding": 0,
        }

    def generate_for_chunks(self, chunks: list[dict]) -> dict:
        """Embed chunk text (+ optional QA pairs) and write to Chroma. Returns progress dict."""
        self._progress.update(
            {"total": len(chunks), "processed": 0, "state": "processing", "start_ts": time.time(), "last_error": None, "skipped_no_embedding": 0}
        )
        ollama = self.ollama_client
        if ollama is None:
            try:
                from backend.services.ollama_client import get_ollama_client

                ollama = get_ollama_client()
            except Exception:
                LOG.warning("embeddings unavailable during vector write", exc_info=True)
                ollama = None

        try:
            self._write_chunks(chunks, ollama)
            if self.qa_pairs_enabled:
                self._write_question_pairs(chunks, ollama)
        except Exception as e:
            self._progress["last_error"] = str(e)
            LOG.error("vector write failed during ingestion", exc_info=True)
        finally:
            self._progress["state"] = "done"
            self._progress["eta_seconds"] = 0
        return dict(self._progress)

    def _write_chunks(self, chunks: list[dict], ollama) -> None:
        """Primary path: embed context-enriched chunk texts in batches, store as kind='chunk'."""
        if not chunks:
            return
        from backend.services.pdf_parser import get_searchable_text

        texts = [c["text"] for c in chunks]
        searchable_texts = [get_searchable_text(c) for c in chunks]
        vecs: list[list[float]] = []
        if ollama is not None:
            try:
                vecs = ollama.embed(searchable_texts)
            except Exception:
                LOG.error("batch embedding failed for %d chunks — skipping vector write", len(chunks), exc_info=True)
                self._progress["last_error"] = "batch embedding failed"
        if len(vecs) != len(texts):
            self._progress["skipped_no_embedding"] += len(texts) - len(vecs)
        ids = [c["chunk_id"] for c in chunks]
        docs = texts
        metas = [_chunk_meta(c, "chunk") for c in chunks]
        self._flush(ids, docs, metas, vecs)
        self._progress["processed"] = len(chunks)

    def _write_question_pairs(self, chunks: list[dict], ollama) -> None:
        """Optional path: LLM questions per chunk, stored as kind='question'."""
        pairs: list[tuple[dict, str]] = []  # (chunk, question)
        for chunk in chunks:
            qs = generate_questions_for_chunk(chunk, ollama_client=ollama)
            for q in qs:
                pairs.append((chunk, q))
            self._progress["processed"] = min(self._progress["processed"] + 1, self._progress["total"])
        if not pairs:
            return
        if ollama is None:
            self._progress["skipped_no_embedding"] += len(pairs)
            return
        questions = [q for _, q in pairs]
        try:
            vecs = ollama.embed(questions)
        except Exception:
            LOG.error("batch embedding failed for %d question pairs — skipped", len(pairs), exc_info=True)
            self._progress["skipped_no_embedding"] += len(pairs)
            return
        ids = [f"{c['chunk_id']}_q{i}" for i, (c, _) in enumerate(pairs)]
        metas = [_chunk_meta(c, "question") for c, _ in pairs]
        self._flush(ids, questions, metas, vecs)

    def _flush(self, ids: list[str], docs: list[str], metas: list[dict], vecs: list[list[float]]) -> None:
        """Write to Chroma in flush-sized batches; failures are logged and surfaced, never silent."""
        if self.collection is None:
            LOG.warning("no collection available — %d items not persisted", len(ids))
            self._progress["last_error"] = "no collection available"
            return
        for start in range(0, len(ids), CHROMA_FLUSH_BATCH):
            end = min(start + CHROMA_FLUSH_BATCH, len(ids))
            batch_vecs = vecs[start:end]
            if len(batch_vecs) != (end - start):
                # embeddings missing for part of the batch — stop persisting;
                # the shortfall is already counted by the caller
                break
            try:
                self.collection.add(
                    ids=ids[start:end],
                    documents=docs[start:end],
                    metadatas=metas[start:end],  # type: ignore[arg-type]
                    embeddings=batch_vecs,  # type: ignore[arg-type]
                )
            except Exception:
                self._progress["last_error"] = "chroma add failed"
                LOG.error("chroma add failed for %d items — corpus incomplete", end - start, exc_info=True)

    def get_progress(self) -> dict:
        return dict(self._progress)
