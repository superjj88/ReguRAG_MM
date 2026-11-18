"""Document link builder — annex/parent + sibling relations.

Deterministic, generic-Hebrew naming patterns (no hardcoded corpus names):
* annex: a document whose folder/file name matches
  ``נספח/נספחים/תוספת לתקנון <X>`` is linked to the best-overlap candidate
  document (token overlap over folder+file names, Hebrew prefix-normalized).
  Non-annex candidates are preferred — an annex should point at its parent
  regulation, not at a sibling annex.
* sibling: PDFs that live in the same folder link to each other.

Links are stored one-directionally (annex→parent, sibling→both ways) in the
document_links table; retrieval and the UI resolve links in both directions.
The builder is idempotent: every rebuild replaces the outgoing links of each
source document.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.core.database import list_documents, replace_links_for_source

LOG = logging.getLogger(__name__)

ANNEX_RE = re.compile(r"^(?:נספחים|נספח|תוספת)\s+לתקנון\s+(.+)$")
_TOKEN_RE = re.compile(r"[\u0590-\u05FF0-9A-Za-z]+")
_PREFIX_LETTERS = "והבלמש"

LINK_THRESHOLD = 0.3  # overlap / min(|candidate|, |fragment|) tokens


def _normalize_token(tok: str) -> str:
    """Strip one leading Hebrew conjunction/prefix letter (generic morphology)."""
    if len(tok) > 2 and tok[0] in _PREFIX_LETTERS:
        return tok[1:]
    return tok


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(t.lower()) for t in _TOKEN_RE.findall(text) if len(t) > 1
    }


def annex_fragment(name: str) -> str | None:
    m = ANNEX_RE.match(name.strip())
    return m.group(1).strip() if m else None


def _score(candidate_tokens: set[str], fragment_tokens: set[str]) -> float:
    if not candidate_tokens or not fragment_tokens:
        return 0.0
    overlap = len(candidate_tokens & fragment_tokens)
    if not overlap:
        return 0.0
    return overlap / min(len(candidate_tokens), len(fragment_tokens))


def _doc_name_tokens(doc: dict) -> set[str]:
    """Folder name + file stem tokens (semantic identity of a document)."""
    p = Path(doc["path"])
    return _tokens(p.stem) | _tokens(p.parent.name)


def _is_annex_doc(doc: dict) -> bool:
    p = Path(doc["path"])
    return annex_fragment(p.stem) is not None or annex_fragment(p.parent.name) is not None


def build_annex_links(fragment: str, candidates: list[dict], self_path: str) -> list[dict]:
    """Best single parent match for an annex fragment (deterministic tie-break)."""
    frag_tokens = _tokens(fragment)
    scored: list[tuple[float, bool, dict]] = []  # (score, is_annex, candidate)
    for c in candidates:
        if c["path"] == self_path:
            continue
        s = _score(_doc_name_tokens(c), frag_tokens)
        if s >= LINK_THRESHOLD:
            scored.append((s, _is_annex_doc(c), c))
    if not scored:
        return []
    # non-annex first, then score desc, then name asc
    scored.sort(key=lambda x: (x[1], -x[0], x[2]["file_name"]))
    best_score, _, best = scored[0]
    return [
        {
            "target_hash": best["file_hash_sha256"],
            "link_type": "annex",
            "basis": f"name-pattern (overlap {best_score:.2f})",
        }
    ]


def build_sibling_links(path: str, candidates: list[dict]) -> list[dict]:
    folder = str(Path(path).parent)
    out: list[dict] = []
    for c in candidates:
        if c["path"] == path:
            continue
        if str(Path(c["path"]).parent) == folder:
            out.append(
                {"target_hash": c["file_hash_sha256"], "link_type": "sibling", "basis": "folder"}
            )
    return out


def _annex_fragment_for(doc: dict) -> str | None:
    """Annex fragment from the folder name first, then the file stem."""
    p = Path(doc["path"])
    return annex_fragment(p.parent.name) or annex_fragment(p.stem)


def rebuild_document_links(conn) -> int:
    """Recompute + replace outgoing links for every document. Returns edge count."""
    docs = list_documents(conn)
    edges = 0
    for d in docs:
        outgoing: list[dict] = []
        fragment = _annex_fragment_for(d)
        if fragment:
            outgoing.extend(build_annex_links(fragment, docs, d["path"]))
        outgoing.extend(build_sibling_links(d["path"], docs))
        edges += replace_links_for_source(conn, d["file_hash_sha256"], outgoing)
    LOG.info("document links rebuilt: %d outgoing edges across %d documents", edges, len(docs))
    return edges
