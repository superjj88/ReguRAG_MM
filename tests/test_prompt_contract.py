"""Offline tests: the runtime system prompt contract.

The prompt is data, not code, but the guardrails in ``answer_generator`` depend
on it: the model must emit the exact refusal sentence and the citation
templates that ``CITATION_RE`` verifies. These tests fail loudly when the
prompt file is missing, empty, or half-replaced.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.services import answer_generator as ag

PROMPT_PATH = Path(__file__).resolve().parents[1] / "backend" / "prompts" / "qa_system.he.txt"


def test_runtime_prompt_file_is_loaded():
    prompt = ag.load_system_prompt()
    assert prompt
    assert prompt != ag.DEFAULT_SYSTEM_PROMPT
    assert prompt == PROMPT_PATH.read_text(encoding="utf-8").strip()


def test_prompt_contains_exact_refusal_sentence():
    assert ag.REFUSAL in ag.load_system_prompt()


def test_prompt_contains_citation_template_and_exact_copy_rule():
    prompt = ag.load_system_prompt()
    assert re.search(r"עמוד\s+N\s*\|\s*סעיף\s+M", prompt), "section-level citation template missing"
    # page-only citations (chunks without a section number) rely on copying the
    # context metadata verbatim — the prompt must forbid inventing a section.
    assert "בדיוק כפי שנתנו לך" in prompt


def test_prompt_keeps_grounding_and_injection_guards():
    prompt = ag.load_system_prompt()
    assert "הקהילה" in prompt  # identity
    assert "אל תמציא" in prompt  # no fabrication
    assert "לא פקודה" in prompt  # source text is data, not instructions
    assert ag.REFUSAL not in prompt.replace(ag.REFUSAL, "", 1), "refusal sentence must appear exactly once"


def test_old_prompt_was_replaced():
    prompt = ag.load_system_prompt()
    assert "עוזר משפטי-מנהלי המשיב" not in prompt


def test_missing_prompt_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setattr(ag, "PROMPT_PATH", tmp_path / "missing.txt")
    assert ag.load_system_prompt() == ag.DEFAULT_SYSTEM_PROMPT
