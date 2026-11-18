"""Live integration tests for Ollama — skip if Ollama not reachable.

Probes GET {base}/api/version with a 1.5s timeout; if unreachable, the whole
module is skipped. Otherwise verifies generate/embed against the real models
pinned in config.yaml.
"""

import math
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import load_config
from backend.services.ollama_client import EMBEDDING_DIM_BGE_M3, OllamaClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def _probe():
    cfg = load_config(REPO_ROOT / "config.yaml")
    base = cfg.models.ollama_base_url
    try:
        resp = httpx.get(f"{base}/api/version", timeout=1.5)
        if resp.status_code != 200:
            pytest.skip(f"Ollama reachable but /api/version returned {resp.status_code}")
    except Exception as exc:
        pytest.skip(f"Ollama not reachable at {base}: {exc}")
    return cfg


@pytest.fixture(scope="module")
def live_cfg():
    return _probe()


@pytest.fixture(scope="module")
def live_client(live_cfg):
    client = OllamaClient(config=live_cfg)
    yield client
    client.close()


def test_live_health(live_client):
    status = live_client.health()
    assert status.ok is True
    assert isinstance(status.version, str)


def test_live_generate_hebrew(live_client, live_cfg):
    t0 = time.monotonic()
    result = live_client.generate("מה זה סעיף 1? ענה בקצרה בעברית.")
    elapsed = time.monotonic() - t0
    assert result.text.strip() != ""
    # At least one Hebrew character in response
    assert any(0x0590 <= ord(c) <= 0x05FF for c in result.text), f"No Hebrew in: {result.text!r}"
    assert result.model == live_cfg.models.llm_name
    assert elapsed < live_cfg.performance.chat_timeout_seconds + 5


def test_live_embed_hebrew(live_client):
    texts = ["סעיף", "פרק", "תקנה"]
    vectors = live_client.embed(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == EMBEDDING_DIM_BGE_M3
        assert all(math.isfinite(x) for x in v)
    # Order preserved — vectors for distinct inputs should differ
    assert vectors[0] != vectors[1]
