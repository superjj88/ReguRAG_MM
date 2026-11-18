import pytest
import yaml
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import (
    ConfigError,
    get_config,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, *, hardware_profile="prod_cpu", **overrides):
    """Build a complete valid 5-section config YAML (documented defaults) at
    ``tmp_path / "config.yaml"`` with optional per-key overrides and return
    the path.
    """
    cfg = {
        "system": {
            "topic_name": "housing_committee",
            "documents_path": "./data/housing_committee",
            "language": "he",
        },
        "hardware_profile": hardware_profile,
        "models": {
            "prod_cpu": {"llm_name": "qwen2.5:3b-instruct-q4_K_M", "num_thread": 8},
            "dev_gpu": {"llm_name": "qwen2.5:7b-instruct-q4_K_M"},
            "embedding_model": "bge-m3",
            "ollama_base_url": "http://127.0.0.1:11434",
        },
        "retrieval": {
            "top_k": 5,
            "hybrid_search_alpha": 0.5,
            "stage3_relevance_threshold": 0.6,
        },
        "performance": {"chat_timeout_seconds": 45, "ingestion_batch_size": 5},  # synthetic fixture, not the repo yaml
    }
    for key in ("hardware_profile",):
        if key in overrides:
            cfg[key] = overrides[key]
    for path, val in overrides.items():
        if path == "hardware_profile":
            continue
        parts = path.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return p


# ---- unit tests (both profiles) ----

def test_load_real_config_dev_gpu_profile():
    actual = load_config(REPO_ROOT / "config.yaml")
    assert actual.hardware_profile == "dev_gpu"
    assert actual.models.llm_name == "ornith-1.5:9b"
    assert actual.models.num_thread is None
    assert actual.models.num_ctx == 32768
    assert actual.models.embedding_model == "bge-m3"
    assert actual.models.ollama_base_url == "http://127.0.0.1:11434"


def test_load_num_ctx_resolved_from_profile():
    p = _write(tmp_path=Path("/tmp"), hardware_profile="dev_gpu", **{"models.dev_gpu.num_ctx": 16384})
    cfg = load_config(p)
    assert cfg.models.num_ctx == 16384


def test_load_num_ctx_absent_defaults_none():
    p = _write(tmp_path=Path("/tmp"), hardware_profile="dev_gpu")
    cfg = load_config(p)
    assert cfg.models.num_ctx is None


def test_invalid_num_ctx_raises():
    p = _write(tmp_path=Path("/tmp"), hardware_profile="prod_cpu", **{"models.prod_cpu.num_ctx": "many"})
    with pytest.raises(ConfigError, match="models\\.prod_cpu\\.num_ctx"):
        load_config(p)
    p2 = _write(tmp_path=Path("/tmp"), hardware_profile="prod_cpu", **{"models.prod_cpu.num_ctx": 0})
    with pytest.raises(ConfigError, match="models\\.prod_cpu\\.num_ctx"):
        load_config(p2)


def test_load_dev_gpu_profile_resolves_7b():
    p = _write(tmp_path=Path("/tmp"), hardware_profile="dev_gpu")
    cfg = load_config(p)
    assert cfg.hardware_profile == "dev_gpu"
    assert cfg.models.llm_name == "qwen2.5:7b-instruct-q4_K_M"
    assert cfg.models.num_thread is None
    assert cfg.models.embedding_model == "bge-m3"
    assert cfg.models.ollama_base_url == "http://127.0.0.1:11434"


# ---- verbatim defaults check ----

def test_all_prd_section8_leaf_defaults():
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    expected = {
        ("system", "topic_name"): "kibbutz_regulations",
        ("system", "documents_path"): "./data/regulations",
        ("system", "language"): "he",
        "hardware_profile": "dev_gpu",
        ("models", "prod_cpu", "llm_name"): "qwen2.5:3b-instruct-q4_K_M",
        ("models", "prod_cpu", "num_thread"): 8,
        ("models", "dev_gpu", "llm_name"): "ornith-1.5:9b",
        ("models", "dev_gpu", "num_ctx"): 32768,
        ("models", "embedding_model"): "bge-m3",
        ("models", "ollama_base_url"): "http://127.0.0.1:11434",
        ("retrieval", "top_k"): 5,
        ("retrieval", "hybrid_search_alpha"): 0.5,
        ("retrieval", "stage3_relevance_threshold"): 0.6,
        ("retrieval", "link_expansion"): True,
        ("retrieval", "link_expansion_max_chunks"): 20,
        ("performance", "chat_timeout_seconds"): 180,
        ("performance", "ingestion_batch_size"): 5,
        ("performance", "qa_pairs_enabled"): False,
        ("performance", "cache_semantic_threshold"): 0.85,
    }
    for path, expected_val in expected.items():
        if len(path) == 2:
            section, key = path
            assert raw[section][key] == expected_val, f"leaf {'.'.join(path)} != {expected_val!r}"
            assert isinstance(raw[section][key], type(expected_val)), \
                f"leaf {'.'.join(path)} type mismatch: {type(raw[section][key]).__name__} vs {type(expected_val).__name__}"
        elif len(path) == 3:
            a, b, c = path
            assert raw[a][b][c] == expected_val, f"leaf {'.'.join(path)} != {expected_val!r}"
            assert isinstance(raw[a][b][c], type(expected_val)), \
                f"leaf {'.'.join(path)} type mismatch: {type(raw[a][b][c]).__name__} vs {type(expected_val).__name__}"
    assert "prod_cpu" in raw["models"] and "dev_gpu" in raw["models"]


# ---- Validation tests ----

def test_unknown_profile_raises():
    p = _write(tmp_path=Path("/tmp"), hardware_profile="no_such_profile")
    with pytest.raises(ConfigError, match="(?i)hardware_profile"):
        load_config(p)


def test_invalid_type_reports_dotted_key_path():
    p = _write(tmp_path=Path("/tmp"), **{"retrieval.top_k": "five"}, hardware_profile="prod_cpu")
    with pytest.raises(ConfigError, match="retrieval\\.top_k"):
        load_config(p)


def test_missing_file_raises_not_found():
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("/tmp/does-not-exist-yaml.yaml"))


# ---- Optional: env override smoke test ----

def test_env_override():
    import os
    os.environ["REGURAG_CONFIG"] = str(REPO_ROOT / "config.yaml")
    try:
        cfg = load_config(path=None)
        assert cfg.hardware_profile == "dev_gpu"
    finally:
        os.environ.pop("REGURAG_CONFIG", None)


# ---- OCR section ----

def test_ocr_defaults_when_section_absent(tmp_path):
    p = _write(tmp_path=tmp_path, hardware_profile="dev_gpu")
    cfg = load_config(p)
    assert cfg.ocr.enabled is True
    assert cfg.ocr.languages == "heb+eng"


def test_ocr_explicit_values(tmp_path):
    p = _write(tmp_path=tmp_path, hardware_profile="dev_gpu", **{"ocr.enabled": False, "ocr.languages": "heb"})
    cfg = load_config(p)
    assert cfg.ocr.enabled is False
    assert cfg.ocr.languages == "heb"


def test_ocr_invalid_values_raise(tmp_path):
    p = _write(tmp_path=tmp_path, hardware_profile="dev_gpu", **{"ocr.enabled": "yes"})
    with pytest.raises(ConfigError, match="ocr\\.enabled"):
        load_config(p)
    p2 = _write(tmp_path=tmp_path, hardware_profile="dev_gpu", **{"ocr.languages": ""})
    with pytest.raises(ConfigError, match="ocr\\.languages"):
        load_config(p2)


def test_real_config_has_ocr_leaves():
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert raw["ocr"]["enabled"] is True
    assert raw["ocr"]["languages"] == "heb+eng"


# ---- Optional: cached accessor smoke test ----

def test_get_config_is_cached(tmp_path):
    p = _write(tmp_path=tmp_path, hardware_profile="dev_gpu")
    # Invalidate any previous cache entry for this path
    get_config.cache_clear()
    a = get_config(p)
    b = get_config(p)
    assert a is b  # same instance — lru_cache keyed on path
