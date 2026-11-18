import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
@dataclass(frozen=True)
class SystemConfig:
    topic_name: str  # "housing_committee"
    documents_path: str  # "./data/housing_committee"
    language: str  # "he"
@dataclass(frozen=True)
class ResolvedModels:
    llm_name: str  # profile-specific
    embedding_model: str  # "bge-m3" (shared)
    ollama_base_url: str  # "http://127.0.0.1:11434" (shared)
    num_thread: int | None  # 8 for prod_cpu; None for dev_gpu
    num_ctx: int | None  # request context window; None → ollama default
@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int  # 5
    hybrid_search_alpha: float  # 0.5
    stage3_relevance_threshold: float  # 0.6
    link_expansion: bool  # True: widen candidate pool with linked documents
    link_expansion_max_chunks: int  # 20: cap on linked-document chunks added
@dataclass(frozen=True)
class PerformanceConfig:
    chat_timeout_seconds: int  # 120
    ingestion_batch_size: int  # 5
    qa_pairs_enabled: bool  # False: optional LLM question-pair embeddings
    cache_semantic_threshold: float  # 0.85: cosine threshold for semantic cache hits
@dataclass(frozen=True)
class OcrConfig:
    enabled: bool  # True: OCR fallback for scanned pages via tesseract
    languages: str  # "heb+eng": tesseract -l value
@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    system: SystemConfig
    hardware_profile: str  # "dev_gpu" | "prod_cpu"
    models: ResolvedModels
    retrieval: RetrievalConfig
    performance: PerformanceConfig
    ocr: OcrConfig

class ConfigError(ValueError):
    pass
def load_config(path: Path | None = None) -> AppConfig:
    """Parse + validate config.yaml and resolve the active hardware profile.
    path=None → default resolution order (R1). Raises ConfigError on any problem."""

    # R1: default path order
    if path is None:
        if "REGURAG_CONFIG" in os.environ:
            path = Path(os.environ["REGURAG_CONFIG"])
        else:
            # from backend/core/config.py → repo root
            path = Path(__file__).resolve().parents[2] / "config.yaml"

    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    # R2: must be a mapping
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    # R3: validate sections
    if "system" not in raw:
        raise ConfigError("missing 'system' section")
    sys_raw = raw["system"]
    if not isinstance(sys_raw, dict):
        raise ConfigError("invalid 'system' section: expected dict")
    for key, expected_type in [("topic_name", str), ("documents_path", str), ("language", str)]:
        if key not in sys_raw:
            raise ConfigError(f"missing 'system.{key}'")
        val = sys_raw[key]
        if not isinstance(val, expected_type) or not val.strip():
            raise ConfigError(f"invalid config at 'system.{key}': expected {expected_type.__name__}, got {type(val).__name__}")

    if "hardware_profile" not in raw:
        raise ConfigError("missing 'hardware_profile'")
    hardware_profile = raw["hardware_profile"]
    if hardware_profile not in ("prod_cpu", "dev_gpu"):
        raise ConfigError(f"invalid config at 'hardware_profile': expected one of ['prod_cpu', 'dev_gpu'], got {hardware_profile!r}")

    if "models" not in raw:
        raise ConfigError("missing 'models' section")
    models_raw = raw["models"]
    if not isinstance(models_raw, dict):
        raise ConfigError("invalid 'models' section: expected dict")

    # Validate per-profile structure
    profile_key = hardware_profile
    if profile_key not in models_raw:
        raise ConfigError(f"missing '{profile_key}' in models")
    profile_cfg = models_raw[profile_key]
    if not isinstance(profile_cfg, dict):
        raise ConfigError(f"invalid 'models.{profile_key}' section: expected dict")
    if "llm_name" not in profile_cfg:
        raise ConfigError(f"missing 'models.{profile_key}.llm_name'")
    llm_name = profile_cfg["llm_name"]
    if not isinstance(llm_name, str) or not llm_name.strip():
        raise ConfigError(f"invalid config at 'models.{profile_key}.llm_name': expected str, got {type(llm_name).__name__}")

    # optional num_thread: can be int or None; presence determines type
    num_thread: int | None
    if "num_thread" in profile_cfg:
        val = profile_cfg["num_thread"]
        if val is None:
            num_thread = None
        elif isinstance(val, int):
            if val <= 0 or isinstance(val, bool):
                raise ConfigError(f"invalid config at 'models.{profile_key}.num_thread': expected int > 0, got {val!r}")
            num_thread = val
        else:
            raise ConfigError(f"invalid config at 'models.{profile_key}.num_thread': expected int or None, got {type(val).__name__}")
    else:
        num_thread = None

    # optional num_ctx: int > 0 or absent; mirrors num_thread
    num_ctx: int | None
    if "num_ctx" in profile_cfg:
        val = profile_cfg["num_ctx"]
        if isinstance(val, int) and not isinstance(val, bool):
            if val <= 0:
                raise ConfigError(f"invalid config at 'models.{profile_key}.num_ctx': expected int > 0, got {val!r}")
            num_ctx = val
        else:
            raise ConfigError(f"invalid config at 'models.{profile_key}.num_ctx': expected int > 0, got {type(val).__name__}")
    else:
        num_ctx = None

    if "embedding_model" not in models_raw:
        raise ConfigError("missing 'models.embedding_model'")
    embedding_model = models_raw["embedding_model"]
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ConfigError(f"invalid config at 'models.embedding_model': expected str, got {type(embedding_model).__name__}")

    if "ollama_base_url" not in models_raw:
        raise ConfigError("missing 'models.ollama_base_url'")
    ollama_base_url = models_raw["ollama_base_url"]
    if not isinstance(ollama_base_url, str) or not ollama_base_url.strip():
        raise ConfigError(f"invalid config at 'models.ollama_base_url': expected str, got {type(ollama_base_url).__name__}")
    # Env override for Docker host wiring (Option 1: reuse host ollama via host.docker.internal)
    # docker-compose.yml sets OLLAMA_BASE_URL=http://host.docker.internal:11434
    if "OLLAMA_BASE_URL" in os.environ and os.environ["OLLAMA_BASE_URL"].strip():
        ollama_base_url = os.environ["OLLAMA_BASE_URL"].strip()

    # R4: retrieval section
    if "retrieval" not in raw:
        raise ConfigError("missing 'retrieval' section")
    retrieval_raw = raw["retrieval"]
    if not isinstance(retrieval_raw, dict):
        raise ConfigError("invalid 'retrieval' section: expected dict")

    # top_k
    if "top_k" not in retrieval_raw:
        raise ConfigError("missing 'retrieval.top_k'")
    top_k = retrieval_raw["top_k"]
    if not isinstance(top_k, int):
        raise ConfigError(f"invalid config at 'retrieval.top_k': expected int, got {type(top_k).__name__}")
    if isinstance(top_k, bool) or top_k <= 0:
        raise ConfigError(f"invalid config at 'retrieval.top_k': expected int > 0, got {top_k!r}")

    # hybrid_search_alpha
    if "hybrid_search_alpha" not in retrieval_raw:
        raise ConfigError("missing 'retrieval.hybrid_search_alpha'")
    alpha = retrieval_raw["hybrid_search_alpha"]
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0.0 or alpha >= 1.0:
        raise ConfigError(f"invalid config at 'retrieval.hybrid_search_alpha': expected float in (0.0, 1.0), got {alpha!r}")

    # stage3_relevance_threshold
    if "stage3_relevance_threshold" not in retrieval_raw:
        raise ConfigError("missing 'retrieval.stage3_relevance_threshold'")
    threshold = retrieval_raw["stage3_relevance_threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0.0 or threshold >= 1.0:
        raise ConfigError(f"invalid config at 'retrieval.stage3_relevance_threshold': expected float in (0.0, 1.0), got {threshold!r}")

    # link_expansion (optional, default True)
    link_expansion = retrieval_raw.get("link_expansion", True)
    if not isinstance(link_expansion, bool):
        raise ConfigError(f"invalid config at 'retrieval.link_expansion': expected bool, got {type(link_expansion).__name__}")

    # link_expansion_max_chunks (optional, default 20)
    link_expansion_max_chunks = retrieval_raw.get("link_expansion_max_chunks", 20)
    if not isinstance(link_expansion_max_chunks, int) or isinstance(link_expansion_max_chunks, bool) or link_expansion_max_chunks <= 0:
        raise ConfigError(f"invalid config at 'retrieval.link_expansion_max_chunks': expected int > 0, got {link_expansion_max_chunks!r}")

    # R4: performance section
    if "performance" not in raw:
        raise ConfigError("missing 'performance' section")
    performance_raw = raw["performance"]
    if not isinstance(performance_raw, dict):
        raise ConfigError("invalid 'performance' section: expected dict")

    # chat_timeout_seconds
    if "chat_timeout_seconds" not in performance_raw:
        raise ConfigError("missing 'performance.chat_timeout_seconds'")
    chat_timeout = performance_raw["chat_timeout_seconds"]
    if not isinstance(chat_timeout, int):
        raise ConfigError(f"invalid config at 'performance.chat_timeout_seconds': expected int, got {type(chat_timeout).__name__}")
    if isinstance(chat_timeout, bool) or chat_timeout <= 0:
        raise ConfigError(f"invalid config at 'performance.chat_timeout_seconds': expected int > 0, got {chat_timeout!r}")

    # ingestion_batch_size
    if "ingestion_batch_size" not in performance_raw:
        raise ConfigError("missing 'performance.ingestion_batch_size'")
    batch_size = performance_raw["ingestion_batch_size"]
    if not isinstance(batch_size, int):
        raise ConfigError(f"invalid config at 'performance.ingestion_batch_size': expected int, got {type(batch_size).__name__}")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ConfigError(f"invalid config at 'performance.ingestion_batch_size': expected int > 0, got {batch_size!r}")

    # qa_pairs_enabled (optional, default False)
    qa_pairs_enabled = performance_raw.get("qa_pairs_enabled", False)
    if not isinstance(qa_pairs_enabled, bool):
        raise ConfigError(f"invalid config at 'performance.qa_pairs_enabled': expected bool, got {type(qa_pairs_enabled).__name__}")

    # cache_semantic_threshold (optional, default 0.85)
    cache_semantic_threshold = performance_raw.get("cache_semantic_threshold", 0.85)
    if not isinstance(cache_semantic_threshold, (int, float)) or isinstance(cache_semantic_threshold, bool) or cache_semantic_threshold <= 0.0 or cache_semantic_threshold >= 1.0:
        raise ConfigError(f"invalid config at 'performance.cache_semantic_threshold': expected float in (0.0, 1.0), got {cache_semantic_threshold!r}")

    # R5: ocr section (optional — defaults keep old configs working)
    ocr_raw = raw.get("ocr", {})
    if not isinstance(ocr_raw, dict):
        raise ConfigError("invalid 'ocr' section: expected dict")
    ocr_enabled = ocr_raw.get("enabled", True)
    if not isinstance(ocr_enabled, bool):
        raise ConfigError(f"invalid config at 'ocr.enabled': expected bool, got {type(ocr_enabled).__name__}")
    ocr_languages = ocr_raw.get("languages", "heb+eng")
    if not isinstance(ocr_languages, str) or not ocr_languages.strip():
        raise ConfigError(f"invalid config at 'ocr.languages': expected str, got {type(ocr_languages).__name__}")

    # All sections validated; build dataclasses (immutable)
    system_cfg = SystemConfig(
        topic_name=sys_raw["topic_name"].strip(),
        documents_path=sys_raw["documents_path"].strip(),
        language=sys_raw["language"].strip()
    )

    models_cfg = ResolvedModels(
        llm_name=llm_name.strip(),
        embedding_model=embedding_model.strip(),
        ollama_base_url=ollama_base_url.strip(),
        num_thread=num_thread,
        num_ctx=num_ctx
    )

    retrieval_cfg = RetrievalConfig(
        top_k=top_k,
        hybrid_search_alpha=float(alpha),
        stage3_relevance_threshold=float(threshold),
        link_expansion=link_expansion,
        link_expansion_max_chunks=link_expansion_max_chunks
    )

    performance_cfg = PerformanceConfig(
        chat_timeout_seconds=chat_timeout,
        ingestion_batch_size=batch_size,
        qa_pairs_enabled=qa_pairs_enabled,
        cache_semantic_threshold=float(cache_semantic_threshold)
    )

    ocr_cfg = OcrConfig(
        enabled=ocr_enabled,
        languages=ocr_languages.strip()
    )

    app_config = AppConfig(
        config_path=path.resolve(),
        system=system_cfg,
        hardware_profile=hardware_profile,
        models=models_cfg,
        retrieval=retrieval_cfg,
        performance=performance_cfg,
        ocr=ocr_cfg
    )

    return app_config
@lru_cache(maxsize=8)
def get_config(path: Path | None = None) -> AppConfig:
    """Cached accessor — consumers share ONE AppConfig. Tests call load_config() directly."""
    return load_config(path)
