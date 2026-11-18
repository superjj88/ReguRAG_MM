# backend.services — Ollama service package
# Provides generate / embed / health helpers for LLM and embedding calls.

__all__ = [
    "OllamaClient",
    "EMBEDDING_DIM_BGE_M3",
    "CONNECT_TIMEOUT_S",
    "build_chat_request",
    "build_embed_requests",
    "fetch_embedding_batch",
    "fetch_completion_batch",
    "generate",
    "get_client",
    "health_check",
    "normalize_embeddings",
    "OllamaError",
    "OllamaClientException",
    "OllamaConnectionError",
    "OllamaConfigError",
    "OllamaHTTPError",
    "OllamaModelMissingError",
]
