"""
Ollama service client.

Shared wrapper over the Ollama REST API used by every downstream consumer:
QA generation, hybrid retrieval / Stage 3,
answer generation, and approved-answers cache probing.

Guarantees:
  * Model identifiers come from AppConfig at call time — never hardcoded.
  * URL is validated at construction; non-loopback emits WARNING (soft).
  * Hebrew UTF-8 safe end-to-end (ensure_ascii=False).
  * Air-gapped: nothing leaves the configured Ollama endpoint.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from collections.abc import Sequence
from urllib.parse import urlparse

import httpx

from backend.core.config import AppConfig, get_config

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM_BGE_M3: int = 1024
CONNECT_TIMEOUT_S: float = 3.0
PROBE_TIMEOUT_S: float = 1.5
DEFAULT_BATCH_SIZE: int = 16
MAX_RETRIES: int = 2
BACKOFF: tuple[float, ...] = (0.5, 1.0)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OllamaError(Exception):
    """Base class for every error raised by this client."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class OllamaConfigError(OllamaError):
    """Raised when construction is invalid (bad URL, missing config)."""


class OllamaConnectionError(OllamaError):
    """Transport-level failure: connect refused, DNS, timeout, etc."""


class OllamaHTTPError(OllamaError):
    """Server returned a non-OK status."""

    status_code: int

    def __init__(self, status_code: int, detail: str, body_excerpt: str = "") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self._body_excerpt = body_excerpt

    @property
    def body_excerpt(self) -> str:
        return self._body_excerpt


class OllamaModelMissingError(OllamaHTTPError):
    """Raised when Ollama reports the configured model is absent.

    In an air-gapped deployment this means the offline
    bundle wasn't loaded. The error carries a human hint — never retried.
    """


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    """Output of a successful generate() call."""

    text: str
    model: str
    eval_count: int | None = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class HealthStatus:
    """Result of health()."""

    ok: bool = True
    version: str = ""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _backoff(index: int) -> float:
    """Compute the retry backoff in seconds (0-indexed retry number)."""
    if index >= len(BACKOFF):
        return BACKOFF[-1]
    return BACKOFF[index]


def _classify_error(
    status_code: int | None,
    body: str,
    called_model: str = "",
) -> tuple[bool, OllamaError]:
    """Pure decision function implementing the retry table.

    Returns (retryable, error).  The caller should raise ``error`` without
    retrying when ``retryable`` is False.
    """
    if status_code is None:
        return True, OllamaConnectionError(
            "Ollama connection error — is Ollama running? "
            "If air-gapped, restore the offline bundle."
        )

    if status_code == 404:
        body_lc = body.lower()
        if "not found" in body_lc:
            hint = (
                f"model '{called_model}' not found — run `ollama pull "
                f"{called_model}` or restore the offline bundle."
            )
            return False, OllamaModelMissingError(404, hint, body[:500])
        hint = (
            f"Ollama returned 404 — {body[:200]}. "
            "Not retried; this looks like a configuration issue."
        )
        return False, OllamaHTTPError(404, hint, body[:500])

    # Retry on 429 and any 5xx (covers 500,501,502,503,504)
    retryable = status_code == 429 or status_code >= 500
    detail = f"Ollama HTTP {status_code} — {body[:200]}"
    if retryable:
        return True, OllamaHTTPError(status_code, detail, body[:500])
    return False, OllamaHTTPError(status_code, detail, body[:500])


def _coerce_vector(v: Any) -> list[float]:
    if not isinstance(v, list) or not v:
        raise OllamaHTTPError(200, "Empty or non-list embedding vector")
    out: list[float] = []
    for x in v:
        try:
            f = float(x)
        except (TypeError, ValueError) as exc:
            raise OllamaHTTPError(200, f"Non-numeric element in embedding: {exc}")
        if not math.isfinite(f):
            raise OllamaHTTPError(200, "Non-finite value in embedding vector")
        out.append(f)
    return out


def _normalize_embeddings(payload: Any) -> list[list[float]]:
    """Normalize one of Ollama's embed response shapes into [[...], ...].

    Handles three known shapes:
      A. {"embeddings": [[…], […]]}                – batched
      B. [{"index": 0, "embedding": [...]}]         – alternate list form
      C. {"embedding": [...]}                       – legacy single
      B2. {"embeddings": [{"embedding": [...]}, …]} – some builds

    Unknown shapes raise ``OllamaHTTPError``.
    """
    # Shape B — top-level list with index/embedding
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if "index" in payload[0]:
            try:
                items = sorted(payload, key=lambda x: int(x.get("index", 0)))
            except (TypeError, ValueError) as exc:
                raise OllamaHTTPError(200, f"Invalid index in embedding response: {exc}")
            out: list[list[float]] = []
            seen: set[int] = set()
            for item in items:
                idx = int(item.get("index", 0))
                if idx in seen:
                    raise OllamaHTTPError(200, f"Duplicate index {idx} in embedding response")
                seen.add(idx)
                emb = item.get("embedding")
                out.append(_coerce_vector(emb))
            return out

    if isinstance(payload, dict):
        # Shape C — legacy single {"embedding": [...]}
        if "embedding" in payload and isinstance(payload["embedding"], list):
            # Distinguish from Shape A/B which use "embeddings" plural
            # If "embeddings" also present, prefer that branch below
            if "embeddings" not in payload:
                return [_coerce_vector(payload["embedding"])]

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list):
            if not embeddings:
                raise OllamaHTTPError(200, "Empty embeddings list in response")
            # Shape A — list of lists
            if isinstance(embeddings[0], list):
                return [_coerce_vector(emb) for emb in embeddings]
            # Dict-wrapped variants
            out2: list[list[float]] = []
            for item in embeddings:
                if isinstance(item, dict):
                    emb = item.get("embedding")
                    if isinstance(emb, list) and emb:
                        out2.append(_coerce_vector(emb))
                    else:
                        raise OllamaHTTPError(200, f"Missing or empty 'embedding' in item: {item}")
                elif isinstance(item, list) and item:
                    out2.append(_coerce_vector(item))
                else:
                    raise OllamaHTTPError(200, f"Unexpected embedding item shape: {item!r:.200}")
            if out2:
                return out2

    raise OllamaHTTPError(
        200, f"Unknown response shape for embedding: {type(payload).__name__}"
    )


def _build_chat_body(
    model: str,
    prompt: str,
    system: str | None = None,
    *,
    temperature: float = 0.0,
    seed: int = 42,
    num_thread: int | None = None,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Build the POST /api/chat request body."""
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "keep_alive": -1,  # keep model resident: embed/generate alternate models
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }
    if system is not None and system.strip():
        body["messages"] = [{"role": "system", "content": system}] + body["messages"]
    if num_thread is not None:
        body["options"]["num_thread"] = num_thread
    if num_ctx is not None:
        body["options"]["num_ctx"] = num_ctx
    return body


def _build_embed_body(model: str, texts: list[str]) -> dict[str, Any]:
    return {"model": model, "input": texts, "keep_alive": -1}


def _build_embed_legacy_body(model: str, text: str) -> dict[str, Any]:
    return {"model": model, "prompt": text, "keep_alive": -1}


# ---------------------------------------------------------------------------
# Loopback guard
# ---------------------------------------------------------------------------


def _is_loopback(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


# ---------------------------------------------------------------------------
# Helpers — extracted for testability
# ---------------------------------------------------------------------------


def _extract_chat_completion(content: str) -> GenerationResult:
    """Parse the final non-streaming `/api/chat` JSON response."""
    text = ""
    model = ""
    eval_count: int | None = None
    try:
        data = json.loads(content)
        message = data.get("message") or {}
        if isinstance(message, dict):
            text = message.get("content", "") or ""
        model = data.get("model", "") or ""
        if "eval_count" in data:
            eval_count = int(data["eval_count"]) if data["eval_count"] is not None else None
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        # Fallback: treat raw content as text
        text = content if isinstance(content, str) else str(content)
        return GenerationResult(text=text, model=model, eval_count=eval_count)
    return GenerationResult(text=text, model=model, eval_count=eval_count)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Synchronous wrapper around the Ollama REST API.

    Construction-time validation:
      * base_url must be ``http`` or ``https``; empty / nonsense rejected.
      * Loopback check logs a ``WARNING`` (soft) for non-loopback hosts.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config is None:
            config = get_config()

        self.config = config
        self._url = base_url or config.models.ollama_base_url
        self._llm_name = config.models.llm_name
        self._embed_model = config.models.embedding_model
        self._num_thread = config.models.num_thread
        self._num_ctx = config.models.num_ctx
        self._chat_timeout = config.performance.chat_timeout_seconds

        if not self._url.startswith("http://") and not self._url.startswith("https://"):
            raise OllamaConfigError(
                f"base_url must start with http:// or https://, got: {self._url!r}"
            )

        if not _is_loopback(self._url):
            LOG.warning(
                "Ollama is expected on loopback, but base_url=%s is external.",
                self._url,
            )

        self._client = httpx.Client(
            base_url=self._url,
            timeout=httpx.Timeout(30.0, connect=CONNECT_TIMEOUT_S),
            trust_env=False,
            transport=transport,
        )
        # Test hook — instance attribute so tests can record sleeps
        self._sleep = time.sleep  # type: ignore[assignment]
        self._http_closed = False

    def close(self) -> None:
        if getattr(self, "_http_closed", False):
            return
        if hasattr(self, "_client") and self._client is not None:
            try:
                self._client.close()
            except Exception:
                LOG.exception("Error closing httpx.Client")
            self._client = None  # type: ignore[assignment]
        self._http_closed = True

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def health(self) -> HealthStatus:
        """Ping ``GET /api/version`` with a short probe timeout.

        No retry. Raises ``OllamaConnectionError`` on transport failure,
        ``OllamaHTTPError`` on any non-2xx response.
        """
        try:
            resp = self._client.get(
                "/api/version",
                timeout=httpx.Timeout(10.0, connect=CONNECT_TIMEOUT_S, read=PROBE_TIMEOUT_S),
            )
            if resp.status_code != 200:
                body = resp.text[:500] if hasattr(resp, "text") else ""
                raise OllamaHTTPError(resp.status_code, f"Ollama health check returned {resp.status_code}", body)
            # Try JSON first, fallback to text
            try:
                data = resp.json()
                version = str(data.get("version", "")).strip()[:50] if isinstance(data, dict) else resp.text.strip()[:50]
            except Exception:
                version = resp.text.strip()[:50] if resp.text.strip() else "unknown"
            if not version:
                version = "unknown"
            LOG.info("ollama /api/version model=health %.0fms attempt=1", 0)
            return HealthStatus(ok=True, version=version)
        except OllamaHTTPError:
            raise
        except httpx.TimeoutException:
            raise OllamaConnectionError(
                f"Ollama health probe timed out ({PROBE_TIMEOUT_S}s). "
                "Is Ollama running on 127.0.0.1:11434?"
            )
        except (httpx.ConnectError, httpx.NetworkError, httpx.ReadError) as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self._url}: {exc!s}"
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        timeout_s: float | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> GenerationResult:
        """POST /api/chat — one-shot non-streaming generation."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty (after stripping)")

        effective_timeout = float(timeout_s if timeout_s is not None else self._chat_timeout)
        model = self._llm_name
        headers = {"Content-Type": "application/json; charset=utf-8"}
        t0 = time.monotonic()

        for attempt in range(max_retries + 1):
            body = _build_chat_body(
                model, prompt, system,
                temperature=temperature, seed=seed,
                num_thread=self._num_thread,
                num_ctx=self._num_ctx,
            )
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")

            try:
                resp = self._client.post(
                    "/api/chat",
                    content=encoded,
                    headers=headers,
                    timeout=httpx.Timeout(10.0, connect=CONNECT_TIMEOUT_S, read=effective_timeout),
                )
                elapsed_ms = (time.monotonic() - t0) * 1000

                if resp.status_code == 200:
                    content = resp.content.decode("utf-8")
                    result = _extract_chat_completion(content)
                    if not result.text.strip():
                        raise OllamaHTTPError(200, f"Ollama returned 200 but empty completion content (model {model}) — treat as failure.", content[:500])
                    # Rebuild with duration + model fallback
                    final_model = result.model or model
                    LOG.info("ollama /api/chat model=%s %.0fms attempt=%d", final_model, elapsed_ms, attempt + 1)
                    return GenerationResult(
                        text=result.text, model=final_model,
                        eval_count=result.eval_count, duration_s=time.monotonic() - t0,
                    )

                body_text = resp.content.decode("utf-8", errors="replace") if resp.content else ""
                retryable, err = _classify_error(
                    resp.status_code, body_text, called_model=model,
                )
                if retryable and attempt < max_retries:
                    LOG.info("ollama /api/chat model=%s %.0fms attempt=%d retryable=%d", model, elapsed_ms, attempt + 1, resp.status_code)
                    self._sleep(_backoff(attempt))
                    continue
                raise err

            except OllamaModelMissingError:
                raise
            except OllamaHTTPError:
                raise
            except httpx.TimeoutException:
                elapsed_ms = (time.monotonic() - t0) * 1000
                if attempt < max_retries:
                    LOG.info("ollama /api/chat model=%s %.0fms attempt=%d timeout", model, elapsed_ms, attempt + 1)
                    self._sleep(_backoff(attempt))
                    continue
                raise OllamaConnectionError(
                    f"Ollama request to /api/chat timed out after {attempt + 1} attempt(s). "
                    f"Is Ollama running on 127.0.0.1:11434?"
                )
            except (httpx.ConnectError, httpx.NetworkError, httpx.ReadError) as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                if attempt < max_retries:
                    LOG.info("ollama /api/chat model=%s %.0fms attempt=%d connect_error", model, elapsed_ms, attempt + 1)
                    self._sleep(_backoff(attempt))
                    continue
                raise OllamaConnectionError(
                    f"Could not reach Ollama on attempt {attempt + 1}: {exc!s}. "
                    "Load the offline model bundle."
                )

        raise OllamaConnectionError(f"Ollama request exhausted all {max_retries + 1} attempts")

    def embed(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout_s: float | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> list[list[float]]:
        """POST /api/embed — batched embedding call."""
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)  # type: ignore[assignment]
        if not texts:
            return []

        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"embed() requires str elements, index {i} has {type(t).__name__}"
                )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        effective_timeout = float(timeout_s if timeout_s is not None else self._chat_timeout)
        model = self._embed_model

        def _do_embed_batch(batch: list[str], url: str) -> list[list[float]]:
            if url == "/api/embeddings":
                # Legacy single-text path: one request per text
                all_vecs: list[list[float]] = []
                for text in batch:
                    lbody = _build_embed_legacy_body(model, text)
                    encoded = json.dumps(lbody, ensure_ascii=False).encode("utf-8")
                    for attempt in range(max_retries + 1):
                        try:
                            resp = self._client.post(
                                url, content=encoded,
                                headers={"Content-Type": "application/json; charset=utf-8"},
                                timeout=httpx.Timeout(10.0, connect=CONNECT_TIMEOUT_S, read=effective_timeout),
                            )
                            if resp.status_code == 200:
                                payload = json.loads(resp.content.decode("utf-8"))
                                vecs = _normalize_embeddings(payload)
                                if len(vecs) != 1:
                                    raise OllamaHTTPError(200, f"Legacy embed expected 1 vector, got {len(vecs)}")
                                all_vecs.append(vecs[0])
                                break
                            body_text = resp.content.decode("utf-8", errors="replace") if resp.content else ""
                            retryable, err = _classify_error(resp.status_code, body_text, called_model=model)
                            if retryable and attempt < max_retries:
                                self._sleep(_backoff(attempt))
                                continue
                            raise err
                        except OllamaHTTPError:
                            raise
                        except httpx.TimeoutException:
                            if attempt < max_retries:
                                self._sleep(_backoff(attempt))
                                continue
                            raise OllamaConnectionError("Ollama embedding request timed out")
                        except (httpx.ConnectError, httpx.NetworkError, httpx.ReadError) as exc:
                            if attempt < max_retries:
                                self._sleep(_backoff(attempt))
                                continue
                            raise OllamaConnectionError(f"Could not reach Ollama embedding endpoint: {exc!s}")
                    else:
                        raise OllamaConnectionError(f"Ollama legacy embed exhausted retries for text {text!r:.50}")
                return all_vecs

            # Normal batch endpoint /api/embed
            body = _build_embed_body(model, batch)
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            for attempt in range(max_retries + 1):
                try:
                    resp = self._client.post(
                        url, content=encoded,
                        headers={"Content-Type": "application/json; charset=utf-8"},
                        timeout=httpx.Timeout(10.0, connect=CONNECT_TIMEOUT_S, read=effective_timeout),
                    )
                    if resp.status_code == 200:
                        payload = json.loads(resp.content.decode("utf-8"))
                        vecs = _normalize_embeddings(payload)
                        if len(vecs) != len(batch):
                            raise OllamaHTTPError(200, f"embed response length {len(vecs)} != input batch size {len(batch)}")
                        LOG.info("ollama %s model=%s %.0fms attempt=%d", url, model, 0, attempt + 1)
                        return vecs
                    body_text = resp.content.decode("utf-8", errors="replace") if resp.content else ""
                    retryable, err = _classify_error(resp.status_code, body_text, called_model=model)
                    if retryable and attempt < max_retries:
                        self._sleep(_backoff(attempt))
                        continue
                    raise err
                except OllamaHTTPError:
                    raise
                except httpx.TimeoutException:
                    if attempt < max_retries:
                        self._sleep(_backoff(attempt))
                        continue
                    raise OllamaConnectionError("Ollama embedding request timed out")
                except (httpx.ConnectError, httpx.NetworkError, httpx.ReadError) as exc:
                    if attempt < max_retries:
                        self._sleep(_backoff(attempt))
                        continue
                    raise OllamaConnectionError(f"Could not reach Ollama embedding endpoint: {exc!s}")
            raise OllamaConnectionError(f"Ollama embed exhausted all {max_retries + 1} attempts for {url}")

        # Batch loop with one-time fallback
        all_vectors: list[list[float]] = []
        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        use_legacy = False

        for batch in batches:
            if use_legacy:
                vecs = _do_embed_batch(batch, "/api/embeddings")
            else:
                try:
                    vecs = _do_embed_batch(batch, "/api/embed")
                except OllamaHTTPError as err:
                    if err.status_code in (404, 405):
                        use_legacy = True
                        vecs = _do_embed_batch(batch, "/api/embeddings")
                    else:
                        raise
            all_vectors.extend(vecs)

        return all_vectors


# ---------------------------------------------------------------------------
# Cached factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_client_cached() -> OllamaClient:
    """Module-level cached accessor — one client per process."""
    return OllamaClient()


def get_ollama_client() -> OllamaClient:
    """Return a single, cached OllamaClient for the process."""
    return _get_client_cached()


def health() -> HealthStatus:
    """Module-level one-liner for Ollama health probing."""
    return get_ollama_client().health()


__all__ = [
    "EMBEDDING_DIM_BGE_M3",
    "OllamaError",
    "OllamaConfigError",
    "OllamaConnectionError",
    "OllamaHTTPError",
    "OllamaModelMissingError",
    "GenerationResult",
    "HealthStatus",
    "OllamaClient",
    "get_ollama_client",
    "health",
]
