#!/usr/bin/env bash
set -euo pipefail

# package_for_offline.sh — build the offline bundle
# On dev machine (with internet): pulls models, builds images, saves bundles,
# and writes SHA256 checksums. Supports --dry-run for CI/tests (no Docker/Ollama needed).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=0
OUTPUT_DIR="${ROOT_DIR}"
MODELS=("qwen2.5:3b-instruct-q4_K_M" "bge-m3")

usage() {
  echo "Usage: $0 [--dry-run] [--output-dir DIR]"
  echo "  --dry-run      create dummy bundles and checksums (no Docker/Ollama)"
  echo "  --output-dir   where to write *.tar / *.tar.gz / SHA256SUMS (default: repo root)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

log() { echo "[package] $*"; }

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY-RUN: creating dummy bundles"
  for f in backend.tar frontend.tar ollama.tar ollama_models_bundle.tar.gz; do
    echo "dummy content for $f (dry-run $(date -Is))" > "$OUTPUT_DIR/$f"
  done
else
  # 1. Pull models
  for m in "${MODELS[@]}"; do
    if command -v ollama >/dev/null 2>&1; then
      log "Pulling model $m"
      ollama pull "$m"
    else
      log "WARNING: ollama not found — skipping pull for $m"
    fi
  done

  # 2. Archive Ollama models directory
  OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"
  if [[ -d "$OLLAMA_MODELS_DIR" ]]; then
    log "Archiving $OLLAMA_MODELS_DIR -> $OUTPUT_DIR/ollama_models_bundle.tar.gz"
    tar -czf "$OUTPUT_DIR/ollama_models_bundle.tar.gz" -C "$(dirname "$OLLAMA_MODELS_DIR")" "$(basename "$OLLAMA_MODELS_DIR")"
  else
    log "WARNING: $OLLAMA_MODELS_DIR not found — creating empty bundle"
    tar -czf "$OUTPUT_DIR/ollama_models_bundle.tar.gz" --files-from /dev/null
  fi

  # 3. Build images
  if command -v docker >/dev/null 2>&1; then
    log "Building Docker images (docker compose build)"
    docker compose build

    log "Saving images"
    docker save -o "$OUTPUT_DIR/backend.tar" regurag-backend:latest 2>/dev/null || docker save -o "$OUTPUT_DIR/backend.tar" "$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -m1 backend || echo 'regurag-backend:latest')"
    docker save -o "$OUTPUT_DIR/frontend.tar" regurag-frontend:latest 2>/dev/null || docker save -o "$OUTPUT_DIR/frontend.tar" "$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -m1 frontend || echo 'regurag-frontend:latest')"
    docker save -o "$OUTPUT_DIR/ollama.tar" ollama/ollama:latest 2>/dev/null || docker save -o "$OUTPUT_DIR/ollama.tar" ollama/ollama:latest
  else
    log "WARNING: docker not found — creating dummy image tars"
    for f in backend.tar frontend.tar ollama.tar; do
      echo "dummy image $f" > "$OUTPUT_DIR/$f"
    done
  fi
fi

# 4. Checksums (always)
log "Generating SHA256SUMS"
(
  cd "$OUTPUT_DIR"
  sha256sum backend.tar frontend.tar ollama.tar ollama_models_bundle.tar.gz > SHA256SUMS
  cat SHA256SUMS
)
log "Verifying checksums"
( cd "$OUTPUT_DIR" && sha256sum -c SHA256SUMS )

log "Done. Artifacts in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*.tar "$OUTPUT_DIR"/*.tar.gz "$OUTPUT_DIR"/SHA256SUMS 2>/dev/null || true
