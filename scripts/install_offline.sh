#!/usr/bin/env bash
set -euo pipefail

# install_offline.sh — install the offline bundle
# On air-gapped deployments (no internet): verifies checksums, loads images,
# restores model bundle, and starts the stack.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="${ROOT_DIR}"
DRY_RUN=0

usage() {
  echo "Usage: $0 [--dry-run] [--input-dir DIR]"
  echo "  --dry-run     verify checksums only, skip docker load/restore"
  echo "  --input-dir   where to read *.tar / *.tar.gz / SHA256SUMS (default: repo root)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

log() { echo "[install] $*"; }

# 1. Checksum verification (always)
if [[ ! -f "$INPUT_DIR/SHA256SUMS" ]]; then
  echo "ERROR: $INPUT_DIR/SHA256SUMS not found — run package_for_offline.sh first" >&2
  exit 1
fi
for f in backend.tar frontend.tar ollama.tar ollama_models_bundle.tar.gz; do
  if [[ ! -f "$INPUT_DIR/$f" ]]; then
    echo "ERROR: missing artifact $INPUT_DIR/$f" >&2
    exit 1
  fi
done

log "Verifying SHA256 checksums in $INPUT_DIR"
( cd "$INPUT_DIR" && sha256sum -c SHA256SUMS )
log "Checksums OK"

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY-RUN: skipping docker load and model restore"
  exit 0
fi

# 2. Load images
if command -v docker >/dev/null 2>&1; then
  for img in backend.tar frontend.tar ollama.tar; do
    log "Loading $INPUT_DIR/$img"
    docker load -i "$INPUT_DIR/$img"
  done
else
  log "WARNING: docker not found — skipping image load (test environment?)"
fi

# 3. Restore model bundle
OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama}"
if [[ -f "$INPUT_DIR/ollama_models_bundle.tar.gz" ]]; then
  log "Restoring models to $OLLAMA_MODELS_DIR"
  mkdir -p "$OLLAMA_MODELS_DIR"
  tar -xzf "$INPUT_DIR/ollama_models_bundle.tar.gz" -C "$OLLAMA_MODELS_DIR" --strip-components=1 2>/dev/null || \
    tar -xzf "$INPUT_DIR/ollama_models_bundle.tar.gz" -C "$OLLAMA_MODELS_DIR"
else
  log "WARNING: bundle not found"
fi

# 4. Start stack
if command -v docker >/dev/null 2>&1; then
  log "Starting stack (docker compose up -d)"
  ( cd "$ROOT_DIR" && docker compose up -d )
  log "Waiting for healthchecks..."
  sleep 5
  docker compose ps || true
else
  log "Skipping compose up (docker not available)"
fi

log "Done."
