#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-data/regulator.db}"
DEST_DIR="${2:-data/backups}"
mkdir -p "$DEST_DIR"
if [[ ! -f "$SRC" ]]; then echo "no db at $SRC"; exit 0; fi
TS=$(date +%Y%m%d_%H%M%S)
DEST="$DEST_DIR/regulator_${TS}.db"
# Checkpoint the WAL so the copy is consistent
python3 - "$SRC" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
conn.close()
PY
cp -a "$SRC" "$DEST"
echo "backup $DEST"
# keep last 10
ls -t "$DEST_DIR"/regulator_*.db 2>/dev/null | tail -n +11 | xargs -r rm
chmod 600 "$DEST"
