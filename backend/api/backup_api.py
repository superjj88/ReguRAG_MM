"""Backup endpoints.

* Uses the SQLite online backup API (consistent snapshots of a live WAL DB)
  instead of shutil.copy2.
* Retention: keeps the last 10 backups (mirrors scripts/sqlite_backup.sh).
* Listing returns file names, not full server paths.
"""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.core.database import backup_database, get_db_path

LOG = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backup", tags=["backup"])

BACKUP_DIR = Path("data/backups")
KEEP_LAST = 10


def _prune_backups() -> None:
    files = sorted(BACKUP_DIR.glob("regulator_*.db"), reverse=True)
    for old in files[KEEP_LAST:]:
        try:
            old.unlink()
        except OSError:
            LOG.warning("could not prune old backup %s", old, exc_info=True)


@router.post("")
def backup():
    src = get_db_path()
    if not src.exists():
        return {"status": "no_db", "path": str(src)}
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"regulator_{ts}.db"
    try:
        backup_database(src, dest)
    except Exception as exc:
        LOG.error("backup failed for %s", src, exc_info=True)
        raise HTTPException(status_code=500, detail=f"backup failed: {exc}") from exc
    _prune_backups()
    dest.chmod(0o600)
    return {"status": "ok", "path": str(dest)}


@router.get("")
def list_backups():
    if not BACKUP_DIR.exists():
        return {"backups": []}
    files = sorted(BACKUP_DIR.glob("*.db"), reverse=True)
    return {"backups": [f.name for f in files]}
