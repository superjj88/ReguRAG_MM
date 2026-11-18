from fastapi import APIRouter

from backend.core.database import get_connection, init_db, list_conflicts, resolve_conflict

router = APIRouter(prefix="/api/conflicts", tags=["conflicts"])

@router.get("")
def get_conflicts(unresolved_only: bool = False):
    init_db()
    conn = get_connection()
    rows = list_conflicts(conn, unresolved_only=unresolved_only)
    conn.close()
    return {"conflicts": rows}

@router.post("/{conflict_id}/resolve")
def post_resolve(conflict_id: int, resolution: str = "approved"):
    init_db()
    conn = get_connection()
    resolve_conflict(conn, conflict_id, resolution)
    conn.close()
    return {"status": "resolved", "id": conflict_id}
