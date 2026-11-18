from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.approved_answers import approve_answer

router = APIRouter(prefix="/api/answers", tags=["answers"])


class ApproveRequest(BaseModel):
    query_text: str
    approved_response: str
    source_hashes: list[str]


class FeedbackRequest(BaseModel):
    query_text: str = Field(min_length=1)
    response_text: str = Field(min_length=1)
    source_hashes: list[str] = []
    comment: str = ""


@router.post("/approve")
def approve(req: ApproveRequest):
    aid = approve_answer(req.query_text, req.approved_response, req.source_hashes)
    return {"status": "approved", "id": aid}


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    """Flag a served answer as incorrect.

    Learning effect: every approved-cache row that serves this answer
    (exact or semantic match) is retired, so the next identical/similar
    query regenerates fresh. The verdict row is kept as the audit trail.
    """
    from backend.services import approved_answers as svc

    try:
        result = svc.record_incorrect(req.query_text, req.response_text, req.source_hashes, comment=req.comment)
    except Exception as exc:
        svc.LOG.error("feedback failed for query=%r: %s", req.query_text[:100], exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"recording feedback failed: {exc}") from exc
    return {"status": "recorded", **result}


@router.get("/feedback")
def list_feedback(limit: int = 50):
    """Newest user verdicts — reviewer visibility into flagged answers."""
    from backend.core.database import get_connection, init_db, list_answer_feedback

    init_db()
    conn = get_connection()
    try:
        return {"feedback": list_answer_feedback(conn, limit=limit)}
    finally:
        conn.close()
