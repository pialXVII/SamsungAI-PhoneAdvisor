"""FastAPI application exposing the chatbot, agents and phone database.

Endpoint groups:

* `/chat`            — RAG chatbot, one question in, grounded answer out
* `/agents/*`        — multi-agent review and comparison workflows
* `/phones*`         — direct access to the scraped database
* `/health`, `/stats`— operational checks

The vector index is built during startup rather than on the first request, so no
user request pays the embedding cost.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

import config
from src.agents.crew import (
    backend_status,
    compare_phones,
    generate_review,
    handle_request,
)
from src.database.db import get_db, healthcheck, init_db
from src.database.models import Phone
from src.database.repository import (
    database_stats,
    find_phone,
    find_phones,
    get_all_phones,
    get_phone_by_id,
)
from src.llm.provider import llm_status
from src.rag.chatbot import get_chatbot

from .schemas import (
    AgentQueryRequest,
    ChatRequest,
    ChatResponseModel,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    PhoneDetail,
    PhoneSummary,
    ReviewRequest,
    ReviewResponse,
)
from .web import INDEX_HTML

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the database schema and vector index before serving traffic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        init_db()
        get_chatbot()
        logger.info("Startup complete — chatbot and vector index ready")
    except Exception as exc:
        # A failed warm-up must not stop the app: /health then reports exactly
        # what is broken, which beats an opaque crash at boot.
        logger.error("Startup warm-up failed: %s", exc)
    yield


app = FastAPI(
    title="Samsung Phone Query and Review System",
    description=(
        "Ask questions about Samsung smartphones, generate AI product reviews, "
        "and compare models. Data scraped from GSMArena; answers grounded in a "
        "local RAG pipeline over open-source models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _phone_summary(phone: Phone) -> dict:
    return {
        "id": phone.id,
        "name": phone.name,
        "release_year": phone.release_year,
        "display_size_inches": phone.display_size_inches,
        "chipset": phone.chipset,
        "main_camera_mp": phone.main_camera_mp,
        "battery_capacity_mah": phone.battery_capacity_mah,
        "url": phone.url,
    }


def _phone_detail(phone: Phone) -> dict:
    grouped: dict[str, dict[str, str]] = {}
    for spec in phone.specifications:
        grouped.setdefault(spec.category, {})[spec.name] = spec.value

    detail = _phone_summary(phone)
    detail.update(
        {
            field: getattr(phone, field)
            for field in (
                "announced", "status", "dimensions", "weight", "build",
                "display_type", "display_size", "display_resolution",
                "display_refresh_rate_hz", "display_protection", "os", "cpu",
                "gpu", "internal_memory", "max_ram_gb", "max_storage_gb",
                "main_camera", "main_camera_video", "selfie_camera",
                "selfie_camera_mp", "battery_type", "charging",
                "charging_watts", "colors",
            )
        }
    )
    detail["prices"] = [p.to_dict() for p in phone.prices]
    detail["specifications"] = grouped
    return detail


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Minimal browser client for trying the system without curl."""
    return INDEX_HTML


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(db: Session = Depends(get_db)) -> dict:
    db_ok, db_message = healthcheck()
    stats = database_stats(db) if db_ok else {}

    try:
        vector_stats = get_chatbot().vector_store.stats()
    except Exception as exc:
        vector_stats = {"ready": False, "error": str(exc)}

    healthy = db_ok and stats.get("phones", 0) > 0 and vector_stats.get("ready")
    return {
        "status": "healthy" if healthy else "degraded",
        "database": {"connected": db_ok, "message": db_message, **stats},
        "vector_store": vector_stats,
        "llm": llm_status(),
        "agents": backend_status(),
    }


@app.get("/stats", tags=["meta"])
def stats(db: Session = Depends(get_db)) -> dict:
    return {
        "database": database_stats(db),
        "models_tracked": config.TARGET_MODELS,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL if config.USE_LLM else None,
        "agent_framework": backend_status(),
    }


# --------------------------------------------------------------------------
# Chatbot
# --------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponseModel, tags=["chatbot"])
def chat(request: ChatRequest) -> dict:
    """Answer a natural-language question using RAG over the phone database."""
    try:
        response = get_chatbot().chat(request.query, top_k=request.top_k)
    except RuntimeError as exc:
        # Raised when the database is empty — a setup problem, not a bad request.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
    return response.to_dict()


@app.get("/chat", response_model=ChatResponseModel, tags=["chatbot"])
def chat_get(q: str = Query(..., min_length=1, description="Your question")) -> dict:
    """GET form of `/chat`, convenient for quick browser testing."""
    return chat(ChatRequest(query=q))


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
@app.post("/agents/review", response_model=ReviewResponse, tags=["agents"])
def agent_review(request: ReviewRequest) -> dict:
    """SpecificationAgent fetches the specs; ReviewAgent writes the review."""
    result = generate_review(request.phone, audience=request.audience)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["review"])
    return result


@app.post("/agents/compare", response_model=CompareResponse, tags=["agents"])
def agent_compare(request: CompareRequest) -> dict:
    """Fetch both spec sheets, then have ComparisonAgent deliver a verdict."""
    result = compare_phones(request.phone_a, request.phone_b, focus=request.focus)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["comparison"])
    return result


@app.post("/agents/query", tags=["agents"])
def agent_query(request: AgentQueryRequest) -> dict:
    """Route free text to the review or comparison crew automatically."""
    result = handle_request(request.query)
    if not result.get("success"):
        raise HTTPException(
            status_code=404, detail=result.get("error", "Could not handle that request")
        )
    return result


# --------------------------------------------------------------------------
# Phones
# --------------------------------------------------------------------------
@app.get("/phones", response_model=list[PhoneSummary], tags=["phones"])
def list_phones(db: Session = Depends(get_db)) -> list[dict]:
    return [_phone_summary(p) for p in get_all_phones(db)]


@app.get("/phones/search", response_model=list[PhoneSummary], tags=["phones"])
def search_phones(
    q: str = Query(..., min_length=1, description="Model name, even partial"),
    limit: int = Query(5, ge=1, le=15),
    db: Session = Depends(get_db),
) -> list[dict]:
    # Declared before /phones/{phone_id} so "search" is not read as an id.
    return [_phone_summary(p) for p in find_phones(db, q, limit=limit)]


@app.get("/phones/{phone_id}", response_model=PhoneDetail, tags=["phones"])
def phone_detail(phone_id: int, db: Session = Depends(get_db)) -> dict:
    phone = get_phone_by_id(db, phone_id)
    if phone is None:
        raise HTTPException(status_code=404, detail=f"No phone with id {phone_id}")
    return _phone_detail(phone)


@app.get("/phones/by-name/{name}", response_model=PhoneDetail, tags=["phones"])
def phone_by_name(name: str, db: Session = Depends(get_db)) -> dict:
    phone = find_phone(db, name)
    if phone is None:
        raise HTTPException(status_code=404, detail=f"No phone matching '{name}'")
    return _phone_detail(phone)
