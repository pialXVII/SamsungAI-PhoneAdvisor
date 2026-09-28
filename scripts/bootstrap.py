"""Prepare a fresh environment, then optionally start the API server.

    python scripts/bootstrap.py            # seed the DB if empty, build the index
    python scripts/bootstrap.py --serve    # ...then serve on $PORT (default 8000)

This is the container entrypoint (see Dockerfile) and the CI seeding step, so a
brand-new environment goes from an empty volume to a serving API with no manual
steps:

1. Create the schema (`init_db`). Safe on an existing database.
2. If the `phones` table is empty, load `data/scraped_phones.json` through the
   same loader `scripts/scrape.py --from-snapshot` uses. It never contacts
   GSMArena: a container start must not depend on a third-party site being up,
   and a polite scrape takes minutes.
3. Load or build the FAISS index, so no request pays for embedding the corpus.
4. With `--serve`, hand over to uvicorn *in this process*. The chatbot singleton
   prepared in step 3 is then the one the app's startup hook picks up, so the
   embedding model and index are loaded once rather than twice.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import config

logger = logging.getLogger("bootstrap")


def _safe_database_url() -> str:
    """The configured URL with any password masked, for the startup log."""
    from sqlalchemy.engine import make_url

    try:
        return make_url(config.DATABASE_URL).render_as_string(hide_password=True)
    except Exception:  # an unparsable URL fails loudly later, in init_db
        return "<unparsable DATABASE_URL>"


def seed_database() -> int:
    """Create the schema and load the snapshot if the database is empty.

    Returns the number of phones in the database afterwards. An existing,
    non-empty database is left untouched, so restarting a container with a
    persistent volume never discards data (including a fresher live scrape).
    """
    from src.database.db import init_db, session_scope
    from src.database.repository import count_phones
    from src.scraper.pipeline import load_from_snapshot

    init_db()
    with session_scope() as session:
        existing = count_phones(session)

    if existing:
        logger.info("Database already holds %s phones — not reseeding", existing)
        return existing

    logger.info("Database is empty — loading data/scraped_phones.json")
    result = load_from_snapshot()
    return int(result.get("stats", {}).get("phones", 0))


def prepare_index() -> dict:
    """Load the cached FAISS index (or build it), then warm the embedder.

    Loading a cached index does not load the embedding model — that normally
    happens lazily on the first vector search, which made the first
    spec-lookup request after a restart take ~18 s instead of milliseconds.
    One throwaway search moves that cost here, before traffic arrives.
    """
    from src.rag.chatbot import get_chatbot

    store = get_chatbot().vector_store
    store.search("battery capacity", top_k=1)
    return store.stats()


def serve(host: str, port: int) -> None:
    import uvicorn

    # One worker on purpose: each worker process would load its own copy of the
    # embedding model (and the LLM when enabled), multiplying memory use.
    uvicorn.run("src.api.main:app", host=host, port=port, log_level="info")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the database and build the index")
    parser.add_argument("--serve", action="store_true", help="Start uvicorn afterwards")
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument(
        "--port",
        type=int,
        # PORT is the variable hosting platforms (Hugging Face Spaces, Render,
        # Fly) set; API_PORT remains the project's own setting.
        default=int(os.getenv("PORT") or config.API_PORT),
    )
    args = parser.parse_args()

    # As a container's PID 1 this process ignores SIGTERM unless it installs a
    # handler, so `docker stop` during seeding would hang for the full grace
    # period before SIGKILL. Uvicorn installs its own handler once serving.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # huggingface_hub logs one INFO line per HEAD request while resolving a
    # cached model — two dozen lines per start that bury the ones that matter.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Database      %s", _safe_database_url())
    logger.info("Vector index  %s", config.VECTOR_INDEX_PATH)
    logger.info(
        "USE_LLM=%s  AGENT_FRAMEWORK=%s  EMBEDDING_MODEL=%s",
        config.USE_LLM,
        config.AGENT_FRAMEWORK,
        config.EMBEDDING_MODEL,
    )

    started = time.perf_counter()
    try:
        phones = seed_database()
    except Exception:
        # Serving an empty database would only produce 503s on every question,
        # so fail the start and let the orchestrator surface the error.
        logger.exception("Could not prepare the database")
        return 1
    if not phones:
        logger.error("Database is still empty after seeding — refusing to start")
        return 1
    seeded_at = time.perf_counter()
    logger.info("Database ready: %s phones (%.1f s)", phones, seeded_at - started)

    try:
        stats = prepare_index()
    except Exception:
        logger.exception("Could not build the vector index")
        return 1
    logger.info(
        "Index ready: %s documents, %s vectors (%.1f s)",
        stats.get("documents"),
        stats.get("vectors"),
        time.perf_counter() - seeded_at,
    )

    if args.serve:
        serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
