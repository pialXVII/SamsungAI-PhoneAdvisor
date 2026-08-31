"""Scrape -> validate -> persist pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import config
from src.database.db import init_db, session_scope
from src.database.repository import database_stats, upsert_phone

from .gsmarena_scraper import GSMArenaScraper, ScrapedPhone

logger = logging.getLogger(__name__)

# A record missing any of these is too thin to answer questions about, so it is
# reported rather than silently stored.
REQUIRED_FIELDS = ("name", "chipset", "display_size", "battery_type")


def validate(scraped: ScrapedPhone) -> list[str]:
    """Return the names of required fields this record is missing."""
    return [field for field in REQUIRED_FIELDS if not scraped.phone.get(field)]


def save_snapshot(records: list[ScrapedPhone]) -> str:
    """Write a JSON copy of the scrape next to the database.

    Re-scraping takes minutes and depends on GSMArena staying reachable; the
    snapshot lets the database be rebuilt offline.
    """
    path = config.DATA_DIR / "scraped_phones.json"
    payload = {
        "scraped_at": datetime.utcnow().isoformat(),
        "source": "GSMArena",
        "count": len(records),
        "phones": [
            {
                **record.phone,
                "raw_specifications": record.raw_specs,
                "prices": record.prices,
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Snapshot written to %s", path)
    return str(path)


def store(records: list[ScrapedPhone]) -> int:
    """Persist scraped records, replacing any existing row for the same model."""
    stored = 0
    with session_scope() as session:
        for record in records:
            missing = validate(record)
            if missing:
                logger.warning(
                    "%s is missing %s — storing anyway", record.name, ", ".join(missing)
                )
            upsert_phone(session, record.phone, record.raw_specs, record.prices)
            stored += 1
    return stored


def run(targets: list[str] | None = None, snapshot: bool = True) -> dict:
    """Full pipeline: create schema, scrape GSMArena, store, report."""
    init_db()

    with GSMArenaScraper() as scraper:
        records = scraper.scrape(targets)

    if not records:
        logger.error("Nothing scraped — check network access to gsmarena.com")
        return {"scraped": 0, "stored": 0, "stats": {}}

    if snapshot:
        save_snapshot(records)

    stored = store(records)

    with session_scope() as session:
        stats = database_stats(session)

    logger.info(
        "Stored %s phones (%s spec rows, %s price rows)",
        stats["phones"],
        stats["specifications"],
        stats["prices"],
    )
    return {"scraped": len(records), "stored": stored, "stats": stats}


def load_from_snapshot() -> dict:
    """Rebuild the database from `data/scraped_phones.json` without network access."""
    path = config.DATA_DIR / "scraped_phones.json"
    if not path.exists():
        raise FileNotFoundError(f"No snapshot at {path}; run the scraper first")

    payload = json.loads(path.read_text(encoding="utf-8"))
    init_db()

    with session_scope() as session:
        for entry in payload["phones"]:
            entry = dict(entry)
            raw_specs = entry.pop("raw_specifications", [])
            prices = entry.pop("prices", [])
            entry.pop("id", None)
            entry.pop("scraped_at", None)
            upsert_phone(session, entry, raw_specs, prices)

    with session_scope() as session:
        stats = database_stats(session)

    logger.info("Restored %s phones from snapshot", stats["phones"])
    return {"scraped": 0, "stored": len(payload["phones"]), "stats": stats}
