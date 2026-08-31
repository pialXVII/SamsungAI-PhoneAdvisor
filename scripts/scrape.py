"""CLI: scrape GSMArena and load the database.

    python scripts/scrape.py                 # scrape all target models
    python scripts/scrape.py --from-snapshot # rebuild from data/scraped_phones.json
    python scripts/scrape.py --reset         # drop and recreate tables first
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# GSMArena specs contain µ, ², € and similar; the default Windows console
# codepage (cp1252) raises UnicodeEncodeError on them mid-scrape.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import config
from src.database.db import init_db
from src.scraper.pipeline import load_from_snapshot, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Samsung phones from GSMArena")
    parser.add_argument(
        "--from-snapshot",
        action="store_true",
        help="Load data/scraped_phones.json instead of hitting the network",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Drop and recreate all tables first"
    )
    parser.add_argument(
        "--models", nargs="*", help="Override the target model list"
    )
    parser.add_argument("--no-snapshot", action="store_true", help="Skip writing JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.reset:
        init_db(drop_existing=True)

    if args.from_snapshot:
        result = load_from_snapshot()
    else:
        result = run(targets=args.models or config.TARGET_MODELS,
                     snapshot=not args.no_snapshot)

    stats = result.get("stats", {})
    print("\n" + "=" * 60)
    print(f"  Phones in database : {stats.get('phones', 0)}")
    print(f"  Specification rows : {stats.get('specifications', 0)}")
    print(f"  Price rows         : {stats.get('prices', 0)}")
    print("=" * 60)

    return 0 if stats.get("phones") else 1


if __name__ == "__main__":
    raise SystemExit(main())
