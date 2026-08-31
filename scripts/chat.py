"""Interactive terminal chatbot.

    python scripts/chat.py

Commands: `review <phone>`, `compare <a> vs <b>`, `phones`, `exit`.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.agents.crew import compare_phones, generate_review
from src.database.db import session_scope
from src.database.repository import get_all_phones
from src.rag.chatbot import SamsungChatbot

BANNER = """
==============================================================
  Samsung Phone Assistant
  Ask about specs, prices or comparisons.
  Commands: review <phone> | compare <a> vs <b> | phones | exit
==============================================================
"""


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    print(BANNER)

    bot = SamsungChatbot()
    print("  Loading models and index…")
    try:
        bot.prepare()
    except RuntimeError as exc:
        print(f"  {exc}")
        return 1
    print("  Ready.\n")

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not line:
            continue
        lowered = line.lower()

        if lowered in ("exit", "quit", "q"):
            print("Bye.")
            return 0

        if lowered == "phones":
            with session_scope() as session:
                for phone in get_all_phones(session):
                    print(f"  - {phone.name}")
            print()
            continue

        if lowered.startswith("review "):
            result = generate_review(line[7:].strip())
            print(f"\n{result['review']}\n")
            continue

        if lowered.startswith("compare "):
            body = line[8:].strip()
            parts = [p.strip() for p in re.split(r"\s+(?:vs\.?|and|versus)\s+", body, flags=re.I)]
            if len(parts) < 2:
                print("  Use:  compare Galaxy S23 vs Galaxy S22\n")
                continue
            result = compare_phones(parts[0], parts[1])
            if result.get("table"):
                print(f"\n{result['table']}")
            print(f"\n{result['comparison']}\n")
            continue

        response = bot.chat(line)
        print(f"\nBot> {response.answer}")
        print(f"     [{response.intent} · {response.generated_by}]\n")


if __name__ == "__main__":
    raise SystemExit(main())
