"""End-to-end demonstration of all four subsystems.

    python scripts/demo.py              # everything
    python scripts/demo.py --part chat  # one section: db | chat | agents

Runs against whatever is already in the database, so `scripts/scrape.py` must
have been run at least once.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.agents.crew import compare_phones, generate_review
from src.database.db import session_scope
from src.database.repository import database_stats, get_all_phones, top_by_column
from src.rag.chatbot import SamsungChatbot

SAMPLE_QUERIES = [
    "What are the camera specs of the Samsung Galaxy S23?",
    "Which Samsung phone has the best battery life?",
    "How does the Galaxy S23 compare to the S22 in terms of performance?",
    "What is the screen size of the Galaxy S22?",
    "Which Samsung phone is the cheapest?",
    "Does the Galaxy S24 Ultra support wireless charging?",
]


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def wrap(text: str, indent: str = "  ") -> str:
    paragraphs = str(text).split("\n")
    return "\n".join(
        textwrap.fill(p, width=76, initial_indent=indent, subsequent_indent=indent)
        if p.strip()
        else ""
        for p in paragraphs
    )


def demo_database() -> None:
    rule("1. SCRAPED DATA IN THE DATABASE")
    with session_scope() as session:
        stats = database_stats(session)
        print(
            f"  {stats['phones']} phones | {stats['specifications']} spec rows "
            f"| {stats['prices']} price rows\n"
        )
        print(f"  {'Model':<28}{'Year':<7}{'Display':<10}{'Battery':<11}{'Camera'}")
        print("  " + "-" * 72)
        for phone in get_all_phones(session):
            print(
                f"  {phone.name:<28}{phone.release_year or '—':<7}"
                f"{str(phone.display_size_inches or '—') + chr(34):<10}"
                f"{str(phone.battery_capacity_mah or '—') + ' mAh':<11}"
                f"{phone.main_camera_mp or '—'} MP"
            )

        print("\n  Largest batteries:")
        for i, phone in enumerate(
            top_by_column(session, "battery_capacity_mah", limit=3), 1
        ):
            print(f"    {i}. {phone.name} — {phone.battery_capacity_mah} mAh")


def demo_chatbot() -> None:
    rule("2. RAG CHATBOT")
    bot = SamsungChatbot()
    bot.prepare()
    print(f"  Vector store: {bot.vector_store.stats()}\n")

    for query in SAMPLE_QUERIES:
        response = bot.chat(query)
        print(f"  Q: {query}")
        print(f"     [intent={response.intent}, via={response.generated_by}]")
        print(wrap(response.answer, indent="     "))
        if response.sources:
            refs = ", ".join(
                f"{s['phone']}/{s['aspect']}" for s in response.sources[:3]
            )
            print(f"     sources: {refs}")
        print()


def demo_agents() -> None:
    from src.agents.crew import backend_status

    rule("3. MULTI-AGENT SYSTEM")
    status = backend_status()
    print(f"  Orchestrator: {status['active']} "
          f"(CrewAI installed: {status['crewai_installed']})")
    if status["active"] == "crewai":
        print("  Note: a CrewAI review runs two agents and takes 1-2 minutes.")

    print("\n  --- Review crew: Data Specialist -> Reviewer ---\n")
    review = generate_review("Galaxy S24 Ultra", audience="a mobile photographer")
    for name, result in review["agents"].items():
        print(
            f"    [{name}] {result['agent']} "
            f"({result['duration_seconds']}s, llm={result['used_llm']})"
        )
    print(
        f"\n  Review of {review['phone']} "
        f"(via {review['generated_by']}, {review['framework']}):\n"
    )
    print(wrap(review["review"], indent="    "))

    print("\n\n  --- Comparison crew: Data Specialist -> Comparison Analyst ---\n")
    comparison = compare_phones("Galaxy S23", "Galaxy S22", focus="performance")
    print(textwrap.indent(comparison["table"], "    "))
    print(f"\n  Verdict (via {comparison['generated_by']}):\n")
    print(wrap(comparison["comparison"], indent="    "))


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo the whole system")
    parser.add_argument(
        "--part",
        choices=["db", "chat", "agents", "all"],
        default="all",
        help="Which section to run",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s | %(name)s | %(message)s",
    )

    with session_scope() as session:
        if database_stats(session)["phones"] == 0:
            print("The database is empty. Run:  python scripts/scrape.py")
            return 1

    if args.part in ("db", "all"):
        demo_database()
    if args.part in ("chat", "all"):
        demo_chatbot()
    if args.part in ("agents", "all"):
        demo_agents()

    rule("DEMO COMPLETE")
    print("  Start the API with:  python scripts/run_api.py")
    print("  Then open:           http://127.0.0.1:8000\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
