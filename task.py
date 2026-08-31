"""Samsung Phone Query and Review System — one-command entry point.

    python task.py setup     # create the database and scrape GSMArena
    python task.py demo      # walk through all four subsystems
    python task.py chat      # interactive terminal chatbot
    python task.py api       # start the FastAPI server + web UI
    python task.py test      # run the test suite

Each subcommand is a thin wrapper over the matching script in `scripts/`.
See README.md for the full documentation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "setup": ("scripts/scrape.py", "Creating the database and scraping GSMArena"),
    "scrape": ("scripts/scrape.py", "Scraping GSMArena"),
    "demo": ("scripts/demo.py", "Running the full demonstration"),
    "chat": ("scripts/chat.py", "Starting the interactive chatbot"),
    "api": ("scripts/run_api.py", "Starting the API server"),
    "test": ("tests/test_system.py", "Running the test suite"),
}


def usage() -> int:
    print(__doc__)
    print("Available commands:", ", ".join(COMMANDS))
    return 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        return usage()

    command = sys.argv[1].lower()
    if command not in COMMANDS:
        print(f"Unknown command: {command!r}\n")
        return usage()

    script, description = COMMANDS[command]
    print(f"\n  {description}…\n")

    # Delegated as a subprocess so each script keeps its own argument parsing
    # and any extra flags pass straight through.
    return subprocess.call(
        [sys.executable, str(ROOT / script), *sys.argv[2:]], cwd=str(ROOT)
    )


if __name__ == "__main__":
    raise SystemExit(main())
